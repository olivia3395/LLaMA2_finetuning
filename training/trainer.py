"""
training/trainer.py — QLoRA SFT (Supervised Fine-Tuning) trainer.

Architecture
────────────
  QLoRATrainer wraps either:
    • trl.SFTTrainer   (preferred — packing, completion masking built in)
    • transformers.Trainer (fallback)

  Both operate the same training loop:
    for batch in dataloader:
        logits = model(input_ids, attention_mask)
        loss   = cross_entropy(logits[:, :-1], input_ids[:, 1:], ignore=-100)
        loss.backward()          # only LoRA params have gradients
        optimiser.step()
        lr_scheduler.step()

Key QLoRA-specific settings
────────────────────────────
  • optim="paged_adamw_32bit" : bitsandbytes paged optimiser keeps states
                                in CPU RAM when not needed → saves ~1 GB
  • gradient_checkpointing=True : recompute activations during backward
                                  → trades compute for memory (saves ~3 GB)
  • bf16=True                 : LoRA params and activations in bfloat16
  • max_grad_norm=0.3         : aggressive clipping prevents instability
                                with quantised weights
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from config import Config
from training.callbacks import (
    LoRACheckpointCallback,
    EarlyStoppingCallback,
    MemoryLogCallback,
    GradientStatsCallback,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TrainingArguments factory
# ---------------------------------------------------------------------------

def build_training_arguments(cfg: Config):
    """
    Build a HuggingFace TrainingArguments object from our TrainingConfig.
    Falls back to a simple namespace for unit testing.
    """
    try:
        from transformers import TrainingArguments

        eval_strategy = "no"
        if cfg.training.evaluation_strategy != "no":
            eval_strategy = cfg.training.evaluation_strategy

        return TrainingArguments(
            output_dir                  = cfg.training.output_dir,
            num_train_epochs            = cfg.training.num_train_epochs,
            max_steps                   = cfg.training.max_steps,
            per_device_train_batch_size = cfg.training.per_device_train_batch_size,
            per_device_eval_batch_size  = cfg.training.per_device_eval_batch_size,
            gradient_accumulation_steps = cfg.training.gradient_accumulation_steps,
            optim                       = cfg.training.optim,
            learning_rate               = cfg.training.learning_rate,
            weight_decay                = cfg.training.weight_decay,
            max_grad_norm               = cfg.training.max_grad_norm,
            lr_scheduler_type           = cfg.training.lr_scheduler_type,
            warmup_ratio                = cfg.training.warmup_ratio,
            warmup_steps                = cfg.training.warmup_steps,
            fp16                        = cfg.training.fp16,
            bf16                        = cfg.training.bf16,
            gradient_checkpointing      = cfg.training.gradient_checkpointing,
            logging_steps               = cfg.training.logging_steps,
            evaluation_strategy         = eval_strategy,
            eval_steps                  = cfg.training.eval_steps,
            save_strategy               = cfg.training.save_strategy,
            save_steps                  = cfg.training.save_steps,
            save_total_limit            = cfg.training.save_total_limit,
            load_best_model_at_end      = cfg.training.load_best_model_at_end,
            group_by_length             = cfg.training.group_by_length,
            report_to                   = cfg.training.report_to,
            seed                        = cfg.training.seed,
        )
    except ImportError:
        return _FakeTrainingArguments(cfg)


class _FakeTrainingArguments:
    """Minimal stand-in for unit tests without transformers installed."""
    def __init__(self, cfg: Config):
        t = cfg.training
        self.output_dir = t.output_dir
        self.num_train_epochs = t.num_train_epochs
        self.max_steps = t.max_steps
        self.per_device_train_batch_size = t.per_device_train_batch_size
        self.gradient_accumulation_steps = t.gradient_accumulation_steps
        self.learning_rate = t.learning_rate
        self.weight_decay  = t.weight_decay
        self.max_grad_norm = t.max_grad_norm
        self.logging_steps = t.logging_steps
        self.seed = t.seed
        self.bf16 = t.bf16
        self.fp16 = t.fp16


# ---------------------------------------------------------------------------
# QLoRA Trainer
# ---------------------------------------------------------------------------

class QLoRATrainer:
    """
    High-level wrapper that constructs and runs the QLoRA training pipeline.

    Usage:
        trainer = QLoRATrainer(model, tokenizer, train_ds, eval_ds, cfg)
        result  = trainer.train()
        trainer.save()
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        train_dataset,
        eval_dataset,
        cfg: Config,
        data_collator=None,
    ):
        self.model         = model
        self.tokenizer     = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset  = eval_dataset
        self.cfg           = cfg
        self.data_collator = data_collator
        self._trainer      = None

        os.makedirs(cfg.training.output_dir, exist_ok=True)

    def _build_callbacks(self) -> list:
        cbs = [
            MemoryLogCallback(),
            GradientStatsCallback(log_every_n_steps=50),
            LoRACheckpointCallback(
                self.cfg.training.output_dir,
                save_total_limit=self.cfg.training.save_total_limit,
            ),
        ]
        return cbs

    def _build_trainer_trl(self, training_args):
        """Build trl.SFTTrainer (preferred)."""
        from trl import SFTTrainer
        from data.dataset import InstructionDataCollator

        collator = self.data_collator or InstructionDataCollator(self.tokenizer)

        return SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=collator,
            args=training_args,
            peft_config=None,          # LoRA already attached
            max_seq_length=self.cfg.model.max_seq_length,
            dataset_text_field=None,   # we pre-tokenise
            callbacks=self._build_callbacks(),
        )

    def _build_trainer_hf(self, training_args):
        """Build transformers.Trainer (fallback)."""
        from transformers import Trainer
        from data.dataset import InstructionDataCollator

        collator = self.data_collator or InstructionDataCollator(self.tokenizer)

        return Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=collator,
            args=training_args,
            callbacks=self._build_callbacks(),
        )

    def setup(self):
        """Build the underlying HF Trainer (call before train())."""
        training_args = build_training_arguments(self.cfg)
        try:
            self._trainer = self._build_trainer_trl(training_args)
            logger.info("Using trl.SFTTrainer")
        except ImportError:
            self._trainer = self._build_trainer_hf(training_args)
            logger.info("Using transformers.Trainer (trl not installed)")
        return self

    def train(self) -> Dict:
        """Run training and return result dict."""
        if self._trainer is None:
            self.setup()

        logger.info("Starting QLoRA training …")
        result = self._trainer.train()
        logger.info(f"Training complete: {result}")
        return result

    def evaluate(self) -> Dict:
        """Run evaluation and return metrics."""
        if self._trainer is None:
            self.setup()
        return self._trainer.evaluate()

    def save(self, output_dir: Optional[str] = None):
        """Save LoRA adapter weights."""
        out = output_dir or os.path.join(self.cfg.training.output_dir, "final_adapter")
        os.makedirs(out, exist_ok=True)
        try:
            self.model.save_pretrained(out)
            self.tokenizer.save_pretrained(out)
        except Exception:
            trainable = {n: p.detach().cpu() for n, p in self.model.named_parameters()
                         if p.requires_grad}
            torch.save(trainable, os.path.join(out, "lora_weights.pt"))
        logger.info(f"Adapter saved → {out}")

    # ── Manual training loop (used when HF Trainer unavailable) ──────────

    def train_manual(
        self,
        num_steps: int = 10,
        log_every: int = 1,
    ) -> List[Dict]:
        """
        Minimal training loop for unit testing without transformers.

        Returns list of per-step loss dicts.
        """
        from torch.utils.data import DataLoader
        from data.dataset import InstructionDataCollator

        device = next(self.model.parameters()).device
        collator = InstructionDataCollator(self.tokenizer)
        loader   = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collator,
        )

        optimiser = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )

        self.model.train()
        history = []
        data_iter = iter(loader)

        for step in range(num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            batch = {k: v.to(device) for k, v in batch.items()}
            optimiser.zero_grad()
            out  = self.model(**batch)
            loss = out.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.cfg.training.max_grad_norm,
            )
            optimiser.step()

            rec = {"step": step + 1, "loss": loss.item()}
            history.append(rec)
            if (step + 1) % log_every == 0:
                logger.info(f"  step {step+1}/{num_steps}  loss={loss.item():.4f}")

        return history
