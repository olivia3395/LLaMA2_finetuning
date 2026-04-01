"""
training/callbacks.py — Custom training callbacks.

Callbacks implemented
─────────────────────
  LoRACheckpointCallback   : saves only adapter weights at each checkpoint
  EarlyStoppingCallback    : stops training when eval loss stops improving
  MemoryLogCallback        : logs GPU VRAM usage at each logging step
  GradientStatsCallback    : logs gradient norms for LoRA params
  GenerationSampleCallback : generates samples periodically (qualitative check)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base callback shim (works without transformers.TrainerCallback import)
# ---------------------------------------------------------------------------

class Callback:
    """Minimal callback base class — matches HuggingFace TrainerCallback API."""

    def on_log(self, args, state, control, logs=None, **kwargs):        pass
    def on_save(self, args, state, control, **kwargs):                  pass
    def on_evaluate(self, args, state, control, metrics=None, **kwargs): pass
    def on_step_end(self, args, state, control, **kwargs):              pass
    def on_train_end(self, args, state, control, **kwargs):             pass
    def on_train_begin(self, args, state, control, **kwargs):           pass


# ---------------------------------------------------------------------------
# LoRA checkpoint callback
# ---------------------------------------------------------------------------

class LoRACheckpointCallback(Callback):
    """
    Saves only the LoRA adapter weights at each checkpoint.

    This is much faster than saving the full quantised model and produces
    much smaller checkpoint files (~30 MB vs 3.5 GB for a 7B model).
    """

    def __init__(self, output_dir: str, save_total_limit: int = 3):
        self.output_dir      = output_dir
        self.save_total_limit = save_total_limit
        self._saved: List[str] = []

    def on_save(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        step_dir = os.path.join(self.output_dir, f"adapter-step-{state.global_step}")
        os.makedirs(step_dir, exist_ok=True)

        try:
            model.save_pretrained(step_dir)
        except Exception:
            # Fallback: save trainable params directly
            trainable = {n: p.detach().cpu() for n, p in model.named_parameters()
                         if p.requires_grad}
            torch.save(trainable, os.path.join(step_dir, "lora_weights.pt"))

        self._saved.append(step_dir)
        logger.info(f"LoRA adapter checkpoint → {step_dir}")

        # Prune oldest checkpoints
        while len(self._saved) > self.save_total_limit:
            old = self._saved.pop(0)
            import shutil
            shutil.rmtree(old, ignore_errors=True)
            logger.info(f"Removed old checkpoint: {old}")


# ---------------------------------------------------------------------------
# Early stopping callback
# ---------------------------------------------------------------------------

class EarlyStoppingCallback(Callback):
    """
    Stop training when eval loss has not improved for `patience` evals.

    Args:
        patience       : number of evaluations without improvement before stopping
        min_delta      : minimum improvement to count as "better"
        monitor        : metric to monitor (default: "eval_loss")
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        monitor: str = "eval_loss",
    ):
        self.patience  = patience
        self.min_delta = min_delta
        self.monitor   = monitor
        self._best     = float("inf")
        self._no_improve = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        value = metrics.get(self.monitor, None)
        if value is None:
            return

        if value < self._best - self.min_delta:
            self._best      = value
            self._no_improve = 0
            logger.info(f"EarlyStopping: new best {self.monitor}={value:.4f}")
        else:
            self._no_improve += 1
            logger.info(
                f"EarlyStopping: no improvement {self._no_improve}/{self.patience}"
            )
            if self._no_improve >= self.patience:
                logger.info("EarlyStopping: stopping training")
                control.should_training_stop = True


# ---------------------------------------------------------------------------
# GPU memory logging callback
# ---------------------------------------------------------------------------

class MemoryLogCallback(Callback):
    """Logs peak GPU memory at each logging step."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self._peak_gb = 0.0

    def on_step_end(self, args, state, control, **kwargs):
        if self.device.type != "cuda":
            return
        allocated = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
        self._peak_gb = max(self._peak_gb, allocated)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and self.device.type == "cuda":
            logs["peak_gpu_gb"] = round(self._peak_gb, 3)
            torch.cuda.reset_peak_memory_stats(self.device)
            self._peak_gb = 0.0


# ---------------------------------------------------------------------------
# Gradient stats callback
# ---------------------------------------------------------------------------

class GradientStatsCallback(Callback):
    """
    Logs gradient norm for LoRA parameters.
    Useful for diagnosing training instability (vanishing/exploding gradients).
    """

    def __init__(self, log_every_n_steps: int = 50):
        self.log_every = log_every_n_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        if state.global_step % self.log_every != 0:
            return

        lora_grads = [
            p.grad.detach().norm().item()
            for n, p in model.named_parameters()
            if "lora_" in n and p.grad is not None
        ]
        if lora_grads:
            import statistics
            logger.info(
                f"  Step {state.global_step} | "
                f"LoRA grad norm  mean={statistics.mean(lora_grads):.4f}  "
                f"max={max(lora_grads):.4f}  "
                f"min={min(lora_grads):.4f}"
            )


# ---------------------------------------------------------------------------
# Generation sample callback
# ---------------------------------------------------------------------------

class GenerationSampleCallback(Callback):
    """
    Generates sample completions at periodic intervals.

    Provides a qualitative check that the model is learning the right format.
    """

    def __init__(
        self,
        tokenizer,
        formatter,
        eval_every_n_steps: int = 100,
        prompts: Optional[List[str]] = None,
        max_new_tokens: int = 128,
        device: Optional[torch.device] = None,
    ):
        self.tokenizer   = tokenizer
        self.formatter   = formatter
        self.eval_every  = eval_every_n_steps
        self.max_new_tokens = max_new_tokens
        self.device      = device
        self.prompts     = prompts or [
            "Explain what a neural network is in simple terms.",
            "Write a Python function to reverse a string.",
            "What is the capital of Japan?",
        ]

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        if state.global_step % self.eval_every != 0:
            return

        model.eval()
        print(f"\n{'─'*56}")
        print(f"  Generation samples @ step {state.global_step}")
        print(f"{'─'*56}")

        for prompt in self.prompts[:2]:   # show 2 samples
            formatted = self.formatter.format_inference(prompt)
            inputs = self.tokenizer(
                formatted, return_tensors="pt", truncation=True, max_length=256
            )
            input_ids = inputs["input_ids"]
            if self.device:
                input_ids = input_ids.to(self.device)

            with torch.no_grad():
                try:
                    output = model.generate(
                        input_ids,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    gen_ids = output[0, input_ids.shape[1]:]
                    text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                    print(f"\n  Prompt: {prompt[:60]!r}")
                    print(f"  Response: {text[:200]!r}")
                except Exception as e:
                    print(f"  [generation failed: {e}]")

        print(f"{'─'*56}\n")
        model.train()
