"""
scripts/train.py — Main QLoRA fine-tuning entry point.

Usage
─────
  # Train LLaMA-2-7B on Alpaca dataset (standard)
  python scripts/train.py

  # Custom model and dataset
  python scripts/train.py --model meta-llama/Llama-2-13b-hf --dataset tatsu-lab/alpaca

  # Multi-turn chat tuning (ShareGPT)
  python scripts/train.py --dataset anon8231489123/ShareGPT_Vicuna_unfiltered --prompt-style chatml

  # Fast offline test (no download, no GPU)
  python scripts/train.py --synthetic --max-steps 2

  # Custom LoRA rank and learning rate
  python scripts/train.py --lora-r 32 --lr 1e-4 --epochs 2

  # Resume from checkpoint
  python scripts/train.py --resume outputs/qlora/adapter-step-200

CLI flags
─────────
  --model           HF model name         (default: meta-llama/Llama-2-7b-hf)
  --dataset         HF dataset name       (default: tatsu-lab/alpaca)
  --prompt-style    alpaca/chatml/llama2   (default: alpaca)
  --lora-r          LoRA rank              (default: 16)
  --lora-alpha      LoRA alpha             (default: 32)
  --epochs          num_train_epochs       (default: 3)
  --max-steps       overrides epochs      (-1 = use epochs)
  --batch-size      per_device_batch_size  (default: 4)
  --grad-accum      gradient_accumulation  (default: 4)
  --lr              learning_rate          (default: 2e-4)
  --seq-len         max_seq_length         (default: 2048)
  --output-dir      checkpoint dir         (default: outputs/qlora)
  --wandb           W&B project name
  --synthetic       use synthetic data    (offline testing)
  --no-4bit         disable 4-bit loading (for CPU testing)
  --hf-token        HuggingFace API token
  --seed            random seed            (default: 42)
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="QLoRA instruction fine-tuning for LLaMA 2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",       default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--dataset",     default="tatsu-lab/alpaca")
    p.add_argument("--prompt-style",default="alpaca",
                   choices=["alpaca","chatml","llama2","simple"])
    p.add_argument("--lora-r",      type=int,   default=16)
    p.add_argument("--lora-alpha",  type=int,   default=32)
    p.add_argument("--lora-dropout",type=float, default=0.05)
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--max-steps",   type=int,   default=-1)
    p.add_argument("--batch-size",  type=int,   default=4)
    p.add_argument("--grad-accum",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--seq-len",     type=int,   default=2048)
    p.add_argument("--output-dir",  default="outputs/qlora")
    p.add_argument("--wandb",       default=None, help="W&B project name")
    p.add_argument("--synthetic",   action="store_true",
                   help="Offline test with synthetic model & data")
    p.add_argument("--no-4bit",     action="store_true",
                   help="Disable 4-bit loading (for CPU/MPS)")
    p.add_argument("--no-grad-ckpt",action="store_true",
                   help="Disable gradient checkpointing")
    p.add_argument("--bf16",        action="store_true", default=True)
    p.add_argument("--fp16",        action="store_true", default=False)
    p.add_argument("--hf-token",    default=None)
    p.add_argument("--max-samples", type=int,   default=None,
                   help="Max training samples (None = all)")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--resume",      default=None,
                   help="Resume training from adapter checkpoint directory")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    # ── Build config ─────────────────────────────────────────────────────
    from config import Config, ModelConfig, LoRAConfig, DataConfig, TrainingConfig

    if args.synthetic:
        from config import fast_test_config
        cfg = fast_test_config()
        logger.info("Running in synthetic/offline mode")
    else:
        cfg = Config(
            model=ModelConfig(
                model_name=args.model,
                load_in_4bit=not args.no_4bit,
                max_seq_length=args.seq_len,
                hf_token=args.hf_token,
            ),
            lora=LoRAConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            ),
            data=DataConfig(
                dataset_name=args.dataset,
                prompt_style=args.prompt_style,
                max_train_samples=args.max_samples,
            ),
            training=TrainingConfig(
                output_dir=args.output_dir,
                num_train_epochs=args.epochs,
                max_steps=args.max_steps,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_accum,
                learning_rate=args.lr,
                bf16=args.bf16 and not args.fp16,
                fp16=args.fp16,
                gradient_checkpointing=not args.no_grad_ckpt,
                report_to="wandb" if args.wandb else "none",
                wandb_project=args.wandb,
                seed=args.seed,
            ),
        )

    print(cfg.summary())

    # ── W&B ──────────────────────────────────────────────────────────────
    if cfg.training.wandb_project:
        try:
            import wandb
            wandb.init(
                project=cfg.training.wandb_project,
                name=cfg.training.wandb_run_name,
                config=cfg.to_dict(),
            )
        except ImportError:
            logger.warning("wandb not installed")

    # ── Device ───────────────────────────────────────────────────────────
    device = (
        "cuda"  if torch.cuda.is_available() else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    logger.info(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    if args.synthetic:
        from model.loader import load_synthetic_model_and_tokenizer
        model, tokenizer = load_synthetic_model_and_tokenizer(device)
    else:
        from model.loader import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(cfg.model)

    # ── Attach LoRA ───────────────────────────────────────────────────────
    if args.synthetic:
        from model.lora import attach_lora_synthetic
        model = attach_lora_synthetic(model, cfg.lora)
    else:
        from model.lora import attach_lora
        model = attach_lora(model, cfg.lora)

    # Resume from checkpoint
    if args.resume:
        logger.info(f"Loading adapter from checkpoint: {args.resume}")
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.resume)
        except Exception as e:
            logger.warning(f"Could not load adapter: {e}")

    # ── Print model summary ───────────────────────────────────────────────
    from model.utils import print_model_summary
    print_model_summary(model, label="QLoRA Model")

    # ── Print memory estimate ─────────────────────────────────────────────
    from model.utils import estimate_training_memory_gb
    n_params = sum(p.numel() for p in model.parameters())
    mem = estimate_training_memory_gb(
        n_params,
        batch_size=cfg.training.per_device_train_batch_size,
        seq_len=cfg.model.max_seq_length,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        r=cfg.lora.r,
    )
    logger.info(
        f"Estimated training memory:\n"
        + "\n".join(f"  {k:<20} {v:.3f} GB" for k, v in mem.items())
    )

    # ── Build datasets ────────────────────────────────────────────────────
    from data.dataset import build_datasets
    train_ds, eval_ds = build_datasets(cfg, tokenizer)
    logger.info(
        f"Dataset: {len(train_ds)} train samples"
        + (f", {len(eval_ds)} eval samples" if eval_ds else "")
    )

    # ── Build trainer ─────────────────────────────────────────────────────
    from training.trainer import QLoRATrainer
    trainer = QLoRATrainer(model, tokenizer, train_ds, eval_ds, cfg)

    # ── Train ─────────────────────────────────────────────────────────────
    if args.synthetic:
        # Lightweight manual loop for offline testing
        n_steps = cfg.training.max_steps if cfg.training.max_steps > 0 else 2
        logger.info(f"Synthetic training loop ({n_steps} steps)")
        history = trainer.train_manual(num_steps=n_steps, log_every=1)
        logger.info(f"Final loss: {history[-1]['loss']:.4f}")
    else:
        trainer.setup()
        trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────
    trainer.save()
    logger.info(f"\n✅  Training complete! Adapter saved to: {cfg.training.output_dir}")


if __name__ == "__main__":
    main()
