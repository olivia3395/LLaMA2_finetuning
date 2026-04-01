"""
config.py — Central configuration for QLoRA instruction fine-tuning.

Design philosophy
─────────────────
All hyperparameters live here as typed dataclasses.  This gives:
  • IDE auto-complete and type checking
  • Easy serialisation to JSON for experiment tracking
  • Clean CLI override in train.py via argparse

QLoRA pipeline summary
──────────────────────
  1. Load LLaMA 2 in 4-bit NF4 (bitsandbytes)     ← ModelConfig
  2. Attach LoRA adapters to q/v projections        ← LoRAConfig
  3. Format instruction dataset                     ← DataConfig
  4. Train with gradient checkpointing              ← TrainingConfig
  5. Merge LoRA → fp16 and export                   ← (merge_and_export.py)

Consumer GPU targets
────────────────────
  RTX 3090 / 4090 (24 GB) : LLaMA-2-7B  — fits easily
  RTX 3080 (10 GB)        : LLaMA-2-7B  — fits with small batch
  RTX 2080 Ti (11 GB)     : LLaMA-2-7B  — fits with batch=1
  M2 MacBook Pro (16 GB)  : LLaMA-2-7B  — MPS backend, batch=1
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """LLaMA 2 loading and 4-bit quantisation settings."""

    # HuggingFace model identifier
    model_name: str = "meta-llama/Llama-2-7b-hf"

    # ── 4-bit NF4 quantisation (bitsandbytes) ──────────────────────────
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"   # compute dtype during forward
    bnb_4bit_quant_type: str = "nf4"            # Normal Float 4 (best quality)
    bnb_4bit_use_double_quant: bool = True       # nested quantisation (saves ~0.4GB)

    # ── Dtype for non-quantised layers ─────────────────────────────────
    torch_dtype: str = "bfloat16"

    # ── Context window ──────────────────────────────────────────────────
    max_seq_length: int = 2048

    # ── Trust remote code (needed for some community models) ────────────
    trust_remote_code: bool = False

    # ── Flash Attention 2 (requires Ampere+ GPU, pip install flash-attn) ─
    use_flash_attention_2: bool = False

    # ── HuggingFace token (needed for Llama-2 gated models) ─────────────
    hf_token: Optional[str] = None


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    """
    Low-Rank Adaptation (LoRA) hyperparameters.

    LoRA parameterises weight updates as: ΔW = B · A
    where A ∈ R^{r×k}, B ∈ R^{d×r}, r << min(d,k)

    Parameters
    ──────────
    r           : rank of the update matrices (8–64 typical)
    lora_alpha  : scaling factor; effective lr ≈ α/r
    target_modules: which Linear layers to add adapters to
    lora_dropout: dropout on adapter outputs (regularisation)
    bias        : whether to train biases ("none"|"all"|"lora_only")
    task_type   : PEFT task type (CAUSAL_LM for instruction tuning)

    Memory impact (7B model, r=16)
    ──────────────────────────────
    LoRA params per layer : 2 × r × d_model = 2 × 16 × 4096 = 131K
    Total trainable       : ~40 layers × 4 modules × 131K ≈ 21M
    vs Base params        : 7,000M
    Trainable fraction    : ~0.3%
    """
    r: int = 16
    lora_alpha: int = 32               # scaling: α/r = 2.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    inference_mode: bool = False


# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Dataset and prompt formatting settings."""

    # Dataset (HF Hub id or "alpaca" | "sharegpt" | "custom")
    dataset_name: str = "tatsu-lab/alpaca"

    # Dataset split
    train_split: str = "train"
    eval_split: Optional[str] = None    # None → no eval split

    # Fraction of training data to use (1.0 = all)
    train_fraction: float = 1.0

    # Prompt template
    # "alpaca"   → instruction/input/output format
    # "chatml"   → <|im_start|>system/user/assistant
    # "llama2"   → [INST] ... [/INST]
    # "simple"   → "### Instruction:\n...\n### Response:\n"
    prompt_style: str = "alpaca"

    # System prompt prepended to every conversation
    system_prompt: str = (
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible."
    )

    # Mask prompt tokens in loss (only train on completions)
    train_on_completions_only: bool = True

    # Eval size carved from training data if no eval_split
    eval_fraction: float = 0.05

    # Max samples to use (None = all)
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = 200

    # Number of dataloader workers
    num_workers: int = 4


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Hugging Face TrainingArguments-compatible settings."""

    # Output directory
    output_dir: str = "outputs/qlora"

    # ── Steps / epochs ──────────────────────────────────────────────────
    num_train_epochs: int = 3
    max_steps: int = -1            # -1 → use epochs; >0 overrides epochs

    # ── Batch size ───────────────────────────────────────────────────────
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4   # effective batch = 4×4 = 16

    # ── Optimiser ────────────────────────────────────────────────────────
    optim: str = "paged_adamw_32bit"       # bitsandbytes paged optimiser
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3

    # ── LR scheduler ─────────────────────────────────────────────────────
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    warmup_steps: int = 0          # 0 → use warmup_ratio

    # ── Precision ────────────────────────────────────────────────────────
    fp16: bool = False
    bf16: bool = True              # bfloat16 preferred on Ampere+ GPUs

    # ── Gradient checkpointing ────────────────────────────────────────────
    gradient_checkpointing: bool = True

    # ── Logging ──────────────────────────────────────────────────────────
    logging_steps: int = 10
    report_to: str = "none"        # "wandb" | "tensorboard" | "none"

    # ── Evaluation ───────────────────────────────────────────────────────
    evaluation_strategy: str = "steps"
    eval_steps: int = 100

    # ── Checkpointing ────────────────────────────────────────────────────
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = False

    # ── Reproducibility ──────────────────────────────────────────────────
    seed: int = 42

    # ── Packing ──────────────────────────────────────────────────────────
    # Concat short samples to fill max_seq_length (higher GPU utilisation)
    packing: bool = False

    # ── Group by length ──────────────────────────────────────────────────
    group_by_length: bool = True

    # ── W&B ──────────────────────────────────────────────────────────────
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Top-level config — aggregates all sub-configs."""

    model: ModelConfig    = field(default_factory=ModelConfig)
    lora: LoRAConfig      = field(default_factory=LoRAConfig)
    data: DataConfig      = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def summary(self) -> str:
        lines = ["=" * 64, "  QLoRA Fine-Tuning — Configuration", "=" * 64]
        for sec in ("model", "lora", "data", "training"):
            obj = getattr(self, sec)
            lines.append(f"\n[{sec.upper()}]")
            for k, v in vars(obj).items():
                lines.append(f"  {k:<35} {v}")
        lines.append("=" * 64)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        import dataclasses
        return {
            sec: dataclasses.asdict(getattr(self, sec))
            for sec in ("model", "lora", "data", "training")
        }


# ---------------------------------------------------------------------------
# Preset factories
# ---------------------------------------------------------------------------

def alpaca_7b_config() -> Config:
    """Standard Alpaca instruction tuning on LLaMA-2-7B."""
    cfg = Config()
    cfg.data.dataset_name  = "tatsu-lab/alpaca"
    cfg.data.prompt_style  = "alpaca"
    cfg.training.num_train_epochs = 3
    cfg.training.per_device_train_batch_size = 4
    cfg.training.gradient_accumulation_steps = 4
    return cfg


def sharegpt_7b_config() -> Config:
    """Multi-turn ChatGPT-style tuning on ShareGPT dataset."""
    cfg = Config()
    cfg.data.dataset_name  = "anon8231489123/ShareGPT_Vicuna_unfiltered"
    cfg.data.prompt_style  = "chatml"
    cfg.lora.r             = 32
    cfg.lora.lora_alpha    = 64
    cfg.training.num_train_epochs = 2
    return cfg


def fast_test_config() -> Config:
    """Tiny config for CI / unit testing (no GPU needed)."""
    cfg = Config()
    cfg.model.model_name   = "__synthetic__"
    cfg.model.load_in_4bit = False
    cfg.data.dataset_name  = "__synthetic__"
    cfg.data.max_train_samples = 32
    cfg.data.max_eval_samples  = 8
    cfg.training.num_train_epochs = 1
    cfg.training.max_steps = 2
    cfg.training.per_device_train_batch_size = 2
    cfg.training.gradient_accumulation_steps = 1
    cfg.training.logging_steps = 1
    cfg.training.eval_steps  = 1
    cfg.training.save_steps  = 1
    cfg.training.output_dir  = "/tmp/qlora_test"
    cfg.training.bf16 = False
    cfg.training.gradient_checkpointing = False
    cfg.lora.r = 4
    cfg.lora.lora_alpha = 8
    return cfg
