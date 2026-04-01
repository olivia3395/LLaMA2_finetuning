"""
model/lora.py — LoRA adapter injection via PEFT.

LoRA (Low-Rank Adaptation) Mathematics
───────────────────────────────────────
For a frozen pre-trained weight matrix W₀ ∈ R^{d×k}, LoRA parameterises
the weight update as a low-rank decomposition:

    h = W₀x + ΔWx = W₀x + (BA)x

where:
    B ∈ R^{d×r}    (initialised to zeros)
    A ∈ R^{r×k}    (initialised with Kaiming uniform)
    r << min(d, k)  (rank, typically 4–64)

During forward:  output = W₀x + (α/r) · BAx
  - W₀ is frozen (4-bit quantised in QLoRA)
  - Only B, A are trained (full fp32/bf16 precision)
  - α/r is the effective learning rate scaling factor

Parameter efficiency (7B model, r=16)
──────────────────────────────────────
  Targeted Linear layers per transformer block:
    q_proj, k_proj, v_proj, o_proj (attention)
    gate_proj, up_proj, down_proj  (MLP)
    Total: 7 layers × 32 blocks = 224 LoRA matrices

  Parameters per LoRA matrix pair (d=4096, r=16):
    A : r × k   =  16 × 4096 = 65,536
    B : d × r   = 4096 × 16  = 65,536
    Per pair    = 131,072

  Total trainable params ≈ 224 × 131,072 ≈ 29.4M
  vs. Base model params  = 7,000M
  Trainable fraction     = 0.42%

Memory during training
──────────────────────
  Frozen 4-bit base:  ~3.5 GB
  LoRA params (bf16): ~0.05 GB
  Gradients (bf16):   ~0.05 GB
  Optimiser states:   ~0.2  GB  (paged AdamW)
  Activations:        ~2–4  GB  (depends on batch/seq length)
  Total:              ~6–8  GB  → fits RTX 3080 (10 GB)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attach LoRA adapters
# ---------------------------------------------------------------------------

def attach_lora(model: nn.Module, lora_cfg) -> nn.Module:
    """
    Wrap a 4-bit quantised model with LoRA adapters using PEFT.

    Args:
        model    : pre-loaded (4-bit) base model
        lora_cfg : LoRAConfig with r, alpha, target_modules, etc.

    Returns:
        PeftModel with trainable LoRA adapters attached
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise ImportError(
            "Install PEFT: pip install peft"
        )

    task_type_map = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        "FEATURE_EXTRACTION": TaskType.FEATURE_EXTRACTION,
    }
    task_type = task_type_map.get(lora_cfg.task_type, TaskType.CAUSAL_LM)

    config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=lora_cfg.target_modules,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=task_type,
        inference_mode=lora_cfg.inference_mode,
    )

    model = get_peft_model(model, config)

    # Log trainable parameter summary
    _log_trainable_params(model)

    return model


def attach_lora_synthetic(model: nn.Module, lora_cfg) -> nn.Module:
    """
    Lightweight LoRA attachment for unit testing (no PEFT dependency).

    Injects trainable A, B matrices alongside the model's linear layers
    using a simple wrapper — not full PEFT, but sufficient for testing
    the training loop end-to-end.
    """
    r     = lora_cfg.r
    alpha = lora_cfg.lora_alpha

    wrapped = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Inject A, B as extra parameters (don't replace the module)
            d, k = module.weight.shape
            module.lora_A = nn.Parameter(torch.randn(r, k) * 0.01)
            module.lora_B = nn.Parameter(torch.zeros(d, r))
            module.lora_scale = alpha / r

            # Monkey-patch forward
            original_forward = module.forward
            def _lora_forward(x, _module=module, _orig=original_forward):
                return _orig(x) + (x @ _module.lora_A.T @ _module.lora_B.T) * _module.lora_scale
            module.forward = _lora_forward

            # Freeze original weight
            module.weight.requires_grad_(False)
            wrapped += 1

    logger.info(f"Synthetic LoRA: wrapped {wrapped} Linear layers (r={r})")
    return model


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _log_trainable_params(model: nn.Module):
    """Print trainable vs total parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    pct       = 100.0 * trainable / max(total, 1)
    logger.info(
        f"LoRA adapters attached:\n"
        f"  Trainable params : {trainable:>15,}  ({pct:.4f}%)\n"
        f"  Frozen params    : {total - trainable:>15,}\n"
        f"  Total params     : {total:>15,}"
    )


def get_trainable_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a dict of all trainable (LoRA) parameters."""
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def freeze_base_model(model: nn.Module):
    """Freeze all non-LoRA parameters (useful for manual setup)."""
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad_(False)


def unfreeze_base_model(model: nn.Module):
    """Unfreeze all parameters (for full fine-tuning or merging)."""
    for param in model.parameters():
        param.requires_grad_(True)


# ---------------------------------------------------------------------------
# LoRA weight inspection
# ---------------------------------------------------------------------------

def lora_weight_stats(model: nn.Module) -> List[Dict]:
    """
    Return statistics about LoRA weights (useful for debugging divergence).

    Returns list of dicts with name, norm, std for each LoRA param.
    """
    stats = []
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            stats.append({
                "name": name,
                "shape": list(param.shape),
                "norm": param.detach().norm().item(),
                "std":  param.detach().std().item(),
                "mean": param.detach().mean().item(),
            })
    return stats


def effective_rank(model: nn.Module) -> Dict[str, float]:
    """
    Compute the effective rank of each LoRA BA product (diagnostic).

    Effective rank = exp(H(singular_values / sum))
    where H is the entropy of the normalised singular value distribution.
    """
    import math
    results = {}
    params = {n: p for n, p in model.named_parameters() if "lora_" in n}

    # Pair lora_A and lora_B by their common prefix
    prefixes = set()
    for name in params:
        if "lora_A" in name:
            prefixes.add(name.replace("lora_A", ""))

    for prefix in prefixes:
        A_key = prefix + "lora_A"
        B_key = prefix + "lora_B"
        if A_key not in params or B_key not in params:
            continue
        A = params[A_key].detach().float()
        B = params[B_key].detach().float()
        W = B @ A   # (d, k)
        try:
            sv = torch.linalg.svdvals(W)
            sv = sv / (sv.sum() + 1e-9)
            entropy = -(sv * (sv + 1e-9).log()).sum().item()
            eff_rank = math.exp(entropy)
            results[prefix.rstrip(".")] = eff_rank
        except Exception:
            pass

    return results
