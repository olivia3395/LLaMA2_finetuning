"""
model/utils.py — Model utilities: parameter counting, VRAM, dtype info.
"""
from __future__ import annotations
import logging
from typing import Dict, Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return trainable / frozen / total parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}


def model_memory_footprint(model: nn.Module) -> Dict[str, float]:
    """
    Estimate model memory usage in GB.
    Counts actual element sizes (4-bit params counted as 0.5 bytes each).
    """
    total_bytes = 0
    for name, param in model.named_parameters():
        dtype = param.dtype
        if dtype in (torch.uint8,):
            # bitsandbytes stores 4-bit in uint8 with 2 values per byte
            bytes_per_param = 0.5
        elif dtype in (torch.float16, torch.bfloat16):
            bytes_per_param = 2
        elif dtype == torch.float32:
            bytes_per_param = 4
        else:
            bytes_per_param = param.element_size()
        total_bytes += param.numel() * bytes_per_param

    return {
        "model_gb":    total_bytes / (1024 ** 3),
        "model_bytes": total_bytes,
    }


def gpu_memory_stats(device: Optional[torch.device] = None) -> Dict[str, float]:
    """Return current GPU memory usage in GB (CUDA only)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "free_gb": 0.0}
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved  = torch.cuda.memory_reserved(device)  / (1024 ** 3)
    total     = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    return {
        "allocated_gb": allocated,
        "reserved_gb":  reserved,
        "free_gb":      total - reserved,
        "total_gb":     total,
    }


def print_model_summary(model: nn.Module, label: str = "Model"):
    """Pretty-print a model summary with parameter and memory info."""
    params = count_parameters(model)
    mem    = model_memory_footprint(model)
    print(f"\n{'─'*56}")
    print(f"  {label}")
    print(f"  Trainable params : {params['trainable']:>15,}")
    print(f"  Frozen params    : {params['frozen']:>15,}")
    print(f"  Total params     : {params['total']:>15,}")
    print(f"  Trainable %      : {100*params['trainable']/max(params['total'],1):>14.4f}%")
    print(f"  Memory (est.)    : {mem['model_gb']:>14.3f} GB")
    if torch.cuda.is_available():
        gpu = gpu_memory_stats()
        print(f"  GPU allocated    : {gpu['allocated_gb']:>14.3f} GB")
        print(f"  GPU free         : {gpu['free_gb']:>14.3f} GB")
    print(f"{'─'*56}\n")


def merge_lora_weights(model: nn.Module, output_path: str) -> nn.Module:
    """
    Merge LoRA adapter weights into the base model and save.

    After merging:
      • The model is equivalent to full fine-tuning at that point
      • No PEFT dependency needed for inference
      • Can be loaded with AutoModelForCausalLM.from_pretrained()

    Args:
        model       : PeftModel with LoRA adapters
        output_path : directory to save the merged model

    Returns:
        merged base model (nn.Module)
    """
    try:
        merged = model.merge_and_unload()
        merged.save_pretrained(output_path, safe_serialization=True)
        logger.info(f"Merged model saved → {output_path}")
        return merged
    except AttributeError:
        logger.warning("Model does not appear to be a PeftModel — saving as-is")
        model.save_pretrained(output_path)
        return model


def save_lora_adapter(model: nn.Module, output_path: str):
    """Save only the LoRA adapter weights (much smaller than full model)."""
    try:
        model.save_pretrained(output_path)
        logger.info(f"LoRA adapter saved → {output_path}")
    except AttributeError:
        import os, torch
        os.makedirs(output_path, exist_ok=True)
        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        torch.save(trainable, os.path.join(output_path, "lora_weights.pt"))
        logger.info(f"LoRA weights saved → {output_path}/lora_weights.pt")


def estimate_training_memory_gb(
    n_params: int,
    batch_size: int,
    seq_len: int,
    gradient_checkpointing: bool = True,
    r: int = 16,
) -> Dict[str, float]:
    """
    Rough estimate of peak GPU memory during QLoRA training.

    Based on empirical measurements from Tim Dettmers' QLoRA paper.
    """
    # 4-bit base model
    base_gb = n_params * 0.5 / (1024 ** 3)

    # LoRA params (bf16)
    lora_params_approx = n_params * 0.0004 * (r / 16)
    lora_gb = lora_params_approx * 2 / (1024 ** 3)

    # Gradients for LoRA params (bf16)
    grad_gb = lora_gb

    # Optimiser states (paged AdamW: 2 states per param, 32-bit)
    optim_gb = lora_params_approx * 8 / (1024 ** 3)

    # Activations (rough estimate; GC reduces this by ~5-10×)
    hidden = 4096  # LLaMA-2-7B hidden size
    layers = 32
    act_per_token = hidden * layers * 2   # bytes in bf16
    act_gb = batch_size * seq_len * act_per_token / (1024 ** 3)
    if gradient_checkpointing:
        act_gb /= 8   # approximate GC reduction

    total_gb = base_gb + lora_gb + grad_gb + optim_gb + act_gb

    return {
        "base_model_gb":  base_gb,
        "lora_params_gb": lora_gb,
        "gradients_gb":   grad_gb,
        "optimizer_gb":   optim_gb,
        "activations_gb": act_gb,
        "total_gb":       total_gb,
    }
