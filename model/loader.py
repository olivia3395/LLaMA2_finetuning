"""
model/loader.py — Load LLaMA 2 in 4-bit NF4 (bitsandbytes QLoRA setup).

4-bit NF4 quantisation
───────────────────────
bitsandbytes implements NF4 (Normal Float 4), a data type optimised for
normally-distributed neural network weights:

  • Quantises weights to 4 bits using a custom non-uniform grid
  • The NF4 grid is information-theoretically optimal for N(0,1) weights
  • Double quantisation further compresses the quantisation constants
    (saves ~0.37 GB on a 7B model)

Memory comparison (7B model)
────────────────────────────
  fp32  :  28.0 GB   (cannot fit on consumer GPU)
  fp16  :  14.0 GB   (RTX 4090 barely fits, no headroom)
  int8  :   7.0 GB   (fits on 8 GB GPU)
  nf4   :   3.5 GB   + activations/KV cache ≈ 5–7 GB total
                       → fits on RTX 3070 (8 GB) with batch=1

Double quantisation savings
───────────────────────────
  Without double quant: quantisation constants in fp32 → 0.5 GB overhead
  With double quant:    quantisation constants in fp8  → 0.13 GB overhead
  Saving: ~0.37 GB for a 7B model
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bitsandbytes quantisation config
# ---------------------------------------------------------------------------

def _bnb_config(model_cfg):
    """Build BitsAndBytesConfig from ModelConfig."""
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise ImportError(
            "Install bitsandbytes: pip install bitsandbytes"
        )

    dtype_map = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }
    compute_dtype = dtype_map.get(model_cfg.bnb_4bit_compute_dtype, torch.bfloat16)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,      # "nf4" or "fp4"
        bnb_4bit_use_double_quant=model_cfg.bnb_4bit_use_double_quant,
    )


# ---------------------------------------------------------------------------
# Main model loader
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_cfg,
    device_map: str = "auto",
) -> Tuple[object, object]:
    """
    Load LLaMA 2 (or any HF causal LM) in 4-bit NF4.

    Steps:
      1. Build BitsAndBytesConfig
      2. Load tokenizer, set padding
      3. Load model with quantisation config
      4. Enable gradient checkpointing (optional)
      5. Prepare model for k-bit training (PEFT utility)

    Args:
        model_cfg  : ModelConfig instance
        device_map : "auto" distributes across available GPUs/CPU

    Returns:
        (model, tokenizer)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("pip install transformers")

    token = model_cfg.hf_token or os.environ.get("HF_TOKEN")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    torch_dtype = dtype_map.get(model_cfg.torch_dtype, torch.bfloat16)

    # ── Tokenizer ────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {model_cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name,
        token=token,
        trust_remote_code=model_cfg.trust_remote_code,
        padding_side="right",     # right-pad for causal LM training
    )
    # LLaMA 2 has no pad token — use EOS
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Model ─────────────────────────────────────────────────────────────
    logger.info(
        f"Loading model: {model_cfg.model_name}  "
        f"(4-bit NF4, double_quant={model_cfg.bnb_4bit_use_double_quant})"
    )

    model_kwargs = dict(
        pretrained_model_name_or_path=model_cfg.model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        token=token,
        trust_remote_code=model_cfg.trust_remote_code,
    )

    if model_cfg.load_in_4bit:
        model_kwargs["quantization_config"] = _bnb_config(model_cfg)

    if model_cfg.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.config.use_cache = False          # must be False for gradient checkpointing
    model.config.pretraining_tp = 1        # disable tensor-parallelism in attention

    # ── Prepare for k-bit training (PEFT utility) ─────────────────────────
    if model_cfg.load_in_4bit:
        try:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
            )
            logger.info("Model prepared for k-bit training (gradient checkpointing enabled)")
        except ImportError:
            logger.warning("PEFT not installed — skipping prepare_model_for_kbit_training")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {n_params / 1e9:.3f}B parameters")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Synthetic model (for offline testing without real weights)
# ---------------------------------------------------------------------------

def load_synthetic_model_and_tokenizer(device: str = "cpu"):
    """
    Return a tiny GPT-2 style model + tokenizer for unit testing.
    Mirrors the same API as load_model_and_tokenizer.
    """
    import torch.nn as nn

    class _FakeTok:
        pad_token    = "<pad>"
        pad_token_id = 0
        eos_token    = "</s>"
        eos_token_id = 2
        bos_token    = "<s>"
        bos_token_id = 1
        padding_side = "right"
        vocab_size   = 1000

        def __call__(self, text, **kw):
            ids = torch.tensor([[hash(c) % 997 + 3 for c in str(text)[:64]]])
            return {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
            }

        def encode(self, text, **kw):
            return [hash(c) % 997 + 3 for c in str(text)[:64]]

        def decode(self, ids, **kw):
            return "synthetic output"

        def batch_decode(self, ids_list, **kw):
            return ["synthetic output"] * len(ids_list)

        def save_pretrained(self, path):
            os.makedirs(path, exist_ok=True)

    class _FakeOutput:
        def __init__(self, logits):
            self.logits = logits
            self.loss   = torch.tensor(2.3)   # ~ ln(10), reasonable initial loss

    class _FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed  = nn.Embedding(1000, 64)
            self.linear = nn.Linear(64, 1000)
            self.config = type("cfg", (), {
                "use_cache": False, "pretraining_tp": 1,
                "hidden_size": 64,
            })()

        def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
            x      = self.embed(input_ids)
            logits = self.linear(x)
            loss   = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(
                    logits.view(-1, 1000),
                    labels.clamp(min=0).view(-1),
                )
            return _FakeOutput(logits) if loss is None else type(
                "Out", (), {"logits": logits, "loss": loss}
            )()

        def generate(self, input_ids, **kw):
            B = input_ids.shape[0]
            extra = torch.randint(3, 100, (B, 10))
            return torch.cat([input_ids, extra], dim=1)

        def save_pretrained(self, path, **kw):
            os.makedirs(path, exist_ok=True)
            torch.save(self.state_dict(), os.path.join(path, "model.pt"))

        def gradient_checkpointing_enable(self, **kw): pass
        def enable_input_require_grads(self): pass
        def get_input_embeddings(self): return self.embed

    model = _FakeModel().to(device)
    return model, _FakeTok()
