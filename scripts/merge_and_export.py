"""
scripts/merge_and_export.py — Merge LoRA adapters into the base model.

Why merge?
──────────
  During training, LoRA weights (BA) are kept separate from the frozen base.
  For deployment, merging folds ΔW = BA back into W:

      W_merged = W_base + (α/r) · B · A

  After merging:
    • No PEFT dependency needed at inference
    • Standard AutoModelForCausalLM.from_pretrained() works
    • Can export to GGUF (llama.cpp), GPTQ, AWQ for further quantisation
    • Inference speed is identical to the base model (no adapter overhead)

  Trade-off: merged model is fp16/bf16 → ~14 GB for 7B vs ~3.5 GB adapter-only

Usage
─────
  # Merge and save as fp16 HuggingFace model
  python scripts/merge_and_export.py \
      --base-model meta-llama/Llama-2-7b-hf \
      --adapter    outputs/qlora/final_adapter \
      --output-dir outputs/qlora/merged_model

  # Also save as GGUF (requires llama.cpp)
  python scripts/merge_and_export.py ... --export-gguf

  # Offline test
  python scripts/merge_and_export.py --synthetic
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    p.add_argument("--base-model", default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--adapter",    default="outputs/qlora/final_adapter")
    p.add_argument("--output-dir", default="outputs/qlora/merged_model")
    p.add_argument("--dtype",      default="float16",
                   choices=["float16", "bfloat16"])
    p.add_argument("--safe-serialization", action="store_true", default=True)
    p.add_argument("--export-gguf", action="store_true",
                   help="Also convert to GGUF via llama.cpp (llama.cpp must be installed)")
    p.add_argument("--synthetic",  action="store_true")
    p.add_argument("--hf-token",   default=None)
    return p.parse_args()


def merge_and_save(args):
    """Load base + adapter, merge, save as standard HF model."""
    os.makedirs(args.output_dir, exist_ok=True)

    if args.synthetic:
        # Offline test: merge with synthetic model
        from model.loader import load_synthetic_model_and_tokenizer
        from model.lora import attach_lora_synthetic
        from config import fast_test_config
        cfg = fast_test_config()
        model, tokenizer = load_synthetic_model_and_tokenizer("cpu")
        model = attach_lora_synthetic(model, cfg.lora)

        logger.info("Synthetic mode: simulating merge")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Saved → {args.output_dir}")
        return

    # ── Real model merge ──────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    token = args.hf_token or os.environ.get("HF_TOKEN")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    logger.info(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
        token=token,
    )

    logger.info(f"Loading LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter, token=token)

    logger.info("Merging LoRA weights into base model …")
    model = model.merge_and_unload()
    model.eval()

    logger.info(f"Saving merged model → {args.output_dir}")
    model.save_pretrained(
        args.output_dir,
        safe_serialization=args.safe_serialization,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, token=token
    )
    tokenizer.save_pretrained(args.output_dir)

    # ── Model card ────────────────────────────────────────────────────────
    card = {
        "base_model":   args.base_model,
        "adapter":      args.adapter,
        "dtype":        args.dtype,
        "merged":       True,
        "framework":    "QLoRA (bitsandbytes + PEFT)",
    }
    with open(os.path.join(args.output_dir, "model_card.json"), "w") as f:
        json.dump(card, f, indent=2)

    logger.info("✅  Merge complete!")
    logger.info(f"   Load with: AutoModelForCausalLM.from_pretrained('{args.output_dir}')")

    # ── GGUF export (optional) ────────────────────────────────────────────
    if args.export_gguf:
        _export_gguf(args.output_dir)


def _export_gguf(model_path: str):
    """Convert merged HF model to GGUF via llama.cpp convert script."""
    import subprocess, shutil

    convert_script = shutil.which("convert.py")  # from llama.cpp
    if convert_script is None:
        # Try common location
        convert_script = os.path.expanduser("~/llama.cpp/convert.py")

    if not os.path.exists(convert_script):
        logger.warning(
            "llama.cpp convert.py not found — skipping GGUF export.\n"
            "Install llama.cpp: git clone https://github.com/ggerganov/llama.cpp"
        )
        return

    gguf_path = model_path + ".gguf"
    cmd = ["python", convert_script, model_path, "--outfile", gguf_path,
           "--outtype", "f16"]
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        logger.info(f"GGUF saved → {gguf_path}")
    else:
        logger.error(f"GGUF conversion failed:\n{result.stderr}")


def main():
    args = parse_args()
    merge_and_save(args)


if __name__ == "__main__":
    main()
