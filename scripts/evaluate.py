"""
scripts/evaluate.py — Evaluate a QLoRA fine-tuned model.

Usage
─────
  # Evaluate adapter on default dataset
  python scripts/evaluate.py --adapter outputs/qlora/final_adapter

  # Compare base vs fine-tuned
  python scripts/evaluate.py --adapter outputs/qlora/final_adapter --compare-base

  # Custom eval dataset
  python scripts/evaluate.py --adapter outputs/qlora/final_adapter \
                              --dataset tatsu-lab/alpaca --split test

  # Offline / synthetic
  python scripts/evaluate.py --synthetic

Outputs
───────
  Console: perplexity, token accuracy, ROUGE-L, sample generations
  JSON:     results saved to --output-dir/eval_results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter",      default=None,
                   help="Path to LoRA adapter directory")
    p.add_argument("--merged",       default=None,
                   help="Path to merged model directory")
    p.add_argument("--base-model",   default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--dataset",      default="tatsu-lab/alpaca")
    p.add_argument("--split",        default="train")
    p.add_argument("--prompt-style", default="alpaca")
    p.add_argument("--max-samples",  type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--compare-base", action="store_true")
    p.add_argument("--output-dir",   default="outputs/qlora/eval")
    p.add_argument("--synthetic",    action="store_true")
    p.add_argument("--hf-token",     default=None)
    return p.parse_args()


def _run_eval(model, tokenizer, cfg, args, label: str) -> dict:
    from data.dataset import build_datasets
    from training.metrics import (
        compute_perplexity, compute_token_accuracy, evaluate_generation
    )
    from data.formatting import get_formatter
    from data.dataset import load_raw_samples

    device = next(model.parameters()).device
    formatter = get_formatter(args.prompt_style, cfg.data.system_prompt)

    # Build eval dataset
    cfg.data.max_train_samples = None
    cfg.data.max_eval_samples  = args.max_samples
    _, eval_ds = build_datasets(cfg, tokenizer)

    if eval_ds is None:
        logger.warning("No eval dataset — using train set subset")
        from data.dataset import build_datasets as bd
        cfg.data.eval_fraction = 0.1
        _, eval_ds = bd(cfg, tokenizer)

    results = {"label": label}

    # PPL
    logger.info(f"  Computing perplexity for [{label}] …")
    ppl_metrics = compute_perplexity(
        model, eval_ds, tokenizer, device, max_samples=args.max_samples
    )
    results.update(ppl_metrics)

    # Token accuracy
    logger.info(f"  Computing token accuracy …")
    acc = compute_token_accuracy(
        model, eval_ds, tokenizer, device, max_samples=min(args.max_samples, 100)
    )
    results["token_accuracy"] = round(acc, 4)

    # ROUGE-L generation eval
    logger.info(f"  Running generation eval (ROUGE-L) …")
    raw_samples = load_raw_samples(
        cfg.data.dataset_name,
        split=cfg.data.train_split,
        max_samples=50,
    )
    gen_metrics = evaluate_generation(
        model, tokenizer, formatter, raw_samples,
        device, max_new_tokens=args.max_new_tokens, num_samples=50,
    )
    results.update(gen_metrics)

    return results


def print_results(results: dict):
    print(f"\n{'═'*56}")
    print(f"  Results: {results['label']}")
    print(f"{'═'*56}")
    for k, v in results.items():
        if k != "label":
            print(f"  {k:<25} {v}")
    print(f"{'═'*56}\n")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Build config ─────────────────────────────────────────────────────
    if args.synthetic:
        from config import fast_test_config
        cfg = fast_test_config()
    else:
        from config import Config, ModelConfig, DataConfig, TrainingConfig, LoRAConfig
        cfg = Config(
            model=ModelConfig(model_name=args.base_model, hf_token=args.hf_token),
            data=DataConfig(dataset_name=args.dataset, train_split=args.split,
                            prompt_style=args.prompt_style,
                            max_train_samples=args.max_samples),
            training=TrainingConfig(output_dir=args.output_dir),
        )

    all_results = []

    # ── Load fine-tuned model ─────────────────────────────────────────────
    if args.synthetic:
        from model.loader import load_synthetic_model_and_tokenizer
        from model.lora import attach_lora_synthetic
        ft_model, tokenizer = load_synthetic_model_and_tokenizer(str(device))
        ft_model = attach_lora_synthetic(ft_model, cfg.lora)
        label = "synthetic_qlora"
    elif args.merged:
        from inference.generate import load_for_inference
        ft_model, tokenizer = load_for_inference(
            base_model_name=args.base_model,
            merged_model_path=args.merged,
            hf_token=args.hf_token,
        )
        label = Path(args.merged).name
    elif args.adapter:
        from inference.generate import load_for_inference
        ft_model, tokenizer = load_for_inference(
            base_model_name=args.base_model,
            adapter_path=args.adapter,
            hf_token=args.hf_token,
        )
        label = f"qlora_{Path(args.adapter).name}"
    else:
        logger.error("Provide --adapter, --merged, or --synthetic")
        sys.exit(1)

    ft_results = _run_eval(ft_model, tokenizer, cfg, args, label)
    print_results(ft_results)
    all_results.append(ft_results)

    # ── Optionally evaluate base model ────────────────────────────────────
    if args.compare_base and not args.synthetic:
        from inference.generate import load_for_inference
        base_model, _ = load_for_inference(
            base_model_name=args.base_model, hf_token=args.hf_token
        )
        base_results = _run_eval(base_model, tokenizer, cfg, args, "base_fp16")
        print_results(base_results)
        all_results.append(base_results)

        # Delta
        d_ppl = ft_results["perplexity"] - base_results["perplexity"]
        d_rouge = ft_results["rouge_l"] - base_results["rouge_l"]
        print(f"  Δ PPL   (FT - base) = {d_ppl:+.4f}")
        print(f"  Δ ROUGE (FT - base) = {d_rouge:+.4f}")

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = Path(args.output_dir) / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
