"""
training/metrics.py — Evaluation metrics for instruction fine-tuning.

Metrics
───────
  perplexity       : exp(mean NLL) on eval set  (primary metric)
  token_accuracy   : fraction of correctly predicted tokens
  completion_loss  : loss only on completion tokens (not prompt)
  rouge_l          : ROUGE-L F1 for text quality (requires rouge-score)
"""

from __future__ import annotations

import math
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    dataset,
    tokenizer,
    device: torch.device,
    max_samples: int = 200,
    batch_size: int = 1,
) -> Dict[str, float]:
    """
    Compute perplexity on an InstructionDataset.

    Only counts loss on non-masked positions (labels != -100).

    Returns dict with:
        perplexity   : exp(mean NLL per token)
        mean_loss    : mean cross-entropy
        total_tokens : number of non-masked tokens evaluated
    """
    from data.dataset import InstructionDataCollator
    from torch.utils.data import DataLoader, Subset

    model.eval()
    n      = min(max_samples, len(dataset))
    subset = Subset(dataset, range(n))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=InstructionDataCollator(tokenizer),
    )

    total_loss   = 0.0
    total_tokens = 0

    for batch in tqdm(loader, desc="PPL eval", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        out   = model(**batch)
        # out.loss averages over non-masked tokens
        n_tokens = (batch["labels"] != -100).sum().item()
        total_loss   += out.loss.item() * n_tokens
        total_tokens += n_tokens

    if total_tokens == 0:
        return {"perplexity": float("inf"), "mean_loss": float("inf"), "total_tokens": 0}

    mean_loss  = total_loss / total_tokens
    perplexity = math.exp(min(mean_loss, 20))   # cap at e^20 to avoid overflow

    return {
        "perplexity":   round(perplexity, 4),
        "mean_loss":    round(mean_loss,  4),
        "total_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Token accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_token_accuracy(
    model: nn.Module,
    dataset,
    tokenizer,
    device: torch.device,
    max_samples: int = 200,
) -> float:
    """
    Fraction of completion tokens the model predicts correctly (argmax).

    This is an overly-optimistic metric (teachers-forced) but useful
    as a quick training health check.
    """
    from data.dataset import InstructionDataCollator
    from torch.utils.data import DataLoader, Subset

    model.eval()
    n      = min(max_samples, len(dataset))
    loader = DataLoader(
        Subset(dataset, range(n)),
        batch_size=1,
        collate_fn=InstructionDataCollator(tokenizer),
    )

    correct = 0
    total   = 0

    for batch in loader:
        batch    = {k: v.to(device) for k, v in batch.items()}
        labels   = batch["labels"]                          # (B, T)
        out      = model(**batch)
        preds    = out.logits.argmax(-1)                   # (B, T)

        mask     = labels != -100
        correct += (preds[mask] == labels[mask]).sum().item()
        total   += mask.sum().item()

    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def compute_rouge_l(references: List[str], hypotheses: List[str]) -> float:
    """Compute mean ROUGE-L F1 (requires rouge-score package)."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [
            scorer.score(ref, hyp)["rougeL"].fmeasure
            for ref, hyp in zip(references, hypotheses)
        ]
        return sum(scores) / max(len(scores), 1)
    except ImportError:
        logger.warning("rouge-score not installed — ROUGE-L unavailable")
        return 0.0


# ---------------------------------------------------------------------------
# Generation-based evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_generation(
    model: nn.Module,
    tokenizer,
    formatter,
    eval_samples: List[Dict],
    device: torch.device,
    max_new_tokens: int = 256,
    num_samples: int = 50,
) -> Dict[str, float]:
    """
    Generate completions for eval samples and compute ROUGE-L.

    Args:
        model        : fine-tuned model
        tokenizer    : matching tokenizer
        formatter    : BaseFormatter used during training
        eval_samples : list of raw {"instruction", "input", "output"} dicts
        device       : inference device
        max_new_tokens: max tokens to generate

    Returns dict with:
        rouge_l      : mean ROUGE-L F1
        mean_len     : mean generated response length (words)
        n_evaluated  : number of samples evaluated
    """
    model.eval()
    references  = []
    hypotheses  = []

    for sample in eval_samples[:num_samples]:
        reference = sample.get("output", "")
        instruction = sample.get("instruction", "")
        input_text  = sample.get("input", "")

        prompt = formatter.format_inference(instruction, input_text)
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        input_ids = inputs["input_ids"].to(device)

        try:
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen_ids = output[0, input_ids.shape[1]:]
            hypothesis = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        except Exception as e:
            logger.debug(f"Generation failed: {e}")
            hypothesis = ""

        references.append(reference)
        hypotheses.append(hypothesis)

    rouge = compute_rouge_l(references, hypotheses)
    mean_len = sum(len(h.split()) for h in hypotheses) / max(len(hypotheses), 1)

    return {
        "rouge_l":     round(rouge,    4),
        "mean_length": round(mean_len, 2),
        "n_evaluated": len(hypotheses),
    }


# ---------------------------------------------------------------------------
# Training curve helper
# ---------------------------------------------------------------------------

def smooth(values: List[float], window: int = 5) -> List[float]:
    """Exponential moving average smoothing for loss curves."""
    if not values:
        return values
    alpha  = 2.0 / (window + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result
