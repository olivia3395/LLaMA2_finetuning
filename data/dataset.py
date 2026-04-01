"""
data/dataset.py — Dataset loading and preprocessing for QLoRA fine-tuning.

Supports
────────
  • HuggingFace Hub datasets (Alpaca, ShareGPT, OpenAssistant, etc.)
  • Local JSON / JSONL files
  • Synthetic dataset for offline unit testing

Preprocessing pipeline
──────────────────────
  raw rows → formatter → {full_text, prompt}
           → tokeniser → {input_ids, attention_mask, labels}
           → loss mask (zero-out prompt tokens if train_on_completions_only)
"""

from __future__ import annotations

import json
import logging
import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from data.formatting import get_formatter, BaseFormatter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic dataset (for offline testing, no HF token needed)
# ---------------------------------------------------------------------------

SYNTHETIC_INSTRUCTIONS = [
    {
        "instruction": "What is the capital of France?",
        "input": "",
        "output": "The capital of France is Paris.",
    },
    {
        "instruction": "Write a Python function to compute the Fibonacci sequence.",
        "input": "n = 10",
        "output": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    },
    {
        "instruction": "Summarise the following text.",
        "input": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet at least once.",
        "output": "The sentence is a pangram — it uses every letter of the alphabet.",
    },
    {
        "instruction": "Explain gradient descent in simple terms.",
        "input": "",
        "output": "Gradient descent is an optimisation algorithm that iteratively adjusts parameters in the direction that most reduces a loss function, like rolling a ball downhill to find the lowest valley.",
    },
    {
        "instruction": "Translate 'Hello, how are you?' into Spanish.",
        "input": "",
        "output": "Hola, ¿cómo estás?",
    },
    {
        "instruction": "List three benefits of regular exercise.",
        "input": "",
        "output": "1. Improved cardiovascular health.\n2. Better mental health and reduced stress.\n3. Stronger muscles and bones.",
    },
    {
        "instruction": "Write a haiku about artificial intelligence.",
        "input": "",
        "output": "Silicon dreams wake,\nThoughts flow through wired neurons—\nMachines learn to see.",
    },
    {
        "instruction": "What is the difference between machine learning and deep learning?",
        "input": "",
        "output": "Machine learning is a subset of AI where models learn patterns from data. Deep learning is a subset of machine learning that uses neural networks with many layers to learn hierarchical representations.",
    },
]


# ---------------------------------------------------------------------------
# Tokenised instruction dataset
# ---------------------------------------------------------------------------

class InstructionDataset(Dataset):
    """
    Tokenised dataset for instruction fine-tuning.

    Each sample is a dict with:
        input_ids      : (seq_len,) LongTensor
        attention_mask : (seq_len,) LongTensor
        labels         : (seq_len,) LongTensor  (-100 for masked positions)
    """

    def __init__(
        self,
        samples: List[Dict],
        tokenizer,
        formatter: BaseFormatter,
        max_seq_length: int = 2048,
        train_on_completions_only: bool = True,
    ):
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.max_seq_length = max_seq_length
        self.train_on_completions_only = train_on_completions_only

        self.data: List[Dict[str, torch.Tensor]] = []
        skipped = 0

        for sample in samples:
            try:
                enc = self._encode(sample)
                if enc is not None:
                    self.data.append(enc)
                else:
                    skipped += 1
            except Exception as e:
                logger.debug(f"Skipping sample: {e}")
                skipped += 1

        logger.info(
            f"Loaded {len(self.data)} samples "
            f"(skipped {skipped} too-long or malformed)"
        )

    def _encode(self, sample: Dict) -> Optional[Dict[str, torch.Tensor]]:
        """Tokenise one sample and build the labels tensor."""
        formatted = self.formatter.format_train(sample)
        full_text = formatted["full_text"]
        prompt    = formatted["prompt"]

        # Tokenise full text
        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        if input_ids.size(0) < 4:
            return None   # too short to be meaningful

        # Build labels: default = copy of input_ids
        labels = input_ids.clone()

        # Mask prompt tokens so loss is only on completions
        if self.train_on_completions_only:
            prompt_enc = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
                return_tensors="pt",
                add_special_tokens=True,
            )
            prompt_len = prompt_enc["input_ids"].size(1)
            # -100 tells cross-entropy to ignore those positions
            labels[:prompt_len] = -100

        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


# ---------------------------------------------------------------------------
# Dataset loading from HuggingFace Hub or local files
# ---------------------------------------------------------------------------

def load_raw_samples(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    fraction: float = 1.0,
    seed: int = 42,
) -> List[Dict]:
    """
    Load raw (untokenised) samples from a dataset source.

    Handles:
      • HuggingFace Hub datasets
      • Local JSON / JSONL files
      • "__synthetic__" fallback (no internet required)

    Returns list of raw dicts (keys depend on the dataset).
    """
    if dataset_name == "__synthetic__":
        samples = SYNTHETIC_INSTRUCTIONS * 4   # 32 samples
        random.seed(seed)
        random.shuffle(samples)
        if max_samples:
            samples = samples[:max_samples]
        logger.info(f"Using {len(samples)} synthetic samples")
        return samples

    # Local JSON / JSONL
    if dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
        return _load_local_json(dataset_name, max_samples)

    # HuggingFace Hub
    try:
        from datasets import load_dataset
        logger.info(f"Loading dataset: {dataset_name} [{split}]")
        ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(
            f"Could not load dataset '{dataset_name}'.\n"
            f"Error: {e}\n"
            f"Tip: use --dataset __synthetic__ for offline testing."
        )

    # Apply fraction
    if fraction < 1.0:
        n = int(len(ds) * fraction)
        ds = ds.select(range(n))

    # Apply max_samples
    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    samples = [dict(row) for row in ds]
    logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
    return samples


def _load_local_json(path: str, max_samples: Optional[int]) -> List[Dict]:
    """Load samples from a local JSON or JSONL file."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for line in f:
                samples.append(json.loads(line.strip()))
        else:
            data = json.load(f)
            samples = data if isinstance(data, list) else [data]
    if max_samples:
        samples = samples[:max_samples]
    logger.info(f"Loaded {len(samples)} samples from {path}")
    return samples


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

class InstructionDataCollator:
    """
    Pads variable-length samples to the longest in the batch.

    Padding strategy:
        input_ids      → pad with tokenizer.pad_token_id
        attention_mask → pad with 0
        labels         → pad with -100 (ignored by loss)
    """

    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer          = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Find max length in batch
        max_len = max(f["input_ids"].size(0) for f in features)

        # Pad to multiple of 8 for tensor core efficiency
        if self.pad_to_multiple_of:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        pad_id = self.tokenizer.pad_token_id or 0
        batch  = {"input_ids": [], "attention_mask": [], "labels": []}

        for f in features:
            seq_len = f["input_ids"].size(0)
            pad_len = max_len - seq_len

            batch["input_ids"].append(
                torch.cat([f["input_ids"],
                           torch.full((pad_len,), pad_id, dtype=torch.long)])
            )
            batch["attention_mask"].append(
                torch.cat([f["attention_mask"],
                           torch.zeros(pad_len, dtype=torch.long)])
            )
            batch["labels"].append(
                torch.cat([f["labels"],
                           torch.full((pad_len,), -100, dtype=torch.long)])
            )

        return {k: torch.stack(v) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Helper: build train/eval datasets
# ---------------------------------------------------------------------------

def build_datasets(
    cfg,
    tokenizer,
) -> tuple:
    """
    Build InstructionDataset for train and (optionally) eval splits.

    Returns (train_dataset, eval_dataset)  where eval_dataset may be None.
    """
    formatter = get_formatter(cfg.data.prompt_style, cfg.data.system_prompt)

    # Load training samples
    train_samples = load_raw_samples(
        cfg.data.dataset_name,
        split=cfg.data.train_split,
        max_samples=cfg.data.max_train_samples,
        fraction=cfg.data.train_fraction,
        seed=cfg.training.seed,
    )

    # Carve out eval set if no separate split
    eval_samples = None
    if cfg.data.eval_split:
        eval_samples = load_raw_samples(
            cfg.data.dataset_name,
            split=cfg.data.eval_split,
            max_samples=cfg.data.max_eval_samples,
            seed=cfg.training.seed,
        )
    elif cfg.data.eval_fraction > 0:
        random.seed(cfg.training.seed)
        random.shuffle(train_samples)
        n_eval = max(1, int(len(train_samples) * cfg.data.eval_fraction))
        n_eval = min(n_eval, cfg.data.max_eval_samples or n_eval)
        eval_samples   = train_samples[:n_eval]
        train_samples  = train_samples[n_eval:]

    train_ds = InstructionDataset(
        train_samples, tokenizer, formatter,
        max_seq_length=cfg.model.max_seq_length,
        train_on_completions_only=cfg.data.train_on_completions_only,
    )
    eval_ds = None
    if eval_samples:
        eval_ds = InstructionDataset(
            eval_samples, tokenizer, formatter,
            max_seq_length=cfg.model.max_seq_length,
            train_on_completions_only=cfg.data.train_on_completions_only,
        )

    return train_ds, eval_ds
