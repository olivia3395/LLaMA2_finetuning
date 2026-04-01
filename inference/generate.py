"""
inference/generate.py — Inference with a QLoRA fine-tuned model.

Two modes
─────────
1. Adapter mode (default during development)
   Load base 4-bit model + LoRA adapter weights separately.
   Fastest to iterate; no merge step needed.
   Memory: ~3.5 GB (base) + tiny adapter weights

2. Merged mode (for deployment)
   Load the merged fp16 model produced by merge_and_export.py.
   No PEFT dependency at inference time.
   Memory: ~14 GB fp16, or re-quantise with llama.cpp / GGUF

Generation parameters
─────────────────────
  temperature   : 0 → greedy; 1 → sampling from full distribution
  top_p         : nucleus sampling (cumulative probability threshold)
  top_k         : top-k sampling
  repetition_penalty : penalise repeated tokens (1.1–1.3 typical)
  max_new_tokens: cap on generated tokens
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generation config
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    max_new_tokens: int      = 512
    temperature: float       = 0.7
    top_p: float             = 0.9
    top_k: int               = 50
    repetition_penalty: float = 1.1
    do_sample: bool          = True
    num_beams: int           = 1     # 1 = no beam search
    num_return_sequences: int = 1


# ---------------------------------------------------------------------------
# Model loader for inference
# ---------------------------------------------------------------------------

def load_for_inference(
    base_model_name: str,
    adapter_path: Optional[str] = None,
    merged_model_path: Optional[str] = None,
    load_in_4bit: bool = True,
    device: str = "auto",
    hf_token: Optional[str] = None,
) -> tuple:
    """
    Load a fine-tuned model for inference.

    Priority:
      1. merged_model_path  — load a standalone merged model
      2. adapter_path       — load base + LoRA adapter
      3. base_model_name only — load untuned base (for comparison)

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = hf_token or os.environ.get("HF_TOKEN")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tok_source = merged_model_path or adapter_path or base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_source, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Model ──────────────────────────────────────────────────────────────
    if merged_model_path:
        logger.info(f"Loading merged model: {merged_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            merged_model_path,
            torch_dtype=torch.float16,
            device_map=device,
            token=token,
        )
    else:
        # Load base in 4-bit
        model_path = base_model_name
        kwargs = dict(torch_dtype=torch.float16, device_map=device, token=token)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

        # Attach adapter
        if adapter_path:
            logger.info(f"Loading LoRA adapter: {adapter_path}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()   # merge for faster inference

    model.eval()
    logger.info("Model ready for inference")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

class InstructionGenerator:
    """
    High-level text generation interface for the fine-tuned model.

    Usage:
        gen = InstructionGenerator(model, tokenizer, formatter)
        resp = gen.generate("Explain quantum computing.")
        for token in gen.stream("Write a poem about Python."):
            print(token, end="", flush=True)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        formatter,
        gen_cfg: Optional[GenerationConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.gen_cfg   = gen_cfg or GenerationConfig()
        self.device    = device or next(model.parameters()).device

    def _build_input(
        self, instruction: str, input_text: str = ""
    ) -> torch.Tensor:
        prompt  = self.formatter.format_inference(instruction, input_text)
        enc     = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        return enc["input_ids"].to(self.device)

    def _generation_kwargs(self) -> Dict:
        cfg = self.gen_cfg
        kw  = dict(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            repetition_penalty=cfg.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if cfg.do_sample:
            kw.update(temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k)
        if cfg.num_beams > 1:
            kw["num_beams"] = cfg.num_beams
        return kw

    @torch.no_grad()
    def generate(
        self,
        instruction: str,
        input_text: str = "",
        **override_kwargs,
    ) -> str:
        """
        Generate a single response.

        Args:
            instruction: the instruction / question
            input_text : optional additional context
            **override_kwargs: override any GenerationConfig field

        Returns:
            Generated response string
        """
        input_ids = self._build_input(instruction, input_text)
        kw        = {**self._generation_kwargs(), **override_kwargs}

        output   = self.model.generate(input_ids, **kw)
        gen_ids  = output[0, input_ids.shape[1]:]
        response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return response.strip()

    @torch.no_grad()
    def generate_batch(
        self,
        instructions: List[str],
        input_texts: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate responses for a batch of instructions."""
        if input_texts is None:
            input_texts = [""] * len(instructions)
        return [
            self.generate(inst, inp)
            for inst, inp in zip(instructions, input_texts)
        ]

    def stream(self, instruction: str, input_text: str = ""):
        """
        Token-by-token streaming generator.

        Usage:
            for token in gen.stream("Tell me a story."):
                print(token, end="", flush=True)
        """
        try:
            from transformers import TextIteratorStreamer
            import threading

            input_ids = self._build_input(instruction, input_text)
            streamer  = TextIteratorStreamer(
                self.tokenizer, skip_special_tokens=True, skip_prompt=True
            )
            kw = {**self._generation_kwargs(), "streamer": streamer}

            thread = threading.Thread(
                target=self.model.generate,
                args=(input_ids,),
                kwargs=kw,
                daemon=True,
            )
            thread.start()
            for token in streamer:
                yield token
            thread.join()

        except ImportError:
            # Fallback: yield full response at once
            yield self.generate(instruction, input_text)

    def interactive(self):
        """REPL loop for interactive testing."""
        print("\n" + "=" * 56)
        print("  QLoRA Fine-Tuned LLaMA 2 — Interactive Mode")
        print("  Type 'quit' or Ctrl-C to exit")
        print("=" * 56 + "\n")

        while True:
            try:
                instruction = input("You: ").strip()
                if instruction.lower() in {"quit", "q", "exit"}:
                    break
                if not instruction:
                    continue

                print("Assistant: ", end="", flush=True)
                for token in self.stream(instruction):
                    print(token, end="", flush=True)
                print("\n")

            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
