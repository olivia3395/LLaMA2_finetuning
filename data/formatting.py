"""
data/formatting.py — Prompt templates for instruction fine-tuning.

Supported formats
─────────────────

1. alpaca  — Stanford Alpaca / Alpaca-cleaned style
   ┌──────────────────────────────────────────────┐
   │ Below is an instruction...                   │
   │                                              │
   │ ### Instruction:                             │
   │ Explain quantum entanglement.                │
   │                                              │
   │ ### Input:                (optional)         │
   │ Focus on the EPR paradox.                    │
   │                                              │
   │ ### Response:                                │
   │ Quantum entanglement is...                   │
   └──────────────────────────────────────────────┘

2. chatml  — OpenAI ChatML / Mistral / Zephyr style
   ┌──────────────────────────────────────────────┐
   │ <|im_start|>system                           │
   │ You are a helpful assistant.<|im_end|>       │
   │ <|im_start|>user                             │
   │ Explain quantum entanglement.<|im_end|>      │
   │ <|im_start|>assistant                        │
   │ Quantum entanglement is...<|im_end|>         │
   └──────────────────────────────────────────────┘

3. llama2  — Meta LLaMA 2 chat format (used by llama.cpp etc.)
   ┌──────────────────────────────────────────────┐
   │ [INST] <<SYS>>                               │
   │ You are a helpful assistant.                 │
   │ <</SYS>>                                     │
   │ Explain quantum entanglement. [/INST]        │
   │ Quantum entanglement is...                   │
   └──────────────────────────────────────────────┘

4. simple  — minimal "### X:\n" format
   ┌──────────────────────────────────────────────┐
   │ ### Instruction:                             │
   │ Explain quantum entanglement.                │
   │ ### Response:                                │
   │ Quantum entanglement is...                   │
   └──────────────────────────────────────────────┘

Each formatter returns:
  full_text : str    — complete input+output string to tokenise
  prompt    : str    — prompt-only prefix (used for masking loss)
"""

from __future__ import annotations
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Base formatter
# ---------------------------------------------------------------------------

class BaseFormatter:
    name: str = "base"

    def format_train(self, sample: Dict) -> Dict[str, str]:
        """
        Format a single training sample.

        Args:
            sample: raw dataset row (keys vary by dataset)

        Returns dict with:
            full_text : complete prompt + completion (for tokenisation)
            prompt    : prompt-only part (for loss masking)
        """
        raise NotImplementedError

    def format_inference(self, instruction: str, input_text: str = "") -> str:
        """Format a prompt for inference (no completion)."""
        raise NotImplementedError

    def completion_start(self) -> str:
        """The string that marks where the completion begins."""
        return ""


# ---------------------------------------------------------------------------
# Alpaca formatter
# ---------------------------------------------------------------------------

ALPACA_SYSTEM = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
)
ALPACA_SYSTEM_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
)

class AlpacaFormatter(BaseFormatter):
    name = "alpaca"

    def __init__(self, system_prompt: Optional[str] = None):
        self.system = system_prompt  # override if provided

    def _build_prompt(self, instruction: str, input_text: str = "") -> str:
        if input_text.strip():
            header = self.system or ALPACA_SYSTEM
            return (
                f"{header}"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n"
            )
        else:
            header = self.system or ALPACA_SYSTEM_NO_INPUT
            return (
                f"{header}"
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n"
            )

    def format_train(self, sample: Dict) -> Dict[str, str]:
        instruction = sample.get("instruction", "")
        input_text  = sample.get("input", "")
        output      = sample.get("output", "")
        prompt    = self._build_prompt(instruction, input_text)
        full_text = prompt + output + "\n"
        return {"full_text": full_text, "prompt": prompt}

    def format_inference(self, instruction: str, input_text: str = "") -> str:
        return self._build_prompt(instruction, input_text)

    def completion_start(self) -> str:
        return "### Response:\n"


# ---------------------------------------------------------------------------
# ChatML formatter
# ---------------------------------------------------------------------------

class ChatMLFormatter(BaseFormatter):
    name = "chatml"

    BOS = "<|im_start|>"
    EOS = "<|im_end|>"

    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.system_prompt = system_prompt

    def _format_messages(
        self, messages: List[Dict[str, str]], add_generation_prompt: bool = False
    ) -> str:
        out = f"{self.BOS}system\n{self.system_prompt}{self.EOS}\n"
        for msg in messages:
            role    = msg["role"]
            content = msg["content"]
            out += f"{self.BOS}{role}\n{content}{self.EOS}\n"
        if add_generation_prompt:
            out += f"{self.BOS}assistant\n"
        return out

    def format_train(self, sample: Dict) -> Dict[str, str]:
        # Handle ShareGPT-style {"conversations": [{"from": "human"/"gpt", "value": ...}]}
        if "conversations" in sample:
            messages = []
            for turn in sample["conversations"]:
                role = "user" if turn["from"] in ("human", "user") else "assistant"
                messages.append({"role": role, "content": turn["value"]})
        # Handle instruction/output style
        elif "instruction" in sample:
            messages = [
                {"role": "user",      "content": sample["instruction"]},
                {"role": "assistant", "content": sample.get("output", "")},
            ]
        # Handle messages list
        elif "messages" in sample:
            messages = sample["messages"]
        else:
            raise ValueError(f"Cannot parse sample with keys: {list(sample.keys())}")

        # Prompt = everything up to the last assistant turn
        prompt_messages = messages[:-1]
        prompt = self._format_messages(prompt_messages, add_generation_prompt=True)

        # Full text = prompt + last assistant content + EOS
        last = messages[-1]
        full_text = self._format_messages(messages[:-1], add_generation_prompt=True)
        full_text += last["content"] + self.EOS + "\n"

        return {"full_text": full_text, "prompt": prompt}

    def format_inference(self, instruction: str, input_text: str = "") -> str:
        user_content = instruction
        if input_text.strip():
            user_content += f"\n\n{input_text}"
        return self._format_messages(
            [{"role": "user", "content": user_content}],
            add_generation_prompt=True,
        )

    def completion_start(self) -> str:
        return f"{self.BOS}assistant\n"


# ---------------------------------------------------------------------------
# LLaMA-2 chat formatter
# ---------------------------------------------------------------------------

class LLaMA2Formatter(BaseFormatter):
    name = "llama2"

    B_INST = "[INST]"
    E_INST = "[/INST]"
    B_SYS  = "<<SYS>>\n"
    E_SYS  = "\n<</SYS>>\n\n"

    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.system_prompt = system_prompt

    def _wrap_system(self, text: str) -> str:
        if self.system_prompt:
            return f"{self.B_SYS}{self.system_prompt}{self.E_SYS}{text}"
        return text

    def format_train(self, sample: Dict) -> Dict[str, str]:
        if "conversations" in sample:
            turns = sample["conversations"]
            prompt = ""
            for i in range(0, len(turns) - 1, 2):
                user_msg = turns[i]["value"]
                if i == 0:
                    user_msg = self._wrap_system(user_msg)
                prompt += f"{self.B_INST} {user_msg} {self.E_INST} "
            # Last user message (without assistant response)
            last_user = turns[-2]["value"] if len(turns) > 1 else ""
            completion = turns[-1]["value"]
            prompt_end = f"{self.B_INST} {self._wrap_system(last_user) if not prompt else last_user} {self.E_INST} "
            full_text  = prompt + prompt_end + completion + " "
            return {"full_text": full_text, "prompt": prompt + prompt_end}
        else:
            instruction = sample.get("instruction", "")
            output      = sample.get("output", "")
            input_text  = sample.get("input", "")
            user_content = self._wrap_system(
                f"{instruction}\n\n{input_text}" if input_text.strip() else instruction
            )
            prompt    = f"{self.B_INST} {user_content} {self.E_INST} "
            full_text = prompt + output + " "
            return {"full_text": full_text, "prompt": prompt}

    def format_inference(self, instruction: str, input_text: str = "") -> str:
        user_content = self._wrap_system(
            f"{instruction}\n\n{input_text}" if input_text.strip() else instruction
        )
        return f"{self.B_INST} {user_content} {self.E_INST} "

    def completion_start(self) -> str:
        return f"{self.E_INST} "


# ---------------------------------------------------------------------------
# Simple formatter
# ---------------------------------------------------------------------------

class SimpleFormatter(BaseFormatter):
    name = "simple"

    def __init__(self, system_prompt: Optional[str] = None):
        self.system_prompt = system_prompt

    def format_train(self, sample: Dict) -> Dict[str, str]:
        instruction = sample.get("instruction", "")
        output      = sample.get("output", "")
        input_text  = sample.get("input", "")

        parts = []
        if self.system_prompt:
            parts.append(f"### System:\n{self.system_prompt}\n\n")
        parts.append(f"### Instruction:\n{instruction}\n\n")
        if input_text.strip():
            parts.append(f"### Input:\n{input_text}\n\n")
        parts.append("### Response:\n")

        prompt    = "".join(parts)
        full_text = prompt + output + "\n"
        return {"full_text": full_text, "prompt": prompt}

    def format_inference(self, instruction: str, input_text: str = "") -> str:
        parts = []
        if self.system_prompt:
            parts.append(f"### System:\n{self.system_prompt}\n\n")
        parts.append(f"### Instruction:\n{instruction}\n\n")
        if input_text.strip():
            parts.append(f"### Input:\n{input_text}\n\n")
        parts.append("### Response:\n")
        return "".join(parts)

    def completion_start(self) -> str:
        return "### Response:\n"


# ---------------------------------------------------------------------------
# Formatter registry
# ---------------------------------------------------------------------------

FORMATTERS = {
    "alpaca": AlpacaFormatter,
    "chatml": ChatMLFormatter,
    "llama2": LLaMA2Formatter,
    "simple": SimpleFormatter,
}


def get_formatter(style: str, system_prompt: Optional[str] = None) -> BaseFormatter:
    """
    Return the appropriate formatter for the given style string.

    Args:
        style         : "alpaca" | "chatml" | "llama2" | "simple"
        system_prompt : optional override for the system message

    Returns:
        BaseFormatter subclass instance
    """
    cls = FORMATTERS.get(style)
    if cls is None:
        raise ValueError(
            f"Unknown prompt style '{style}'. "
            f"Valid options: {list(FORMATTERS.keys())}"
        )
    if system_prompt is not None:
        return cls(system_prompt=system_prompt)
    return cls()
