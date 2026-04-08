<div align="center">

<img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
<img src="https://img.shields.io/badge/PEFT-LoRA-8b5cf6?style=for-the-badge"/>
<img src="https://img.shields.io/badge/GPU-10GB+-22c55e?style=for-the-badge&logo=nvidia&logoColor=white"/>

<br/><br/>

# 🦙 QLoRA Fine-Tuning
### LLaMA-2 Instruction Tuning on Consumer GPUs

<br/>

> Fine-tune a **7B parameter model** on a single **10 GB GPU** in 2–5 hours  
> by combining 4-bit quantization with low-rank adapter training.

<br/>

[🚀 Quick Start](#-quick-start) · [💾 Memory Guide](#-memory-requirements) · [🏗️ Architecture](#️-project-architecture) · [🎯 Prompt Formats](#-prompt-formats) · [📊 Evaluation](#-evaluation) · [📎 References](#-references)

<br/>



</div>

## ✨ What is QLoRA?

**QLoRA = Quantization + LoRA.** Two complementary techniques stacked together to make large-model fine-tuning accessible on commodity hardware:

<br/>

<div align="center">

| Technique | Memory Savings | Quality Cost |
|:---|:---:|:---:|
| 4-bit NF4 quantization (bitsandbytes) | **4× vs fp16** | ~0.1% PPL increase |
| Double quantization | +0.37 GB extra | negligible |
| LoRA adapters (r = 16) | train only **0.4% of params** | minimal for instruction tuning |
| Paged AdamW optimizer | offloads optimizer states to CPU | none |
| Gradient checkpointing | **5–10× activation savings** | 20–30% slower |

</div>

<br/>



## 💾 Memory Requirements

### VRAM by Model Size

<div align="center">

| Model | Min VRAM | Recommended | Batch Size |
|:---|:---:|:---:|:---:|
| LLaMA-2-7B | **8 GB** | 16 GB | 1–4 |
| LLaMA-2-13B | **12 GB** | 24 GB | 1–2 |
| LLaMA-2-70B | **48 GB** | 80 GB | 1 (multi-GPU) |

</div>

<br/>

### Memory Breakdown — 7B, batch = 4, seq = 2048

```
4-bit base model weights        ~3.5 GB
LoRA parameters  (r=16)         ~0.05 GB
Gradients        (LoRA only)    ~0.05 GB
Paged AdamW optimizer           ~0.2  GB   (partially on CPU)
Activations      (GC enabled)   ~2.5  GB
KV cache + overhead             ~0.7  GB
────────────────────────────────────────
Total                           ~7.0  GB   ← fits RTX 3080 (10 GB) ✅
```

<br/>



## 🏗️ Project Architecture

```
qlora_ft/
│
├── config.py                        ModelConfig · LoRAConfig · DataConfig · TrainingConfig
│
├── data/
│   ├── formatting.py                AlpacaFormatter · ChatMLFormatter · LLaMA2Formatter
│   └── dataset.py                   InstructionDataset · InstructionDataCollator · build_datasets
│
├── model/
│   ├── loader.py                    load_model_and_tokenizer()  [4-bit NF4 + bitsandbytes]
│   ├── lora.py                      attach_lora()  [PEFT], weight stats, effective rank
│   └── utils.py                     count_parameters · memory_footprint · merge_lora_weights
│
├── training/
│   ├── trainer.py                   QLoRATrainer  [wraps trl.SFTTrainer / HF Trainer]
│   ├── callbacks.py                 LoRACheckpoint · EarlyStopping · MemoryLog · GradientStats
│   └── metrics.py                   compute_perplexity · token_accuracy · ROUGE-L
│
├── inference/
│   └── generate.py                  InstructionGenerator  [single / batch / streaming]
│
├── scripts/
│   ├── train.py                     Main training CLI
│   ├── evaluate.py                  Evaluation CLI
│   └── merge_and_export.py          Merge LoRA → fp16 + optional GGUF export
│
└── tests/
    └── test_all.py                  40+ tests across 12 groups
```

<br/>

### LoRA Target Layers

LoRA adapters are attached to **7 linear projections** in each transformer block:

```
LLaMA-2 Transformer Block
│
├── Self-Attention
│   ├── q_proj   ← LoRA   query transformation
│   ├── k_proj   ← LoRA   key transformation
│   ├── v_proj   ← LoRA   value transformation
│   └── o_proj   ← LoRA   output projection
│
└── MLP (SwiGLU)
    ├── gate_proj  ← LoRA  gating branch
    ├── up_proj    ← LoRA  expansion branch
    └── down_proj  ← LoRA  contraction branch

Skipped (frozen):  embed_tokens · lm_head · layernorm / RMSNorm
```

<br/>



## 🎯 Prompt Formats

Three formats are supported out of the box — choose based on your dataset.

<br/>

### Alpaca *(default — single-turn instruction)*
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Explain quantum entanglement in simple terms.

### Response:
Quantum entanglement is a phenomenon where two particles...
```

### ChatML *(ShareGPT / multi-turn)*
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Explain quantum entanglement.<|im_end|>
<|im_start|>assistant
Quantum entanglement is...<|im_end|>
```

### LLaMA-2 Chat
```
[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

Explain quantum entanglement. [/INST] Quantum entanglement is...
```

<br/>



## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Offline test — no model download, no GPU required

```bash
python scripts/train.py --synthetic
python tests/test_all.py
```

### Real training — requires HuggingFace token + GPU

```bash
export HF_TOKEN=hf_...

# Standard Alpaca instruction tuning
python scripts/train.py --model meta-llama/Llama-2-7b-hf

# Faster: smaller rank, 1 epoch, 5k samples
python scripts/train.py --lora-r 8 --epochs 1 --max-samples 5000
```

<br/>



## 🏋️ Training

### CLI Reference

```
python scripts/train.py [OPTIONS]

  --model          meta-llama/Llama-2-7b-hf   HuggingFace model name
  --dataset        tatsu-lab/alpaca            Dataset identifier
  --prompt-style   alpaca                      alpaca · chatml · llama2 · simple
  --lora-r         16                          LoRA rank
  --lora-alpha     32                          LoRA alpha  (scaling = α/r)
  --epochs         3                           Training epochs
  --batch-size     4                           Per-device batch size
  --grad-accum     4                           Gradient accumulation steps
  --lr             2e-4                        Learning rate
  --seq-len        2048                        Max sequence length
  --output-dir     outputs/qlora               Checkpoint directory
  --wandb          my-project                  W&B project name  (optional)
  --synthetic      (flag)                      Offline test mode
  --max-samples    5000                        Limit dataset size
```

### Example — Multi-turn Chat Tuning

```bash
python scripts/train.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --dataset anon8231489123/ShareGPT_Vicuna_unfiltered \
  --prompt-style chatml \
  --lora-r 32 \
  --lora-alpha 64 \
  --epochs 2 \
  --wandb my-qlora-run
```

<br/>



## 📊 Evaluation

```bash
# Evaluate fine-tuned adapter
python scripts/evaluate.py --adapter outputs/qlora/final_adapter

# Compare with base model
python scripts/evaluate.py --adapter outputs/qlora/final_adapter --compare-base

# Offline mode
python scripts/evaluate.py --synthetic
```

**Metrics reported:**

| Metric | Description |
|:---|:---|
| `perplexity` | exp(NLL) on completion tokens |
| `token_accuracy` | Teacher-forced next-token prediction accuracy |
| `rouge_l` | ROUGE-L F1 on generated vs. reference outputs |
| `mean_length` | Mean generated response length in tokens |

<br/>



## 💬 Inference

### Interactive REPL

```python
from inference.generate import load_for_inference, InstructionGenerator, GenerationConfig
from data.formatting import get_formatter

model, tokenizer = load_for_inference(
    base_model_name="meta-llama/Llama-2-7b-hf",
    adapter_path="outputs/qlora/final_adapter",
)
gen = InstructionGenerator(
    model, tokenizer, get_formatter("alpaca"),
    GenerationConfig(max_new_tokens=512, temperature=0.7),
)
gen.interactive()
```

### Python API

```python
# Single response
response = gen.generate("Explain gradient descent")

# Batch inference
responses = gen.generate_batch(["Q1", "Q2", "Q3"])

# Streaming output
for token in gen.stream("Write a poem about Python."):
    print(token, end="", flush=True)
```

<br/>



## 📦 Merge and Export

Merge the LoRA adapter back into the base weights for standalone deployment:

```bash
# Merge adapter → fp16 model
python scripts/merge_and_export.py \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter    outputs/qlora/final_adapter \
  --output-dir outputs/qlora/merged_model

# Also export to GGUF (requires llama.cpp)
python scripts/merge_and_export.py ... --export-gguf
```

After merging, load like any standard HuggingFace model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("outputs/qlora/merged_model")
```

<br/>


## ⚙️ Configuration Reference

### ModelConfig

| Parameter | Default | Description |
|:---|:---:|:---|
| `model_name` | `meta-llama/Llama-2-7b-hf` | HuggingFace model identifier |
| `load_in_4bit` | `True` | Enable 4-bit NF4 quantization |
| `bnb_4bit_quant_type` | `nf4` | `nf4` (best) or `fp4` |
| `bnb_4bit_use_double_quant` | `True` | Double quantization |
| `max_seq_length` | `2048` | Context window |

### LoRAConfig

| Parameter | Default | Description |
|:---|:---:|:---|
| `r` | `16` | LoRA rank |
| `lora_alpha` | `32` | Scaling factor (α/r = 2.0) |
| `target_modules` | 7 LLaMA linear layers | Which layers receive adapters |
| `lora_dropout` | `0.05` | Dropout on adapter outputs |

### TrainingConfig

| Parameter | Default | Description |
|:---|:---:|:---|
| `optim` | `paged_adamw_32bit` | Paged optimizer (saves VRAM) |
| `learning_rate` | `2e-4` | Adam learning rate |
| `gradient_checkpointing` | `True` | Trade compute for memory |
| `bf16` | `True` | bfloat16 mixed precision |
| `gradient_accumulation_steps` | `4` | Effective batch = batch × accum |

<br/>



## 🧪 Testing

```bash
# Full test suite
python -m pytest tests/ -v

# Or directly
python tests/test_all.py
```

**40+ tests across 12 groups:**

| Group | Covers |
|:---:|:---|
| A | Config presets and validation |
| B | Alpaca · ChatML · LLaMA-2 · Simple formatters |
| C | Dataset loading and preprocessing |
| D | Data collator and padding |
| E | LoRA utilities and rank analysis |
| F | Model utilities and memory footprint |
| G | Synthetic model end-to-end |
| H | `InstructionDataset` correctness |
| I | Training callbacks |
| J | Metrics (PPL, ROUGE, accuracy) |
| K | Manual training loop |
| L | Inference generator (single / batch / stream) |

<br/>



## 📎 References

1. **Dettmers et al. (2023)** — *QLoRA: Efficient Finetuning of Quantized LLMs*  
   https://arxiv.org/abs/2305.14314

2. **Hu et al. (2022)** — *LoRA: Low-Rank Adaptation of Large Language Models*  
   https://arxiv.org/abs/2106.09685

3. **Touvron et al. (2023)** — *Llama 2: Open Foundation and Fine-Tuned Chat Models*  
   https://arxiv.org/abs/2307.09288

4. **Dettmers et al. (2022)** — *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*  
   https://arxiv.org/abs/2208.07339

5. **Wei et al. (2022)** — *Finetuned Language Models Are Zero-Shot Learners*  
   https://arxiv.org/abs/2109.01652

6. **PEFT Library** — HuggingFace Parameter-Efficient Fine-Tuning  
   https://github.com/huggingface/peft

7. **TRL Library** — Transformer Reinforcement Learning (SFTTrainer)  
   https://github.com/huggingface/trl

8. **bitsandbytes** — 4-bit and 8-bit quantization for PyTorch  
   https://github.com/TimDettmers/bitsandbytes

<br/>



<div align="center">



</div>
