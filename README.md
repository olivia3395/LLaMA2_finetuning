# QLoRA Fine-Tuning — LLaMA 2 Instruction Tuning on Consumer GPUs

## Overview

**QLoRA** = Quantization + LoRA. It enables fine-tuning of 7B+ parameter models
on a single 8–24 GB consumer GPU by combining:

| Technique | Memory Savings | Quality Cost |
|-----------|---------------|-------------|
| 4-bit NF4 quantization (bitsandbytes) | 4× vs fp16 | ~0.1% PPL increase |
| Double quantization | +0.37 GB extra savings | negligible |
| LoRA adapters (r=16) | train 0.4% of params | minimal for instruction tuning |
| Paged AdamW optimizer | offload optimizer states to CPU | none |
| Gradient checkpointing | 5–10× activation savings | 20–30% slower |

**Result**: Fine-tune LLaMA-2-7B on a **10 GB GPU** (RTX 3080) in 2–5 hours.



## Theory: QLoRA in Depth

### 4-bit NF4 Quantization

**Normal Float 4 (NF4)** is a data type designed specifically for neural network weights:

```
Insight: Pre-trained neural network weights follow a normal distribution N(0, σ²)

NF4 grid: choose 16 values that minimize quantization error under N(0,1)
         (information-theoretically optimal for normally-distributed data)

Standard INT4: uniform grid → wastes precision in the tails
NF4:           non-uniform grid → concentrates precision where weights are dense
```

Quantization formula:
```
W_nf4 = nf4_quantize(W / absmax(W))   # normalize then quantize to 16 levels
W_fp  = nf4_dequantize(W_nf4) * absmax(W)   # dequantize for computation
```

**Double quantization**: quantize the quantization constants (`absmax` values)
themselves using 8-bit — saves an additional 0.37 GB for 7B models.

### LoRA Mathematics

For a frozen weight W₀ ∈ R^{d×k}, LoRA parameterises updates as:

```
h = W₀x + ΔWx = W₀x + (α/r) · B · A · x

where:
  A ∈ R^{r×k}   initialized with N(0, 1/r²)   [randomly initialized]
  B ∈ R^{d×r}   initialized with zeros         [so ΔW = 0 at start]
  r << min(d,k) [rank, typically 4-64]
  α              [scaling, typically 2r]
```

During training:
- W₀ is **frozen** (stored in 4-bit, never updated)
- Only A, B are trained in **full fp32/bf16** precision
- The asymmetric init ensures training is stable (ΔW starts at 0)

### Why Target These Layers?

```
LLaMA-2 Transformer Block:
  Self-Attention:
    q_proj  ←  LoRA  (query transformation)
    k_proj  ←  LoRA  (key transformation)
    v_proj  ←  LoRA  (value transformation)
    o_proj  ←  LoRA  (output projection)
  MLP (SwiGLU):
    gate_proj  ←  LoRA  (gating branch)
    up_proj    ←  LoRA  (expansion branch)
    down_proj  ←  LoRA  (contraction branch)

  Skipped (kept frozen):
    embed_tokens   (embedding lookup, no matmul benefit)
    lm_head        (sensitive to accuracy, small)
    layernorm/RMSNorm  (tiny, few parameters)
```

### Training Objective

Standard next-token prediction (language modeling):

```
L = -1/|C| · Σ_{t∈C} log P(w_t | w_{<t})

where C is the set of completion token positions
(prompt tokens are masked with label=-100, excluded from loss)
```

This is called **instruction masking** — we only compute loss on the response,
not the instruction. This prevents the model from "forgetting" the instruction
format by over-fitting on prompt tokens.


## Memory Requirements

### GPU VRAM by Model Size

| Model | Min VRAM | Recommended | Batch Size |
|-------|----------|-------------|-----------|
| LLaMA-2-7B  | 8 GB  | 16 GB | 1-4 |
| LLaMA-2-13B | 12 GB | 24 GB | 1-2 |
| LLaMA-2-70B | 48 GB | 80 GB | 1 (multi-GPU) |

### Memory Breakdown (7B, batch=4, seq=2048)

```
4-bit base model:          ~3.5 GB
LoRA parameters (r=16):    ~0.05 GB
Gradients (LoRA only):     ~0.05 GB
Paged AdamW optimizer:     ~0.2  GB  (partially on CPU)
Activations (GC enabled):  ~2.5  GB
KV cache + overhead:       ~0.7  GB
─────────────────────────────────────
Total:                     ~7.0  GB  ← fits RTX 3080 (10 GB)
```



## Architecture

```
qlora_ft/
├── config.py                    ← ModelConfig, LoRAConfig, DataConfig, TrainingConfig
│
├── data/
│   ├── formatting.py            ← AlpacaFormatter, ChatMLFormatter, LLaMA2Formatter, SimpleFormatter
│   └── dataset.py               ← InstructionDataset, InstructionDataCollator, build_datasets
│
├── model/
│   ├── loader.py                ← load_model_and_tokenizer() (4-bit NF4 + bitsandbytes)
│   ├── lora.py                  ← attach_lora() (PEFT), weight stats, effective rank
│   └── utils.py                 ← count_parameters, memory footprint, merge_lora_weights
│
├── training/
│   ├── trainer.py               ← QLoRATrainer (wraps trl.SFTTrainer / HF Trainer)
│   ├── callbacks.py             ← LoRACheckpoint, EarlyStopping, MemoryLog, GradientStats
│   └── metrics.py               ← compute_perplexity, token_accuracy, ROUGE-L, evaluate_generation
│
├── inference/
│   └── generate.py              ← InstructionGenerator (single/batch/streaming), load_for_inference
│
├── scripts/
│   ├── train.py                 ← Main training CLI
│   ├── evaluate.py              ← Evaluation CLI (PPL, ROUGE, generation quality)
│   └── merge_and_export.py      ← Merge LoRA → fp16 + optional GGUF export
│
└── tests/
    └── test_all.py              ← 40+ tests across 12 groups
```



## Prompt Formats

### Alpaca (default)
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Explain quantum entanglement in simple terms.

### Response:
Quantum entanglement is a phenomenon where two particles...
```

### ChatML (for ShareGPT / multi-turn)
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



## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Offline test (no model download, no GPU needed)
```bash
python scripts/train.py --synthetic
python tests/test_all.py
```

### Real training (requires HF token + GPU)
```bash
export HF_TOKEN=hf_...

# Standard Alpaca instruction tuning
python scripts/train.py --model meta-llama/Llama-2-7b-hf

# Faster: smaller rank, 1 epoch
python scripts/train.py --lora-r 8 --epochs 1 --max-samples 5000
```



## Training

```bash
python scripts/train.py [OPTIONS]

  --model          meta-llama/Llama-2-7b-hf   HF model name
  --dataset        tatsu-lab/alpaca            Dataset
  --prompt-style   alpaca                      alpaca/chatml/llama2/simple
  --lora-r         16                          LoRA rank
  --lora-alpha     32                          LoRA alpha (scaling = α/r)
  --epochs         3                           Training epochs
  --batch-size     4                           Per-device batch size
  --grad-accum     4                           Gradient accumulation (eff. batch = 16)
  --lr             2e-4                        Learning rate
  --seq-len        2048                        Max sequence length
  --output-dir     outputs/qlora               Checkpoint directory
  --wandb          my-project                  W&B project name
  --synthetic      (flag)                      Offline test mode
  --max-samples    5000                        Limit dataset size
```

**Example: Multi-turn chat tuning**
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


## Evaluation

```bash
# Evaluate fine-tuned adapter
python scripts/evaluate.py --adapter outputs/qlora/final_adapter

# Compare with base model
python scripts/evaluate.py --adapter outputs/qlora/final_adapter --compare-base

# Offline
python scripts/evaluate.py --synthetic
```

**Metrics reported:**
- `perplexity` — exp(NLL) on completion tokens
- `token_accuracy` — teacher-forced next-token prediction accuracy
- `rouge_l` — ROUGE-L F1 on generated vs reference outputs
- `mean_length` — mean generated response length


## Inference

```bash
# Interactive REPL
python -c "
from inference.generate import load_for_inference, InstructionGenerator, GenerationConfig
from data.formatting import get_formatter

model, tokenizer = load_for_inference(
    base_model_name='meta-llama/Llama-2-7b-hf',
    adapter_path='outputs/qlora/final_adapter',
)
gen = InstructionGenerator(
    model, tokenizer, get_formatter('alpaca'),
    GenerationConfig(max_new_tokens=512, temperature=0.7),
)
gen.interactive()
"
```

**Python API:**
```python
from inference.generate import InstructionGenerator, GenerationConfig

# Single response
response = gen.generate("Explain gradient descent")

# Batch
responses = gen.generate_batch(["Q1", "Q2", "Q3"])

# Streaming
for token in gen.stream("Write a poem about Python."):
    print(token, end="", flush=True)
```



## Merge and Export

```bash
# Merge adapter into base model (fp16)
python scripts/merge_and_export.py \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter    outputs/qlora/final_adapter \
  --output-dir outputs/qlora/merged_model

# Also export to GGUF (requires llama.cpp)
python scripts/merge_and_export.py ... --export-gguf
```

After merging, load like any standard HF model:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("outputs/qlora/merged_model")
```


## Configuration Reference

### ModelConfig
| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `meta-llama/Llama-2-7b-hf` | HF model identifier |
| `load_in_4bit` | `True` | Enable 4-bit NF4 quantization |
| `bnb_4bit_quant_type` | `nf4` | `nf4` (best) or `fp4` |
| `bnb_4bit_use_double_quant` | `True` | Double quantization |
| `max_seq_length` | `2048` | Context window |

### LoRAConfig
| Parameter | Default | Description |
|-----------|---------|-------------|
| `r` | `16` | LoRA rank |
| `lora_alpha` | `32` | Scaling factor (α/r = 2.0) |
| `target_modules` | 7 LLaMA linear layers | Which layers to add adapters |
| `lora_dropout` | `0.05` | Dropout on adapter outputs |

### TrainingConfig
| Parameter | Default | Description |
|-----------|---------|-------------|
| `optim` | `paged_adamw_32bit` | Paged optimizer (saves VRAM) |
| `learning_rate` | `2e-4` | Adam LR |
| `gradient_checkpointing` | `True` | Trade compute for memory |
| `bf16` | `True` | bfloat16 training |
| `gradient_accumulation_steps` | `4` | Effective batch = batch × accum |


## Testing

```bash
python -m pytest tests/ -v
# or
python tests/test_all.py
```

**40+ tests across 12 groups:**
A — Config, B — Alpaca/ChatML/LLaMA2/Simple formatters,
C — Dataset loading, D — Data collator, E — LoRA utilities,
F — Model utilities, G — Synthetic model, H — InstructionDataset,
I — Training callbacks, J — Metrics, K — Manual training loop,
L — Inference generator



## References

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
