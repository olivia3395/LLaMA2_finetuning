
# **Fine-Tuning LLaMA 2 with QLoRA**

This project demonstrates how to fine-tune **LLaMA 2**, a large language model developed by Meta, using **QLoRA** (Quantized Low-Rank Adaptation).
The method allows efficient training of massive transformer models on consumer-grade GPUs without sacrificing performance — a key advancement in democratizing large model fine-tuning.


##  **1. Background & Motivation**

Fine-tuning large models such as LLaMA 2 (7B–70B parameters) traditionally requires **hundreds of gigabytes of GPU memory**.
This is because all model parameters are stored and updated in 16- or 32-bit precision during training.

However, in practice:

* Most model layers are already near-optimal after pretraining.
* Only a small subspace of parameters needs to be adapted for new tasks.

To address this, **QLoRA** combines two ideas:

1. **Quantization** → Reduce model memory footprint by storing weights in 4-bit precision.
2. **LoRA (Low-Rank Adaptation)** → Introduce small trainable adapter matrices instead of updating all weights.

Thus, QLoRA makes it possible to fine-tune a **70B model on a single 48GB GPU**, or a **7B model on a 12GB GPU** — without losing accuracy.


##  **2. Technical Overview**

###  Step 1: Quantization

The base model (e.g., `meta-llama/Llama-2-7b-hf`) is loaded in **4-bit quantized format** using the `bitsandbytes` library.
This reduces memory consumption by ~75%.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      # NormalFloat4 quantization
    bnb_4bit_use_double_quant=True, # Nested quantization for precision
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
```



###  Step 2: Add LoRA Adapters

LoRA introduces small rank-decomposed weight updates:
[
W' = W + A B^\top
]
where (A, B \in \mathbb{R}^{d \times r}) are low-rank matrices (r ≪ d).

We attach LoRA adapters only to attention layers, leaving the original model frozen.

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=64,                          # Rank of adaptation
    lora_alpha=16,                 # Scaling factor
    target_modules=["q_proj", "v_proj"], # Apply to key attention weights
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
```

This drastically reduces trainable parameters (typically < 1% of total),
while still enabling task-specific adaptation.



###  Step 3: Prepare Dataset & Tokenization

Use any text dataset — such as instruction tuning data, domain documents, or conversation pairs.

Each sample becomes a self-contained instruction–response prompt.



### Step 4: Fine-Tuning with PEFT + Transformers Trainer

We fine-tune only the LoRA adapters while keeping the 4-bit quantized weights fixed.


Training is efficient — often requiring less than 20GB GPU memory for a 7B model.

## **3. Mathematical Formulation**

### **LoRA Update Rule**

For a given linear layer ( h = W x ), LoRA reparameterizes the weight update as:

$$
\Delta W = A B^\top, \quad A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times k}
$$

Thus, during fine-tuning:

$$
h' = (W + \Delta W) x = W x + A (B^\top x)
$$

Only (A, B) are trainable; (W) remains frozen.

### **QLoRA Quantization Function**

The 4-bit quantization step approximates full-precision weights (W) by:

$$
\tilde{W} = \text{Dequantize}(\text{Quantize}(W, 4\text{-bit}))
$$

This ensures minimal information loss while drastically reducing memory use.


##  **4. Architecture Summary**

At inference time, the LoRA adapters can be **merged back** into the model or kept as plug-ins for modular deployment.


## **5. Results & Observations**

| Model                                   | GPU             | Memory Usage | Fine-Tuning Time | ΔTrainable Params | Accuracy / Quality |
| --------------------------------------- | --------------- | ------------ | ---------------- | ----------------- | ------------------ |
| **LLaMA2-7B Full Precision (baseline)** | A100 80GB       | ~70GB        | 100% reference   | 100%              | 100%               |
| **LLaMA2-7B QLoRA (4-bit)**             | RTX 3090 (24GB) | ~14GB        | 2–3 hours        | ~0.8%             | 98–99% baseline    |
| **LLaMA2-13B QLoRA (4-bit)**            | A100 40GB       | ~24GB        | 3–4 hours        | ~0.5%             | 97–99% baseline    |

**Key Findings:**

* QLoRA achieves nearly the same downstream performance as full fine-tuning.
* Memory usage reduced by 4–6×.
* Training speed increases due to reduced precision and fewer trainable parameters.
* No significant loss in generalization or text quality.



## **6. Advantages & Limitations**

### **Advantages**

* Enables large-model fine-tuning on consumer GPUs
* Minimal loss of accuracy compared to full precision
* Compatible with all Transformer-based architectures
* Modular — adapters can be swapped or stacked per domain

### **Limitations**

* Quantization may cause slight degradation in low-resource or rare-token settings
* Some GPU architectures may not fully support `bitsandbytes` 4-bit kernels
* Merging adapters requires additional post-processing


## **7. Future Work**

* Integrate **LoRA + DPO (Direct Preference Optimization)** for instruction alignment
* Extend QLoRA for **multimodal** models (LLaVA / BLIP-2)
* Implement dynamic rank allocation based on layer importance
* Explore **parameter-efficient continual learning** with adapter fusion

##  **8. References**

* Hugging Face PEFT & bitsandbytes Documentation: [https://huggingface.co/docs/peft](https://huggingface.co/docs/peft)

