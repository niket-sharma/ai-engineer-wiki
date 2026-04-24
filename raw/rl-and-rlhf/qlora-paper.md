# QLoRA: Efficient Finetuning of Quantized LLMs

**Paper:** "QLoRA: Efficient Finetuning of Quantized LLMs"
**Authors:** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (University of Washington)
**Published:** 2023-05-23
**arXiv ID:** 2305.14314
**Key result:** Finetune a 65B model on a single 48GB GPU with no performance degradation vs full 16-bit finetuning

---

## Problem

Full finetuning of a 65B model requires ~780 GB of GPU memory (model weights + optimizer states + gradients). LoRA in float16 still requires ~30 GB for a 7B model. Neither is feasible for most practitioners.

**Goal:** Reduce memory enough to finetune 65B on a single consumer/research GPU.

---

## Core Innovations

### 1. 4-bit NormalFloat (NF4)

The key insight: pretrained neural network weights follow a **normal distribution** (mean 0, std ~1 after weight normalization). Standard int4 quantization uses uniform bins — wasteful for a non-uniform distribution.

**NF4 construction:**
1. Compute the 2^k + 1 quantiles of a standard normal distribution N(0, 1)
2. These become the NF4 codebook values (e.g., for 4-bit: 16 values)
3. Normalize each weight tensor to [-1, 1]
4. Map each weight to its nearest NF4 codebook value (nearest-neighbor quantization)

**Why it's better than int4:**
- More quantization levels where weights are dense (near 0)
- Fewer levels where weights are sparse (extremes)
- Information-theoretically optimal for normal distributions
- Experimentally matches or exceeds FP4 and int4 across all models tested

```python
# NF4 uses non-uniform quantization levels, not evenly spaced integers
# int4 levels: [-8, -7, -6, ..., 6, 7]  (uniform)
# NF4 levels: [-1.0, -0.694, -0.509, -0.394, ..., 0.694, 1.0]  (quantile-spaced for N(0,1))
```

### 2. Double Quantization

NF4 requires storing quantization constants (one per block, typically float32). These constants themselves take memory.

**Double quantization:** quantize the quantization constants.
- First quantization: weights → NF4 (4-bit), one quantization constant per 64-weight block
- Second quantization: those constants → float8, one constant per 256 blocks
- Memory saving: 0.373 bits per parameter (≈ 3 GB for a 65B model)

### 3. Paged Optimizers

CPU RAM is cheap; GPU VRAM is the bottleneck. Long sequences cause occasional GPU memory spikes that can crash training.

**Paged optimizer:** Use NVIDIA unified memory to automatically page optimizer states (Adam momentum, variance) to CPU when GPU memory is full, page back when needed.
- Implemented via `torch.cuda.make_non_blocking_copies()`
- Near-zero overhead for normal sequences; saves from OOM on anomalous long inputs

### 4. LoRA on the Quantized Base

Combine NF4 base model with LoRA adapters in BFloat16:

```
forward(x) = base_weight_NF4 · x  +  Δ_BF16 · x
           = dequantize(base_NF4) · x  +  (B_BF16 @ A_BF16) · x
```

**Key:** Gradients only flow through the BF16 LoRA adapters. The NF4 base weights are frozen. No gradient computation through the quantized base = massive memory savings.

**Dequantization at compute time:** NF4 weights are dequantized to BF16 only when needed for matrix multiplication, then immediately discarded.

---

## Memory Breakdown (Llama 65B)

| Component | Full Finetune | QLoRA |
|---|---|---|
| Model weights | 65B × 2 bytes = 130 GB | 65B × 0.5 bytes = 33 GB |
| Optimizer states (Adam) | 260 GB | Minimal (LoRA only) |
| Gradients | 130 GB | Minimal (LoRA only) |
| LoRA adapters | — | ~600 MB |
| **Total** | **~780 GB** | **~48 GB** |

Fits on a single A100 80GB or two A100 40GB GPUs.

---

## Guanaco: The Benchmark Model

The paper trained **Guanaco** — a 65B Llama model finetuned with QLoRA on 9,000 samples from OASST1 (Open Assistant).

**Results on Vicuna benchmark:**
- Guanaco-65B: 99.3% of ChatGPT quality (human evaluation)
- Guanaco-33B: 97.8%
- Guanaco-7B: 87.7%

**Training cost:** Guanaco-65B trained in ~36 hours on a single A100. Full cost: < $1,000 on cloud.

---

## QLoRA Implementation (bitsandbytes + PEFT)

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
import torch

# Step 1: Load model in NF4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",       # NF4 (not int4 or fp4)
    bnb_4bit_use_double_quant=True,  # double quantization
    bnb_4bit_compute_dtype=torch.bfloat16  # dequantize to bf16 for compute
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Step 2: Prepare for k-bit training (freeze base, set up gradient checkpointing)
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)

# Step 3: Add LoRA adapters
lora_config = LoraConfig(
    r=64,              # rank — paper uses r=64 for best quality
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # all linear layers
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 160,251,904 || all params: 6,914,734,080 || trainable%: 2.317
```

---

## Hyperparameter Recommendations

From the paper's ablations:

| Hyperparameter | Recommendation | Reasoning |
|---|---|---|
| **Rank r** | 64 | Higher rank = better on harder tasks; 16 is sufficient for easy tasks |
| **Target modules** | All linear layers | Paper shows all layers > q,v only |
| **alpha** | r (or 2r) | α/r scales learning rate — keep ~1 or 2 |
| **Batch size** | 16 total (micro × accum) | Larger is generally better |
| **Max seq length** | 512–2048 | Longer = paged optimizer more important |
| **Quantization** | NF4 + DQ | Always; int4 is worse |

---

## Key Findings from Ablations

1. **NF4 > int4 and FP4** on all benchmarks tested (MMLU, HellaSwag, etc.)
2. **Double quantization saves ~3GB** on 65B with < 0.01% degradation
3. **All linear layers > q+v only**: including FFN layers in LoRA improves quality
4. **Higher rank matters for harder tasks**: r=64 for mathematical reasoning; r=16 fine for instruction following
5. **16-bit LoRA base ≈ NF4 LoRA base**: no quality degradation from 4-bit quantization

---

## When to Use QLoRA vs Full LoRA

| Scenario | Use |
|---|---|
| < 7B model, ample GPU RAM | Full precision LoRA |
| 7B–13B, single GPU (24 GB) | QLoRA (NF4) |
| 30B+, single GPU or limited GPUs | QLoRA required |
| Production inference after finetuning | Merge LoRA → quantize separately for inference |
| Highest quality, no constraints | Full finetuning |

**Inference after QLoRA:** Two options:
1. Keep NF4 model + LoRA adapter separate (smaller, inference still requires dequantization overhead)
2. Merge adapter into dequantized FP16 model → then apply inference-only quantization (GPTQ/AWQ)

---

## Common  Questions

- "What is QLoRA and how does it differ from LoRA?"
- "Explain NF4 quantization. Why is it better than int4 for transformer weights?"
- "What is double quantization and how much memory does it save?"
- "How do paged optimizers prevent OOM errors in QLoRA training?"
- "Walk me through the memory savings that let QLoRA finetune a 65B model on one GPU."
- "What target modules should you apply LoRA to in a Llama model?"
