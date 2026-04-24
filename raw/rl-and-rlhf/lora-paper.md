# LoRA: Low-Rank Adaptation of Large Language Models

**Authors:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
**Published:** 2021-06-17
**arXiv ID:** 2106.09685
**URL:** https://arxiv.org/abs/2106.09685
**Venue:** ICLR 2022

---

## Abstract Summary

An important paradigm of NLP is large-scale pretraining followed by task-specific adaptation. Full fine-tuning updates all parameters, which is prohibitively expensive for large models. LoRA proposes to freeze the pretrained model weights and inject trainable low-rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. LoRA performs on-par or better than full fine-tuning on several benchmarks while having no additional inference latency.

---

## Core Motivation

Full fine-tuning a 175B parameter model (GPT-3):
- Stores a full copy of gradients and optimizer states: ~700B parameters worth of states (3× model size for AdamW)
- Prohibitively expensive and slow

**Key hypothesis:** The weight update during fine-tuning has **low intrinsic rank** — the task adaptation can be expressed in a much lower-dimensional subspace than the full weight matrix.

Evidence: Aghajanyan et al. (2020) showed that pre-trained language models have a low intrinsic dimensionality — they can be fine-tuned effectively in a small subspace.

---

## LoRA Formulation

For a pretrained weight matrix W_0 ∈ R^(d×k), LoRA constrains the update by representing it with a low-rank decomposition:

```
W_0 + ΔW = W_0 + B·A
```

Where:
- B ∈ R^(d×r), A ∈ R^(r×k)
- Rank r << min(d, k) — typically r = 4, 8, 16, or 64
- W_0 is **frozen** during training
- B and A are the only parameters being trained

**Forward pass:**
```
h = W_0·x + ΔW·x = W_0·x + B·A·x
```

**Initialization:**
- A: initialized from Gaussian distribution N(0, σ²)
- B: initialized to **zero**

Why zero initialization for B? At the start of training, B·A = 0 → the update ΔW = 0 → the model starts from exactly the pretrained weights. Ensures training stability.

**Scaling:**
```
h = W_0·x + (α/r) · B·A·x
```
- α is a constant (often set equal to r, making α/r = 1)
- Scaling by α/r makes the update magnitude independent of r — easier hyperparameter tuning

---

## Parameter Count Analysis

| Component | Full fine-tuning | LoRA (r=16) | Ratio |
|---|---|---|---|
| d×k weight matrix | d×k | r×d + r×k = r(d+k) | r/(min(d,k)) |
| GPT-3 175B (d=12288) | 175B | ~4.7M (r=4, Wq+Wv only) | 0.0027% |
| Llama 3 8B (d=4096, r=16) | 8B | ~16M | 0.2% |

**Adapter count (Llama 3 8B, r=16, Wq+Wk+Wv+Wo only):**
```
4 matrices × 32 layers × 2 × 4096 × 16 = 16.7M params
```

---

## Which Layers to Adapt

The paper studies adapting different weight matrices in attention layers (W_q, W_k, W_v, W_o) and FFN layers.

**Paper findings:**
- Adapting W_q and W_v gives the best results vs. parameter budget
- Adapting all four attention matrices (W_q, W_k, W_v, W_o) is better when you have budget
- FFN layer adaptation helps further

**Modern practice (post-paper):**
- Adapt all linear layers including FFN up/down/gate projections
- `lm_head` and embedding are usually excluded (or included only if domain vocab shift is large)
- Hugging Face PEFT default: `target_modules="all-linear"`

---

## Rank Selection Guidelines

| r | Trainable params (8B) | When |
|---|---|---|
| 4 | ~8M | Very simple task, minimal compute |
| 8 | ~16M | Light instruction following |
| 16 | ~33M | General PEFT baseline (most common) |
| 32 | ~65M | More complex domain adaptation |
| 64 | ~131M | Complex tasks, available compute |
| 128+ | ~260M+ | Diminishing returns; consider full fine-tune |

**Insight:** The paper shows r=4 often suffices for GLUE tasks. For instruction following (SFT), r=8–16 is typical. For complex reasoning or large domain shift, r=32–64.

---

## Inference: Zero Additional Latency

After training, merge the adapter back into the base weights:
```python
W_merged = W_0 + (alpha/r) * B @ A
```

This is a one-time operation. The merged model is identical in size and inference speed to the original pretrained model — no adapter overhead at serving time.

**Alternatively:** Keep adapters separate for serving multiple adapters (e.g., different LoRA adapters for different tasks on the same base model). This requires adapter switching at inference time.

---

## Experimental Results

**GPT-3 on NLU benchmarks:**

| Method | WikiSQL | MNLI-m | SAMSum |
|---|---|---|---|
| Full fine-tune | 73.8 | 89.7 | 53.0 |
| LoRA (r=4) | 73.4 | 91.7 | 53.8 |
| LoRA (r=8) | 74.0 | 91.6 | 53.4 |

LoRA matches or exceeds full fine-tuning at <0.5% of the parameters.

**Compared to other PEFT methods:**

| Method | Trainable params | Quality |
|---|---|---|
| Prefix Tuning | Sequence length overhead | Lower quality, hard to optimize |
| Adapter layers | Per-layer overhead (~5%) | Inference latency overhead |
| LoRA | Weight matrices only | No inference overhead, best quality |

---

## Why LoRA Works: Theoretical Perspective

**Low-rank hypothesis:** The paper shows empirically that W_q and W_v updates during full fine-tuning have low "intrinsic rank" — most of the update's information is captured in the top few singular values.

Specifically: for GPT-3 fine-tuned on WikiSQL:
- W_q update ΔW has effective rank ~5
- W_v update ΔW has effective rank ~2

This validates LoRA's constraint — you're not losing much by forcing rank r=4 or r=8.

**Why not just use SVD on the full update?** You'd need to do full fine-tuning first to get ΔW. LoRA trains the low-rank factors directly from scratch.

---

## Practical Implementation

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                           # rank
    lora_alpha=32,                  # scaling (alpha/r = 2)
    target_modules="all-linear",    # which modules to adapt
    lora_dropout=0.05,              # regularization
    bias="none",                    # whether to adapt bias terms
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 16,777,216 || all params: 8,046,927,872 || trainable%: 0.2085
```

---

## Variants and Extensions

| Method | Change | Key Benefit |
|---|---|---|
| **QLoRA** (Dettmers 2023) | 4-bit NF4 base model + LoRA | 4× memory reduction |
| **DoRA** (Liu 2024) | Decompose into magnitude + direction | Better than LoRA on many tasks |
| **AdaLoRA** (Zhang 2023) | SVD-based adaptive rank per layer | Better quality at same param budget |
| **LoRA+** (Hayou 2024) | Different LR for A and B matrices | 1.2× faster convergence |
| **LoftQ** (Li 2023) | Initialize LoRA to compensate quant error | Better QLoRA starting point |
| **FLORA** | Full-rank gradient with low-rank update | Theoretical: best of both |

---

## -Relevant Insights

**The non-obvious question: "Why two matrices instead of one?"**
If we used a single matrix C ∈ R^(d×k) with rank constraint, we'd need to do SVD during training to maintain the rank constraint — computationally expensive. Using B·A explicitly factorizes the low-rank update: the product B·A always has rank ≤ r by construction, and training is just SGD on B and A.

**The weight merging insight:** Many interviewers don't know that LoRA has zero inference overhead after merging. Be ready to explain: after training, you just add `(α/r)·B·A` to `W_0` — one matrix add, then the model is back to its original structure.

**Practical wisdom:** r=16 is the go-to default. Increase if results are disappointing, decrease if memory is tight. `alpha=r` (scaling=1) is a safe default.

---

## Common  Questions From This Paper

- "Walk me through LoRA mathematically. Why two matrices B and A?"
- "Why is B initialized to zero in LoRA?"
- "How does LoRA achieve zero inference latency?"
- "What rank should you choose for LoRA? What factors affect this?"
- "When would you use full fine-tuning instead of LoRA?"
- "What is QLoRA and how does it extend LoRA?"
