---
title: "LoRA & QLoRA"
aliases: ["LoRA", "QLoRA", "low-rank adaptation", "parameter-efficient fine-tuning", "PEFT", "DoRA"]
tags: [fine-tuning, alignment, peft]
related: ["[[rlhf]]", "[[transformer-architecture]]", "[[dpo]]", "[[grpo]]"]
sources: ["training-knowledge"]
interview_relevance: high
last_updated: 2026-04-22
status: current
---

# LoRA & QLoRA

## TL;DR
LoRA fine-tunes a pretrained model by adding low-rank decomposition matrices to frozen weight matrices — instead of updating all W (d×d), you train two small matrices A (d×r) and B (r×d) where r << d. QLoRA extends this by quantizing the frozen base model to 4-bit (NF4), enabling fine-tuning of 65B+ models on a single A100. Both are now the default approach for instruction tuning and preference optimization.

## Intuition
Full fine-tuning updates every weight in the model. For a 7B model that's 7 billion gradient updates per step — expensive in compute and memory.

LoRA's insight: the weight update ΔW during fine-tuning has low intrinsic rank. If the task adaptation lives in a low-dimensional subspace, we don't need to represent ΔW as a full d×d matrix. Instead:
```
W' = W + ΔW = W + B·A   where B ∈ R^(d×r), A ∈ R^(r×d), r << d
```
W is frozen. Only A and B are trained. At inference, B·A is merged into W — zero additional latency.

QLoRA stacks 4-bit quantization on top: the frozen W is stored in NF4 (a new 4-bit datatype optimized for normally-distributed weights), reducing memory 4× with minimal quality loss.

## Technical Detail

**LoRA parameterization:**
```
h = Wx + BAx = Wx + ΔWx
```
- A initialized from Gaussian (N(0, σ²)), B initialized to zeros → ΔW = 0 at start
- Scaling factor: ΔW is multiplied by α/r to control update magnitude
- Only trains 2 × r × d parameters per adapted layer instead of d × d

**Parameter count example — Llama 3 8B, r=16, α=32:**
- Adapted layers: Q, K, V, O projections × 32 layers = 128 weight matrices
- Each: 2 × 16 × 4096 = ~131k params
- Total LoRA params: ~16M (0.2% of 8B) — yet achieves >90% of full fine-tune quality

**Which layers to adapt:**
- Common: Q, V projections (original LoRA paper)
- Better: Q, K, V, O, and sometimes up/gate/down projections in FFN
- QLoRA paper shows adapting all linear layers is best

**QLoRA specifics:**
- NF4 quantization: 4-bit datatype with bins optimized for N(0,1) distribution (LLM weights are approximately normal)
- Double quantization: quantize the quantization constants themselves for extra memory savings
- Paged optimizers: use CPU RAM as overflow for optimizer states
- Result: fine-tune 65B model on a single 48GB A100

**LoRA rank selection:**
| Rank r | Trainable params | When to use |
|---|---|---|
| 4–8 | Minimal | Simple instruction following |
| 16 | Moderate | General PEFT |
| 64–128 | Higher | Complex tasks, when you can afford it |
| Full rank | = full fine-tune | Rarely needed with PEFT |

## Variants & Extensions

| Method | Change from LoRA | Benefit |
|---|---|---|
| QLoRA | 4-bit NF4 quantization of base model | 4× memory reduction |
| DoRA | Decompose weight into magnitude + direction, apply LoRA to direction | Better than LoRA on some tasks |
| LoRA+ | Different learning rates for A and B | 1.2× faster convergence |
| AdaLoRA | SVD-based adaptive rank allocation per layer | Better quality at same param count |
| LoftQ | Initialize LoRA to compensate for quantization error | Better QLoRA starting point |
| ORPO | LoRA with preference optimization built into SFT loss | No separate reward model needed |

## Tradeoffs
| Advantage | Disadvantage |
|---|---|
| 100–10000× fewer trainable parameters | Lower capacity than full fine-tune for very complex tasks |
| Can merge into base model — zero inference latency | Rank is a fixed hyperparameter (AdaLoRA helps) |
| Works with 4-bit base models (QLoRA) | QLoRA: bfloat16 compute with 4-bit storage → still needs GPU with large VRAM |
| Multiple LoRA adapters can be swapped at serving time | Adapter merging can cause interference if not careful |

## Interview Angles

**What interviewers are really testing:**
- Can you explain the low-rank decomposition mathematically?
- Do you understand why ΔW = BA and not just a single matrix?
- Can you explain the memory savings from QLoRA end-to-end?
- Do you know which layers to adapt and what rank to use?

**Common follow-up questions:**
- "Walk me through LoRA mathematically. Why two matrices instead of one?"
- "Why is B initialized to zeros in LoRA?"
- "How does QLoRA allow fine-tuning a 70B model on one GPU?"
- "What is NF4? Why use it instead of int4?"
- "When would you choose full fine-tuning over LoRA?"
- "How do you serve multiple LoRA adapters efficiently in production?"

**Gotchas / misconceptions:**
- LoRA merged weights have ZERO additional inference latency — the adapter is merged before serving
- The α/r scaling factor is NOT the same as learning rate — don't confuse them
- QLoRA stores the base model in 4-bit but computes in bfloat16 — dequantize on the fly
- LoRA doesn't update the embedding or LM head by default — worth checking in implementations

## Connections
- [[rlhf]] — LoRA is the standard way to fine-tune the policy model in RLHF pipelines
- [[dpo]] — DPO is commonly run with LoRA as the fine-tuning backbone
- [[transformer-architecture]] — LoRA adapters are inserted into transformer linear weight matrices

## Sources
- Training knowledge (Hu et al. 2022 "LoRA"; Dettmers et al. 2023 "QLoRA")
