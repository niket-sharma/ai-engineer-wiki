---
title: "Math & Notation Cheatsheet"
aliases: ["math cheatsheet", "transformer math", "attention formula"]
tags: [cheatsheet, math, transformers]
related: ["[[attention-mechanism]]", "[[transformer-architecture]]", "[[lora-qlora]]"]
sources: ["training-knowledge"]
interview_relevance: high
last_updated: 2026-04-22
status: current
---

# Math & Notation Cheatsheet

Quick-reference for formulas you need to recall on a whiteboard.

---

## Attention

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V
```

| Symbol | Meaning | Typical value |
|---|---|---|
| Q, K | Query, Key matrices | R^(n × d_k) |
| V | Value matrix | R^(n × d_v) |
| d_k | Key/Query dimension | d_model / n_heads |
| n | Sequence length | 2k–128k |
| d_model | Model dimension | 768–8192 |

**Multi-Head:**
```
MHA(Q,K,V) = Concat(head_1,...,head_h) · W_O
head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)
```

**Complexity:** Time O(n²d), Space O(n²) → Flash Attention: Space O(n)

---

## Parameter Count (Decoder-Only Transformer)

```
Total params ≈ 12 · n_layers · d_model²  +  vocab_size · d_model
```

| Component | Params |
|---|---|
| Each attention layer | 4 · d_model² (W_Q, W_K, W_V, W_O) |
| Each FFN layer | 8 · d_model² (W_1 up/gate, W_2 down, SwiGLU) |
| Per layer total | ~12 · d_model² |
| Embedding | vocab_size × d_model |

**Examples:**
| Model | d_model | n_layers | Approx params |
|---|---|---|---|
| GPT-2 small | 768 | 12 | ~117M |
| Llama 3 8B | 4096 | 32 | ~8B |
| Llama 3 70B | 8192 | 80 | ~70B |

---

## KV Cache Memory

```
KV cache bytes = 2 × n_layers × n_kv_heads × d_head × seq_len × bytes_per_element
```

**Llama 3 8B example (bfloat16, 4096 tokens):**
```
= 2 × 32 × 8 × 128 × 4096 × 2 = 536 MB
```

| Context | KV Cache (Llama 3 8B) |
|---|---|
| 4k tokens | 536 MB |
| 8k tokens | 1.07 GB |
| 32k tokens | 4.3 GB |
| 128k tokens | 17 GB |

---

## LoRA

```
W' = W + ΔW = W + B·A
B ∈ R^(d×r),  A ∈ R^(r×d),  r << d
```

Trainable params per layer: `2 · d · r`

Scaling: effective ΔW is `(α/r) · B·A`  (α typically = r, making scale = 1)

**LoRA param count (Llama 3 8B, r=16, adapting Q/K/V/O × 32 layers):**
```
4 layers × 32 layers × 2 × 4096 × 16 = ~16.8M params  (0.21% of 8B)
```

---

## RLHF / DPO

**RLHF objective:**
```
max_π E_{x,y~π}[r(x,y)] - β · KL[π(y|x) || π_ref(y|x)]
```

**DPO loss:**
```
L_DPO = -E[log σ(β · log[π_θ(y_w|x)/π_ref(y_w|x)] - β · log[π_θ(y_l|x)/π_ref(y_l|x)])]
```

**Bradley-Terry RM loss:**
```
L_RM = -log σ(r(y_chosen) - r(y_rejected))
```

---

## RoPE (Rotary Position Embedding)

```
q_m = R_m · q,   k_n = R_n · k
q_m^T · k_n = q^T · R_{n-m} · k    (encodes relative position n-m)
```

R_m is a rotation matrix parameterized by position m and frequency θ.

---

## Softmax & Cross-Entropy

```
softmax(z_i) = exp(z_i) / Σ_j exp(z_j)
```

```
L_CE = -Σ_i y_i · log(p_i) = -log(p_correct)
```

**Perplexity:**
```
PPL = exp(-1/N · Σ_i log p(token_i))
```
Lower is better. GPT-2 ~30 PPL on WikiText-103. Llama 3 8B ~7 PPL.

---

## Transformer FFN (SwiGLU variant — modern LLMs)

```
FFN(x) = (SiLU(x·W_gate) ⊙ (x·W_up)) · W_down
```

Standard ReLU variant: `FFN(x) = max(0, xW_1) · W_2`

SwiGLU: `SiLU(x) = x · σ(x)` — smooth activation, empirically better.

---

## Retrieval (RRF)

```
RRF(d) = Σ_r  1 / (k + rank_r(d))     k = 60 standard
```

Fuses rankings from dense and sparse retrieval without score normalization.

---

## GPU Memory Quick Reference

| Model | FP16 | INT8 | INT4 |
|---|---|---|---|
| Llama 3 8B | 16 GB | 8 GB | 4 GB |
| Llama 3 70B | 140 GB | 70 GB | 35 GB |
| Llama 3 405B | 810 GB | 405 GB | ~200 GB |

Minimum GPU to serve (weights only, no KV cache):
- 8B in FP16: 1× A100-80GB (tight), 2× A100-40GB
- 70B in FP16: 2× A100-80GB (TP=2)
- 70B in INT4: 1× A100-80GB (just fits)
