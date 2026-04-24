---
title: "Attention Mechanism"
aliases: ["self-attention", "scaled dot-product attention", "multi-head attention", "MHA"]
tags: [transformers, architecture, core-concept]
related: ["[[transformer-architecture]]", "[[kv-cache]]", "[[flash-attention]]", "[[positional-encoding]]"]
sources: ["training-knowledge"]
interview_relevance: high
last_updated: 2026-04-22
status: current
---

# Attention Mechanism

## TL;DR
Attention lets each token in a sequence selectively gather information from every other token by computing weighted sums of value vectors. It is the core operation that gives transformers their power and is the dominant  topic for any AI engineer role.

## Intuition
Think of attention as a soft lookup table. You have a **query** (what you're looking for), **keys** (labels on each item in a library), and **values** (the actual content of each item). The query scores against every key; those scores become weights; you take a weighted sum of the values. Every token does this simultaneously, and the model learns which keys to pay attention to.

The "scaled" part divides by √d_k to prevent dot products from growing so large that softmax saturates into near-hard argmaxes (which kills gradients).

## Technical Detail

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V
```
- Q ∈ R^(n × d_k), K ∈ R^(n × d_k), V ∈ R^(n × d_v)
- n = sequence length, d_k = key/query dimension, d_v = value dimension
- Time complexity: O(n² · d)  — the n² factor is the bottleneck for long sequences
- Space complexity: O(n²) for the attention matrix (Flash Attention reduces this to O(n))

**Multi-Head Attention (MHA):**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)
```
- h heads with d_model/h dimension each — same total compute, richer representation
- Each head can attend to different positional patterns or semantic relations

**Causal (masked) self-attention:** for decoder/GPT-style models, future positions are masked to -∞ before softmax, so each position only attends to earlier positions.

**Cross-attention:** Q comes from the decoder, K and V come from the encoder output. Used in encoder-decoder models (T5, original transformer for MT).

## Variants & Extensions

| Variant | Change | Benefit |
|---|---|---|
| MHA | h parallel heads | Richer subspace learning |
| MQA (Multi-Query Attention) | One shared K/V head, h Q heads | Much smaller KV cache |
| GQA (Grouped-Query Attention) | g shared K/V groups | Balance between MHA quality and MQA speed |
| Sparse attention | Attend only to local/strided positions | O(n√n) instead of O(n²) |
| Linear attention | Kernel approximation of softmax | O(n) but quality loss |
| Flash Attention | IO-aware CUDA kernel | Same math, 2–4× faster, O(n) memory |

## Tradeoffs
| Advantage | Disadvantage |
|---|---|
| Global receptive field in one layer | O(n²) compute and memory cost |
| Parallelizable (vs RNNs) | No inherent positional order — needs PE |
| Expressive: any token can attend to any token | KV cache grows linearly with context length |
| Differentiable soft lookup | Large d_model → large weight matrices |

##  Angles

**What interviewers are really testing:**
- Can you derive the attention formula and explain each component?
- Do you understand *why* we scale by √d_k?
- Can you explain MHA vs MQA vs GQA and the KV cache motivation?
- Do you understand causal masking and why it's needed for generation?

**Common follow-up questions:**
- "What's the complexity of attention and how does Flash Attention improve it?"
- "Why does MQA reduce the KV cache size? Walk me through the math."
- "What happens if you remove the scaling factor?"
- "How would you implement efficient attention for a 100k token context?"
- "What's the difference between self-attention and cross-attention?"

**Gotchas / misconceptions:**
- Scaling by √d_k prevents softmax saturation — NOT about normalizing gradients
- Flash Attention does NOT change the mathematical output — it's purely an IO optimization
- GQA is used in most modern LLMs (Llama 3, Mistral) — MHA is legacy for large models
- The "attention weights" after softmax sum to 1 per query position, not globally

## Connections
- [[transformer-architecture]] — attention is the core building block; FFN + attention = transformer layer
- [[kv-cache]] — caches K and V projections to avoid recomputation during autoregressive generation
- [[flash-attention]] — IO-efficient CUDA implementation of the same math
- [[positional-encoding]] — RoPE applies rotations to Q and K inside attention to inject position info

## Sources
- Training knowledge (Vaswani et al. 2017 "Attention Is All You Need"; Ainslie et al. 2023 GQA paper)
