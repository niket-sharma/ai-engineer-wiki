---
title: "Positional Encoding"
aliases: ["PE", "RoPE", "ALiBi", "sinusoidal encoding", "rotary position embedding"]
tags: [transformers, architecture]
related: ["[[transformer-architecture]]", "[[attention-mechanism]]"]
sources: ["training-knowledge"]
interview_relevance: high
last_updated: 2026-04-22
status: current
---

# Positional Encoding

## TL;DR
Attention is permutation-invariant — shuffle the input tokens and you get the same output. Positional encodings inject order information. The field has evolved from fixed sinusoidal (original transformer) to learned absolute to RoPE (now dominant) to ALiBi (good length extrapolation).

## Intuition
Attention computes dot products between token representations — it doesn't know if two tokens are adjacent or 1000 positions apart. PE fixes this by adding (or multiplying) position-dependent information into the representations before or during attention.

The key challenge is **length generalization**: training on sequences of length 2048 and then needing to handle 4096 or 128k tokens at inference. Different PE methods handle this very differently.

## Technical Detail

**Absolute Sinusoidal PE (original Vaswani et al.):**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
- Fixed (not learned), added to token embeddings before first layer
- Different frequencies encode different scales of position
- Poor length extrapolation — model never saw positions > train length

**Learned Absolute PE:**
- Same shape as sinusoidal but trained as an embedding table
- Used in original BERT and GPT-2
- Hard upper bound: can't extend beyond max position seen in training

**RoPE (Rotary Position Embedding) — Dominant in modern LLMs:**
RoPE rotates Q and K vectors by a position-dependent angle before computing attention:
```
q_m = R(m) · q    k_n = R(n) · k
→ q_m · k_n = (R(m) · q)^T (R(n) · k) = q^T · R(n-m) · k
```
The dot product depends only on the **relative position** (n-m), not absolute positions. This means:
- No explicit PE added to embeddings — PE is applied inside attention
- Relative position is inherent to the formulation
- Can extend beyond training length with tricks (YaRN, RoPE scaling)
- Used in: Llama, Mistral, Falcon, Qwen — essentially all modern open-source LLMs

**ALiBi (Attention with Linear Biases):**
Instead of modifying embeddings, adds a linear bias to attention logits:
```
softmax(QK^T / sqrt(d_k) + m · [-|i-j|])
```
Where m is a head-specific slope. Closer tokens get higher attention scores. No position info added to embeddings. Excellent length extrapolation — trained at 1024 tokens, works well at 4096+.

## Variants & Extensions

| Method | Type | Length Extrapolation | Notes |
|---|---|---|---|
| Sinusoidal | Absolute, fixed | Poor | Original transformer |
| Learned Absolute | Absolute, trained | None (hard cutoff) | BERT, GPT-2 |
| RoPE | Relative, applied in attention | Good with YaRN/scaling | Llama 3, Mistral |
| ALiBi | Bias-based, relative | Excellent | MPT, BLOOM |
| YaRN | RoPE extension | Excellent | Extends RoPE to 128k+ |

## Tradeoffs
| Method | Advantage | Disadvantage |
|---|---|---|
| Sinusoidal | No parameters, simple | Poor extrapolation |
| Learned | Expressive | Hard length limit |
| RoPE | Relative positions, extendable | Slightly more complex |
| ALiBi | Best extrapolation | Position biases may not capture all patterns |

## Interview Angles

**What interviewers are really testing:**
- Do you understand WHY positional encoding is needed (permutation invariance of attention)?
- Can you explain RoPE at a conceptual level and why it became dominant?
- Do you know the difference between absolute and relative PE?
- Can you explain how YaRN or RoPE scaling extends context windows?

**Common follow-up questions:**
- "Why does attention need positional encoding at all?"
- "What's the difference between absolute and relative positional encoding?"
- "Why did RoPE replace learned absolute PE in modern LLMs?"
- "How does Llama 3 handle 128k context? What changes to RoPE were needed?"
- "What is ALiBi and in what scenarios does it outperform RoPE?"

**Gotchas / misconceptions:**
- RoPE is NOT added to embeddings before the layer — it's applied inside attention to Q and K
- The dot product with RoPE encodes only relative positions, not absolute
- "Context length extension" of Llama models uses RoPE scaling tricks, not a fundamentally different PE
- ALiBi doesn't use any position-dependent embedding at all — just attention biases

## Connections
- [[transformer-architecture]] — PE is applied to token embeddings at the input, or (RoPE) inside attention
- [[attention-mechanism]] — RoPE modifies Q and K projections; ALiBi modifies attention logits

## Sources
- Training knowledge (Vaswani et al. 2017; Su et al. 2022 RoPE; Press et al. 2022 ALiBi)
