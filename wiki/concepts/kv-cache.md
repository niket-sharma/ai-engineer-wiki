---
title: "KV Cache"
aliases: ["key-value cache", "kv-caching", "paged attention"]
tags: [transformers, inference, optimization]
related: ["[[attention-mechanism]]", "[[transformer-architecture]]", "[[flash-attention]]", "[[llm-serving-infra]]"]
sources: ["training-knowledge"]
relevance: high
last_updated: 2026-04-22
status: current
---

# KV Cache

## TL;DR
KV cache stores the Key and Value projections for previously generated tokens so they don't need to be recomputed at each decoding step. Without it, generating token N requires recomputing K and V for all N-1 previous tokens — O(n²) total work. With it, generating token N requires only computing K and V for token N — O(n) total work. It's the most important inference optimization for autoregressive LLMs.

## Intuition
During autoregressive generation, the model is asked "given tokens 1..N, predict token N+1." For the attention operation at layer L, each token needs to attend to every previous token. The Keys and Values for tokens 1..N-1 were already computed on prior steps — there's no reason to recompute them. Cache them.

We cache K and V but NOT Q because Q for the current step depends on the current token embedding, which changes every step. K and V from past tokens are fixed once computed.

## Technical Detail

**Memory cost of KV cache:**
```
KV cache size = 2 × n_layers × n_kv_heads × d_head × seq_len × dtype_bytes
```
Example — Llama 3 8B, fp16, 4096 token context:
```
2 × 32 layers × 8 kv_heads × 128 d_head × 4096 tokens × 2 bytes
= 2 × 32 × 8 × 128 × 4096 × 2 = 536 MB
```
At 8k tokens: 1 GB. At 128k tokens: 16 GB. KV cache competes directly with model weights for GPU memory.

**Why GQA/MQA reduce KV cache:**
- MHA: n_kv_heads = n_heads (e.g., 32). KV cache ∝ 32 heads.
- MQA: n_kv_heads = 1. KV cache ∝ 1 head — 32× smaller.
- GQA: n_kv_heads = g (e.g., 8). KV cache ∝ 8 heads — 4× smaller than MHA.

Llama 3 8B uses GQA with 8 KV heads vs 32 Q heads — 4× KV cache reduction.

**Paged Attention (vLLM):**
Rather than allocating a contiguous block of GPU memory for the full max sequence length upfront, paged attention allocates KV cache in fixed-size pages (like OS virtual memory). This allows:
- Different requests to share GPU memory pages (e.g., shared system prompts → prefix caching)
- No memory fragmentation from variable-length sequences
- Higher GPU utilization → higher throughput

## Variants & Extensions

| Technique | What it does |
|---|---|
| MQA | Single KV head — smallest cache, some quality loss |
| GQA | Grouped KV heads — balance quality vs cache size |
| Paged Attention (vLLM) | Non-contiguous memory pages for KV cache |
| Prefix caching | Cache KV for shared system prompt across requests |
| KV cache quantization | Store KV in int8/fp8 to halve memory |
| Sliding window | Only cache last W tokens (Mistral) — bounded memory |

## Tradeoffs
| Advantage | Disadvantage |
|---|---|
| O(1) per-step compute instead of O(n) | Significant GPU memory cost grows with context length |
| Essential for real-time generation | Must be carefully managed in multi-request serving |
| Enables long-context generation | Paged/quantized KV cache adds implementation complexity |

##  Angles

**What to understand deeply:**
- Do you understand *why* only K and V are cached (not Q)?
- Can you compute KV cache memory size for a given model?
- Do you understand the connection between GQA and KV cache reduction?
- Can you explain paged attention and why it improves serving throughput?

**Common follow-up questions:**
- "Walk me through why we cache K and V but not Q."
- "How does GQA reduce the KV cache? Show the math."
- "What is paged attention and how does it improve GPU utilization?"
- "How would you implement prefix caching for a shared system prompt?"
- "At what context length does KV cache become larger than the model weights?"

**Gotchas / misconceptions:**
- KV cache grows with SEQUENCE LENGTH, not batch size directly — though both matter
- Paged attention is a memory management strategy, not a new attention algorithm
- Prefix caching only helps when requests share a common prefix (e.g., system prompt)
- Flash Attention and KV cache are complementary — FA speeds up the attention computation; KV cache reduces the number of computations needed

## Connections
- [[attention-mechanism]] — KV cache stores K and V tensors from the attention projection
- [[flash-attention]] — Flash Attention is the compute kernel; KV cache reduces work
- [[llm-serving-infra]] — serving systems (vLLM, TGI) are built around KV cache management
- [[transformer-architecture]] — KV cache size is determined by n_layers × n_kv_heads × d_head

## Sources
- Training knowledge (Kwon et al. 2023 "Efficient Memory Management for Large Language Model Serving with PagedAttention")
