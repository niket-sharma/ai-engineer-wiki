---
title: "Flash Attention"
aliases: ["FlashAttention", "FA", "FA2", "FA3", "IO-aware attention"]
tags: [transformers, optimization, inference, training]
related: ["[[attention-mechanism]]", "[[kv-cache]]", "[[transformer-architecture]]"]
sources: ["training-knowledge"]
interview_relevance: high
last_updated: 2026-04-22
status: current
---

# Flash Attention

## TL;DR
Flash Attention is an IO-aware implementation of standard scaled dot-product attention that avoids materializing the full n×n attention matrix in GPU HBM (high-bandwidth memory). It produces bit-identical results to standard attention but is 2–4× faster and uses O(n) memory instead of O(n²). It's the default attention kernel in all major training frameworks and LLM serving systems.

## Intuition
Standard attention on a GPU:
1. Compute QK^T → write n×n matrix to HBM (slow)
2. Apply softmax → read n×n from HBM, write back (slow)
3. Multiply by V → read n×n from HBM (slow)

The bottleneck is not compute — it's HBM reads/writes. Modern GPUs are compute-bound on matrix multiplications but memory-bandwidth-bound on pointwise operations like softmax.

Flash Attention tiles the computation to fit in fast SRAM (on-chip memory):
- Process blocks of Q, K, V at a time
- Compute a numerically stable running softmax (online softmax algorithm)
- Never write the full attention matrix to HBM
- Result: same math, much less HBM traffic

## Technical Detail

**GPU memory hierarchy:**
- SRAM (on-chip, shared memory): ~20 MB, fast (~19 TB/s)
- HBM (GPU DRAM): ~40–80 GB, slow (~2 TB/s)
- Standard attention materializes O(n²) in HBM; FA keeps it in SRAM

**Online Softmax:**
To compute softmax without materializing the full row, maintain running max m and running sum l:
```
for each block:
    m_new = max(m_prev, max(block_scores))
    l_new = exp(m_prev - m_new) * l_prev + sum(exp(block_scores - m_new))
    output_accum = rescale(output_accum) + exp(block_scores - m_new) * V_block
```
This produces exact softmax without ever having the full row in memory.

**Complexity:**
| Version | HBM reads/writes | Memory | Speed vs Standard |
|---|---|---|---|
| Standard | O(n²) | O(n²) | 1× |
| Flash Attention v1 | O(n² / M) where M = SRAM size | O(n) | 2–3× |
| Flash Attention v2 | Better parallelism over heads | O(n) | 3–4× |
| Flash Attention v3 | Exploits H100 async tensor cores | O(n) | 5–8× |

**Backward pass:** FA also fuses the attention backward pass — it recomputes attention weights on the fly during backprop rather than storing them, trading compute for memory.

## Variants & Extensions

- **FA v1** (Dao et al. 2022): Original tiled SRAM approach, CUDA
- **FA v2** (Dao 2023): Better parallelism over sequence/head dimensions, 2× speedup over FA1
- **FA v3** (2024): Exploits H100 async compute (warpgroups), FP8 support
- **FlashDecoding**: Extension for inference decoding where seq len is 1 but batch is large
- **FlashInfer**: Production-grade FA implementation with paged KV cache support

## Tradeoffs
| Advantage | Disadvantage |
|---|---|
| Exact same math as standard attention | Custom CUDA kernels — complex to implement |
| 2–8× faster wall-clock training/inference | SRAM size limits block size (hardware-dependent) |
| O(n) memory — enables very long contexts | Backward recomputation increases FLOPs slightly |
| No quality loss (bit-identical on bfloat16) | Hardware-specific (optimized for A100/H100) |

## Interview Angles

**What interviewers are really testing:**
- Do you understand the GPU memory hierarchy and why IO matters?
- Can you explain the online softmax trick at a high level?
- Do you know the difference between FA's contribution (IO efficiency) vs KV cache (compute reduction)?

**Common follow-up questions:**
- "What problem does Flash Attention solve? Is it about compute or memory?"
- "Why can't we just use a faster matrix multiply for attention?"
- "How does FA enable training on longer sequences?"
- "What's the difference between Flash Attention v1 and v2?"
- "Flash Attention and KV cache — how do they complement each other?"

**Gotchas / misconceptions:**
- FA does NOT change the mathematical result of attention — it's an optimization, not an approximation
- The bottleneck FA solves is HBM bandwidth, not FLOP count
- FA improves both training AND inference (different code paths, both benefit)
- "Memory-efficient attention" from xformers is a different but related approach

## Connections
- [[attention-mechanism]] — FA is an optimized implementation of scaled dot-product attention
- [[kv-cache]] — KV cache reduces the number of attention computations; FA makes each computation faster
- [[transformer-architecture]] — FA is a drop-in replacement in any transformer layer

## Sources
- Training knowledge (Dao et al. 2022 FlashAttention; Dao 2023 FlashAttention-2)
