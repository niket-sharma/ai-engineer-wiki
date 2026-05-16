---
title: "Paged Attention"
aliases: ["PagedAttention", "vLLM paging", "block-based KV cache"]
tags: [inference, serving, kv-cache, vllm]
related:
- "[[kv-cache]]"
- "[[continuous-batching]]"
- "[[quantization]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Paged Attention

## TL;DR
Manages KV cache in fixed-size blocks (like OS virtual memory paging) to eliminate fragmentation and enable dynamic sharing.

## Intuition
Traditional KV cache pre-allocates contiguous GPU memory for the maximum sequence length — most of it wasted for short sequences. PagedAttention (used by vLLM) stores KV cache in non-contiguous blocks, like OS virtual memory paging. This eliminates internal fragmentation, allows sharing KV blocks across parallel sequences (beam search, prefix caching), and dramatically improves GPU memory utilization.

## Technical Detail
<!-- to be filled -->

## Variants & Extensions
<!-- to be filled -->

## Tradeoffs
| Advantage | Disadvantage |
|---|---|
| ... | ... |

## Practical Applications
- Common use cases and when to apply
- Common follow-up questions
- Gotchas / misconceptions to avoid

## Connections
- [[kv-cache]] — PagedAttention is an implementation strategy for KV cache management
- [[continuous-batching]] — PagedAttention enables efficient continuous batching by freeing memory as sequences complete
- [[flash-attention]] — Both are memory-efficiency techniques; Flash Attention is for training/compute, PagedAttention is for serving/memory

## Sources
<!-- Add raw/ source paths after ingestion -->
