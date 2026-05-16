---
title: "Quantization"
aliases: ["model quantization", "INT8", "INT4", "GPTQ", "AWQ"]
tags: [inference, serving, compression, quantization]
related:
- "[[kv-cache]]"
- "[[flash-attention]]"
- "[[paged-attention]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Quantization

## TL;DR
Reducing model weight/activation precision to shrink memory and increase throughput at the cost of some accuracy.

## Intuition
A 70B model in FP16 needs ~140GB of GPU memory. INT4 quantization cuts that to ~35GB — same model, 4× smaller, runs on a single A100 80GB instead of two. The tradeoff: quantization introduces rounding error that can degrade generation quality, especially for reasoning tasks.

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
- [[kv-cache]] — KV cache can also be quantized (FP8 KV cache) to save memory
- [[continuous-batching]] — Lower memory per model = more batch slots available
- [[tensor-parallelism]] — Quantization and TP are complementary memory-reduction strategies

## Sources
<!-- Add raw/ source paths after ingestion -->
