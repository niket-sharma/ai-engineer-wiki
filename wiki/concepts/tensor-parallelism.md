---
title: "Tensor Parallelism"
aliases: ["TP", "model parallelism", "Megatron-LM parallelism"]
tags: [inference, serving, distributed, parallelism]
related:
- "[[quantization]]"
- "[[continuous-batching]]"
- "[[paged-attention]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Tensor Parallelism

## TL;DR
Split individual layers across GPUs (column/row partitioning) to fit models that don't fit on a single device.

## Intuition
A 70B model at FP16 needs ~140GB; a single H100 has 80GB. Tensor parallelism (TP) splits each weight matrix across N GPUs — each GPU holds a shard and computes its partial result, then an all-reduce syncs. Pipeline parallelism (PP) splits by layer instead. TP has lower latency (one all-reduce per layer); PP has lower communication overhead but adds pipeline bubbles.

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
- [[quantization]] — Complementary strategy; TP reduces per-GPU model size, quantization reduces per-weight size
- [[continuous-batching]] — Batching strategies must account for TP communication overhead

## Sources
<!-- Add raw/ source paths after ingestion -->
