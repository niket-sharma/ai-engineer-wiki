---
title: "Continuous Batching"
aliases: ["in-flight batching", "dynamic batching", "iteration-level scheduling"]
tags: [inference, serving, throughput, batching]
related:
- "[[paged-attention]]"
- "[[quantization]]"
- "[[tensor-parallelism]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Continuous Batching

## TL;DR
Process new requests at each decode step rather than waiting for the whole batch to finish — higher GPU utilization.

## Intuition
Static batching waits for every sequence in a batch to finish before accepting new requests — late-finishing sequences hold up the GPU for everyone. Continuous (in-flight) batching inserts new sequences into the batch at every decode iteration. Sequences that finish free their slots immediately. This keeps GPU utilization near 100% and dramatically improves throughput for variable-length workloads.

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
- [[paged-attention]] — PagedAttention's block-based memory management makes continuous batching practical
- [[quantization]] — Lower memory per model leaves more room for larger batch sizes

## Sources
<!-- Add raw/ source paths after ingestion -->
