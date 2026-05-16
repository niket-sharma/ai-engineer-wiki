---
title: "Distributed Training"
aliases: ["DDP", "FSDP", "ZeRO", "data parallelism", "model parallelism"]
tags: [training, distributed, parallelism]
related:
- "[[tensor-parallelism]]"
- "[[mixed-precision-training]]"
- "[[gradient-checkpointing]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Distributed Training

## TL;DR
Splitting training computation across multiple GPUs — via data parallelism (DDP), model parallelism (FSDP/ZeRO), or both.

## Intuition
DDP (Distributed Data Parallel): each GPU has a full model copy, processes different data shards, gradients are all-reduced. Simple but memory-inefficient — each GPU stores the full model. FSDP / ZeRO: shards model parameters, gradients, and optimizer states across GPUs. ZeRO-3 (fully sharded) eliminates all redundancy — each GPU only stores 1/N of everything, requires all-gather before each layer's forward/backward.

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
- [[tensor-parallelism]] — TP splits within a layer; FSDP/ZeRO shards across layers
- [[mixed-precision-training]] — Gradient communication in DDP is done in FP16 to save bandwidth

## Sources
<!-- Add raw/ source paths after ingestion -->
