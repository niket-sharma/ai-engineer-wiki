---
title: "Gradient Checkpointing"
aliases: ["activation checkpointing", "rematerialization"]
tags: [training, memory, efficiency]
related:
- "[[mixed-precision-training]]"
- "[[distributed-training]]"
- "[[lora-qlora]]"
sources: []
relevance: medium
last_updated: 2026-05-16
status: stub
---

# Gradient Checkpointing

## TL;DR
Trade compute for memory by not storing all activations — recompute them during the backward pass instead.

## Intuition
Training a transformer requires storing all activations from the forward pass to compute gradients in the backward pass. For a 7B model, this can require 80+ GB. Gradient checkpointing saves only a subset of activations (checkpoints) and recomputes the rest on demand during backward. Memory usage drops from O(L) to O(√L) where L is layers, at the cost of ~33% more compute.

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
- [[mixed-precision-training]] — Combined with BF16, enables training much larger models
- [[lora-qlora]] — QLoRA uses gradient checkpointing + 4-bit quantization to fine-tune 65B models on a single GPU

## Sources
<!-- Add raw/ source paths after ingestion -->
