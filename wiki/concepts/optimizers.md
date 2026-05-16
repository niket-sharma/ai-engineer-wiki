---
title: "Optimizers"
aliases: ["Adam", "AdamW", "Lion", "Shampoo", "weight decay"]
tags: [training, optimization, gradient-descent]
related:
- "[[learning-rate-schedules]]"
- "[[distributed-training]]"
- "[[lora-qlora]]"
sources: []
relevance: medium
last_updated: 2026-05-16
status: stub
---

# Optimizers

## TL;DR
Algorithms that update model weights from gradients — Adam/AdamW dominate, with memory-efficient alternatives emerging.

## Intuition
Adam tracks per-parameter first (m) and second (v) moment estimates of gradients. AdamW decouples weight decay from the gradient update (L2 regularization via Adam scales weight decay by the learning rate — AdamW applies it directly). Memory: Adam stores 3 copies of parameters (weights + m + v) = 3x parameter memory. Lion (from Google Brain) uses only the sign of the gradient update — 1.5x parameter memory, competitive quality. Shampoo uses matrix preconditioners — better for large batches.

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
- [[learning-rate-schedules]] — Optimizer choice interacts with LR schedule; AdamW + cosine-with-warmup is the standard
- [[lora-qlora]] — LoRA reduces optimizer state memory by only training adapter weights

## Sources
<!-- Add raw/ source paths after ingestion -->
