---
title: "Mixed Precision Training"
aliases: ["FP16", "BF16", "FP8", "loss scaling", "AMP"]
tags: [training, precision, efficiency]
related:
- "[[gradient-checkpointing]]"
- "[[distributed-training]]"
- "[[quantization]]"
sources: []
relevance: medium
last_updated: 2026-05-16
status: stub
---

# Mixed Precision Training

## TL;DR
Training with lower-precision floats (FP16/BF16) for speed and memory savings, with FP32 master weights for stability.

## Intuition
FP32 training is slow and memory-hungry. Mixed precision keeps weights in FP16/BF16 for compute (faster, smaller) but accumulates gradients in FP32 (stable). BF16 (Brain Float) has the same exponent range as FP32 (avoids overflow/underflow) but only 7 mantissa bits vs. FP16's 10. For modern hardware (A100+, H100), BF16 is preferred over FP16 because it doesn't require manual loss scaling. FP8 is emerging for H100 forward passes.

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
- [[quantization]] — Mixed precision for inference is called quantization; mixed precision for training keeps higher precision for gradients
- [[distributed-training]] — Precision interacts with gradient all-reduce communication bandwidth

## Sources
<!-- Add raw/ source paths after ingestion -->
