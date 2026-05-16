---
title: "Learning Rate Schedules"
aliases: ["warmup", "cosine decay", "WSD", "linear decay", "cyclical LR"]
tags: [training, optimization, learning-rate]
related:
- "[[optimizers]]"
- "[[distributed-training]]"
sources: []
relevance: medium
last_updated: 2026-05-16
status: stub
---

# Learning Rate Schedules

## TL;DR
How the learning rate changes during training — warmup + cosine decay is the de facto standard for LLMs.

## Intuition
Starting with a large LR causes instability (gradients are noisy early); ending with a large LR prevents convergence. Linear warmup gradually increases LR from 0 to peak over the first N steps. Cosine decay then brings it down to 0 (or a small floor) following a cosine curve. WSD (Warmup-Stable-Decay, from MiniCPM) keeps LR constant in the middle phase for long training runs, then decays sharply at the end — better for compute-optimal training.

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
- [[optimizers]] — LR schedule is chosen in conjunction with optimizer (AdamW + cosine is standard)

## Sources
<!-- Add raw/ source paths after ingestion -->
