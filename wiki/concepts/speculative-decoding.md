---
title: "Speculative Decoding"
aliases: ["speculative sampling", "draft-and-verify"]
tags: [inference, serving, latency, throughput]
related:
- "[[kv-cache]]"
- "[[continuous-batching]]"
- "[[paged-attention]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Speculative Decoding

## TL;DR
Use a cheap draft model to propose multiple tokens, then verify in parallel with the large model — same quality, lower latency.

## Intuition
LLM decoding is sequential by default: one token at a time. Speculative decoding breaks this by having a small fast model (the 'draft') generate k tokens cheaply, then the large model verifies all k in one forward pass. Accepted tokens are free throughput; rejected tokens cost only the draft model's time.

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
- [[kv-cache]] — Both the draft and verifier model use KV cache; managing two caches adds complexity
- [[continuous-batching]] — Speculative decoding interacts with batching strategies in non-trivial ways

## Sources
<!-- Add raw/ source paths after ingestion -->
