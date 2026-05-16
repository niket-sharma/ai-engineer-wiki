---
title: "LLM Cost Optimization"
aliases: ["prompt caching", "semantic caching", "model routing", "cost reduction"]
tags: [production-ai, cost, caching, routing]
related:
- "[[model-routing]]"
- "[[quantization]]"
- "[[observability-llm]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# LLM Cost Optimization

## TL;DR
Techniques to reduce LLM API and inference costs: caching, routing, batching, distillation.

## Intuition
At scale, LLM costs dominate. The lever hierarchy: (1) Prompt caching — reuse KV cache for identical prefixes (Anthropic, OpenAI both support this); (2) Semantic caching — cache responses for semantically similar queries; (3) Model routing — send simple queries to cheap small models, complex ones to expensive large models; (4) Distillation — train a small specialized model on the large model's outputs for a specific task; (5) Batch APIs — async processing at 50% cost discount when latency is not critical.

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
- [[model-routing]] — Routing is a key cost-optimization lever
- [[quantization]] — Smaller quantized models cost less to run
- [[observability-llm]] — Cost tracking is a core observability metric

## Sources
<!-- Add raw/ source paths after ingestion -->
