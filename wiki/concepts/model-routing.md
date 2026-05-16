---
title: "Model Routing"
aliases: ["LLM routing", "model gateway", "cost-quality routing"]
tags: [production-ai, routing, cost, serving]
related:
- "[[cost-optimization]]"
- "[[observability-llm]]"
- "[[quantization]]"
sources: []
relevance: medium
last_updated: 2026-05-16
status: stub
---

# Model Routing

## TL;DR
Dynamically selecting which model to use for each request based on cost, latency, and complexity.

## Intuition
Not every query needs GPT-4. A routing layer classifies incoming queries (by complexity, topic, or user tier) and sends them to the cheapest model that can handle them. Simple factual queries go to a 7B model; complex reasoning or creative tasks go to a frontier model. Routers can be learned (a small classifier trained on examples) or rule-based (keyword/length heuristics).

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
- [[cost-optimization]] — Routing is one of the highest-leverage cost levers
- [[observability-llm]] — Routing decisions must be logged to measure effectiveness
- [[safety-and-guardrails]] — Guardrails can also act as routing signals (e.g., flagged queries go to a safer model)

## Sources
<!-- Add raw/ source paths after ingestion -->
