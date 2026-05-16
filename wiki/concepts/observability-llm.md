---
title: "LLM Observability"
aliases: ["LLM monitoring", "LLM tracing", "Langfuse", "LangSmith", "Arize"]
tags: [production-ai, observability, monitoring, tracing]
related:
- "[[cost-optimization]]"
- "[[safety-and-guardrails]]"
- "[[langgraph-agents]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# LLM Observability

## TL;DR
Tracking token usage, latency, cost, and quality across LLM calls — enabling debugging and optimization.

## Intuition
LLM systems are opaque: a 'wrong' answer could be caused by a bad prompt, bad retrieval, bad model, or bad guardrail. Observability means instrumenting every LLM call with: input/output (for debugging), token counts (for cost), latency (for SLA monitoring), and quality scores (for regression detection). Tools like Langfuse and LangSmith provide trace UIs; Arize adds drift detection and evaluation at scale.

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
- [[cost-optimization]] — Token/cost tracking is the first step to optimization
- [[safety-and-guardrails]] — Guardrail decisions are a key event to log
- [[rag-evaluation]] — Observability surfaces the data needed for offline eval

## Sources
<!-- Add raw/ source paths after ingestion -->
