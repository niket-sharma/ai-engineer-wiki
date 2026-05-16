---
title: "Safety and Guardrails"
aliases: ["LLM safety", "guardrails", "content filtering", "Llama Guard", "NeMo Guardrails"]
tags: [production-ai, safety, content-filtering]
related:
- "[[prompt-injection]]"
- "[[observability-llm]]"
- "[[cost-optimization]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Safety and Guardrails

## TL;DR
Input and output filtering layers that prevent harmful, off-topic, or policy-violating LLM behavior.

## Intuition
Guardrails work in two places: input (before the LLM sees the message — detect jailbreaks, PII, off-topic queries) and output (before the response reaches the user — detect toxicity, hallucinations, policy violations). Tools like Llama Guard (a classifier LLM) and NeMo Guardrails (a rule-based + LLM hybrid) provide these layers. The tradeoff is latency (extra inference) and false positive rate (blocking legitimate queries).

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
- [[prompt-injection]] — Guardrails are the primary defense against injection attacks
- [[observability-llm]] — Guardrail decisions should be logged and monitored
- [[cost-optimization]] — Guardrail calls add latency and cost; fast classifiers (not full LLMs) are preferred

## Sources
<!-- Add raw/ source paths after ingestion -->
