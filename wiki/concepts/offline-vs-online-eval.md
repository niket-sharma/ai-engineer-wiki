---
title: "Offline vs Online Evaluation"
aliases: ["A/B testing", "counterfactual eval", "interleaving"]
tags: [evaluation, a-b-testing, experimentation]
related:
- "[[llm-evaluation]]"
- "[[rag-evaluation]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Offline vs Online Evaluation

## TL;DR
Offline eval is fast and cheap; online eval (A/B tests) is the ground truth but slow and risky.

## Intuition
Offline eval uses static datasets and automated metrics — fast but may not reflect real user behavior. Online eval (A/B tests) measures actual user outcomes (clicks, satisfaction, retention) but requires traffic, takes time, and risks degrading user experience. The gap between offline and online metrics is the fundamental challenge in production ML.

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
- [[llm-evaluation]] — Offline benchmarks are a form of offline eval
- [[rag-evaluation]] — RAGAS and similar are offline eval frameworks

## Sources
<!-- Add raw/ source paths after ingestion -->
