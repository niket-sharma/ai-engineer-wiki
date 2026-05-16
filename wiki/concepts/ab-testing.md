---
title: "A/B Testing & Experimentation"
aliases: ["A/B test", "randomized controlled trial", "CUPED", "sequential testing"]
tags: [classic-ml, experimentation, statistics]
related:
- "[[offline-vs-online-eval]]"
- "[[model-monitoring]]"
- "[[causal-inference]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# A/B Testing & Experimentation

## TL;DR
Randomized experiments to measure the causal effect of changes (model updates, UI changes) on business metrics.

## Intuition
An A/B test randomly assigns users to control (A) or treatment (B) and measures the difference in outcomes. The key decisions: power analysis (how many users needed to detect the target effect size?), metric choice (primary and guardrail metrics), and duration (long enough to avoid novelty effects). CUPED reduces variance by controlling for pre-experiment outcomes, allowing smaller or shorter experiments. Sequential testing (always-valid p-values) lets you peek at results without inflating false positive rates.

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
- [[offline-vs-online-eval]] — A/B testing is the canonical online evaluation method
- [[causal-inference]] — A/B tests are the gold standard; causal inference handles cases where randomization is impossible

## Sources
<!-- Add raw/ source paths after ingestion -->
