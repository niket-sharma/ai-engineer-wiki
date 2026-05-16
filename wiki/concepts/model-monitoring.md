---
title: "Model Monitoring & Drift Detection"
aliases: ["data drift", "concept drift", "PSI", "covariate shift"]
tags: [classic-ml, mlops, monitoring, drift]
related:
- "[[feature-engineering]]"
- "[[observability-llm]]"
- "[[calibration]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Model Monitoring & Drift Detection

## TL;DR
Detecting when a deployed model's inputs or performance change over time, requiring retraining or intervention.

## Intuition
Models degrade in production because the world changes. Two types: covariate shift (input distribution X changes but the relationship P(Y|X) stays the same — detectable without labels) and concept drift (the relationship P(Y|X) changes — requires labels to detect). PSI (Population Stability Index) is the industry standard for covariate shift; KS test and KL divergence are alternatives. Concept drift requires monitoring business metrics or ground truth labels (when available).

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
- [[feature-engineering]] — Feature distributions are the primary unit of drift detection
- [[observability-llm]] — The same observability principles apply to both LLM and classical ML systems
- [[calibration]] — Concept drift degrades calibration before it degrades AUC

## Sources
<!-- Add raw/ source paths after ingestion -->
