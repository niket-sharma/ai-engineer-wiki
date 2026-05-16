---
title: "Causal Inference"
aliases: ["causal inference", "DiD", "difference-in-differences", "IV", "propensity scores", "uplift modeling"]
tags: [classic-ml, causal-inference, statistics, experimentation]
related:
- "[[ab-testing]]"
- "[[model-monitoring]]"
- "[[feature-engineering]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Causal Inference

## TL;DR
Methods for estimating causal effects when randomized experiments are infeasible — DiD, IV, propensity scores.

## Intuition
Correlation ≠ causation. Causal inference asks: what would have happened to the treated units if they had not been treated (the counterfactual)? Methods: Difference-in-Differences (DiD) — compare trend change between treated and control groups pre/post intervention; Instrumental Variables (IV) — use an instrument that affects treatment but not outcome directly; Propensity Score Matching — match treated and control units with similar probability of treatment; Uplift modeling — estimate individual treatment effects (heterogeneous effects).

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
- [[ab-testing]] — A/B tests are the gold standard; causal inference handles cases where randomization isn't possible
- [[model-monitoring]] — Causal methods are needed to attribute observed metric changes to specific model updates

## Sources
raw/statistics-and-ml/causal-inference.md
