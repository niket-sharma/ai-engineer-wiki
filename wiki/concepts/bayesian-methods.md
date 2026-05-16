---
title: "Bayesian Methods"
aliases: ["Bayesian inference", "MCMC", "variational inference", "posterior", "prior"]
tags: [classic-ml, statistics, bayesian]
related:
- "[[calibration]]"
- "[[ab-testing]]"
- "[[ensemble-methods]]"
sources: []
relevance: medium
last_updated: 2026-05-16
status: stub
---

# Bayesian Methods

## TL;DR
Probabilistic reasoning that incorporates prior beliefs and updates them with data — principled uncertainty quantification.

## Intuition
Bayesian inference computes the posterior P(θ|data) ∝ P(data|θ) × P(θ). In practice, the posterior is intractable for complex models. MCMC (Markov Chain Monte Carlo) samples from it — asymptotically exact but slow. Variational Inference (VI) approximates the posterior with a simpler distribution and optimizes the ELBO — faster but approximate. Pyro and Stan are the main frameworks. Practical value: uncertainty quantification (credible intervals vs. confidence intervals), hierarchical models (partial pooling), and Bayesian A/B testing (sequential, no p-hacking).

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
- [[calibration]] — Bayesian models are naturally calibrated; frequentist models require post-hoc calibration
- [[ab-testing]] — Bayesian A/B testing gives probability that B > A directly, without fixed sample sizes

## Sources
raw/statistics-and-ml/bayesian-inference.md
