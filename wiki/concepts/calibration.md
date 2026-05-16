---
title: "Probability Calibration"
aliases: ["calibration", "Platt scaling", "isotonic regression", "ECE"]
tags: [classic-ml, calibration, evaluation]
related:
- "[[gradient-boosting]]"
- "[[evaluation-metrics]]"
- "[[imbalanced-classification]]"
sources: []
relevance: medium
last_updated: 2026-05-16
status: stub
---

# Probability Calibration

## TL;DR
Making model-predicted probabilities match empirical frequencies — essential for risk-sensitive decisions.

## Intuition
A well-calibrated model saying '70% probability' should be right 70% of the time. Most classifiers (especially GBDTs and neural nets) are not calibrated by default — GBDTs tend to be overconfident near 0 and 1, neural nets can be underconfident. Calibration matters when the probability itself is used (fraud risk score, medical diagnosis), not just the ranking.

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
- [[gradient-boosting]] — GBDTs are notoriously poorly calibrated; always calibrate before using as probabilities
- [[imbalanced-classification]] — Class imbalance worsens calibration; fix imbalance first, then calibrate

## Sources
<!-- Add raw/ source paths after ingestion -->
