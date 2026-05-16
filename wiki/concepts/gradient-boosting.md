---
title: "Gradient Boosting"
aliases: ["XGBoost", "LightGBM", "CatBoost", "GBDT"]
tags: [classic-ml, ensemble, trees]
related:
- "[[ensemble-methods]]"
- "[[calibration]]"
- "[[feature-engineering]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Gradient Boosting

## TL;DR
Sequentially fitting decision trees to residuals — the dominant algorithm for tabular data.

## Intuition
Gradient boosting builds an ensemble one tree at a time, where each new tree fits the gradient of the loss with respect to the current ensemble's predictions. This is 'boosting in function space': you're doing gradient descent on the space of functions, not parameters. XGBoost adds L1/L2 regularization to the tree structure; LightGBM uses histogram-based splits and leaf-wise growth for speed; CatBoost handles categorical features natively with ordered target encoding.

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
- [[ensemble-methods]] — Boosting is one of the three main ensemble paradigms (alongside bagging and stacking)
- [[calibration]] — GBDT outputs are not calibrated probabilities; Platt scaling or isotonic regression is needed
- [[feature-engineering]] — GBDTs handle monotone constraints natively, useful for financial and medical features

## Sources
<!-- Add raw/ source paths after ingestion -->
