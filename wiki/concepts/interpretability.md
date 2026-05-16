---
title: "Model Interpretability"
aliases: ["interpretability", "explainability", "SHAP", "LIME", "PDP", "feature importance"]
tags: [classic-ml, interpretability, explainability]
related:
- "[[gradient-boosting]]"
- "[[feature-engineering]]"
- "[[calibration]]"
sources: []
relevance: medium
last_updated: 2026-05-16
status: stub
---

# Model Interpretability

## TL;DR
Methods for understanding why a model makes a specific prediction — from global feature importance to local explanations.

## Intuition
Two types: global (what features matter most overall?) and local (why did the model predict X for this specific instance?). Global: permutation importance (shuffle a feature, measure accuracy drop), SHAP summary plots, PDP (partial dependence plots — show marginal effect of a feature). Local: SHAP values (game-theoretic, additive feature attributions — the current gold standard), LIME (local linear approximation around a point). SHAP is preferred because it's consistent, locally accurate, and decomposes predictions into feature contributions that sum to the model output.

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
- [[gradient-boosting]] — GBDTs have native feature importance; SHAP provides more reliable estimates
- [[feature-engineering]] — Interpretability reveals which engineered features matter
- [[calibration]] — Well-calibrated models are easier to interpret because outputs are meaningful probabilities

## Sources
<!-- Add raw/ source paths after ingestion -->
