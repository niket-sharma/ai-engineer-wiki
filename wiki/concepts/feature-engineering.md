---
title: "Feature Engineering"
aliases: ["target encoding", "feature leakage", "feature interactions"]
tags: [classic-ml, features, data]
related:
- "[[gradient-boosting]]"
- "[[imbalanced-classification]]"
- "[[model-monitoring]]"
sources: []
relevance: medium
last_updated: 2026-05-16
status: stub
---

# Feature Engineering

## TL;DR
Transforming raw data into model-ready features — often the highest-leverage activity in applied ML.

## Intuition
Good features encode domain knowledge that the model cannot easily learn from raw data alone. Key patterns: target encoding (encode categorical variables with their target mean — but use cross-validation folds to prevent leakage); interaction features (explicit multiplication of related features); monotone constraints (force a feature to have monotone effect on the output); lag features for time series (previous period values as features).

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
- [[gradient-boosting]] — GBDTs handle raw features well but benefit from careful encoding
- [[model-monitoring]] — Feature distributions should be monitored for drift in production

## Sources
<!-- Add raw/ source paths after ingestion -->
