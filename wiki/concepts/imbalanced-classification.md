---
title: "Imbalanced Classification"
aliases: ["class imbalance", "SMOTE", "oversampling", "undersampling"]
tags: [classic-ml, classification, imbalance]
related:
- "[[calibration]]"
- "[[gradient-boosting]]"
- "[[evaluation-metrics]]"
sources: []
relevance: medium
last_updated: 2026-05-16
status: stub
---

# Imbalanced Classification

## TL;DR
Techniques for learning from datasets where one class is far rarer than another (e.g., fraud: 0.1% positive rate).

## Intuition
With 99% negatives, a model predicting 'always negative' gets 99% accuracy but 0% recall — useless. Fixes: (1) threshold tuning — move the decision threshold down to catch more positives; (2) class weights — penalize misclassifying the minority class more; (3) oversampling — SMOTE generates synthetic minority samples; (4) undersampling — downsample the majority. Threshold tuning and class weights are almost always better than resampling for gradient boosting.

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
- [[calibration]] — Resampling breaks calibration; calibrate after any resampling
- [[gradient-boosting]] — GBDTs support class weights natively via `scale_pos_weight`
- [[evaluation-metrics]] — Use precision/recall/F1/AUC-PR, not accuracy, for imbalanced data

## Sources
<!-- Add raw/ source paths after ingestion -->
