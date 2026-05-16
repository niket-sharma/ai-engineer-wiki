---
title: "Time Series Analysis"
aliases: ["time series", "ARIMA", "forecasting", "Prophet", "temporal data"]
tags: [classic-ml, time-series, forecasting, statistics]
related:
- "[[gradient-boosting]]"
- "[[model-monitoring]]"
- "[[feature-engineering]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Time Series Analysis

## TL;DR
Methods for modeling and forecasting sequential temporal data — from ARIMA to gradient boosting with lag features.

## Intuition
Time series requires special handling because observations are not i.i.d. — they have autocorrelation, trend, and seasonality. Classical approaches: ARIMA models linear autocorrelation; Prophet decomposes into trend + seasonality + holidays (additive model, robust to missing data). Modern practice: gradient boosting on lag features (yesterday's value, 7-day rolling mean) often beats deep learning for tabular time series. DL (LSTM, Transformer-based) wins for high-frequency, multivariate settings (electricity, traffic).

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
- [[gradient-boosting]] — GBDT with lag features is the go-to for tabular time series
- [[feature-engineering]] — Lag features, rolling statistics, and calendar features are the core
- [[model-monitoring]] — Temporal models decay faster than cross-sectional models; monitoring is critical

## Sources
raw/statistics-and-ml/
