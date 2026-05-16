---
title: "ML Feature Pipeline Design"
aliases: ["feature pipeline", "streaming features", "batch features", "feature store design"]
tags: [system-design, mlops, features, streaming]
related:
- "[[feature-store]]"
- "[[ml-platform]]"
- "[[model-monitoring]]"
sources: []
relevance: medium
last_updated: 2026-05-16
status: stub
---

# ML Feature Pipeline Design

## TL;DR
Design of pipelines that compute, serve, and monitor ML features across batch and streaming paths.

## Intuition
Feature pipelines have two latency classes: (1) batch — daily/hourly jobs (Spark, dbt) that compute aggregations like '30-day purchase count'; (2) streaming — sub-second Flink/Kafka jobs for real-time signals like 'last 5 clicks'. The feature store bridges them: offline store (Parquet/Delta, point-in-time correct joins for training) and online store (Redis/Cassandra, low-latency serving for inference). The hardest problem: training-serving skew — features computed differently at training time vs. serving time.

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
- [[feature-store]] — The storage layer this pipeline feeds
- [[ml-platform]] — The broader platform context
- [[model-monitoring]] — Feature drift detection is the first line of model monitoring
- [[ab-testing]] — Feature pipeline changes require careful A/B validation

## Sources
<!-- Add raw/ source paths after ingestion -->
