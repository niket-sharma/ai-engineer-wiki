---
title: "Recommendation System Design"
aliases: ["RecSys", "two-tower", "collaborative filtering", "candidate generation"]
tags: [system-design, recommender, two-tower, ranking]
related:
- "[[rag-pipeline-design]]"
- "[[embedding-models]]"
- "[[vector-databases]]"
- "[[ml-platform]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Recommendation System Design

## TL;DR
End-to-end design of a scalable recommender: two-tower retrieval, candidate generation, and multi-stage ranking.

## Intuition
Production recommenders split into two stages: (1) candidate generation — retrieve O(100) candidates from millions using a two-tower model (user embedding × item embedding) and ANN search; (2) ranking — score candidates with a heavier model using cross-features, then apply business rules. This two-stage design balances recall (cheap retrieval) with precision (expensive ranking) at scale.

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
- [[embedding-models]] — User and item towers produce the embeddings
- [[vector-databases]] — ANN search over item embeddings for candidate retrieval
- [[ml-platform]] — Feature store serves real-time user and item features to the ranking model
- [[ab-testing]] — A/B testing is the primary evaluation method for recommender improvements

## Sources
<!-- Add raw/ source paths after ingestion -->
