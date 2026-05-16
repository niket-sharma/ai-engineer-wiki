---
title: "Embedding Evaluation"
aliases: ["embedding eval", "retrieval eval", "MTEB"]
tags: [evaluation, embeddings, retrieval, mteb]
related:
- "[[rag-evaluation]]"
- "[[vector-databases]]"
- "[[reranking]]"
sources: []
relevance: medium
last_updated: 2026-05-16
status: stub
---

# Embedding Evaluation

## TL;DR
Measuring embedding model quality via retrieval benchmarks (MTEB) and ranking metrics (NDCG, MRR).

## Intuition
Embeddings are measured by how well they support downstream retrieval. MTEB (Massive Text Embedding Benchmark) is the standard suite. Key metrics: NDCG@k (quality-weighted ranking), MRR (where does the first relevant result appear), Recall@k (what fraction of relevant items are in the top k).

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
- [[rag-evaluation]] — Embedding quality directly affects RAG context precision/recall
- [[vector-databases]] — The infrastructure that serves embeddings at query time
- [[reranking]] — Reranking compensates for imperfect embedding retrieval

## Sources
<!-- Add raw/ source paths after ingestion -->
