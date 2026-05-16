---
title: "BM25 and Sparse Retrieval"
aliases: ["BM25", "TF-IDF", "keyword search", "sparse retrieval"]
tags: [rag, retrieval, search, bm25]
related:
- "[[hybrid-search]]"
- "[[embedding-models]]"
- "[[reranking]]"
sources: []
relevance: medium
last_updated: 2026-05-16
status: stub
---

# BM25 and Sparse Retrieval

## TL;DR
Keyword-based retrieval using term frequency and inverse document frequency — fast, interpretable, and still competitive.

## Intuition
BM25 scores a document for a query by summing IDF-weighted term frequency scores, with saturation (tf never grows unbounded) and length normalization. Despite being a 1994 algorithm, BM25 is competitive with modern embeddings on many benchmarks — especially for exact keyword matches, rare terms, and out-of-domain queries. It's the 'B' in most hybrid search systems and the baseline every dense retrieval system should beat.

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
- [[hybrid-search]] — BM25 is one of the two legs of hybrid search
- [[embedding-models]] — Dense retrieval complements BM25 for semantic matching
- [[reranking]] — After BM25/hybrid retrieval, reranking improves precision

## Sources
<!-- Add raw/ source paths after ingestion -->
