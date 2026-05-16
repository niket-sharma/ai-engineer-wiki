---
title: "Hybrid Search"
aliases: ["hybrid retrieval", "sparse-dense fusion", "RRF", "reciprocal rank fusion"]
tags: [rag, retrieval, search, bm25]
related:
- "[[rag-systems]]"
- "[[embedding-models]]"
- "[[bm25-and-sparse-retrieval]]"
- "[[reranking]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Hybrid Search

## TL;DR
Combining dense (semantic) and sparse (keyword) retrieval to get the best of both — better recall than either alone.

## Intuition
Dense retrieval (embeddings) handles semantic similarity and paraphrase; sparse retrieval (BM25) handles exact keyword matches and rare terms. Neither dominates in all cases. Hybrid search combines them: retrieve top-k from each, then merge results with Reciprocal Rank Fusion (RRF: score = Σ 1/(rank + k)) or weighted score fusion. RRF is parameter-free and robust; weighted fusion requires tuning but can be more accurate when calibrated.

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
- [[rag-systems]] — Hybrid search is the recommended retrieval strategy for production RAG
- [[bm25-and-sparse-retrieval]] — BM25 is the dominant sparse retrieval method
- [[reranking]] — Hybrid search improves recall; reranking improves precision of the top results

## Sources
<!-- Add raw/ source paths after ingestion -->
