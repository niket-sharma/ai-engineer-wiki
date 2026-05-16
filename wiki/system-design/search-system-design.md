---
title: "Search System Design"
aliases: ["search", "query understanding", "retrieval pipeline", "ranking"]
tags: [system-design, search, retrieval, ranking]
related:
- "[[hybrid-search]]"
- "[[reranking]]"
- "[[bm25-and-sparse-retrieval]]"
- "[[embedding-models]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Search System Design

## TL;DR
End-to-end design of a production search system: query understanding, multi-stage retrieval, learning-to-rank.

## Intuition
Search decomposes into: (1) query understanding (spell correction, entity recognition, intent classification, query rewriting); (2) retrieval (BM25 + dense retrieval in parallel, merged with RRF); (3) ranking (learning-to-rank with GBDT or neural rankers, personalization signals); (4) result presentation (snippets, facets, spell suggestions). Each stage trades precision for recall or vice versa.

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
- [[hybrid-search]] — The retrieval layer combines sparse and dense signals
- [[reranking]] — Cross-encoder reranking at the top of the funnel
- [[query-rewriting]] — Query understanding includes rewriting for better retrieval
- [[bm25-and-sparse-retrieval]] — BM25 is the sparse retrieval backbone

## Sources
<!-- Add raw/ source paths after ingestion -->
