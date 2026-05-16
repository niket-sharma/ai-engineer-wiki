---
title: "Query Rewriting"
aliases: ["HyDE", "query decomposition", "step-back prompting", "multi-query retrieval"]
tags: [rag, retrieval, query-rewriting]
related:
- "[[rag-systems]]"
- "[[hybrid-search]]"
- "[[embedding-models]]"
sources: []
relevance: medium
last_updated: 2026-05-16
status: stub
---

# Query Rewriting

## TL;DR
Transforming user queries before retrieval to improve recall — from HyDE to decomposition to multi-query.

## Intuition
User queries are often poor retrieval queries: too short, ambiguous, or phrased differently than documents. Query rewriting transforms them before retrieval. HyDE (Hypothetical Document Embeddings) generates a hypothetical answer and embeds that instead of the query — the answer's embedding is closer to real answer embeddings. Multi-query generates multiple paraphrases and merges results. Step-back prompting abstracts the query to a higher-level concept to retrieve background context.

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
- [[rag-systems]] — Query rewriting is a pre-retrieval step in the RAG pipeline
- [[hybrid-search]] — Query rewriting benefits both dense and sparse retrieval legs
- [[rag-evaluation]] — Query rewriting's impact is measured via context recall improvements

## Sources
<!-- Add raw/ source paths after ingestion -->
