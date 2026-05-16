---
title: "Chunking Strategies"
aliases: ["text chunking", "document splitting", "semantic chunking", "late chunking"]
tags: [rag, retrieval, chunking, preprocessing]
related:
- "[[rag-systems]]"
- "[[embedding-models]]"
- "[[rag-evaluation]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Chunking Strategies

## TL;DR
How to split documents into retrieval units — one of the highest-leverage decisions in RAG pipeline design.

## Intuition
Chunks that are too small lose context; chunks too large hurt retrieval precision and exceed model context windows. Key strategies: fixed-size (split every N tokens with overlap); recursive (split on paragraph → sentence → word until small enough); semantic (split at topic boundaries using embedding similarity); late chunking (embed the whole document, then pool for each chunk — preserves global context). The right strategy depends on document structure and query type.

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
- [[rag-systems]] — Chunking is a critical step in the ingestion pipeline
- [[embedding-models]] — Chunk size must match the embedding model's optimal input length
- [[rag-evaluation]] — Poor chunking directly degrades context precision and recall

## Sources
<!-- Add raw/ source paths after ingestion -->
