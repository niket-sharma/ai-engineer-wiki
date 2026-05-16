---
title: "Embedding Models"
aliases: ["text embeddings", "sentence transformers", "SBERT", "E5", "BGE"]
tags: [rag, retrieval, embeddings, nlp]
related:
- "[[rag-systems]]"
- "[[vector-databases]]"
- "[[embedding-evaluation]]"
- "[[hybrid-search]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Embedding Models

## TL;DR
Models that encode text into dense vectors for semantic search — the retrieval backbone of RAG systems.

## Intuition
Embedding models (SBERT, E5, BGE, Voyage) produce fixed-length vectors where semantically similar texts are close in vector space. They're trained with contrastive objectives: pull positive pairs (question + relevant passage) together, push negative pairs apart. Hard negatives (passages that look relevant but aren't) are crucial for training quality. Choosing the right embedding model depends on domain (general vs. domain-specific), query-document asymmetry, and multilingual requirements.

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
- [[rag-systems]] — Embedding models generate the query and document vectors that power retrieval
- [[vector-databases]] — Vector DBs store and index embeddings at scale
- [[embedding-evaluation]] — MTEB benchmarks compare embedding models across tasks
- [[hybrid-search]] — Embedding models provide the dense retrieval signal in hybrid search

## Sources
<!-- Add raw/ source paths after ingestion -->
