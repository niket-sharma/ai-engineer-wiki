---
title: "RAG Evaluation"
aliases: ["RAGAS", "RAG eval", "retrieval-augmented generation evaluation"]
tags: [evaluation, rag, ragas, retrieval]
related:
- "[[rag-systems]]"
- "[[llm-evaluation]]"
- "[[embedding-evaluation]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# RAG Evaluation

## TL;DR
Frameworks and metrics for evaluating retrieval-augmented generation pipelines end-to-end.

## Intuition
RAG has two failure modes: bad retrieval (right answer not fetched) and bad generation (right context, wrong answer). RAGAS separates these with component-level metrics: context precision/recall measure retrieval quality; faithfulness and answer relevance measure generation quality.

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
- [[rag-systems]] — The pipeline being evaluated
- [[llm-evaluation]] — General LLM eval methods apply to the generation step
- [[reranking]] — Reranking improves context precision

## Sources
<!-- Add raw/ source paths after ingestion -->
