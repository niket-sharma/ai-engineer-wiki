---
title: "Agentic RAG"
aliases: ["adaptive retrieval", "corrective RAG", "self-RAG", "query routing"]
tags: [agents, rag, retrieval]
related:
- "[[rag-systems]]"
- "[[agent-architectures]]"
- "[[query-rewriting]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Agentic RAG

## TL;DR
RAG systems where an agent decides when, what, and how to retrieve — rather than a fixed pipeline.

## Intuition
Standard RAG always retrieves before generating. Agentic RAG makes retrieval a decision: the agent determines if retrieval is needed, what query to use, how many rounds to retrieve, and whether retrieved context is sufficient (corrective RAG rejects bad context and re-retrieves). Self-RAG adds a token-level critic that evaluates retrieved passages mid-generation.

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
- [[rag-systems]] — Agentic RAG extends the standard RAG pipeline with decision-making
- [[agent-architectures]] — The retrieval decision is part of the agent's reasoning loop
- [[query-rewriting]] — Query rewriting is a key agentic retrieval enhancement

## Sources
<!-- Add raw/ source paths after ingestion -->
