---
title: "Agent Memory"
aliases: ["short-term memory", "long-term memory", "episodic memory", "semantic memory"]
tags: [agents, memory, context]
related:
- "[[agent-architectures]]"
- "[[rag-systems]]"
- "[[langgraph-agents]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Agent Memory

## TL;DR
How agents persist and access information across steps and sessions — from in-context to external stores.

## Intuition
In-context memory (the agent's current context window) is fast but limited. Episodic memory (stored interaction history, retrieved by similarity) enables learning across sessions. Semantic memory (a knowledge base of facts) is long-term and structured. Procedural memory (learned skills, fine-tuned into the model weights) is persistent but expensive to update. RAG is the most common implementation of episodic + semantic memory for agents.

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
- [[rag-systems]] — RAG implements external memory for agents
- [[langgraph-agents]] — LangGraph's StateGraph manages short-term memory as state
- [[agent-architectures]] — Memory type choice is a key architectural decision

## Sources
<!-- Add raw/ source paths after ingestion -->
