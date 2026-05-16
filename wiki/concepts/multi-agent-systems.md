---
title: "Multi-Agent Systems"
aliases: ["multi-agent", "AutoGen", "CrewAI", "supervisor pattern"]
tags: [agents, multi-agent, orchestration]
related:
- "[[agent-architectures]]"
- "[[langgraph-agents]]"
- "[[tool-use]]"
sources: []
relevance: medium
last_updated: 2026-05-16
status: stub
---

# Multi-Agent Systems

## TL;DR
Systems where multiple LLM agents collaborate — each specialized, with a coordinator managing task routing.

## Intuition
Single agents struggle with long tasks that require parallel work or deep specialization. Multi-agent systems divide tasks: a supervisor routes subtasks to specialized worker agents (coder, researcher, reviewer), collects results, and synthesizes. Key challenge: error propagation (one agent's mistake cascades) and communication overhead (passing context between agents consumes tokens). LangGraph's multi-agent support and AutoGen/CrewAI are the main frameworks.

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
- [[langgraph-agents]] — LangGraph's multi-agent graphs support supervisor and peer-to-peer patterns
- [[agent-architectures]] — Multi-agent is an extension of Plan-and-Execute with distributed execution
- [[agent-evaluation]] — Multi-agent systems require trajectory-level evaluation

## Sources
<!-- Add raw/ source paths after ingestion -->
