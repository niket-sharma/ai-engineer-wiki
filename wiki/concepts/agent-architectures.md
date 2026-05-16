---
title: "Agent Architectures"
aliases: ["ReAct", "Reflexion", "Plan-and-Execute", "agent patterns"]
tags: [agents, architecture, reasoning]
related:
- "[[langgraph-agents]]"
- "[[tool-use]]"
- "[[agent-memory]]"
- "[[multi-agent-systems]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Agent Architectures

## TL;DR
Patterns for structuring LLM reasoning + action loops — from simple ReAct to multi-step planners.

## Intuition
ReAct (Reason + Act): alternates between a thought (reasoning) and an action (tool call) until done. Simple, general, but can lose track of goals in long chains. Plan-and-Execute: separate planner (makes a multi-step plan) and executor (carries out each step). More robust for complex tasks. Reflexion: adds a reflection step after each episode where the agent critiques its own performance and updates a 'scratchpad' for the next attempt.

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
- [[langgraph-agents]] — LangGraph implements these patterns as state machines
- [[tool-use]] — Agent architectures orchestrate tool calls
- [[agent-memory]] — Memory types determine how context persists across steps
- [[agentic-rag]] — Agents with retrieval capabilities require specialized architectures

## Sources
<!-- Add raw/ source paths after ingestion -->
