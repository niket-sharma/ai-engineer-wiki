---
title: "Agent Evaluation"
aliases: ["agentic eval", "LLM agent benchmarks"]
tags: [evaluation, agents, benchmarks]
related:
- "[[langgraph-agents]]"
- "[[llm-evaluation]]"
- "[[rag-evaluation]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Agent Evaluation

## TL;DR
Measuring agent performance via task success, trajectory quality, and tool-use correctness.

## Intuition
Unlike single-turn LLM eval, agents take multiple steps. Evaluation must account for the full trajectory — did it use the right tools in the right order? Did it recover from errors? Task success (did it complete the goal?) and trajectory quality (did it do so efficiently?) are the two primary axes.

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
- [[langgraph-agents]] — The agent architecture being evaluated
- [[llm-evaluation]] — Single-turn eval methods as building blocks
- [[tool-use]] — Tool schemas and correctness are core eval dimensions

## Sources
<!-- Add raw/ source paths after ingestion -->
