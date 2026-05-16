---
title: "Tool Use"
aliases: ["tool calling", "function calling", "tool schemas"]
tags: [agents, tools, function-calling]
related:
- "[[function-calling]]"
- "[[agent-architectures]]"
- "[[mcp-protocol]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Tool Use

## TL;DR
Giving LLMs access to external functions (search, code execution, APIs) to extend their capabilities.

## Intuition
Tools let agents take actions beyond text generation. Key design decisions: schema design (tool names and descriptions are part of the prompt — clear descriptions improve selection accuracy); error handling (tools fail; agents need explicit error states and retry logic); tool selection (with many tools, a router or retrieval step selects the relevant subset to avoid polluting context).

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
- [[function-calling]] — The API-level mechanism for structured tool invocation
- [[mcp-protocol]] — MCP standardizes tool discovery and execution across agent frameworks
- [[agent-architectures]] — Tool calls are the action step in ReAct and related patterns

## Sources
<!-- Add raw/ source paths after ingestion -->
