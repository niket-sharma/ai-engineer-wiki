---
title: "Function Calling"
aliases: ["tool calling", "tool use", "JSON mode", "parallel tool calls"]
tags: [prompting, agents, tools, structured-output]
related:
- "[[mcp-protocol]]"
- "[[structured-output]]"
- "[[langgraph-agents]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Function Calling

## TL;DR
Structured output format where the LLM produces a JSON function call that a host application executes.

## Intuition
Function calling gives LLMs a well-typed interface to external systems. The model is given tool schemas (JSON Schema format) and can choose to call a tool by outputting a structured JSON object instead of text. The host application executes the call and returns results. This is more reliable than free-text tool invocation because the output is parsed, validated, and typed. Parallel function calls let the model call multiple tools in one step.

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
- [[mcp-protocol]] — MCP extends function calling with a standard protocol for tool discovery and execution
- [[structured-output]] — Function calling is a special case of structured output generation
- [[langgraph-agents]] — LangGraph orchestrates multi-step function calling

## Sources
<!-- Add raw/ source paths after ingestion -->
