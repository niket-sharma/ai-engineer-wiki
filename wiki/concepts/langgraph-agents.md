---
title: "LangGraph & Agents"
aliases: ["LangGraph", "agent orchestration", "ReAct", "tool use", "agentic AI", "multi-agent"]
tags: [agents, orchestration, rag]
related: ["[[rag-systems]]", "[[mcp-protocol]]"]
sources: ["training-knowledge"]
interview_relevance: high
last_updated: 2026-04-22
status: current
---

# LangGraph & Agents

## TL;DR
LLM agents combine a language model with tools, memory, and a control loop — the LLM decides what to do next, calls tools, observes results, and repeats. LangGraph implements this as a stateful directed graph (StateGraph) where nodes are Python functions and edges can be conditional. It's the dominant production framework for multi-step, agentic workflows as of 2025.

## Intuition
A single LLM call is like a single function call — input in, output out, stateless. An agent is like a program: it has state, can call external functions (tools), loop, branch, and make decisions. The LLM serves as the "reasoning engine" that decides what action to take at each step.

LangGraph's key contribution: making this control flow explicit, inspectable, and interruptible. Instead of implicit chains, you define a graph where you can see exactly what's happening, checkpoint state, replay, and stream outputs.

## Technical Detail

**ReAct Pattern (Reason + Act):**
```
Thought: I need to find the current price of AAPL.
Action: search_web("AAPL stock price today")
Observation: AAPL is $189.50
Thought: Now I can answer.
Answer: AAPL is currently $189.50
```
The LLM is prompted to alternate between reasoning (Thought) and action (Action). The framework executes the action and feeds the result back as Observation.

**LangGraph StateGraph:**
```python
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    messages: list[BaseMessage]
    tool_calls: list[dict]

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tools", execute_tools)
graph.add_conditional_edges("llm", should_continue, {"continue": "tools", "end": END})
graph.add_edge("tools", "llm")
graph.set_entry_point("llm")

app = graph.compile(checkpointer=MemorySaver())
```

Key concepts:
- **State**: TypedDict that flows through the graph and accumulates
- **Nodes**: Python functions that take state and return state updates
- **Edges**: Control flow — can be conditional (LLM decides next step)
- **Checkpointing**: Save state at each step → resumable, fault-tolerant
- **Interrupts**: Pause graph execution for human-in-the-loop approval

**Tool use pattern:**
```python
tools = [search_web, calculate, query_database]
llm_with_tools = llm.bind_tools(tools)
# LLM outputs tool_calls → parse → execute → feed back
```

**Multi-agent patterns:**
- **Supervisor**: One orchestrator LLM routes to specialized subagents
- **Swarm**: Agents hand off to each other directly (LangGraph Swarm library)
- **Parallel**: Spawn multiple agents in parallel, aggregate results

**Memory types in agents:**
| Type | Scope | Implementation |
|---|---|---|
| In-context | Current conversation | Messages in state |
| Episodic | Cross-session recall | Vector DB (user memories) |
| Semantic | Long-term knowledge | RAG / wiki |
| Procedural | How to do things | System prompt |

## Variants & Extensions

| Framework | Key Characteristic |
|---|---|
| LangGraph | Stateful graphs, checkpointing, production-grade |
| CrewAI | Role-based multi-agent, simple API |
| AutoGen | Microsoft, code execution, debate patterns |
| Agno (formerly Phidata) | Lightweight, good for quick prototyping |
| Claude Agent SDK | Native Anthropic, tool use built-in |
| OpenAI Agents SDK | Native OpenAI, similar to LangGraph patterns |

## Tradeoffs
| Advantage | Disadvantage |
|---|---|
| Explicit control flow — inspectable and debuggable | More complex than simple chain calls |
| Checkpointing enables fault tolerance and replay | LLM-based routing can be unpredictable |
| Human-in-the-loop via interrupts | Cost scales with number of LLM calls per task |
| Streaming built-in | Debugging multi-agent systems is hard |
| Works with any LLM that supports tool/function calling | Long-running agents can drift from original intent |

## Interview Angles

**What interviewers are really testing:**
- Do you understand the ReAct pattern and why agents need it?
- Can you design a multi-step agentic RAG pipeline?
- Do you understand the tradeoffs of agentic vs single-call approaches?
- Do you know how to handle failures, loops, and human escalation?

**Common follow-up questions:**
- "Walk me through how a ReAct agent processes a complex query."
- "How does LangGraph differ from LangChain LCEL chains?"
- "How would you design a multi-agent system for financial document analysis?"
- "What is agent memory and how would you implement cross-session memory?"
- "How do you prevent agent loops and runaway costs?"
- "When would you NOT use an agent? What's the cost?"

**Gotchas / misconceptions:**
- Agents are not always better — a single LLM call is cheaper and more deterministic
- LLM routing in agents fails on ambiguous tasks — design explicit fallbacks
- "Human in the loop" via LangGraph interrupts is production-ready, not just a demo feature
- Checkpointing is critical for production agents — without it you can't resume or debug

## Connections
- [[rag-systems]] — agents can orchestrate multi-step RAG with iterative retrieval
- [[mcp-protocol]] — MCP standardizes how agents connect to external tools/resources

## Sources
- Training knowledge (Yao et al. 2022 "ReAct"; LangGraph docs 2024–2025)
