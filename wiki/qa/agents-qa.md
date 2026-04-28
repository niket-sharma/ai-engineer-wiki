---
title: "Agents & Orchestration Q&A"
tags: [qa, agents, orchestration]
related: ["[[langgraph-agents]]", "[[mcp-protocol]]", "[[rag-systems]]"]
last_updated: 2026-04-22
---

# Agents & Orchestration Q&A

---

## L1 — Conceptual

### Q1. What is an LLM agent and how does it differ from a simple LLM call?

**A:** A single LLM call: input → output, stateless, one step. An agent adds:
- **Tools**: ability to call external functions (search, code execution, database queries)
- **Memory**: state that persists across multiple steps (in-context or external)
- **Control loop**: the LLM decides what to do next, executes it, observes results, repeats
- **Planning**: multi-step reasoning toward a goal

The LLM acts as the "reasoning engine" — it generates text describing what action to take, the framework parses that into an actual function call, executes it, and feeds the result back. This loop continues until the LLM decides the task is complete.

**When NOT to use an agent:** When a single LLM call is sufficient. Agents add latency, cost, and complexity. Many "agent" use cases are better served by a well-crafted single prompt.

---

### Q2. What is the ReAct pattern?

**A:** ReAct (Reason + Act) is a prompting pattern that interleaves reasoning and action:
```
Thought: I need to find today's stock price for AAPL.
Action: search_web("AAPL stock price 2026-04-22")
Observation: AAPL is trading at $189.50.
Thought: Now I can answer the question.
Answer: AAPL is currently $189.50.
```
The model is prompted to verbalize its reasoning before each action. This improves performance by forcing step-by-step reasoning, makes the agent's behavior interpretable, and allows the framework to parse action strings into actual function calls.

ReAct is the basis for most modern agent frameworks (LangGraph, Claude tool use, OpenAI Assistants).

---

### Q3. What are the main failure modes of LLM agents?

**A:**
1. **Infinite loops**: agent keeps calling tools without terminating → cost explosion, needs iteration limits
2. **Tool hallucination**: agent calls a non-existent tool or passes wrong arguments
3. **Context accumulation**: after many tool calls, the context window fills up, degrading coherence
4. **Planning failures**: agent takes locally reasonable steps that don't lead to the goal
5. **Unreliable tool outputs**: agent trusts incorrect tool results and builds on them
6. **Scope creep**: agent takes unintended actions (deletes files, sends emails) without confirmation

Mitigations: explicit iteration limits, tool schema validation, human-in-the-loop checkpoints for irreversible actions, context summarization.

---

## L2 — Technical

### Q4. How does LangGraph differ from LangChain's LCEL chains?

**A:**

**LangChain LCEL chains:**
- Linear (or branching) pipeline: A → B → C
- Implicit state: output of A is input to B
- Good for: simple, predictable pipelines
- Not good for: loops, dynamic routing, stateful multi-step processes

**LangGraph:**
- Explicit stateful graph: nodes + edges + state TypedDict
- Supports cycles (loops), conditional edges, checkpointing, streaming
- State is explicit and accumulates throughout the graph
- Supports human-in-the-loop via interrupt() before/after nodes

```python
# LangGraph StateGraph — state accumulates, graph can loop
class State(TypedDict):
    messages: Annotated[list, add_messages]
    
graph.add_node("agent", call_agent)
graph.add_node("tools", call_tools)
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")  # loop back
```

**When to use LangGraph over LCEL:**
- Any workflow with loops or retries
- Multi-agent systems with handoffs
- Long-running workflows needing checkpointing
- Human-in-the-loop approval flows

---

### Q5. Walk me through how you'd implement memory in a multi-session LLM agent.

**A:**

Memory types and implementation:

**In-context (within session):**
- Just the message history — already built into any agent framework
- Limit: context window fills up after many exchanges → summarize old messages

**Cross-session episodic memory:**
```python
# On session end: extract and store key facts
facts = llm.extract_facts(conversation_history)
for fact in facts:
    memory_store.upsert(
        user_id=user_id,
        embedding=embed(fact.text),
        content=fact.text,
        timestamp=now()
    )

# On session start: retrieve relevant past facts
user_context = memory_store.search(
    user_id=user_id,
    query=first_message,
    top_k=5
)
```

**Semantic memory (knowledge base):**
- RAG over documents, not user-specific
- Separate from episodic memory

**Procedural memory (how to do things):**
- System prompt — defines agent's skills and persona
- Updated by deploying new system prompt versions, not dynamically

**Production considerations:**
- Scope memories per user — never leak cross-user
- Staleness: memories from 2 years ago may be wrong — add timestamps and decay
- Storage: vector DB for retrieval (Qdrant), metadata in SQL

---

### Q6. Design a multi-agent system for financial document analysis.

**A:**

**Use case:** User uploads an earnings report and asks "Is this company a good investment?"

**Why multi-agent (not single):**
- Task has distinct specialized subtasks: extraction, analysis, synthesis
- Parallel execution possible
- Each subagent can be a smaller, faster, cheaper model

**Architecture:**

```
Orchestrator Agent (Claude Opus)
    ├── Extraction Agent → "Extract all financial metrics from pages 1-50"
    ├── Risk Analysis Agent → "Identify risk factors and rate severity"
    ├── Competitor Comparison Agent → "Compare metrics to industry benchmarks"
    └── Synthesis Agent → "Combine all analyses into investment memo"
```

**LangGraph implementation:**

```python
class State(TypedDict):
    document: str
    extracted_metrics: dict
    risk_analysis: str
    competitor_comparison: str
    final_memo: str

graph.add_node("orchestrator", route_to_subagents)
graph.add_node("extract", extraction_agent)
graph.add_node("risk", risk_agent)
graph.add_node("compare", comparison_agent)
graph.add_node("synthesize", synthesis_agent)

# Parallel execution: extract + risk + compare run simultaneously
graph.add_edge("orchestrator", "extract")
graph.add_edge("orchestrator", "risk")
graph.add_edge("orchestrator", "compare")
# After all three complete → synthesize
graph.add_edge(["extract", "risk", "compare"], "synthesize")
```

**Guardrails:**
- Each subagent output is validated (schema check) before passing to synthesizer
- Human review checkpoint before final memo is delivered
- Max execution time: 60 seconds, then timeout with partial results
- Audit log of all subagent calls and outputs

---

## L3 — Applied

### Q7. How would you build a customer-facing financial Q&A agent with compliance guardrails?

**A:**

**Key requirements for financial domain:**
- Must not give specific investment advice (regulatory)
- Must cite sources
- Must refuse questions outside scope
- Must be auditable

**Architecture:**
```
User query
→ Intent classifier: is this investment advice / out of scope?
→ If flagged: route to "I can provide information but not advice" response
→ If in scope: RAG retrieval from approved document corpus
→ LLM with strict system prompt
→ Response + citations + compliance disclaimer
→ Log everything
```

**System prompt guardrails:**
```
You are a financial information assistant. You MUST:
- Only use information from the provided context
- Never recommend buying/selling specific securities
- Always include: "This is for informational purposes only, not investment advice"
- Cite your sources with document names and sections
If asked for investment advice, redirect to a licensed advisor.
```

**Multi-layer safety:**
1. Intent classification (fast, cheap model) → catch 90% of out-of-scope queries before hitting expensive LLM
2. System prompt constraints → LLM-level guardrail
3. Output classification: scan response for prohibited phrases ("you should buy", "I recommend")
4. Human review queue: flag low-confidence responses for human review

**Monitoring:**
- Track refusal rate by query category
- Random sampling of responses for compliance review
- Alert if any response contains specific ticker + buy/sell recommendation
