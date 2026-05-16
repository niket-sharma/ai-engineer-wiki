# LangGraph: Build Stateful, Multi-Actor Applications with LLMs

**Source:** LangGraph documentation (langchain-ai.github.io/langgraph), GitHub README
**URL:** https://github.com/langchain-ai/langgraph
**Version notes:** LangGraph 0.2+ (2024–2025)

---

## What is LangGraph?

LangGraph is a library for building stateful, multi-actor LLM applications using directed graphs. It extends LangChain with:
- **Explicit state**: typed state flows through the graph, accumulating across steps
- **Cycles**: graphs can loop (agents can retry, reflect, iterate)
- **Checkpointing**: save and resume graph state (fault tolerance, human-in-the-loop)
- **Streaming**: stream outputs at any node
- **Interrupts**: pause graph before/after a node for human approval

**Why not just LangChain LCEL?**
- LCEL is a linear pipeline (A → B → C) — no cycles, no stateful accumulation
- LangGraph adds loops, branching, stateful checkpointing — essential for real agents

---

## Core Concepts

### State

The central data structure that flows through the graph:

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Messages accumulate (add_messages = append to list, don't replace)
    messages: Annotated[list, add_messages]
    # Other state fields replace (default behavior)
    tool_calls_made: int
    current_plan: str
    retrieved_docs: list[str]
```

`Annotated[list, add_messages]`: special reducer that appends new messages to the list instead of replacing it. Other fields use the default reducer (replace with new value).

Custom reducers can be defined for any field: `Annotated[T, my_reducer_function]`.

### Nodes

Python functions that take state and return state updates:

```python
def call_llm(state: AgentState) -> dict:
    """Node: call the LLM with current messages."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}  # add_messages will append this

def execute_tools(state: AgentState) -> dict:
    """Node: execute tool calls from the last LLM message."""
    last_message = state["messages"][-1]
    tool_results = []
    for tool_call in last_message.tool_calls:
        result = tools_map[tool_call["name"]].invoke(tool_call["args"])
        tool_results.append(ToolMessage(content=str(result), 
                                         tool_call_id=tool_call["id"]))
    return {"messages": tool_results, "tool_calls_made": state["tool_calls_made"] + 1}
```

### Edges

**Regular edges:** Always go from node A to node B.
```python
graph.add_edge("tools", "agent")  # after tools, always go back to agent
```

**Conditional edges:** Function decides which node to go to next.
```python
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """If the last message has tool calls, execute them. Otherwise, end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"

graph.add_conditional_edges("agent", should_continue)
```

### Graph Assembly

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Build
graph = StateGraph(AgentState)
graph.add_node("agent", call_llm)
graph.add_node("tools", execute_tools)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
graph.add_edge("tools", "agent")

# Compile with checkpointing
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
```

---

## Running the Graph

### Basic Invocation

```python
config = {"configurable": {"thread_id": "user-123"}}  # for checkpointing

# Stream events
for event in app.stream(
    {"messages": [HumanMessage(content="What's the weather in NYC?")]},
    config=config
):
    for node, updates in event.items():
        print(f"Node '{node}':", updates)
```

### Streaming Tokens

```python
# Stream LLM tokens as they're generated
for chunk in app.stream(..., stream_mode="messages"):
    print(chunk.content, end="", flush=True)
```

### Getting Final State

```python
result = app.invoke(initial_state, config=config)
final_messages = result["messages"]
```

---

## Checkpointing: State Persistence

Checkpointing saves the full graph state after each node execution:

```python
# In-memory (dev/testing)
checkpointer = MemorySaver()

# Postgres (production)
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string("postgresql://...")

# SQLite (simple production)
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

app = graph.compile(checkpointer=checkpointer)
```

**What checkpointing enables:**
1. **Resume**: crash mid-workflow → restart from last checkpoint
2. **Human-in-the-loop**: pause, let human review, continue
3. **Time travel**: replay the graph from any past state
4. **Debugging**: inspect state at any node

**Thread IDs:** Each `thread_id` is an independent "conversation" with its own state. Same `thread_id` = continue existing conversation. Different `thread_id` = new conversation.

---

## Human-in-the-Loop Patterns

### Interrupts (Before/After a Node)

```python
# Compile with interrupt before "tools" node
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"]  # pause before executing tools
)

# Run until interrupt
result = app.invoke(initial_state, config=config)
# result["__interrupt__"] contains interrupt info

# Human reviews, approves, resumes
app.invoke(None, config=config)  # continue from checkpoint
```

### Approval Workflow

```python
def human_review_node(state: AgentState) -> dict:
    """Special node that pauses for human input."""
    # LangGraph's interrupt() function pauses execution
    from langgraph.types import interrupt
    
    human_input = interrupt({
        "message": "Please review the following plan:",
        "plan": state["current_plan"],
    })
    
    if human_input["approved"]:
        return {"approved": True}
    else:
        return {"approved": False, "feedback": human_input["feedback"]}
```

---

## Multi-Agent Patterns

### Supervisor Pattern

```python
class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str  # which subagent to call next

def supervisor(state: SupervisorState) -> dict:
    """Orchestrator decides which agent to call next."""
    response = supervisor_llm.invoke(state["messages"])
    return {"next": response.content}  # "researcher", "writer", "FINISH"

def route_to_agent(state: SupervisorState) -> str:
    return state["next"]

graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", research_agent)
graph.add_node("writer", writing_agent)
graph.add_conditional_edges("supervisor", route_to_agent,
    {"researcher": "researcher", "writer": "writer", "FINISH": END})
graph.add_edge("researcher", "supervisor")
graph.add_edge("writer", "supervisor")
```

### Subgraph Pattern

```python
# Each subagent can be its own graph
research_subgraph = StateGraph(ResearchState)
# ... build the subgraph

# Compile subgraph
research_agent = research_subgraph.compile()

# Use as a node in the supervisor graph
supervisor_graph.add_node("research", research_agent)
```

---

## LangGraph vs Alternatives

| | LangGraph | CrewAI | AutoGen | LangChain LCEL |
|---|---|---|---|---|
| **State management** | Explicit TypedDict | Implicit | Implicit | No |
| **Cycles** | Yes | Yes | Yes | No |
| **Checkpointing** | Built-in | No | No | No |
| **Human-in-loop** | Built-in interrupt | No | Manual | No |
| **Streaming** | Built-in | Limited | Limited | Yes |
| **Multi-agent** | First-class | Role-based | Conversation | No |
| **Debugging** | LangSmith integration | Limited | Limited | LangSmith |
| **Production readiness** | High | Medium | Medium | High |

---

## Common Patterns for Production

### Rate Limiting / Budget Control

```python
def check_budget(state: AgentState) -> str:
    if state["tool_calls_made"] >= 10:  # max 10 tool calls
        return "end"
    if state["estimated_cost_usd"] > 1.0:  # max $1
        return "end"
    return "continue"
```

### Error Handling

```python
def safe_tool_execution(state: AgentState) -> dict:
    try:
        result = execute_tools(state)
        return result
    except Exception as e:
        # Return error as tool result, let agent decide what to do
        return {"messages": [ToolMessage(content=f"Error: {e}", ...)]}
```

### Memory Integration

```python
def retrieve_memories(state: AgentState) -> dict:
    """Pre-process node: inject relevant user memories."""
    user_id = state["user_id"]
    recent_facts = memory_store.search(user_id, state["messages"][-1].content)
    memory_context = "\n".join(f["content"] for f in recent_facts)
    system_message = SystemMessage(content=f"User context:\n{memory_context}")
    return {"messages": [system_message]}
```

---

## -Relevant Insights

**When to use LangGraph over a simple LLM call:**
- Multi-step tasks requiring loops (retry logic, iterative refinement)
- Human approval in the middle of a workflow
- Long-running tasks that need fault tolerance
- Multiple agents with handoffs

**When NOT to use LangGraph:**
- Single-call tasks → just use `llm.invoke(...)`
- Simple pipeline → LangChain LCEL is simpler
- Stateless batch processing → doesn't need checkpointing

**The key differentiator:** Checkpointing + interrupts. No other major framework has built-in, production-ready state persistence and human-in-the-loop this seamlessly.

---

## Common  Questions

- "What is LangGraph and how does it differ from LangChain chains?"
- "Walk me through how you'd implement a ReAct agent in LangGraph."
- "What is checkpointing in LangGraph and why does it matter for production?"
- "How do you implement human-in-the-loop approval in LangGraph?"
- "Design a multi-agent system for financial document analysis using LangGraph."
- "How would you prevent an agent from running forever or spending too much?"
