---
title: "MCP Protocol"
aliases: ["Model Context Protocol", "MCP", "MCP server", "MCP client"]
tags: [agents, orchestration, tooling, protocol]
related: ["[[langgraph-agents]]"]
sources: ["training-knowledge"]
interview_relevance: medium
last_updated: 2026-04-22
status: current
---

# MCP Protocol

## TL;DR
Model Context Protocol (MCP) is an open standard (Anthropic, 2024) for connecting LLMs to external tools, data sources, and resources via a client-server protocol. It's analogous to HTTP for web services — a common interface so any LLM host can talk to any MCP server without custom integration code. As of 2025 it's gaining adoption rapidly (Cursor, Claude Desktop, many IDE integrations use it).

## Intuition
Before MCP: every LLM app needed custom code to integrate with each tool. Want GitHub + Slack + database? Write three different integrations, each in your app's format.

With MCP: each tool exposes an MCP server. Any LLM client that speaks MCP can use any MCP server. Build once, use everywhere. It's the "USB standard" for LLM tools.

## Technical Detail

**Architecture:**
```
LLM Host (Claude Desktop, Cursor, your app)
    ↕ MCP Client (built into host)
    ↕ MCP Protocol (JSON-RPC 2.0 over stdio or SSE)
    ↕ MCP Server (exposes tools/resources)
        → External systems (GitHub, DBs, files, APIs)
```

**Core primitives:**
| Primitive | What it is | Example |
|---|---|---|
| Tools | Functions the LLM can call | `search_web(query)`, `execute_sql(query)` |
| Resources | Data the LLM can read | File contents, DB row, API response |
| Prompts | Reusable prompt templates | System prompt for a specific persona |
| Sampling | Server asks host to call the LLM | Allows recursive LLM calls |

**Transport options:**
- **stdio**: MCP server runs as a subprocess, communicates via stdin/stdout — for local tools
- **SSE (Server-Sent Events)**: HTTP-based, for remote MCP servers

**Tool definition (server-side):**
```python
@mcp.tool()
def search_database(query: str, table: str) -> str:
    """Search the financial database."""
    results = db.execute(f"SELECT * FROM {table} WHERE ...")
    return json.dumps(results)
```

**Tool call flow:**
1. LLM receives user message
2. LLM decides to call a tool (outputs tool call JSON)
3. MCP client finds the right MCP server for that tool
4. Sends call over JSON-RPC
5. MCP server executes, returns result
6. Result injected back into LLM context
7. LLM generates final response

## Variants & Extensions
- **FastMCP**: Python library for quick MCP server creation
- **MCP Inspector**: Debug tool for testing MCP servers
- **Claude Desktop MCP config**: JSON config to register MCP servers locally
- **Remote MCP servers**: Hosted servers (Cloudflare Workers) for cloud tools

## Tradeoffs
| Advantage | Disadvantage |
|---|---|
| Standardized — build once, use with any MCP-compatible client | Still maturing — ecosystem is early (2024–2025) |
| Clean separation of tool logic from LLM app | Adds network/subprocess overhead vs direct function calls |
| Supports resource reading, not just tool calling | Security model is not yet fully standardized |
| Growing ecosystem (100s of MCP servers on GitHub) | Debugging distributed MCP servers is non-trivial |

## Interview Angles

**What interviewers are really testing:**
- Are you aware of MCP as an emerging standard?
- Can you explain why a protocol like MCP is valuable (avoiding N×M integrations)?
- Do you understand the primitives (tools, resources, prompts)?

**Common follow-up questions:**
- "What is MCP and why was it created?"
- "How does MCP differ from OpenAI's function calling?"
- "How would you build an MCP server to expose your company's internal database?"
- "What are the security considerations when building MCP servers?"

**Gotchas / misconceptions:**
- MCP is a protocol/standard, not a specific implementation
- OpenAI function calling is similar but proprietary — MCP is open and cross-provider
- MCP does not require Anthropic's Claude — any LLM host can implement the client side

## Connections
- [[langgraph-agents]] — LangGraph agents can consume MCP servers as tool providers

## Sources
- Training knowledge (Anthropic MCP specification 2024; modelcontextprotocol.io)
