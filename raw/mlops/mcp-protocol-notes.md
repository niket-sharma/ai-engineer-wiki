# Model Context Protocol (MCP): Specification and Implementation Notes

**Source:** Anthropic MCP Specification (modelcontextprotocol.io), GitHub: modelcontextprotocol/specification
**Published:** November 2024 (Anthropic open-source release)
**Version:** MCP 2024-11-05 (current as of early 2025)

---

## What MCP Is

MCP (Model Context Protocol) is an open standard for connecting LLMs (hosted in "clients") to external tools, data sources, and capabilities (provided by "servers"). Anthropic developed MCP and open-sourced it in November 2024 — it's now maintained by the community.

**The N×M integration problem (before MCP):**
- N LLM apps (Claude Desktop, Cursor, your internal tool, VS Code extension...)
- M external tools (GitHub, Slack, databases, file systems, APIs...)
- Without a standard: N×M custom integrations required
- With MCP: N clients implement MCP client protocol, M servers implement MCP server protocol → N+M implementations total

**Analogy:** MCP is to LLM tools what USB is to peripheral devices, or what HTTP is to web services.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ MCP Host (e.g., Claude Desktop, Cursor IDE, your app)       │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ MCP Client (built into the host)                    │    │
│  │ - Manages connections to MCP servers                │    │
│  │ - Translates tool calls between LLM and server      │    │
│  └───────────────────────┬─────────────────────────────┘    │
│                          │  MCP Protocol                    │
└──────────────────────────┼─────────────────────────────────-┘
                           │  JSON-RPC 2.0 over stdio or SSE
              ┌────────────▼────────────┐
              │ MCP Server              │
              │ - Exposes tools         │
              │ - Exposes resources     │
              │ - Exposes prompts       │
              │           │            │
              └───────────┼────────────┘
                          │
              ┌───────────▼───────────────┐
              │ External System           │
              │ (GitHub, DB, file system) │
              └───────────────────────────┘
```

---

## Core Primitives

### 1. Tools (Most Important)

Functions the LLM can call. Server exposes a list of tools with JSON Schema definitions.

**Server definition:**
```python
@mcp.tool()
def search_database(
    query: str,
    table: str = "products",
    limit: int = 10
) -> str:
    """Search the product database using natural language.
    
    Args:
        query: Natural language search query
        table: Database table to search (products, orders, customers)
        limit: Maximum number of results to return
    
    Returns:
        JSON string with matching records
    """
    results = db.execute(f"SELECT * FROM {table} WHERE ... LIMIT {limit}")
    return json.dumps(results, indent=2)
```

**Tool call flow:**
1. LLM generates: `{"tool": "search_database", "args": {"query": "red sneakers", "limit": 5}}`
2. MCP client forwards to MCP server
3. Server executes the function
4. Server returns result
5. MCP client injects result into LLM context as a tool result message
6. LLM continues generating

**Security consideration:** Tools can have side effects. MCP doesn't enforce safety — it's the server's responsibility to validate inputs and the host's responsibility to implement authorization.

### 2. Resources

Data sources the LLM can read (analogous to GET requests — should be read-only, no side effects).

```python
@mcp.resource("file://{path}")
def read_file(path: str) -> str:
    """Read a file from the local filesystem."""
    return Path(path).read_text()

@mcp.resource("db://tables/{table_name}/schema")
def get_table_schema(table_name: str) -> str:
    """Get the schema for a database table."""
    return db.get_schema(table_name)
```

Resources have URIs (like REST endpoints). The LLM can list available resources and read specific ones.

**Difference from tools:** Tools are actions (can have side effects). Resources are data access (should be read-only).

### 3. Prompts

Reusable prompt templates that the server exposes:

```python
@mcp.prompt()
def code_review_prompt(code: str, language: str) -> list[Message]:
    """Generate a code review prompt."""
    return [
        Message(role="user", content=f"""
        Please review the following {language} code:
        
        ```{language}
        {code}
        ```
        
        Focus on: correctness, security, performance, readability.
        """)
    ]
```

The host can offer these prompts to users as slash commands or quick actions.

### 4. Sampling (Server → Client LLM Call)

Allows the MCP server to ask the host to make an LLM call. Enables recursive/agentic patterns:

```
Server: "I need to analyze this document. Please have the LLM summarize it first."
Client: OK, I'll call the LLM → return summary → you continue
Server: Now I have the summary, I can proceed with my task.
```

Less commonly used, but important for complex agentic servers.

---

## Transport Protocols

### stdio Transport (Local)

```
Host process → spawns MCP server as subprocess
                ↕ stdin/stdout (JSON-RPC 2.0 messages)
              MCP server process
```

Configuration in Claude Desktop (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/Documents"]
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]
    }
  }
}
```

**When:** Local tools, development, trusted environments.

### SSE Transport (Remote)

```
Host → HTTP connection → MCP server (running as web service)
       Server-Sent Events for server→client messages
       HTTP POST for client→server messages
```

**When:** Remote/cloud tools, shared team servers, SaaS integrations.

---

## Building an MCP Server

### Python (using FastMCP)

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My Company's Internal Tools")

@mcp.tool()
def query_sales_db(
    sql: str,
    max_rows: int = 100
) -> str:
    """Execute a read-only SQL query against the sales database."""
    # Safety: only allow SELECT
    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed")
    
    results = sales_db.execute(sql, max_rows=max_rows)
    return results.to_json()

@mcp.tool()
def search_confluence(query: str, space: str = "ENGINEERING") -> str:
    """Search Confluence documentation."""
    results = confluence_client.search(query, space=space)
    return json.dumps([{"title": r.title, "url": r.url, "excerpt": r.excerpt} 
                       for r in results[:5]])

@mcp.resource("confluence://{page_id}")
def get_confluence_page(page_id: str) -> str:
    """Get the full content of a Confluence page."""
    return confluence_client.get_page(page_id).content

if __name__ == "__main__":
    mcp.run()  # stdio transport by default
```

### TypeScript (official SDK)

```typescript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new Server({ name: "my-server", version: "1.0.0" });

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [{
    name: "get_weather",
    description: "Get current weather for a location",
    inputSchema: {
      type: "object",
      properties: {
        location: { type: "string", description: "City name" }
      },
      required: ["location"]
    }
  }]
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name === "get_weather") {
    const { location } = request.params.arguments;
    const weather = await fetchWeather(location);
    return { content: [{ type: "text", text: JSON.stringify(weather) }] };
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
```

---

## Security Considerations

**Tool injection:** A malicious MCP server could expose tools that, when called by the LLM, exfiltrate data or perform unauthorized actions. Hosts should:
- Only connect to trusted MCP servers
- Implement approval flows for sensitive tool calls
- Scope tool permissions (e.g., read-only DB access)

**Input validation:** MCP doesn't validate tool arguments — the server must validate. SQL injection, path traversal, command injection are all possible if the server doesn't sanitize.

**Authentication:** MCP doesn't specify authentication. For remote servers, implement OAuth or API key authentication at the HTTP layer.

**Secrets:** Never expose secrets through MCP tools. The LLM can be prompted to log or exfiltrate sensitive data.

---

## MCP Ecosystem (Early 2025)

**Official Anthropic servers:**
- `@modelcontextprotocol/server-filesystem`: file system access
- `@modelcontextprotocol/server-github`: GitHub API
- `@modelcontextprotocol/server-postgres`: PostgreSQL queries
- `@modelcontextprotocol/server-brave-search`: web search

**Third-party servers:**
- Cloudflare, Atlassian, Zapier, Linear have announced MCP servers
- Community-built: 500+ servers on GitHub as of early 2025

**Clients:**
- Claude Desktop: native MCP support
- Cursor: MCP support in 2024
- Claude Code (this tool): MCP client built-in

---

## MCP vs OpenAI Function Calling vs LangChain Tools

| | MCP | OpenAI Functions | LangChain Tools |
|---|---|---|---|
| **Standard** | Open (Anthropic) | Proprietary (OpenAI) | Library-specific |
| **Transport** | stdio, SSE | API | In-process |
| **Server/client split** | Yes | No | No |
| **Cross-provider** | Yes | No | Adapter needed |
| **Resources** | Yes | No | Limited |
| **Prompts** | Yes | No | No |
| **Maturity** | Early (2024–) | Mature (2023–) | Mature (2023–) |

**Key advantage of MCP:** Build a tool once → use with any MCP-compatible client. OpenAI functions and LangChain tools require reimplementing for each framework.

---

## Common Interview Questions

- "What is MCP and why was it created? What problem does it solve?"
- "Explain the difference between Tools, Resources, and Prompts in MCP."
- "How does MCP differ from OpenAI function calling?"
- "What are the security risks of MCP and how would you mitigate them?"
- "How would you build an MCP server that exposes your company's internal database?"
- "What transport protocols does MCP support and when would you use each?"
