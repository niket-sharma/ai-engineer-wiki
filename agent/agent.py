import json
import os
from pathlib import Path
from typing import Any

import openai

from wiki_tool import (
    APPEND_LOG_TOOL_SCHEMA,
    AUDIT_WIKI_TOOL_SCHEMA,
    LIST_PAGES_TOOL_SCHEMA,
    READ_RAW_TOOL_SCHEMA,
    READ_WIKI_FILE_TOOL_SCHEMA,
    SEARCH_WIKI_TOOL_SCHEMA,
    WIKI_TOOL_SCHEMA,
    WRITE_WIKI_FILE_TOOL_SCHEMA,
    append_wiki_log,
    audit_wiki,
    list_wiki_pages,
    read_raw_source,
    read_wiki_file,
    search_wiki,
    wiki_lookup,
    write_wiki_file,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = REPO_ROOT / "wiki" / "index.md"
SKILL_PATH = REPO_ROOT / "skill.md"

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")
MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "4096"))

client = openai.OpenAI()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _build_system_prompt() -> str:
    index = _read_text(INDEX_PATH)
    skill_excerpt = _read_text(SKILL_PATH)

    return f"""You are the operating agent for the AI Engineer Wiki codebase.

Primary goal:
- Execute wiki operations end-to-end with repository changes when needed.

Supported operations:
- OP-1 INGEST: read raw source, update/create wiki pages, update wiki/index.md, append wiki/log.md.
- OP-2 QUERY: answer from compiled wiki only, cite pages as [[page-slug]].
- OP-3 AUDIT: produce contradiction/orphan/stub/gap/staleness report.
- OP-4 GENERATE: create or refresh topic Q&A pages in wiki/qa/.
- OP-5 CHEATSHEET: create/update concise reference pages in wiki/cheatsheets/.

Mandatory rules:
- Never write outside wiki/.
- Never modify files under raw/.
- When updating wiki content, preserve/refresh frontmatter fields.
- Always log non-trivial write operations using append_wiki_log.
- For queries, always call wiki_lookup before final answer.
- If wiki coverage is missing, say so and suggest ingest.

Tooling guidance:
- Use list_wiki_pages + search_wiki to discover existing pages.
- Use read_raw_source for ingestion input.
- Use read_wiki_file / write_wiki_file for deterministic edits.
- Use audit_wiki when user asks for health checks.

Current wiki index:
{index}

Project operating spec:
{skill_excerpt[:24000]}
"""


SYSTEM_PROMPT = _build_system_prompt()
TOOLS = [
    WIKI_TOOL_SCHEMA,
    LIST_PAGES_TOOL_SCHEMA,
    READ_RAW_TOOL_SCHEMA,
    READ_WIKI_FILE_TOOL_SCHEMA,
    WRITE_WIKI_FILE_TOOL_SCHEMA,
    APPEND_LOG_TOOL_SCHEMA,
    SEARCH_WIKI_TOOL_SCHEMA,
    AUDIT_WIKI_TOOL_SCHEMA,
]


def _dispatch_tool_call(name: str, payload: dict[str, Any]) -> str:
    if name == "wiki_lookup":
        return wiki_lookup(payload.get("page_names", []), payload.get("sections"))
    if name == "list_wiki_pages":
        pages = list_wiki_pages()
        return "\n".join(f"- {p['slug']} ({p['path']})" for p in pages)
    if name == "read_raw_source":
        return read_raw_source(payload.get("raw_path", ""))
    if name == "read_wiki_file":
        return read_wiki_file(payload.get("path", ""))
    if name == "write_wiki_file":
        return write_wiki_file(payload.get("path", ""), payload.get("content", ""))
    if name == "append_wiki_log":
        return append_wiki_log(payload.get("entry", ""))
    if name == "search_wiki":
        return search_wiki(payload.get("query", ""), int(payload.get("limit", 20)))
    if name == "audit_wiki":
        return audit_wiki()
    raise ValueError(f"Unknown tool: {name}")


def run_agent(user_query: str, history: list) -> tuple[str, list]:
    """Single agent turn with tool use."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it in your shell or add it to .env at repo root."
        )

    history.append({"role": "user", "content": user_query})

    while True:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            tools=TOOLS,
            messages=messages,
        )

        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            assistant_msg = choice.message
            history.append({
                "role": "assistant",
                "content": assistant_msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in (assistant_msg.tool_calls or [])
                ],
            })

            for tc in (assistant_msg.tool_calls or []):
                try:
                    payload = json.loads(tc.function.arguments)
                    result = _dispatch_tool_call(tc.function.name, payload)
                except Exception as exc:  # noqa: BLE001
                    result = f"Tool error in {tc.function.name}: {exc}"

                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            continue

        answer = choice.message.content or "I could not produce a text response for that request."
        history.append({"role": "assistant", "content": answer})
        return answer, history
