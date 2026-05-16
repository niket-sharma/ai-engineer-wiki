import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
WIKI_DIR = REPO_ROOT / "wiki"
RAW_DIR = REPO_ROOT / "raw"
INDEX_PATH = WIKI_DIR / "index.md"
LOG_PATH = WIKI_DIR / "log.md"

# Preferred precedence when duplicate slugs exist.
PREFERRED_WIKI_SUBDIR_ORDER = [
    "qa",
    "interview-qa",
    "-qa",
    "concepts",
    "system-design",
    "companies",
    "cheatsheets",
]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalize_slug(value: str) -> str:
    return value.strip().removesuffix(".md").lower()


def _safe_resolve(path: Path) -> Path:
    return path.resolve(strict=False)


def _is_under(base: Path, target: Path) -> bool:
    try:
        _safe_resolve(target).relative_to(_safe_resolve(base))
        return True
    except ValueError:
        return False


def _wiki_sort_key(path: Path) -> tuple[int, str]:
    rel = path.relative_to(WIKI_DIR)
    top = rel.parts[0] if rel.parts else ""
    if top in PREFERRED_WIKI_SUBDIR_ORDER:
        rank = PREFERRED_WIKI_SUBDIR_ORDER.index(top)
    else:
        rank = len(PREFERRED_WIKI_SUBDIR_ORDER)
    return rank, str(rel).replace("\\", "/")


def search_index(query: str) -> tuple[list[str], str]:
    """Read index.md and return all page names + raw index text."""
    index = _read_text(INDEX_PATH)
    all_pages = re.findall(r"\[\[([^\]]+)\]\]", index)
    if query:
        q = query.lower()
        all_pages = [p for p in all_pages if q in p.lower()]
    return all_pages, index


def resolve_page_path(page_name: str) -> Path | None:
    slug = _normalize_slug(page_name)
    candidates = [p for p in WIKI_DIR.rglob("*.md") if p.stem.lower() == slug]
    if not candidates:
        return None
    candidates.sort(key=_wiki_sort_key)
    return candidates[0]


def list_wiki_pages() -> list[dict[str, str]]:
    pages: list[dict[str, str]] = []
    for path in sorted(WIKI_DIR.rglob("*.md"), key=_wiki_sort_key):
        rel = path.relative_to(WIKI_DIR).as_posix()
        if rel in {"index.md", "log.md"}:
            continue
        pages.append({"slug": path.stem, "path": f"wiki/{rel}"})
    return pages


def load_page(page_name: str) -> str | None:
    """Load a full wiki page by slug name, searching all subdirectories."""
    path = resolve_page_path(page_name)
    if path is None:
        return None
    return _read_text(path)


def extract_sections(page_content: str, headings: list[str] | None = None) -> str:
    """Extract specific ## sections from page content."""
    if not headings:
        return page_content

    requested = {h.strip().lower() for h in headings}
    extracted: list[str] = []
    current_section: str | None = None
    current_lines: list[str] = []

    def _matches(section_name: str) -> bool:
        key = section_name.lower().strip()
        return key in requested or any(req in key for req in requested)

    for line in page_content.split("\n"):
        if line.startswith("## "):
            if current_section and _matches(current_section):
                extracted.append(f"### {current_section}\n" + "\n".join(current_lines).rstrip())
            current_section = line.lstrip("# ").strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_section and _matches(current_section):
        extracted.append(f"### {current_section}\n" + "\n".join(current_lines).rstrip())

    if extracted:
        return "\n\n".join(extracted)

    available = [
        line.lstrip("# ").strip()
        for line in page_content.split("\n")
        if line.startswith("## ")
    ]
    return (
        "Requested sections were not found. "
        f"Requested={sorted(requested)}. Available={available}.\n\n"
        f"{page_content}"
    )


def wiki_lookup(page_names: list[str], sections: list[str] | None = None) -> str:
    """Main tool function. Loads pages and optionally filters to specific sections."""
    results = []
    for name in page_names:
        path = resolve_page_path(name)
        if path:
            content = _read_text(path)
            extracted = extract_sections(content, sections)
            rel = path.relative_to(REPO_ROOT).as_posix()
            results.append(f"# [{name}] ({rel})\n{extracted}")
        else:
            results.append(f"# [{name}]\n(Page not found in wiki)")
    return "\n\n---\n\n".join(results)


def read_raw_source(raw_path: str) -> str:
    target = _safe_resolve(REPO_ROOT / raw_path)
    if not _is_under(RAW_DIR, target):
        raise ValueError("raw_path must be under raw/.")
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"File not found: {raw_path}")
    return _read_text(target)


def read_wiki_file(path: str) -> str:
    target = _safe_resolve(REPO_ROOT / path)
    if not _is_under(WIKI_DIR, target):
        raise ValueError("path must be under wiki/.")
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return _read_text(target)


def write_wiki_file(path: str, content: str) -> str:
    target = _safe_resolve(REPO_ROOT / path)
    if not _is_under(WIKI_DIR, target):
        raise ValueError("path must be under wiki/.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {target.relative_to(REPO_ROOT).as_posix()}"


def append_wiki_log(entry: str) -> str:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.write_text("# Operation Log\nAppend-only. Most recent entry at top.\n\n", encoding="utf-8")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    chunk = f"## [OP] {now}\n{entry.strip()}\n\n"
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(chunk)
    return f"Appended log entry at {now}"


def search_wiki(query: str, limit: int = 20) -> str:
    q = query.strip().lower()
    if not q:
        return "Query is empty."

    hits: list[str] = []
    for path in sorted(WIKI_DIR.rglob("*.md"), key=_wiki_sort_key):
        text = _read_text(path)
        if q in text.lower():
            rel = path.relative_to(REPO_ROOT).as_posix()
            lines = text.splitlines()
            snippet = ""
            for line in lines:
                if q in line.lower():
                    snippet = line.strip()
                    break
            hits.append(f"- {rel}: {snippet}")
            if len(hits) >= limit:
                break

    if not hits:
        return f"No matches found for '{query}'."
    return "\n".join(hits)


@dataclass
class FrontmatterInfo:
    relevance: str | None = None
    status: str | None = None
    last_updated: date | None = None


def _extract_frontmatter_info(text: str) -> FrontmatterInfo:
    info = FrontmatterInfo()
    if not text.startswith("---\n"):
        return info

    end = text.find("\n---", 4)
    if end == -1:
        return info

    fm = text[4:end]
    rel = re.search(r"^relevance:\s*(\S+)\s*$", fm, flags=re.MULTILINE)
    if rel:
        raw = rel.group(1).strip().lower()
        try:
            n = int(raw)
            info.relevance = "high" if n >= 8 else ("medium" if n >= 5 else "low")
        except ValueError:
            info.relevance = raw

    status = re.search(r"^status:\s*(\S+)\s*$", fm, flags=re.MULTILINE)
    if status:
        info.status = status.group(1).strip().lower()

    updated = re.search(r"^last_updated:\s*(\d{4}-\d{2}-\d{2})\s*$", fm, flags=re.MULTILINE)
    if updated:
        try:
            info.last_updated = datetime.strptime(updated.group(1), "%Y-%m-%d").date()
        except ValueError:
            pass

    return info


def _all_wiki_links(text: str) -> list[str]:
    return re.findall(r"\[\[([^\]]+)\]\]", text)


def _iter_wiki_markdown() -> Iterable[Path]:
    for p in WIKI_DIR.rglob("*.md"):
        if p.name in {"index.md", "log.md"}:
            continue
        yield p


def audit_wiki() -> str:
    pages = list(_iter_wiki_markdown())
    page_slugs = {p.stem.lower() for p in pages}

    contradiction_entries: list[str] = []
    inbound: dict[str, int] = {p.stem.lower(): 0 for p in pages}
    missing_links: dict[str, set[str]] = {}
    stub_pages: list[str] = []
    stale_high_priority: list[str] = []

    for path in pages:
        rel = path.relative_to(REPO_ROOT).as_posix()
        text = _read_text(path)

        for i, line in enumerate(text.splitlines(), start=1):
            if "CONTRADICTION" in line.upper():
                contradiction_entries.append(f"- {rel}:{i} {line.strip()}")

        for link in _all_wiki_links(text):
            slug = _normalize_slug(link)
            if slug in inbound:
                inbound[slug] += 1
            else:
                missing_links.setdefault(slug, set()).add(rel)

        fm = _extract_frontmatter_info(text)
        if fm.status == "stub":
            stub_pages.append(rel)

        if fm.relevance == "high" and fm.last_updated:
            if fm.last_updated < (date.today() - timedelta(days=60)):
                stale_high_priority.append(f"- {rel} (last_updated={fm.last_updated.isoformat()})")

    orphan_pages = [
        p.relative_to(REPO_ROOT).as_posix()
        for p in pages
        if inbound.get(p.stem.lower(), 0) == 0
    ]

    missing_pages_lines: list[str] = []
    for slug, refs in sorted(missing_links.items()):
        missing_pages_lines.append(f"- [[{slug}]] referenced in: {', '.join(sorted(refs))}")

    report = [f"## Wiki Audit Report - {date.today().isoformat()}"]
    report.append(f"### Unresolved Contradictions: {len(contradiction_entries)}")
    report.extend(contradiction_entries[:50] if contradiction_entries else ["- None"])
    report.append(f"### Orphan Pages: {len(orphan_pages)}")
    report.extend([f"- {p}" for p in orphan_pages[:100]] if orphan_pages else ["- None"])
    report.append(f"### Stub Pages: {len(stub_pages)}")
    report.extend([f"- {p}" for p in stub_pages[:100]] if stub_pages else ["- None"])
    report.append(f"### Missing Pages (referenced but absent): {len(missing_pages_lines)}")
    report.extend(missing_pages_lines[:100] if missing_pages_lines else ["- None"])
    report.append(f"### Stale High-Priority Pages: {len(stale_high_priority)}")
    report.extend(stale_high_priority[:100] if stale_high_priority else ["- None"])
    return "\n".join(report)


WIKI_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "wiki_lookup",
        "description": (
            "Look up compiled knowledge from the AI Engineer Wiki. "
            "Returns full synthesized pages or specific sections. "
            "Use this before answering wiki questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "page_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Wiki page slugs to load, e.g. ['attention-mechanism', 'kv-cache']",
                },
                "sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional. Specific ## section headings to extract, "
                        "e.g. ['TL;DR', 'Tradeoffs']. If omitted, full pages are returned."
                    ),
                },
            },
            "required": ["page_names"],
        },
    },
}

LIST_PAGES_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_wiki_pages",
        "description": "List all wiki markdown pages with slug and path.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}

READ_RAW_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_raw_source",
        "description": "Read a raw source file. Path must be under raw/.",
        "parameters": {
            "type": "object",
            "properties": {
                "raw_path": {
                    "type": "string",
                    "description": "Repo-relative path like raw/transformers/attention-is-all-you-need.md",
                }
            },
            "required": ["raw_path"],
        },
    },
}

READ_WIKI_FILE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_wiki_file",
        "description": "Read a wiki file. Path must be under wiki/.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Repo-relative path like wiki/concepts/kv-cache.md",
                }
            },
            "required": ["path"],
        },
    },
}

WRITE_WIKI_FILE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_wiki_file",
        "description": "Create or update a wiki file. Path must be under wiki/.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Repo-relative path under wiki/",
                },
                "content": {
                    "type": "string",
                    "description": "Full file content to write.",
                },
            },
            "required": ["path", "content"],
        },
    },
}

APPEND_LOG_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "append_wiki_log",
        "description": "Append an operation entry to wiki/log.md.",
        "parameters": {
            "type": "object",
            "properties": {
                "entry": {
                    "type": "string",
                    "description": "Log entry markdown body.",
                }
            },
            "required": ["entry"],
        },
    },
}

SEARCH_WIKI_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_wiki",
        "description": "Search all wiki markdown files for a text query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
            },
            "required": ["query"],
        },
    },
}

AUDIT_WIKI_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "audit_wiki",
        "description": "Run wiki health audit (contradictions, orphans, stubs, gaps, stale pages).",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}
