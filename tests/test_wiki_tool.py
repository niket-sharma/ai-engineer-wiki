"""Tests for wiki_tool.py core functions."""
import sys
from pathlib import Path

# Allow importing from agent/ without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from wiki_tool import (  # noqa: E402
    _extract_frontmatter_info,
    audit_wiki,
    list_wiki_pages,
    resolve_page_path,
)


def test_audit_wiki_runs():
    result = audit_wiki()
    assert isinstance(result, str)
    assert "Wiki Audit Report" in result
    assert "Orphan Pages" in result
    assert "Stub Pages" in result


def test_list_wiki_pages():
    pages = list_wiki_pages()
    assert isinstance(pages, list)
    assert len(pages) > 0
    for page in pages:
        assert "slug" in page
        assert "path" in page


def test_resolve_page_path():
    path = resolve_page_path("attention-mechanism")
    assert path is not None
    assert path.exists()
    assert path.suffix == ".md"


def test_frontmatter_parses_numeric_relevance():
    """Regression test: numeric relevance values must be mapped to string form."""
    page_high = "---\ntitle: T\nrelevance: 10\nstatus: current\nlast_updated: 2026-01-01\n---\n"
    page_med = "---\ntitle: T\nrelevance: 6\nstatus: current\nlast_updated: 2026-01-01\n---\n"
    page_low = "---\ntitle: T\nrelevance: 3\nstatus: current\nlast_updated: 2026-01-01\n---\n"
    page_str = "---\ntitle: T\nrelevance: high\nstatus: current\nlast_updated: 2026-01-01\n---\n"

    assert _extract_frontmatter_info(page_high).relevance == "high"
    assert _extract_frontmatter_info(page_med).relevance == "medium"
    assert _extract_frontmatter_info(page_low).relevance == "low"
    assert _extract_frontmatter_info(page_str).relevance == "high"


def test_frontmatter_parses_numeric_relevance_boundary():
    """8 → high, 5 → medium, 4 → low."""
    assert _extract_frontmatter_info("---\nrelevance: 8\n---\n").relevance == "high"
    assert _extract_frontmatter_info("---\nrelevance: 5\n---\n").relevance == "medium"
    assert _extract_frontmatter_info("---\nrelevance: 4\n---\n").relevance == "low"
