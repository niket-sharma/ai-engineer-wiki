"""Tests for scripts/validate_wiki.py — the MAINTAIN/CI integrity gate."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "validate_wiki.py"

spec = importlib.util.spec_from_file_location("validate_wiki", SCRIPT)
validate_wiki = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate_wiki)

GOOD_PAGE = """---
title: "KV Cache"
aliases: []
tags: [inference]
related: [[flash-attention]]
sources: []
relevance: high
last_updated: 2026-06-11
status: current
---

# KV Cache

See [[flash-attention]].
"""


@pytest.fixture
def fake_wiki(tmp_path, monkeypatch):
    (tmp_path / "wiki" / "concepts").mkdir(parents=True)
    monkeypatch.setattr(validate_wiki, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validate_wiki, "WIKI_DIR", tmp_path / "wiki")
    return tmp_path / "wiki"


def test_frontmatter_parser():
    fm = validate_wiki.frontmatter(GOOD_PAGE)
    assert fm["title"] == '"KV Cache"'
    assert fm["status"] == "current"
    assert validate_wiki.frontmatter("# no frontmatter\n") is None


def test_valid_page_passes(fake_wiki):
    (fake_wiki / "concepts" / "kv-cache.md").write_text(GOOD_PAGE,
                                                        encoding="utf-8")
    (fake_wiki / "concepts" / "flash-attention.md").write_text(
        GOOD_PAGE.replace("KV Cache", "Flash Attention"), encoding="utf-8")
    errors = []
    validate_wiki.check_frontmatter(errors)
    assert errors == []


def test_missing_field_is_error(fake_wiki):
    bad = GOOD_PAGE.replace("relevance: high\n", "")
    (fake_wiki / "concepts" / "bad.md").write_text(bad, encoding="utf-8")
    errors = []
    validate_wiki.check_frontmatter(errors)
    assert any("missing frontmatter field 'relevance'" in e for e in errors)


def test_invalid_status_and_date_are_errors(fake_wiki):
    bad = GOOD_PAGE.replace("status: current", "status: shiny") \
                   .replace("last_updated: 2026-06-11", "last_updated: June 11")
    (fake_wiki / "concepts" / "bad.md").write_text(bad, encoding="utf-8")
    errors = []
    validate_wiki.check_frontmatter(errors)
    assert any("status 'shiny'" in e for e in errors)
    assert any("not YYYY-MM-DD" in e for e in errors)


def test_numeric_relevance_allowed(fake_wiki):
    page = GOOD_PAGE.replace("relevance: high", "relevance: 9")
    (fake_wiki / "concepts" / "num.md").write_text(page, encoding="utf-8")
    errors = []
    validate_wiki.check_frontmatter(errors)
    assert errors == []


def test_broken_link_is_warning(fake_wiki):
    (fake_wiki / "concepts" / "kv-cache.md").write_text(
        GOOD_PAGE + "\nAlso see [[no-such-page]].\n", encoding="utf-8")
    (fake_wiki / "concepts" / "flash-attention.md").write_text(
        GOOD_PAGE, encoding="utf-8")
    warnings = []
    validate_wiki.check_links(warnings)
    assert any("[[no-such-page]]" in w for w in warnings)
    assert not any("[[flash-attention]]" in w for w in warnings)


def test_index_coverage_warns_on_missing_reference(fake_wiki):
    (fake_wiki / "concepts" / "kv-cache.md").write_text(GOOD_PAGE,
                                                        encoding="utf-8")
    (fake_wiki / "index.md").write_text("# Index\n(nothing)\n",
                                        encoding="utf-8")
    warnings = []
    validate_wiki.check_index_coverage(warnings)
    assert any("kv-cache.md" in w for w in warnings)


def test_real_wiki_passes_validation():
    """The repo's actual wiki must pass (this is what CI gates on)."""
    result = subprocess.run([sys.executable, str(SCRIPT)], cwd=REPO_ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
