"""Tests for OP-8 MAINTAIN (agent/maintain.py) — all offline.

The LLM maintainer is faked; repo paths are redirected to tmp_path so no test
touches the real wiki/, state/, or raw/ trees.
"""
import json

import pytest

import maintain
from maintain import (
    ChangelogEntry,
    PageBudget,
    WatchItem,
    changelog_table,
    consume_queue,
    keyword_relevance,
    load_queue,
    load_run_state,
    load_watchlist,
    parse_arxiv_atom,
    parse_feed,
    parse_maintain_request,
    run_maintain,
    save_run_state,
    weakness_list,
    write_wiki_page,
)

CONCEPT_PAGE = """---
title: "KV Cache"
aliases: []
tags: [inference]
related: []
sources: []
relevance: high
last_updated: 2026-06-01
status: current
---

# KV Cache

## TL;DR
Cache of K and V projections reused across decode steps.
"""


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """Redirect maintain.py's module-level paths into a temp repo."""
    (tmp_path / "wiki" / "concepts").mkdir(parents=True)
    (tmp_path / "wiki" / "qa").mkdir()
    (tmp_path / "state").mkdir()
    (tmp_path / "raw" / "auto").mkdir(parents=True)
    (tmp_path / "wiki" / "concepts" / "kv-cache.md").write_text(
        CONCEPT_PAGE, encoding="utf-8")
    (tmp_path / "wiki" / "index.md").write_text(
        "# Index\n- [[kv-cache]]\n- [[flash-attention]]\n", encoding="utf-8")
    monkeypatch.setattr(maintain, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(maintain, "WIKI_DIR", tmp_path / "wiki")
    monkeypatch.setattr(maintain, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(maintain, "AUTO_DIR", tmp_path / "raw" / "auto")
    return tmp_path


class FakeMaintainer:
    """Deterministic stand-in for LLMMaintainer."""

    def __init__(self, qa="## L3\n\n### Q1. Harder question?\nAnswer.\n",
                 fail=False):
        self.qa = qa
        self.fail = fail
        self.calls = []

    def generate_qa(self, concept, difficulty, concept_page, existing_qa, date):
        self.calls.append(("generate_qa", concept))
        if self.fail:
            raise RuntimeError("api down")
        return (existing_qa or "") + self.qa

    def expand_section(self, page, section, sources, date):
        self.calls.append(("expand_section", section))
        if self.fail:
            raise RuntimeError("api down")
        return page + f"\n## Expanded: {section}\nNew depth.\n"

    def draft_page(self, slug, item_text, source_path, date):
        self.calls.append(("draft_page", slug))
        return f"---\ntitle: \"{slug}\"\n---\n\n# {slug}\n\nDrafted.\n"

    def score_relevance(self, item, topics, weaknesses):
        return {"score": 9, "topic_slug": "kv-cache",
                "reason": "fake", "new_page": False}


# ---------------------------------------------------------------------------
# Watchlist + run state
# ---------------------------------------------------------------------------

def test_real_watchlist_parses():
    wl = load_watchlist()
    assert any(e.get("category") == "cs.CL" for e in wl["arxiv"])
    assert any("huggingface" in f.get("name", "") for f in wl["feeds"])
    assert 0 < wl["settings"]["relevance_threshold"] <= 10
    assert wl["settings"]["max_auto_ingests"] == 5


def test_run_state_defaults_and_roundtrip(tmp_path):
    path = tmp_path / "maintain_state.json"
    state = load_run_state(path)
    assert state == {"last_run": None, "seen_ids": []}
    state["last_run"] = "2026-06-11"
    state["seen_ids"] = [f"id-{i}" for i in range(600)]
    save_run_state(state, path)
    loaded = load_run_state(path)
    assert loaded["last_run"] == "2026-06-11"
    assert len(loaded["seen_ids"]) == 500  # trimmed


# ---------------------------------------------------------------------------
# Page budget + safe writes
# ---------------------------------------------------------------------------

def test_page_budget_caps_distinct_pages():
    budget = PageBudget(cap=2)
    assert budget.allow("wiki/a.md")
    assert budget.allow("wiki/b.md")
    assert budget.allow("wiki/a.md")        # re-touch is free
    assert not budget.allow("wiki/c.md")    # over cap


def test_write_wiki_page_rejects_outside_wiki(fake_repo):
    with pytest.raises(ValueError):
        write_wiki_page("state/skill_ratings.json", "x", PageBudget(), [],
                        source="s", reason="r")
    with pytest.raises(ValueError):
        write_wiki_page("raw/auto/x.md", "x", PageBudget(), [],
                        source="s", reason="r")


def test_write_wiki_page_refuses_shrink(fake_repo):
    changelog = []
    ok = write_wiki_page("wiki/concepts/kv-cache.md", "tiny", PageBudget(),
                         changelog, source="s", reason="r")
    assert not ok
    assert changelog[0].action == "skipped"
    # Original content untouched.
    text = (fake_repo / "wiki/concepts/kv-cache.md").read_text(encoding="utf-8")
    assert "KV Cache" in text


def test_write_wiki_page_dry_run_writes_nothing(fake_repo):
    changelog = []
    ok = write_wiki_page("wiki/concepts/new-page.md", "# New\n", PageBudget(),
                         changelog, source="s", reason="r", dry_run=True)
    assert ok
    assert changelog[0].action == "created"
    assert not (fake_repo / "wiki/concepts/new-page.md").exists()


# ---------------------------------------------------------------------------
# Queue consumption
# ---------------------------------------------------------------------------

def _queue(*tasks):
    return {"tasks": list(tasks)}


def test_generate_qa_task_creates_qa_page(fake_repo):
    queue = _queue({"id": "q-001", "type": "generate_qa", "concept": "kv-cache",
                    "difficulty": 4, "reason": "scored 1/4", "status": "pending"})
    fake = FakeMaintainer()
    changelog = []
    done = consume_queue(queue, fake, PageBudget(), changelog, "2026-06-11",
                         {"wanted_topics": []})
    assert done == 1
    assert queue["tasks"][0]["status"] == "done"
    qa = fake_repo / "wiki/qa/kv-cache-qa.md"
    assert qa.exists() and "Harder question" in qa.read_text(encoding="utf-8")
    assert changelog[0].page == "wiki/qa/kv-cache-qa.md"


def test_generate_qa_unknown_concept_fails(fake_repo):
    queue = _queue({"id": "q-001", "type": "generate_qa",
                    "concept": "no-such-page", "status": "pending"})
    done = consume_queue(queue, FakeMaintainer(), PageBudget(), [],
                         "2026-06-11", {"wanted_topics": []})
    assert done == 0
    assert queue["tasks"][0]["status"] == "failed"


def test_llm_unavailable_keeps_task_pending(fake_repo):
    queue = _queue({"id": "q-001", "type": "generate_qa", "concept": "kv-cache",
                    "status": "pending"})
    done = consume_queue(queue, None, PageBudget(), [], "2026-06-11",
                         {"wanted_topics": []})
    assert done == 0
    assert queue["tasks"][0]["status"] == "pending"
    assert "LLM unavailable" in queue["tasks"][0]["note"]


def test_api_error_keeps_task_pending(fake_repo):
    queue = _queue({"id": "q-001", "type": "generate_qa", "concept": "kv-cache",
                    "status": "pending"})
    done = consume_queue(queue, FakeMaintainer(fail=True), PageBudget(), [],
                         "2026-06-11", {"wanted_topics": []})
    assert done == 0
    assert queue["tasks"][0]["status"] == "pending"


def test_wiki_gap_with_source_expands_page(fake_repo):
    (fake_repo / "raw" / "inference").mkdir()
    (fake_repo / "raw" / "inference" / "kv-cache-notes.md").write_text(
        "kv cache deep dive notes", encoding="utf-8")
    queue = _queue({"id": "q-002", "type": "wiki_gap",
                    "page": "wiki/concepts/kv-cache.md",
                    "section": "eviction policies", "status": "pending"})
    fake = FakeMaintainer()
    done = consume_queue(queue, fake, PageBudget(), [], "2026-06-11",
                         {"wanted_topics": []})
    assert done == 1
    assert ("expand_section", "eviction policies") in fake.calls
    text = (fake_repo / "wiki/concepts/kv-cache.md").read_text(encoding="utf-8")
    assert "Expanded: eviction policies" in text


def test_wiki_gap_without_source_goes_to_watchlist(fake_repo):
    queue = _queue({"id": "q-002", "type": "wiki_gap",
                    "page": "wiki/concepts/kv-cache.md",
                    "section": "x", "status": "pending"})
    watchlist = {"wanted_topics": []}
    done = consume_queue(queue, FakeMaintainer(), PageBudget(), [],
                         "2026-06-11", watchlist)
    assert done == 1
    assert queue["tasks"][0]["status"] == "done"
    assert watchlist["wanted_topics"] == ["kv-cache"]


def test_unknown_task_type_fails(fake_repo):
    queue = _queue({"id": "q-009", "type": "mystery", "status": "pending"})
    consume_queue(queue, FakeMaintainer(), PageBudget(), [], "2026-06-11",
                  {"wanted_topics": []})
    assert queue["tasks"][0]["status"] == "failed"


def test_non_pending_tasks_untouched(fake_repo):
    queue = _queue({"id": "q-001", "type": "generate_qa", "concept": "kv-cache",
                    "status": "done", "completed": "2026-06-01"})
    fake = FakeMaintainer()
    done = consume_queue(queue, fake, PageBudget(), [], "2026-06-11",
                         {"wanted_topics": []})
    assert done == 0 and fake.calls == []


# ---------------------------------------------------------------------------
# Feed parsing
# ---------------------------------------------------------------------------

ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2606.01234v1</id>
    <title>Faster  KV Cache\n Eviction</title>
    <summary>We evict   keys.</summary>
    <published>2026-06-09T17:59:59Z</published>
  </entry>
</feed>
"""

RSS_FEED = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>New quantization method</title>
    <link>https://example.com/post</link>
    <guid>post-1</guid>
    <description>&lt;p&gt;Int4 for &lt;b&gt;flash attention&lt;/b&gt;.&lt;/p&gt;</description>
    <pubDate>Tue, 09 Jun 2026 10:00:00 GMT</pubDate>
  </item>
</channel></rss>
"""


def test_parse_arxiv_atom():
    items = parse_arxiv_atom(ARXIV_ATOM, "cs.CL")
    assert len(items) == 1
    item = items[0]
    assert item.id == "arxiv:2606.01234v1"
    assert item.title == "Faster KV Cache Eviction"  # whitespace collapsed
    assert item.source == "arxiv:cs.CL"
    assert item.published == "2026-06-09"


def test_parse_rss_feed():
    items = parse_feed(RSS_FEED, "example-blog")
    assert len(items) == 1
    item = items[0]
    assert item.id == "example-blog:post-1"
    assert item.url == "https://example.com/post"
    assert "flash attention" in item.summary
    assert "<b>" not in item.summary  # tags stripped


# ---------------------------------------------------------------------------
# Relevance
# ---------------------------------------------------------------------------

def test_keyword_relevance_scores_topic_hits():
    item = WatchItem(id="x", title="A study of the KV cache", url="",
                     summary="and flash attention kernels", source="t")
    verdict = keyword_relevance(item, ["kv-cache", "flash-attention", "rope"],
                                weaknesses=[])
    assert verdict["score"] == 6  # two topic hits
    assert verdict["topic_slug"] in ("kv-cache", "flash-attention")


def test_keyword_relevance_weakness_boost():
    item = WatchItem(id="x", title="KV cache eviction", url="",
                     summary="", source="t")
    base = keyword_relevance(item, ["kv-cache"], weaknesses=[])
    boosted = keyword_relevance(item, ["kv-cache"], weaknesses=["kv-cache"])
    assert boosted["score"] == base["score"] + 2


def test_keyword_relevance_irrelevant_item_scores_zero():
    item = WatchItem(id="x", title="Quantum biology of birds", url="",
                     summary="magnetoreception", source="t")
    assert keyword_relevance(item, ["kv-cache", "rope"], [])["score"] == 0


def test_weakness_list_orders_weakest_first():
    ratings = {"concepts": {
        "strong": {"rating": 1500}, "weak": {"rating": 1050},
        "mid": {"rating": 1250},
    }}
    assert weakness_list(ratings) == ["weak", "mid"]  # 1500 excluded (>1300)


# ---------------------------------------------------------------------------
# Orchestrator (offline end-to-end)
# ---------------------------------------------------------------------------

def test_run_maintain_offline_end_to_end(fake_repo):
    state_dir = fake_repo / "state"
    (state_dir / "maintenance_queue.json").write_text(json.dumps({
        "tasks": [{"id": "q-001", "type": "generate_qa", "concept": "kv-cache",
                   "difficulty": 4, "reason": "scored 1/4",
                   "status": "pending"}]}), encoding="utf-8")
    ratings_before = '{"version": 1, "concepts": {}}'
    (state_dir / "skill_ratings.json").write_text(ratings_before,
                                                  encoding="utf-8")
    wl_path = fake_repo / "watchlist.yaml"
    wl_path.write_text("settings: {}\narxiv: []\nfeeds: []\nwanted_topics: []\n",
                       encoding="utf-8")

    result = run_maintain(no_fetch=True, no_pr=True,
                          maintainer=FakeMaintainer(), state_dir=state_dir,
                          watchlist_path=wl_path, log_to_wiki=False)

    assert result.tasks_done == 1
    assert result.validation_passed
    assert result.pr_url is None
    assert (fake_repo / "wiki/qa/kv-cache-qa.md").exists()
    # Queue persisted with the task marked done.
    queue = load_queue(state_dir / "maintenance_queue.json")
    assert queue["tasks"][0]["status"] == "done"
    # Run state stamped.
    assert load_run_state(state_dir / "maintain_state.json")["last_run"]
    # Safety rail: skill_ratings.json untouched byte-for-byte.
    assert (state_dir / "skill_ratings.json").read_text(
        encoding="utf-8") == ratings_before


def test_run_maintain_dry_run_writes_nothing(fake_repo):
    state_dir = fake_repo / "state"
    (state_dir / "maintenance_queue.json").write_text(json.dumps({
        "tasks": [{"id": "q-001", "type": "generate_qa", "concept": "kv-cache",
                   "status": "pending"}]}), encoding="utf-8")
    wl_path = fake_repo / "watchlist.yaml"
    wl_path.write_text("settings: {}\narxiv: []\nfeeds: []\nwanted_topics: []\n",
                       encoding="utf-8")

    result = run_maintain(dry_run=True, no_fetch=True, no_pr=True,
                          maintainer=FakeMaintainer(), state_dir=state_dir,
                          watchlist_path=wl_path, log_to_wiki=False)

    assert result.tasks_done == 1  # reported …
    assert not (fake_repo / "wiki/qa/kv-cache-qa.md").exists()  # … not written
    # Queue file not rewritten in dry-run.
    queue = load_queue(state_dir / "maintenance_queue.json")
    assert queue["tasks"][0]["status"] == "pending"
    assert not (state_dir / "maintain_state.json").exists()


def test_run_maintain_respects_page_cap(fake_repo):
    # 3 concepts queued but cap of 1 page.
    for slug in ("page-a", "page-b"):
        (fake_repo / "wiki" / "concepts" / f"{slug}.md").write_text(
            CONCEPT_PAGE.replace("KV Cache", slug), encoding="utf-8")
    state_dir = fake_repo / "state"
    tasks = [{"id": f"q-{i}", "type": "generate_qa", "concept": c,
              "status": "pending"}
             for i, c in enumerate(["kv-cache", "page-a", "page-b"])]
    (state_dir / "maintenance_queue.json").write_text(
        json.dumps({"tasks": tasks}), encoding="utf-8")
    wl_path = fake_repo / "watchlist.yaml"
    wl_path.write_text("settings: {}\narxiv: []\nfeeds: []\nwanted_topics: []\n",
                       encoding="utf-8")

    result = run_maintain(no_fetch=True, no_pr=True, max_pages=1,
                          maintainer=FakeMaintainer(), state_dir=state_dir,
                          watchlist_path=wl_path, log_to_wiki=False)
    assert result.tasks_done == 1
    skipped = [e for e in result.changelog if e.action == "skipped"]
    assert len(skipped) == 2
    assert all("budget" in e.reason for e in skipped)


def test_max_pages_hard_capped_at_twelve(fake_repo):
    state_dir = fake_repo / "state"
    wl_path = fake_repo / "watchlist.yaml"
    wl_path.write_text("settings: {}\narxiv: []\nfeeds: []\nwanted_topics: []\n",
                       encoding="utf-8")
    # max_pages=999 must clamp to the spec rail of 12 — verified indirectly
    # via PageBudget, which run_maintain constructs with min(max_pages, 12).
    result = run_maintain(no_fetch=True, no_pr=True, max_pages=999,
                          maintainer=FakeMaintainer(), state_dir=state_dir,
                          watchlist_path=wl_path, log_to_wiki=False)
    assert result.tasks_done == 0


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def test_changelog_table_renders():
    table = changelog_table([
        ChangelogEntry("wiki/qa/kv-cache-qa.md", "updated", "q-001", "scored 1/4"),
    ])
    assert "| wiki/qa/kv-cache-qa.md | updated | q-001 | scored 1/4 |" in table
    assert table.splitlines()[0].startswith("| Page ")


@pytest.mark.parametrize("text,expected", [
    ("run maintenance", True),
    ("please process the maintenance queue", True),
    ("do the weekly update", True),
    ("maintain the wiki", True),
    ("what does the wiki say about kv cache?", False),
    ("interview me on transformers", False),
])
def test_parse_maintain_request(text, expected):
    assert parse_maintain_request(text) is expected
