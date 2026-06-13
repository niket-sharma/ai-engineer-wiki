"""OP-8 MAINTAIN — autonomous weekly updater.

Spec: interview-agent-spec.md §3.3, skill.md OP-8.

Pipeline: consume state/maintenance_queue.json → fetch watchlist sources into
raw/auto/ → relevance-filter → draft/update wiki pages → validate → open a PR
(never push main) → append a run summary to wiki/log.md.

Deterministic parts (queue consumption, watchlist parsing, feed parsing,
keyword relevance fallback, page budget, changelog, branch/PR plumbing) are
unit-testable offline; the LLM maintainer is injected so tests substitute a
fake. With no API key the run degrades gracefully: fetching and filtering
still work, LLM-dependent tasks stay `pending`.

Safety rails (spec §3.3): never delete pages; never write
state/skill_ratings.json (ASSESS-only); cap pages touched per run at 12;
validate before opening a PR.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from interview import _load_prompt, load_skill_ratings

REPO_ROOT = Path(__file__).resolve().parent.parent
WIKI_DIR = REPO_ROOT / "wiki"
STATE_DIR = REPO_ROOT / "state"
AUTO_DIR = REPO_ROOT / "raw" / "auto"
WATCHLIST_PATH = Path(__file__).resolve().parent / "watchlist.yaml"

MAX_PAGES_PER_RUN = 12          # spec §3.3 safety rail
DEFAULT_RELEVANCE_THRESHOLD = 6
DEFAULT_MAX_AUTO_INGESTS = 5    # spec §8 cost rail
DEFAULT_MAX_ITEMS_PER_SOURCE = 10

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"


# ---------------------------------------------------------------------------
# Watchlist + run state
# ---------------------------------------------------------------------------

def load_watchlist(path: Path | None = None) -> dict:
    path = path or WATCHLIST_PATH
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (ImportError, FileNotFoundError):
        return {"settings": {}, "arxiv": [], "feeds": [], "wanted_topics": []}
    data.setdefault("settings", {})
    data.setdefault("arxiv", [])
    data.setdefault("feeds", [])
    data.setdefault("wanted_topics", [])
    return data


def save_watchlist(watchlist: dict, path: Path | None = None) -> None:
    import yaml

    path = path or WATCHLIST_PATH
    path.write_text(
        "# Watchlist for OP-8 MAINTAIN (spec §3.3 step 2).\n"
        + yaml.safe_dump(watchlist, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def load_run_state(path: Path | None = None) -> dict:
    """state/maintain_state.json — last run date + already-seen item ids."""
    path = path or (STATE_DIR / "maintain_state.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    data.setdefault("last_run", None)
    data.setdefault("seen_ids", [])
    return data


def save_run_state(state: dict, path: Path | None = None) -> Path:
    path = path or (STATE_DIR / "maintain_state.json")
    state["seen_ids"] = state.get("seen_ids", [])[-500:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Page budget + changelog (safety rails)
# ---------------------------------------------------------------------------

class PageBudget:
    """Caps distinct wiki pages touched per run (spec: 12)."""

    def __init__(self, cap: int = MAX_PAGES_PER_RUN):
        self.cap = cap
        self.touched: list[str] = []

    def allow(self, rel_path: str) -> bool:
        if rel_path in self.touched:
            return True
        if len(self.touched) >= self.cap:
            return False
        self.touched.append(rel_path)
        return True


@dataclass
class ChangelogEntry:
    page: str
    action: str        # created | updated | skipped
    source: str
    reason: str


def write_wiki_page(rel_path: str, content: str, budget: PageBudget,
                    changelog: list[ChangelogEntry], source: str,
                    reason: str, dry_run: bool = False) -> bool:
    """Single write path for MAINTAIN: wiki/-only, budget-capped, no deletes,
    never shrinks an existing page below half its size (anti-clobber)."""
    target = (REPO_ROOT / rel_path).resolve()
    try:
        target.relative_to(WIKI_DIR.resolve())
    except ValueError:
        raise ValueError(f"MAINTAIN may only write under wiki/: {rel_path}")

    if not budget.allow(rel_path):
        changelog.append(ChangelogEntry(rel_path, "skipped", source,
                                        f"page budget ({budget.cap}) exhausted"))
        return False

    action = "updated" if target.exists() else "created"
    if target.exists():
        old = target.read_text(encoding="utf-8")
        if len(content) < len(old) * 0.5:
            changelog.append(ChangelogEntry(
                rel_path, "skipped", source,
                "draft shrank page by >50% — refusing (no-deletion rail)"))
            return False

    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    changelog.append(ChangelogEntry(rel_path, action, source, reason))
    return True


# ---------------------------------------------------------------------------
# LLM maintainer (injected; optional)
# ---------------------------------------------------------------------------

class LLMMaintainer:
    """Generative steps: harder Q&A, section expansion, page drafting,
    relevance scoring. All grounded in full pages / full sources (router
    pattern — whole documents, never chunks)."""

    def __init__(self, model: str | None = None):
        import openai

        self.client = openai.OpenAI()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.drafter_system = _load_prompt("page_drafter_system.md")
        self.relevance_system = _load_prompt("relevance_filter.md")

    def _chat(self, system: str, prompt: str, max_tokens: int = 3000) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def generate_qa(self, concept: str, difficulty: int, concept_page: str,
                    existing_qa: str | None, date: str) -> str:
        existing = (
            f"Current Q&A page (preserve all of it, append to it):\n---\n"
            f"{existing_qa[:12000]}\n---\n" if existing_qa else
            "There is no Q&A page for this concept yet — create one with the "
            "standard frontmatter and ## L1 / ## L2 / ## L3 sections.\n"
        )
        return self._chat(
            self.drafter_system,
            f"Today's date: {date}\n"
            f"Task: add {difficulty - 1}–{difficulty} HARDER interview questions "
            f"(difficulty level {difficulty} of 5) for the concept "
            f"[[{concept}]] to its Q&A page. Use '### Qn.' headings under the "
            f"matching ## L-section (L3 holds level 4–5 questions).\n"
            f"{existing}"
            f"Ground every question and answer strictly in this concept page:\n"
            f"---\n{concept_page[:16000]}\n---\n"
            f"Output the FULL updated Q&A page markdown.",
        )

    def expand_section(self, page: str, section: str, sources: list[str],
                       date: str) -> str:
        joined = "\n\n===\n\n".join(s[:8000] for s in sources[:3])
        return self._chat(
            self.drafter_system,
            f"Today's date: {date}\n"
            f"Task: expand the section '{section}' of this wiki page using the "
            f"raw sources below. Preserve all existing content; only add depth. "
            f"Update last_updated and sources in the frontmatter.\n"
            f"Current page:\n---\n{page[:16000]}\n---\n"
            f"Raw sources:\n---\n{joined}\n---\n"
            f"Output the FULL updated page markdown.",
        )

    def draft_page(self, slug: str, item_text: str, source_path: str,
                   date: str) -> str:
        return self._chat(
            self.drafter_system,
            f"Today's date: {date}\n"
            f"Task: draft the wiki concept page [[{slug}]] from this source. "
            f"Set sources: [{source_path}] in the frontmatter.\n"
            f"Source material:\n---\n{item_text[:16000]}\n---\n"
            f"Output the FULL page markdown.",
        )

    def score_relevance(self, item: "WatchItem", topics: list[str],
                        weaknesses: list[str]) -> dict:
        raw = self._chat(
            self.relevance_system,
            f"Item:\nTitle: {item.title}\nSource: {item.source}\n"
            f"Summary: {item.summary[:2000]}\n\n"
            f"Wiki topics: {', '.join(topics[:200])}\n"
            f"Weakness list (weakest first): {', '.join(weaknesses) or '(none)'}\n"
            "Respond ONLY in JSON.",
            max_tokens=200,
        )
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {"score": 0, "topic_slug": "", "reason": "unparseable", "new_page": False}
        return {
            "score": max(0, min(10, int(data.get("score", 0)))),
            "topic_slug": str(data.get("topic_slug", "")),
            "reason": str(data.get("reason", "")),
            "new_page": bool(data.get("new_page", False)),
        }


# ---------------------------------------------------------------------------
# Step 1 — consume the queue
# ---------------------------------------------------------------------------

def load_queue(path: Path | None = None) -> dict:
    path = path or (STATE_DIR / "maintenance_queue.json")
    try:
        queue = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        queue = {"tasks": []}
    queue.setdefault("tasks", [])
    return queue


def save_queue(queue: dict, path: Path | None = None) -> None:
    path = path or (STATE_DIR / "maintenance_queue.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")


def find_raw_sources(concept: str, raw_dir: Path | None = None,
                     limit: int = 3) -> list[Path]:
    """raw/ files whose name or content mentions the concept."""
    raw_dir = raw_dir or (REPO_ROOT / "raw")
    if not raw_dir.is_dir():
        return []
    needle = concept.lower().replace("-", " ")
    name_hits, content_hits = [], []
    for path in sorted(raw_dir.rglob("*.md")):
        stem = path.stem.lower().replace("-", " ")
        if needle in stem or concept.lower() in path.stem.lower():
            name_hits.append(path)
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        if needle in text or concept.lower() in text:
            content_hits.append(path)
    return (name_hits + content_hits)[:limit]


def _find_concept_page(concept: str) -> Path | None:
    for sub in ("concepts", "system-design"):
        candidate = WIKI_DIR / sub / f"{concept}.md"
        if candidate.exists():
            return candidate
    return None


def consume_queue(queue: dict, maintainer, budget: PageBudget,
                  changelog: list[ChangelogEntry], date: str,
                  watchlist: dict, dry_run: bool = False) -> int:
    """Process pending tasks. Returns the number completed. Tasks needing the
    LLM stay pending when maintainer is None."""
    done = 0
    for task in queue["tasks"]:
        if task.get("status") != "pending":
            continue

        if task.get("type") == "generate_qa":
            concept = task.get("concept", "")
            page_path = _find_concept_page(concept)
            if page_path is None:
                task["status"] = "failed"
                task["note"] = f"no concept page for '{concept}'"
                continue
            if maintainer is None:
                task["note"] = "pending: LLM unavailable"
                continue
            qa_rel = f"wiki/qa/{concept}-qa.md"
            qa_path = REPO_ROOT / qa_rel
            existing = qa_path.read_text(encoding="utf-8") if qa_path.exists() else None
            try:
                content = maintainer.generate_qa(
                    concept, int(task.get("difficulty", 4)),
                    page_path.read_text(encoding="utf-8"), existing, date)
            except Exception as exc:  # noqa: BLE001 — keep the task for next run
                task["note"] = f"pending: {exc}"
                continue
            if content and write_wiki_page(
                    qa_rel, content, budget, changelog,
                    source=task.get("id", "queue"),
                    reason=task.get("reason", "generate_qa"), dry_run=dry_run):
                task["status"] = "done"
                task["completed"] = date
                done += 1

        elif task.get("type") == "wiki_gap":
            page_rel = task.get("page", "")
            page_path = REPO_ROOT / page_rel
            if not page_path.exists():
                task["status"] = "failed"
                task["note"] = f"page not found: {page_rel}"
                continue
            concept = page_path.stem
            sources = find_raw_sources(concept)
            if not sources:
                # No raw/ coverage → add to the watchlist (spec §3.3 step 1).
                wanted = watchlist.setdefault("wanted_topics", [])
                if concept not in wanted:
                    wanted.append(concept)
                task["status"] = "done"
                task["completed"] = date
                task["note"] = "no raw source — added to watchlist wanted_topics"
                done += 1
                continue
            if maintainer is None:
                task["note"] = "pending: LLM unavailable"
                continue
            try:
                content = maintainer.expand_section(
                    page_path.read_text(encoding="utf-8"),
                    task.get("section", ""),
                    [s.read_text(encoding="utf-8", errors="ignore") for s in sources],
                    date)
            except Exception as exc:  # noqa: BLE001
                task["note"] = f"pending: {exc}"
                continue
            src_rel = ", ".join(s.relative_to(REPO_ROOT).as_posix() for s in sources)
            if content and write_wiki_page(
                    page_rel, content, budget, changelog,
                    source=src_rel, reason=task.get("reason", "wiki_gap"),
                    dry_run=dry_run):
                task["status"] = "done"
                task["completed"] = date
                done += 1

        else:
            task["status"] = "failed"
            task["note"] = f"unknown task type: {task.get('type')}"
    return done


# ---------------------------------------------------------------------------
# Step 2 — monitor sources (watchlist → raw/auto/)
# ---------------------------------------------------------------------------

@dataclass
class WatchItem:
    id: str
    title: str
    url: str
    summary: str
    source: str            # e.g. "arxiv:cs.CL", "huggingface-blog"
    published: str = ""    # ISO date when known
    text: str = ""         # fuller body when available


def _et_text(elem, tag: str) -> str:
    child = elem.find(tag)
    return (child.text or "").strip() if child is not None else ""


def parse_arxiv_atom(xml_text: str, category: str) -> list[WatchItem]:
    items = []
    root = ET.fromstring(xml_text)
    for entry in root.findall(f"{ATOM_NS}entry"):
        raw_id = _et_text(entry, f"{ATOM_NS}id")
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1] or raw_id
        items.append(WatchItem(
            id=f"arxiv:{arxiv_id}",
            title=re.sub(r"\s+", " ", _et_text(entry, f"{ATOM_NS}title")),
            url=raw_id,
            summary=re.sub(r"\s+", " ", _et_text(entry, f"{ATOM_NS}summary")),
            source=f"arxiv:{category}",
            published=_et_text(entry, f"{ATOM_NS}published")[:10],
        ))
    return items


def parse_feed(xml_text: str, source_name: str) -> list[WatchItem]:
    """Parse RSS 2.0 <item> or Atom <entry> feeds."""
    items: list[WatchItem] = []
    root = ET.fromstring(xml_text)
    for node in root.iter("item"):  # RSS 2.0
        link = _et_text(node, "link")
        items.append(WatchItem(
            id=f"{source_name}:{_et_text(node, 'guid') or link}",
            title=_et_text(node, "title"),
            url=link,
            summary=re.sub(r"<[^>]+>", " ", _et_text(node, "description"))[:2000],
            source=source_name,
            published=_et_text(node, "pubDate"),
        ))
    if not items:
        for node in root.iter(f"{ATOM_NS}entry"):  # Atom
            link_el = node.find(f"{ATOM_NS}link")
            link = link_el.get("href", "") if link_el is not None else ""
            items.append(WatchItem(
                id=f"{source_name}:{_et_text(node, ATOM_NS + 'id') or link}",
                title=_et_text(node, f"{ATOM_NS}title"),
                url=link,
                summary=re.sub(r"<[^>]+>", " ",
                               _et_text(node, f"{ATOM_NS}summary")
                               or _et_text(node, f"{ATOM_NS}content"))[:2000],
                source=source_name,
                published=_et_text(node, f"{ATOM_NS}updated")[:10],
            ))
    return items


def fetch_watchlist(watchlist: dict, seen_ids: set[str],
                    verbose: bool = True) -> list[WatchItem]:
    """Fetch new items from all watchlist sources. Network errors are
    per-source and non-fatal (headless cron must survive a dead feed)."""
    import requests

    settings = watchlist.get("settings", {})
    per_source = int(settings.get("max_items_per_source",
                                  DEFAULT_MAX_ITEMS_PER_SOURCE))
    items: list[WatchItem] = []

    for entry in watchlist.get("arxiv", []):
        cat = entry.get("category")
        if not cat:
            continue
        try:
            resp = requests.get(ARXIV_API, params={
                "search_query": f"cat:{cat}",
                "sortBy": "submittedDate", "sortOrder": "descending",
                "max_results": per_source,
            }, timeout=30)
            resp.raise_for_status()
            items.extend(parse_arxiv_atom(resp.text, cat))
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  ⚠ arxiv {cat} fetch failed: {exc}")

    for feed in watchlist.get("feeds", []):
        url, name = feed.get("url"), feed.get("name", "feed")
        if not url:
            continue
        try:
            resp = requests.get(url, timeout=30,
                                headers={"User-Agent": "ai-engineer-wiki-maintain/1.0"})
            resp.raise_for_status()
            items.extend(parse_feed(resp.text, name)[:per_source])
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  ⚠ feed {name} fetch failed: {exc}")

    fresh = [i for i in items if i.id not in seen_ids]
    if verbose:
        print(f"  Fetched {len(items)} items, {len(fresh)} new.")
    return fresh


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")[:80]


def save_item_to_raw(item: WatchItem, date: str,
                     auto_dir: Path | None = None) -> Path:
    auto_dir = auto_dir or AUTO_DIR
    auto_dir.mkdir(parents=True, exist_ok=True)
    path = auto_dir / f"{date}-{_slugify(item.title) or 'item'}.md"
    counter = 2
    while path.exists():
        path = auto_dir / f"{date}-{_slugify(item.title)}-{counter}.md"
        counter += 1
    path.write_text(
        f"# {item.title}\n\n"
        f"**Source:** {item.source}\n"
        f"**URL:** {item.url}\n"
        f"**Published:** {item.published}\n"
        f"**Fetched:** {date} by maintain.py (watchlist)\n\n"
        f"---\n\n## Summary\n\n{item.summary}\n"
        + (f"\n---\n\n## Content\n\n{item.text}\n" if item.text else ""),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Step 3 — relevance filter
# ---------------------------------------------------------------------------

def index_topics() -> list[str]:
    """All [[slugs]] in wiki/index.md."""
    try:
        index = (WIKI_DIR / "index.md").read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    seen: list[str] = []
    for slug in re.findall(r"\[\[([^\]|]+)", index):
        slug = slug.strip().lower()
        if slug and slug not in seen:
            seen.append(slug)
    return seen


def weakness_list(ratings: dict | None = None, n: int = 10) -> list[str]:
    ratings = ratings if ratings is not None else load_skill_ratings()
    concepts = ratings.get("concepts", {})
    ranked = sorted(concepts, key=lambda c: concepts[c].get("rating", 1200))
    return [c for c in ranked if concepts[c].get("rating", 1200) <= 1300][:n]


def keyword_relevance(item: WatchItem, topics: list[str],
                      weaknesses: list[str],
                      wanted_topics: list[str] | None = None) -> dict:
    """Offline fallback scorer: topic-term overlap + weakness/wanted boost.
    Same output shape as LLMMaintainer.score_relevance."""
    text = f"{item.title} {item.summary}".lower()
    best_slug, hits = "", 0
    for slug in topics:
        phrase = slug.replace("-", " ")
        if len(phrase) < 3:
            continue
        if phrase in text:
            hits += 1
            if not best_slug or len(phrase) > len(best_slug.replace("-", " ")):
                best_slug = slug
    score = min(8, hits * 3)  # 1 topic hit → 3, 2 → 6, 3+ → 8
    boost_pool = set(weaknesses) | set(wanted_topics or [])
    boosted = best_slug in boost_pool or any(
        w.replace("-", " ") in text for w in boost_pool)
    if boosted:
        score = min(10, score + 2)
    return {"score": score, "topic_slug": best_slug,
            "reason": f"keyword overlap ({hits} topic hits"
                      + (", weakness boost)" if boosted else ")"),
            "new_page": False}


# ---------------------------------------------------------------------------
# Step 5 — PR plumbing (never push main)
# ---------------------------------------------------------------------------

def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=check,
                          capture_output=True, text=True)


def changelog_table(changelog: list[ChangelogEntry]) -> str:
    lines = ["| Page | Action | Source | Reason |",
             "|------|--------|--------|--------|"]
    for e in changelog:
        lines.append(f"| {e.page} | {e.action} | {e.source} | {e.reason} |")
    return "\n".join(lines)


def open_pr(date: str, changelog: list[ChangelogEntry],
            summary: str) -> str | None:
    """Branch maintain/<date>, commit, push, open PR via gh. Returns the PR
    URL, or None when there is nothing to commit or gh/push fails."""
    branch = f"maintain/{date}"
    status = _git("status", "--porcelain").stdout.strip()
    if not status:
        print("Nothing to commit — no PR opened.")
        return None

    current = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    try:
        _git("checkout", "-B", branch)
        _git("add", "wiki/", "state/", "raw/auto/", "agent/watchlist.yaml")
        _git("commit", "-m", f"maintain: weekly update {date}")
        _git("push", "-u", "origin", branch, "--force-with-lease")
        body = (f"Automated weekly MAINTAIN run ({date}).\n\n"
                f"{summary}\n\n## Changelog\n\n{changelog_table(changelog)}\n")
        result = subprocess.run(
            ["gh", "pr", "create", "--title", f"maintain: weekly update {date}",
             "--body", body, "--base", "main", "--head", branch],
            cwd=REPO_ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"gh pr create failed: {result.stderr.strip()}")
            return None
        url = result.stdout.strip().splitlines()[-1]
        print(f"PR opened: {url}")
        return url
    except subprocess.CalledProcessError as exc:
        print(f"git step failed: {exc.stderr if hasattr(exc, 'stderr') else exc}")
        return None
    finally:
        _git("checkout", current, check=False)


def run_validation() -> bool:
    script = REPO_ROOT / "scripts" / "validate_wiki.py"
    if not script.exists():
        return True
    result = subprocess.run(["python3", str(script)], cwd=REPO_ROOT,
                            capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip())
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class MaintainResult:
    tasks_done: int
    items_fetched: int
    items_ingested: int
    changelog: list[ChangelogEntry] = field(default_factory=list)
    pr_url: str | None = None
    validation_passed: bool = True


def run_maintain(
    dry_run: bool = False,
    no_fetch: bool = False,
    no_pr: bool = False,
    max_pages: int = MAX_PAGES_PER_RUN,
    maintainer=None,
    use_llm: bool | None = None,
    state_dir: Path | None = None,
    watchlist_path: Path | None = None,
    log_to_wiki: bool = True,
) -> MaintainResult:
    state_dir = state_dir or STATE_DIR
    date = datetime.now().strftime("%Y-%m-%d")
    budget = PageBudget(min(max_pages, MAX_PAGES_PER_RUN))
    changelog: list[ChangelogEntry] = []

    if maintainer is None:
        if use_llm is None:
            use_llm = bool(os.getenv("OPENAI_API_KEY"))
        if use_llm:
            try:
                maintainer = LLMMaintainer()
            except Exception as exc:  # noqa: BLE001
                print(f"(LLM unavailable — LLM-dependent tasks stay pending: {exc})")

    watchlist = load_watchlist(watchlist_path)
    settings = watchlist.get("settings", {})
    threshold = int(settings.get("relevance_threshold", DEFAULT_RELEVANCE_THRESHOLD))
    max_ingests = int(settings.get("max_auto_ingests", DEFAULT_MAX_AUTO_INGESTS))

    # 1. Consume the queue.
    print("== Step 1: maintenance queue ==")
    queue_path = state_dir / "maintenance_queue.json"
    queue = load_queue(queue_path)
    pending = sum(1 for t in queue["tasks"] if t.get("status") == "pending")
    tasks_done = consume_queue(queue, maintainer, budget, changelog, date,
                               watchlist, dry_run=dry_run)
    print(f"  {tasks_done}/{pending} pending task(s) completed.")
    if not dry_run:
        save_queue(queue, queue_path)

    # 2. Monitor sources.
    items_fetched = 0
    scored: list[tuple[dict, WatchItem, Path]] = []
    run_state = load_run_state(state_dir / "maintain_state.json")
    if not no_fetch:
        print("== Step 2: watchlist fetch ==")
        seen = set(run_state.get("seen_ids", []))
        fresh = fetch_watchlist(watchlist, seen)
        items_fetched = len(fresh)

        # 3. Relevance filter.
        print("== Step 3: relevance filter ==")
        topics = index_topics()
        weaknesses = weakness_list(
            load_skill_ratings(state_dir / "skill_ratings.json"))
        wanted = watchlist.get("wanted_topics", [])
        for item in fresh:
            if maintainer is not None:
                try:
                    verdict = maintainer.score_relevance(item, topics, weaknesses)
                except Exception:  # noqa: BLE001
                    verdict = keyword_relevance(item, topics, weaknesses, wanted)
            else:
                verdict = keyword_relevance(item, topics, weaknesses, wanted)
            run_state["seen_ids"].append(item.id)
            if verdict["score"] < threshold:
                continue
            path = (save_item_to_raw(item, date) if not dry_run
                    else AUTO_DIR / "(dry-run)")
            scored.append((verdict, item, path))
            print(f"  ✓ [{verdict['score']}/10] {item.title[:80]}")
        scored.sort(key=lambda t: -t[0]["score"])
        scored = scored[:max_ingests]

    # 4. Draft pages from relevant items.
    items_ingested = 0
    if scored and maintainer is not None:
        print("== Step 4: draft pages ==")
        for verdict, item, raw_path in scored:
            slug = _slugify(verdict.get("topic_slug") or item.title)
            if not slug:
                continue
            page_rel = f"wiki/concepts/{slug}.md"
            page_path = REPO_ROOT / page_rel
            raw_rel = (raw_path.relative_to(REPO_ROOT).as_posix()
                       if raw_path.is_absolute() and raw_path.exists()
                       else str(raw_path))
            source_text = f"# {item.title}\n\n{item.summary}\n\n{item.text}"
            try:
                if page_path.exists():
                    content = maintainer.expand_section(
                        page_path.read_text(encoding="utf-8"),
                        "Variants & Extensions", [source_text], date)
                else:
                    content = maintainer.draft_page(slug, source_text, raw_rel, date)
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠ draft failed for {slug}: {exc}")
                continue
            if content and write_wiki_page(
                    page_rel, content, budget, changelog,
                    source=raw_rel, reason=verdict.get("reason", "watchlist"),
                    dry_run=dry_run):
                items_ingested += 1
    elif scored:
        print(f"  ({len(scored)} relevant item(s) saved to raw/auto/ — "
              "no LLM available to draft pages; ingest them manually.)")

    if not dry_run:
        run_state["last_run"] = date
        save_run_state(run_state, state_dir / "maintain_state.json")
        save_watchlist(watchlist, watchlist_path)

    # Validate before any PR.
    print("== Step 5: validation ==")
    validation_passed = run_validation()

    summary = (f"MAINTAIN {date}: {tasks_done} queue task(s) done, "
               f"{items_fetched} item(s) fetched, {items_ingested} ingested, "
               f"{len(budget.touched)} page(s) touched"
               f"{' [DRY RUN]' if dry_run else ''}.")
    print(summary)

    pr_url = None
    if not dry_run and not no_pr and changelog:
        if validation_passed:
            pr_url = open_pr(date, changelog, summary)
        else:
            print("Validation failed — PR not opened (fix and rerun).")

    if not dry_run and log_to_wiki:
        from wiki_tool import append_wiki_log

        append_wiki_log("MAINTAIN: " + summary
                        + (f" PR: {pr_url}" if pr_url else ""))

    return MaintainResult(
        tasks_done=tasks_done,
        items_fetched=items_fetched,
        items_ingested=items_ingested,
        changelog=changelog,
        pr_url=pr_url,
        validation_passed=validation_passed,
    )


def parse_maintain_request(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(
        r"\brun maintenance\b|\bprocess the maintenance queue\b"
        r"|\bweekly update\b|\bmaintain the wiki\b", lowered))
