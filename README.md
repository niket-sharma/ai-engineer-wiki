# AI Engineer Wiki

A living, compounding knowledge base for AI engineering — transformers, alignment, RAG, agents, inference/serving, evaluation, system design, and classic ML — with an **adaptive mock-interview layer** that assesses you against it and a **self-maintenance layer** that decides what the wiki learns next.

Compiled from primary sources into interlinked concept pages (the LLM Wiki pattern: compile once, query fast). Knowledge compounds; never re-derive from scratch.

## Architecture

```
            ┌─────────────────────────────────────────────┐
            │                  WIKI (knowledge)            │
            │   concepts/  qa/  system-design/  reports/   │
            └──────┬──────────────────────────▲────────────┘
                   │ questions + rubrics      │ new Q&A, page updates, PRs
                   ▼                          │
            ┌──────────────┐   transcript   ┌─┴───────────────┐
            │  INTERVIEWER │ ─────────────▶ │  ASSESS + REPORT │
            │  (adaptive)  │                │  (grader)        │
            └──────────────┘                └─┬───────────────┘
                   ▲                          │ weakness report + queue
                   │ Elo difficulty state     ▼
            ┌──────┴──────────────────────────────────────┐
            │        MAINTAINER (autonomous, weekly)       │
            │  arXiv/release monitor → draft pages → PRs   │
            │  weakness-driven GENERATE prioritization     │
            └──────────────────────────────────────────────┘
```

The loop: interviews produce transcripts (`raw/interviews/`) → ASSESS grades them against wiki pages (the page **is** the rubric), updates per-concept Elo ratings, and queues maintenance tasks → MAINTAIN consumes the queue weekly, monitors arXiv/blog feeds, drafts pages, and opens a PR for human review. Weak concepts automatically get harder questions and richer pages.

## Run your first interview in 3 commands

```bash
cd agent && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
export OPENAI_API_KEY="your_key_here"
python cli.py interview --topic kv-cache --questions 5
```

Then grade it:

```bash
python cli.py assess
```

You get a scored report in `wiki/reports/`, updated Elo ratings in `state/skill_ratings.json`, and queued follow-up work in `state/maintenance_queue.json`.

## Operations

| Op | Trigger | What happens |
|---|---|---|
| **INGEST** | `Ingest raw/...` | Read raw source, update/create wiki pages, update index, append log |
| **QUERY** | `What does the wiki say about...` | Answer from compiled wiki only, cite `[[page-slug]]` |
| **AUDIT** | `Run a full wiki audit` | Contradictions, orphans, stubs, missing pages, stale pages |
| **GENERATE** | `Generate questions on...` | L1/L2/L3 Q&A pages in `wiki/qa/` |
| **CHEATSHEET** | `Make a cheatsheet for...` | Quick-reference pages in `wiki/cheatsheets/` |
| **INTERVIEW** | `python cli.py interview` or "interview me on X" | Adaptive mock interview; Elo-banded difficulty, transcript to `raw/interviews/` |
| **ASSESS** | `python cli.py assess` or "assess my interview" | Grade vs wiki rubrics, Elo updates, weakness report, queue follow-ups |
| **MAINTAIN** | `python cli.py maintain` or weekly GitHub Action | Consume queue, fetch watchlist, draft pages, open a PR (never pushes main) |

## Adaptive difficulty (Elo)

Each concept carries an Elo rating (start 1200). Question levels 1–5 map to opponent ratings 1000–1800; questions are picked within ±150 of your rating (the productive-struggle zone). Scores 0–1/2/3–4 map to loss/draw/win; K=32 for a concept's first 5 sessions, then 16. Two strong sessions on a concept measurably raise its question difficulty. Only ASSESS writes `state/skill_ratings.json`.

## Self-maintenance

`python cli.py maintain` (or the weekly `maintain.yml` Action) runs headless:

1. Consumes `state/maintenance_queue.json` — harder Q&A for weak concepts, page expansion for `wiki_gap` flags.
2. Polls `agent/watchlist.yaml` (arXiv cs.CL/cs.LG, Anthropic/OpenAI/Meta/HF blogs) into `raw/auto/`.
3. Relevance-filters against wiki topics, boosting your current weaknesses.
4. Drafts pages in the wiki's house style, then opens a PR on branch `maintain/YYYY-MM-DD`.

Safety rails: never deletes pages, never touches skill ratings, ≤12 pages per run, and `scripts/validate_wiki.py` gates every PR in CI.

## Repo layout

```text
wiki/                 compiled knowledge (the agent writes only here)
  concepts/  system-design/  qa/  cheatsheets/  reports/  index.md  log.md
raw/                  immutable source material (+ interviews/, auto/)
state/                Elo ratings, maintenance queue, assessment log (git-versioned)
agent/                agent.py · wiki_tool.py · interview.py · assess.py ·
                      maintain.py · cli.py · app.py · prompts/ · watchlist.yaml
scripts/              validate_wiki.py (CI gate) · wikilinks_to_md.py (MkDocs)
tests/                offline test suite (LLM calls faked)
```

## Setup

```bash
cd agent
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
export OPENAI_API_KEY="your_key_here"
export OPENAI_MODEL="gpt-4o"      # default
export AGENT_MAX_TOKENS="4096"    # default
```

## Run

```bash
make chat                    # terminal REPL (chat + "interview me on <topic>")
make ui                      # Streamlit: Chat / Interview / History tabs
make interview TOPIC=kv-cache  # run interview on a topic (TOPIC= required)
make assess                  # grade the latest unassessed transcript
make maintain                # weekly maintainer, local (no PR)
make test                    # offline test suite
make validate                # wiki integrity gate
cd agent && python fetch_sources.py --list   # starter sources
```

## Testing & running the full stack

### 1. Unit tests (no API key needed)

```bash
make test
# or: cd agent && source .venv/bin/activate && pytest tests/ -q
```

All tests run offline (LLM calls are faked). Should complete in under 1 second.

### 2. Terminal chat REPL

```bash
cd agent && source .venv/bin/activate
python cli.py
```

| Goal | Prompt |
|---|---|
| QUERY | `what does the wiki say about kv-cache?` |
| INGEST | `ingest raw/transformers/attention-is-all-you-need.md` |
| GENERATE | `generate questions on kv-cache` |
| CHEATSHEET | `make a cheatsheet for transformers` |
| AUDIT | `audit the wiki` |
| INTERVIEW | `interview me on kv-cache` |
| ASSESS | `assess my interview` |
| MAINTAIN | `run maintenance` |

### 3. Interview subcommand

```bash
python cli.py interview --topic kv-cache --style drill --questions 3
python cli.py interview --topic transformers --style deep
python cli.py interview --weakest               # picks your lowest-rated concepts
python cli.py interview --topic kv-cache --no-llm  # offline: question-bank only, no API calls
```

Styles: `drill` (rapid-fire), `deep` (one question + follow-ups), `system-design`, `behavioral`.

All flags:

| Flag | Default | Description |
|---|---|---|
| `--topic <slug>` | — | Wiki concept slug to interview on (e.g. `kv-cache`, `transformers`) |
| `--style` | `drill` | Session style: `drill`, `deep`, `system-design`, `behavioral` |
| `--questions N` | 5 | Max questions per session |
| `--duration N` | none | Time-cap in minutes (ends session when elapsed) |
| `--level N` | auto | Override starting difficulty 1–5 (bypasses Elo; 1=recall, 5=open design) |
| `--weakest` | false | Interview on your 3 lowest-rated concepts instead of a specific topic |
| `--tutor` | false | Allow corrective nudges mid-session (less strict) |
| `--company <slug>` | none | Bias questions toward a company's interview style |
| `--no-llm` | false | Question-bank only; skips all API calls (useful for offline testing) |

> **No API key?** Pass `--no-llm` and the session falls back to `wiki/qa/` question bank at a fixed difficulty. All transcript and assessment logic still works; only novel question generation and adaptive follow-ups are skipped.

### 4. Assess subcommand

```bash
python cli.py assess                       # auto-picks latest unassessed transcript
python cli.py assess --transcript raw/interviews/2026-06-11-kv-cache-drill.md
```

Writes a scored report to `wiki/reports/`, updates Elo ratings in `state/skill_ratings.json`, and queues follow-up tasks in `state/maintenance_queue.json`.

### 5. Maintain subcommand

```bash
python cli.py maintain --dry-run --no-fetch   # safe: process queue only, no network, no PR
python cli.py maintain --dry-run              # fetch watchlist but don't open a PR
python cli.py maintain --no-llm              # offline: queue tasks only, skip LLM page drafts
python cli.py maintain                        # full run: fetch + draft + open PR (needs `gh` CLI)
```

All flags:

| Flag | Description |
|---|---|
| `--dry-run` | Report what would change without writing anything |
| `--no-fetch` | Skip the arXiv/blog watchlist fetch; process queue tasks only |
| `--no-pr` | Apply changes to working tree without creating a branch or PR |
| `--max-pages N` | Cap on wiki pages touched per run (hard max 12) |
| `--no-llm` | Offline mode — LLM-dependent tasks stay pending, no API calls |

### 6. Streamlit UI

```bash
cd agent && source .venv/bin/activate
streamlit run app.py
# opens http://localhost:8501
```

Three tabs:
- **Chat** — same operations as the REPL
- **Interview** — topic/style/company presets, live adaptive session, "Assess now" button
- **History** — Elo radar chart + per-concept trend lines from `state/assessment_log.jsonl`

### End-to-end smoke test

```bash
python cli.py interview --topic kv-cache --style drill --questions 3
python cli.py assess
cat state/maintenance_queue.json              # verify tasks were queued
python cli.py maintain --dry-run --no-fetch  # verify queue is consumed
```

## Topics covered

Transformers · RLHF/DPO/GRPO · RAG & Retrieval · LLM Agents · Inference & Serving · Evaluation · Production AI · System Design · Statistics · Algorithms
