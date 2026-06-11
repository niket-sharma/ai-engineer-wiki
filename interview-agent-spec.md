# Spec: Adaptive Interview Agent + Self-Maintaining Wiki

**Project:** `ai-engineer-wiki` — Phase 2 evolution
**Repo:** github.com/niket-sharma/ai-engineer-wiki
**Author:** Niket Sharma
**Status:** Draft v1.0 — ready for Claude Code execution
**Target environment:** WSL2 Ubuntu, Python 3.11+, existing `agent/` venv

---

## 1. Vision

Close the learning loop. Today the wiki compounds knowledge passively (INGEST → QUERY).
This spec adds an **adaptive mock-interview layer** that assesses Niket against the wiki,
and a **self-maintenance layer** that uses interview weakness signals + weekly source
monitoring to decide what the wiki ingests and generates next.

```
            ┌─────────────────────────────────────────────┐
            │                  WIKI (knowledge)            │
            │  concepts/  qa/  companies/  cheatsheets/    │
            └──────┬──────────────────────────▲────────────┘
                   │ questions + rubrics      │ new Q&A, page updates, PRs
                   ▼                          │
            ┌──────────────┐   transcript   ┌─┴───────────────┐
            │  INTERVIEWER │ ─────────────▶ │  ASSESS + REPORT │
            │  (adaptive)  │                │  (grader)        │
            └──────────────┘                └─┬───────────────┘
                   ▲                          │ weakness report (JSON + md)
                   │ Elo difficulty state     ▼
            ┌──────┴──────────────────────────────────────┐
            │        MAINTAINER (autonomous, weekly)       │
            │  arXiv/release monitor → draft pages → PRs   │
            │  weakness-driven GENERATE prioritization     │
            └──────────────────────────────────────────────┘
```

**Core principle (unchanged):** Knowledge compounds, never re-derive from scratch.
Interview transcripts and wrong answers become raw source material (`raw/interviews/`)
that flows through the existing INGEST pipeline.

---

## 2. Current State (baseline — do not break)

Existing agent operations defined in `skill.md`:

| Op | Behavior |
|---|---|
| `INGEST` | read `raw/...`, update/create wiki pages, update index, append log |
| `QUERY` | answer from wiki pages with page citations (full-page router, no chunked RAG) |
| `AUDIT` | contradictions, orphans, stubs, missing pages, stale high-priority pages |
| `GENERATE` | create/update topic Q&A pages |
| `COMPANY` | create/update company prep pages |
| `CHEATSHEET` | create/update quick-reference summaries |

Existing files: `agent/agent.py`, `agent/wiki_tool.py`, `agent/cli.py`, `agent/app.py`
(Streamlit), `agent/fetch_sources.py`. Wiki layout: `wiki/{concepts,companies,system-design,cheatsheets,qa}/`, `wiki/index.md`, `wiki/log.md`.

**Constraint:** All new operations must reuse the full-page router pattern — load whole
pages as context, never chunk. All new state lives in version-controlled files (no DB).

---

## 3. New Capabilities

### 3.1 `INTERVIEW` operation

Conduct an adaptive mock technical interview against wiki content.

**Invocation (CLI/Streamlit/skill):**
```
Interview me on transformers, 30 minutes, company=capital-one
Interview me on my weakest topics
Run a system design interview, hard mode
```

**Behavior:**
1. **Session setup.** Resolve scope:
   - `topic=X` → load `wiki/concepts/` pages tagged/linked to X plus `wiki/qa/` pages for X.
   - `company=Y` → additionally load `wiki/companies/Y.md` and bias question style/topics
     to that page's interview-format notes.
   - `weakest` → read `state/skill_ratings.json`, pick the N lowest-rated concepts.
2. **Question selection.** Pull from `wiki/qa/` first; if coverage is thin, generate
   novel questions grounded in the loaded concept pages (and tag them `generated:true`
   so ASSESS can later persist good ones back into `wiki/qa/`).
3. **Adaptive difficulty.** Maintain a per-concept Elo-style rating (see §4.1).
   - Correct + confident answer → escalate (e.g., KV cache basics → GQA/MQA tradeoffs →
     paged attention / continuous batching).
   - Wrong/partial → de-escalate one level and probe the prerequisite concept.
   - Question difficulty levels: 1 (recall) … 5 (open-ended design under constraints).
4. **Interview styles** (flag `style=`):
   - `drill` — rapid-fire Q&A, short answers.
   - `deep` — one question, multiple follow-up probes ("why", "what breaks if...").
   - `system-design` — single scenario, 20–40 min, whiteboard-style with checkpoints.
   - `behavioral` — STAR-format, sourced from `wiki/companies/` behavioral notes.
5. **Session output.** Write a full transcript to
   `raw/interviews/YYYY-MM-DD-<topic>-<style>.md` with frontmatter (see §4.2).
   Transcripts are INGEST-compatible by design.

**Anti-leak rule:** During the session, never reveal rubric content or wiki citations
until ASSESS. The interviewer asks and probes; it does not teach mid-session
(unless `mode=tutor` is explicitly set).

### 3.2 `ASSESS` operation

Grade a completed interview session against the wiki.

**Behavior:**
1. Load the transcript + every concept page referenced by the session's questions.
   The **concept page is the rubric** (full-page router pattern).
2. For each Q/A pair, produce:
   - `score` 0–4 (0 = no answer, 4 = senior-level complete with tradeoffs)
   - `gaps`: bullet list of missing key points, each citing the wiki page + heading
   - `misconceptions`: anything stated that contradicts the wiki (these are gold —
     flag them prominently)
   - `wiki_gap`: boolean — set true when Niket's answer was reasonable but the wiki
     page lacked the depth to grade it properly (**this is the self-maintenance hook**)
3. Update `state/skill_ratings.json` (Elo update per concept, see §4.1).
4. Write `wiki/reports/YYYY-MM-DD-<topic>.md` — human-readable weakness report:
   summary scores, top 3 weaknesses, top 3 strengths, misconception list,
   and a **7-day micro study plan** with links to existing wiki pages.
5. Append a machine-readable block to `state/assessment_log.jsonl`.
6. **Trigger follow-ups automatically:**
   - For each concept scoring ≤ 2 → queue a `GENERATE` task (harder Q&A for that topic).
   - For each `wiki_gap=true` → queue an `AUDIT`-style stub flag on that concept page.
   - Queue file: `state/maintenance_queue.json` (consumed by the Maintainer, §3.3).

### 3.3 `MAINTAIN` operation (autonomous weekly updater)

Runs headless via GitHub Actions (cron) or manually via CLI.

**Behavior:**
1. **Consume the queue.** Process `state/maintenance_queue.json`:
   - `generate_qa` tasks → run GENERATE for the topic at the requested difficulty.
   - `wiki_gap` tasks → expand the flagged concept page section using `raw/` sources;
     if no source exists, add to the watchlist below.
2. **Monitor sources.** Extend `fetch_sources.py` with a watchlist
   (`agent/watchlist.yaml`): arXiv categories (cs.CL, cs.LG), Anthropic/OpenAI/Meta
   release blogs, HF blog. Fetch new items since last run into `raw/auto/`.
3. **Relevance filter.** For each new item, score relevance against `wiki/index.md`
   topics + current weakness ratings. Ingest only items above threshold; weakness-related
   topics get priority boost.
4. **Draft in Niket's style.** INGEST relevant items → draft/update concept pages.
   Style guide: mirror existing pages (bottom-up explanations, shape-annotated tensor
   walkthroughs before intuition, explicit prerequisites section, Q&A at bottom).
5. **Open a PR, never push to main.** Branch `maintain/YYYY-MM-DD`, one PR per run,
   PR body = changelog table (page, action, source, reason) + diff stats.
   Niket reviews and merges. New/changed pages automatically become interviewable.
6. Append run summary to `wiki/log.md`.

**Safety rails:** never delete pages; never modify `state/skill_ratings.json` outside
ASSESS; cap pages-touched per run at 12; PRs must pass `scripts/validate_wiki.py` (§5.4).

### 3.4 Streamlit interview UI (extend `agent/app.py`)

- New "Interview" tab: topic/company/style/duration pickers, live chat session,
  visible timer, per-question "submit answer" flow.
- Post-session: render the ASSESS report inline + radar chart of concept ratings
  (matplotlib/plotly from `state/skill_ratings.json`).
- "History" tab: rating trends over time per concept (line chart from
  `state/assessment_log.jsonl`).
- **Voice (Phase 4, optional):** browser `webkitSpeechRecognition` for answer dictation
  + TTS for question reading via `streamlit-webrtc` or a simple JS component.
  Keep text as the canonical record either way.

---

## 4. Data Schemas

### 4.1 `state/skill_ratings.json`
```json
{
  "version": 1,
  "updated": "2026-06-10T14:00:00Z",
  "concepts": {
    "kv-cache": {
      "rating": 1340,
      "sessions": 4,
      "last_assessed": "2026-06-08",
      "trend": [1200, 1265, 1310, 1340],
      "wiki_page": "wiki/concepts/kv-cache.md"
    }
  }
}
```
- Elo: start 1200. Question difficulty maps to opponent rating
  (level 1 = 1000 … level 5 = 1800). K-factor 32 for first 5 sessions, then 16.
- Score 0–4 maps to Elo outcome: 0–1 → loss, 2 → draw, 3–4 → win.
- Difficulty selection rule: pick questions whose level rating is within ±150 of the
  concept's current rating (keeps sessions in the productive struggle zone).

### 4.2 Interview transcript frontmatter (`raw/interviews/*.md`)
```yaml
---
type: interview-transcript
date: 2026-06-10
topic: transformers
style: deep
company: capital-one        # optional
duration_min: 30
questions: 7
concepts_touched: [self-attention, kv-cache, rope]
assessed: false             # ASSESS flips this to true
---
```

### 4.3 `state/maintenance_queue.json`
```json
{
  "tasks": [
    {"id": "q-001", "type": "generate_qa", "concept": "gqa",
     "difficulty": 4, "reason": "scored 1/4 on 2026-06-10", "status": "pending"},
    {"id": "q-002", "type": "wiki_gap", "page": "wiki/concepts/rope.md",
     "section": "extrapolation behavior", "reason": "rubric too thin to grade",
     "status": "pending"}
  ]
}
```

### 4.4 `state/assessment_log.jsonl` — one JSON object per assessed session
```json
{"date": "2026-06-10", "topic": "transformers", "style": "deep",
 "overall": 2.7, "per_concept": {"kv-cache": 3, "rope": 1},
 "misconceptions": ["claimed RoPE is applied to V"],
 "report": "wiki/reports/2026-06-10-transformers.md"}
```

---

## 5. Implementation Plan (phased, with Claude Code execution strategy)

> **Claude Code usage notes (June 2026 features):**
> - **Ultracode / Dynamic Workflows** (`/effort ultracode` or the `workflow` keyword,
>   Opus 4.8 / Fable 5): best for Phase 0 (codebase mapping) and Phase 3 (the
>   multi-file Maintainer + CI work). It fans out parallel subagents with adversarial
>   verification — ideal when the split strategy isn't known upfront. Note: workflow
>   subagents run in acceptEdits mode, so review the git diff after.
> - **Ralph Loop / `/loop`**: best for Phase 1–2 tasks with crisp completion signals —
>   "loop until `pytest tests/` is green and `scripts/validate_wiki.py` passes."
>   Define max iterations (suggest 8) and the success command explicitly.
> - **Plain single-agent sessions**: best for prompt-craft work (interviewer persona,
>   grading rubric prompts) where you want to iterate by feel, not by test signal.
> - Avoid parallel fan-out on tasks that mutate the same file (e.g., `agent.py`) —
>   sequence those, parallelize only across independent files.

### Phase 0 — Recon & scaffolding (½ day) — *ultracode, read-mostly*
- [ ] Map current `agent/agent.py` operation dispatch; document extension points in
      `plan.md`.
- [ ] Create `state/`, `wiki/reports/`, `raw/interviews/`, `raw/auto/` dirs with
      `.gitkeep`; add `state/*.json` schemas as empty seeds.
- [ ] Add `tests/` with pytest scaffolding; CI workflow `.github/workflows/test.yml`.
- [ ] Update `skill.md` with the three new operation specs (copy §3 of this doc).

### Phase 1 — INTERVIEW (2–3 days) — */loop against pytest*
- [ ] `agent/interview.py`: session manager (scope resolution, question selection,
      Elo-based difficulty picker, transcript writer).
- [ ] Interviewer system prompt: persona = senior interviewer at target company,
      probing follow-ups, no teaching, no rubric leakage.
- [ ] Wire into `cli.py` (`interview` command) and operation dispatch in `agent.py`.
- [ ] Tests: scope resolution, difficulty picker stays within ±150 band, transcript
      frontmatter validity, question dedup within a session.
- **Done when:** a full 5-question drill session on "kv-cache" runs end-to-end in CLI
  and produces a valid transcript file.

### Phase 2 — ASSESS + feedback bridge (2 days) — */loop against pytest*
- [ ] `agent/assess.py`: rubric loading (full pages), per-answer grading prompt,
      Elo updates, report writer, queue writer.
- [ ] Grading prompt must require: cite the wiki heading for every gap; output strict
      JSON (use the "respond only in JSON" pattern + safe parse with fence stripping).
- [ ] Tests: Elo math (golden cases), JSONL append integrity, queue task generation
      for score ≤ 2 and wiki_gap=true, report renders with study plan section.
- **Done when:** assessing a fixture transcript produces a report, updated ratings,
  and ≥1 queued maintenance task.

### Phase 3 — MAINTAIN + automation (3–4 days) — *ultracode workflow*
- [ ] `agent/maintain.py`: queue consumer, watchlist fetcher (extend
      `fetch_sources.py`), relevance scorer, page drafter, PR opener (use `gh` CLI).
- [ ] `agent/watchlist.yaml` with initial sources (arXiv cs.CL/cs.LG RSS, Anthropic
      news, HF blog).
- [ ] `.github/workflows/maintain.yml`: weekly cron (Sun 06:00 ET), runs MAINTAIN
      headless with `ANTHROPIC_API_KEY` secret; opens PR. Mirror the GitHub Actions
      pattern from `trading-agent-compass`.
- [ ] `scripts/validate_wiki.py`: link integrity, frontmatter schema, index coverage,
      no-deletion check. Run in CI on every PR.
- **Done when:** a dry-run MAINTAIN on a seeded queue opens a draft PR touching ≤ 12
  pages, and validation passes.

### Phase 4 — UI + polish (2 days, optional/parallel)
- [ ] Streamlit Interview tab + report rendering + radar/trend charts.
- [ ] Company mode preset buttons (Capital One, MassMutual, Fidelity, Exxon) pulling
      from existing `wiki/companies/` pages.
- [ ] Voice input/output (browser speech APIs) — nice-to-have, text stays canonical.
- [ ] README rewrite: new architecture diagram, demo GIF, "run your first interview
      in 3 commands" quickstart.

---

## 6. Prompts to Build (treat as first-class artifacts)

Store under `agent/prompts/` as versioned .md files:

1. `interviewer_system.md` — persona, style modes, anti-leak rules, escalation logic.
2. `grader_system.md` — rubric-from-page instructions, strict JSON output schema,
   misconception detection, wiki_gap criteria.
3. `page_drafter_system.md` — Niket's style guide (bottom-up, shape-annotated,
   prerequisites first), frontmatter requirements, interlinking rules.
4. `relevance_filter.md` — scoring instructions for watchlist items vs. index topics
   + weakness boost.

---

## 7. Acceptance Criteria (project-level)

1. `python cli.py` → "interview me on transformers" runs a complete adaptive session.
2. ASSESS produces a report with cited gaps and updates ratings; two consecutive
   strong sessions on a concept measurably raise its question difficulty.
3. A weakness automatically results in new harder Q&A appearing in `wiki/qa/`
   within one MAINTAIN cycle.
4. Weekly Action opens a PR with drafted pages; `validate_wiki.py` gates the merge.
5. All state is plain files in git; deleting `state/` and re-running degrades
   gracefully to defaults (rating 1200, empty queue).
6. Existing operations (INGEST/QUERY/AUDIT/GENERATE/COMPANY/CHEATSHEET) unchanged
   and passing their current behavior.

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Grader inflates scores (sycophancy) | Grading prompt requires citing a wiki heading per point awarded; spot-check with a "harsh mode" second pass on 1 in 5 sessions |
| Interview questions leak the answer | Anti-leak rule in prompt + test fixture checking no rubric text appears in questions |
| Maintainer PR slop | 12-page cap, validation script, PR-only (never push main), human merge |
| Elo noise from few sessions | K=32 early then 16; show confidence (session count) in UI |
| API cost of weekly runs | Relevance filter before INGEST; cap auto-ingests at 5 items/run; prompt caching on concept pages |
| Parallel agents clobbering `agent.py` | Sequence edits to shared files; reserve fan-out for independent modules/tests |

---

## 9. Suggested First Claude Code Prompt

```
Read INTERVIEW_AGENT_SPEC.md and plan.md. Execute Phase 0 exactly as specified:
map agent.py's dispatch, create the new directories and state seeds, scaffold
pytest + CI, and update skill.md with the INTERVIEW, ASSESS, and MAINTAIN
operation specs from §3. Do not modify existing operations. When done, run
the test suite and show me the diff summary.
```

Then Phase 1 with: `/loop` — success = `pytest tests/ -q` green + a real CLI drill
session producing a valid transcript; max 8 iterations.