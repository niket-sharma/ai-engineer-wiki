# AI Engineer Wiki — Build Plan

> A step-by-step implementation guide for building the wiki from scratch using
> Claude Code. Follow phases in order. Each phase is self-contained and
> produces a working artifact before moving to the next.

---

## Project Overview

| Item | Detail |
|---|---|
| Project name | ai-engineer-wiki |
| Primary tool | Claude Code (claude CLI) |
| Storage format | Plain Markdown (.md) |
| Viewer (optional) | Obsidian |
| Version control | Git |
| Estimated setup time | 30–45 minutes |
| Ongoing maintenance | 15–20 min per source ingested |

---

## Prerequisites

Before starting, ensure you have:

- [ ] Claude Code installed (`npm install -g @anthropic-ai/claude-code`)
- [ ] Git installed
- [ ] A directory where you want the repo to live (e.g. `~/projects/`)
- [ ] Obsidian installed (optional but recommended for graph view)
- [ ] A folder of study materials to seed with (PDFs, notes, transcripts, URLs)

---

## Phase 0 — Repo Initialization (Day 1, ~15 min)

### 0.1 Create the repository

```bash
cd ~/projects
mkdir ai-engineer-wiki
cd ai-engineer-wiki
git init
```

### 0.2 Place SKILL.md

Copy the `SKILL.md` file into the repo root. This is the instruction file
Claude Code reads to understand all operations.

```bash
# SKILL.md should already be in the root
ls SKILL.md   # confirm it's there
```

### 0.3 Create .gitignore

```bash
cat > .gitignore << 'EOF'
.obsidian/
.DS_Store
*.swp
*.tmp
__pycache__/
.env
EOF
```

### 0.4 Bootstrap with Claude Code

Open Claude Code in the repo root and run the bootstrap command:

```bash
claude
```

Then say:

> "Read SKILL.md thoroughly. Then bootstrap the wiki: create the full directory
> structure, wiki/index.md, wiki/log.md, and stub pages for every concept in
> the Topic Coverage Map. Do not hallucinate content — stubs are fine."

Claude Code will:
- Create all `raw/` subdirectories
- Create all `wiki/` subdirectories
- Create `wiki/index.md` from the index template
- Create `wiki/log.md`
- Create ~20 stub concept pages with correct frontmatter

### 0.5 Initial commit

```bash
git add .
git commit -m "chore: bootstrap wiki structure and stub pages"
```

**Exit criteria for Phase 0:**
- [ ] Directory tree matches the layout in SKILL.md Section 2
- [ ] `wiki/index.md` exists with all sections
- [ ] `wiki/log.md` exists
- [ ] At least 15 stub pages exist in `wiki/concepts/`
- [ ] Git history has first commit

---

## Phase 1 — Seed the Wiki (Week 1, ~3–4 hours total)

Ingest your highest-priority sources first. The goal is to have the core
transformer + RL + RAG concepts fully compiled as a solid foundation.

### 1.1 Priority Source List

Ingest in this order (highest ROI first):

**Batch A — Transformers (Day 1–2)**

| Source | Where to get it | raw/ path |
|---|---|---|
| "Attention Is All You Need" | arxiv.org/abs/1706.03762 | `raw/transformers/attention-is-all-you-need.md` |
| Karpathy's GPT lecture notes / transcript | youtube / your own notes | `raw/transformers/karpathy-gpt-lecture-notes.md` |
| Karpathy's nanoGPT repo README | github.com/karpathy/nanoGPT | `raw/transformers/nanogpt-readme.md` |
| Flash Attention paper summary | arxiv.org/abs/2205.14135 | `raw/transformers/flash-attention.md` |
| RoPE / ALiBi positional encoding notes | your study notes | `raw/transformers/positional-encoding-notes.md` |

**Batch B — Fine-tuning & Alignment (Day 3–4)**

| Source | Where to get it | raw/ path |
|---|---|---|
| LoRA paper summary | arxiv.org/abs/2106.09685 | `raw/rl-and-rlhf/lora-paper.md` |
| RLHF overview | InstructGPT paper / your notes | `raw/rl-and-rlhf/rlhf-overview.md` |
| DPO paper summary | arxiv.org/abs/2305.18290 | `raw/rl-and-rlhf/dpo-paper.md` |
| GRPO / DeepSeek-R1 notes | your notes | `raw/rl-and-rlhf/grpo-deepseek-notes.md` |
| PPO algorithm notes | your study notes | `raw/rl-and-rlhf/ppo-notes.md` |

**Batch C — RAG & Agents (Day 5–7)**

| Source | Where to get it | raw/ path |
|---|---|---|
| LangGraph documentation summary | langchain docs | `raw/mlops/langgraph-overview.md` |
| RAG survey / notes | your study notes | `raw/mlops/rag-systems-notes.md` |
| MCP protocol notes | your notes | `raw/mlops/mcp-protocol-notes.md` |
| Vector DB comparison notes | your notes | `raw/mlops/vector-db-comparison.md` |
| Reranking / ColBERT notes | your notes | `raw/mlops/reranking-notes.md` |

### 1.2 How to ingest each source

For each source in the list above:

1. Prepare the source as a Markdown file in the correct `raw/` subdirectory.
   If it's a paper, write a summary. If it's your notes, paste them as-is.
   If it's a URL, paste the key content.

2. In Claude Code, say:

   > "Ingest `raw/transformers/attention-is-all-you-need.md` into the wiki.
   > Follow the INGEST operation from SKILL.md exactly."

3. Review the pages Claude Code creates or updates. Check:
   - Frontmatter is complete
   - "Practical Applications" section is substantive
   - Wiki-links `[[like-this]]` are used
   - Log entry was added to `wiki/log.md`

4. Commit:
   ```bash
   git add .
   git commit -m "feat(ingest): attention-is-all-you-need → wiki"
   ```

### 1.3 After each batch — run a mini audit

After completing each batch (A, B, C), say:

> "Run a mini audit: list any contradictions flagged, any stubs that got
> promoted, and what concepts were referenced but not yet paged."

Use the gap list to decide what to ingest next.

**Exit criteria for Phase 1:**
- [ ] All 15 sources from batches A/B/C ingested
- [ ] Core concept pages are `status: current` (not stubs)
- [ ] `wiki/log.md` has an entry for every ingest
- [ ] Git has a commit per source or per batch
- [ ] Zero unresolved contradictions from batch ingests

---

## Phase 2 — Q&A Generation (Week 1–2, ~2 hours)

Once core concept pages exist, generate the Q&A layer.

### 2.1 Generate Q&A for each domain

Run each of the following in Claude Code:

```
"Generate Q&A for transformers. Cover all concept pages in
wiki/concepts/ tagged [transformers]. Create L1/L2/L3 questions.
Save to wiki/qa/transformers-qa.md"

"Generate Q&A for RL and alignment (RLHF, PPO, DPO, GRPO, LoRA).
Save to wiki/qa/rl-qa.md"

"Generate Q&A for RAG systems, vector databases, and reranking.
Save to wiki/qa/rag-qa.md"

"Generate Q&A for LLM agents, LangGraph, and MCP protocol.
Save to wiki/qa/agents-qa.md"

"Generate Q&A for MLOps: serving, quantization, monitoring, A/B
testing. Save to wiki/qa/mlops-qa.md"

"Generate system design Q&A. Include full design walkthroughs for:
RAG pipeline, LLM serving infrastructure, ML feature store, recommendation
system. Save to wiki/qa/system-design-qa.md"

"Generate behavioral Q&A. Include leadership principles, conflict
resolution, project failure, cross-functional collaboration.
Save to wiki/qa/behavioral-qa.md"
```

### 2.2 Review and enrich Q&A pages

For each generated Q&A file:
- Read through the L3 questions — these are the ones that trip candidates up
- Add any question you've been asked before that's missing
- Mark any question you're weak on with `> 🔴 WEAK AREA` blockquote

### 2.3 Generate cheatsheets

```
"Make a cheatsheet for transformer math: attention formula, complexity,
parameter counts. Save to wiki/cheatsheets/transformer-math.md"

"Make a cheatsheet for all fine-tuning methods covered in the wiki.
Table format: method | approach | memory cost | when to use.
Save to wiki/cheatsheets/finetuning-methods.md"

"Make a cheatsheet for algorithm complexity.
Save to wiki/cheatsheets/complexity-guide.md"

"Make an acronyms cheatsheet covering all acronyms in the wiki.
Save to wiki/cheatsheets/acronyms.md"
```

**Exit criteria for Phase 2:**
- [ ] 7 Q&A files exist in `wiki/qa/`
- [ ] Each Q&A file has L1, L2, and L3 questions
- [ ] Weak areas are flagged with 🔴
- [ ] 4 cheatsheets exist in `wiki/cheatsheets/`



---

## Phase 3 — Ongoing Maintenance (Weekly)

### Weekly Routine (~20 min/week)

**Every Monday:**
```
"Run a full wiki audit. Report: unresolved contradictions, orphan pages,
stubs, missing pages, stale high-priority pages."
```

Review the report. Prioritize what to fix.

**As you study:**
- Any time you read something new → INGEST it
- Any time you get a question wrong in mock  → note it, ask Claude
  Code to strengthen that wiki page
- Any time you read a conflicting claim → let the wiki flag it, then resolve


### Adding new sources over time

Good ongoing source types to ingest:
- Papers you read
- Blog posts (Lilian Weng, Hugging Face, OpenAI, Anthropic)
- Conference talk transcripts (NeurIPS, ICML)
- Mock feedback
- Leetcode editorial notes for patterns you struggle with
- Your own implementation notes (e.g., from building nanoGPT)

---

## Phase 4 — Obsidian Integration (Optional, ~10 min setup)

Obsidian gives you a visual graph of all wiki pages and their connections.
Especially useful for seeing knowledge gaps.

### 4.1 Open vault

```bash
# In Obsidian: Open folder as vault → select ai-engineer-wiki/
```

### 4.2 Recommended plugins

- **Graph View** — visual map of all `[[wiki-links]]`
- **Dataview** — query frontmatter fields (e.g., "show all high-relevance stubs")
- **Templater** — enforce the page template on new pages

### 4.3 Useful Dataview queries

Add these to `wiki/index.md` as live dashboards:

````markdown
## Stubs to fill

```dataview
TABLE relevance, last_updated
FROM "wiki/concepts"
WHERE status = "stub"
SORT relevance DESC
```

## High-priority pages not updated recently

```dataview
TABLE last_updated
FROM "wiki"
WHERE relevance = "high" AND last_updated < date(today) - dur(30 days)
SORT last_updated ASC
```
````

---

## Phase 5 — Publishing (Optional)

If you want to share the wiki publicly (great for portfolio):

### Option A — GitHub Pages with MkDocs

```bash
pip install mkdocs mkdocs-material
mkdocs new .
# Configure mkdocs.yml to point at wiki/ directory
mkdocs gh-deploy
```

### Option B — Keep private, reference in resume/portfolio

Even a private wiki is valuable to mention:
> *"I maintain a self-updating AI knowledge base using the LLM Wiki
> pattern — 50+ interlinked concept pages compiled from primary sources."*

---

## Quick Reference — Claude Code Commands

Save these prompts. Use them verbatim for reliable results.

### Bootstrap
```
Read SKILL.md thoroughly. Bootstrap the wiki: create full directory structure,
wiki/index.md, wiki/log.md, and stub pages for every concept in the Topic
Coverage Map. Set all stubs to status: stub. Log the operation.
```

### Ingest
```
Ingest raw/<path/to/source.md> into the wiki. Follow OP-1 from SKILL.md exactly.
Update all affected pages, create new pages if needed, flag contradictions,
update the index, and log the operation.
```

### Query
```
Query the wiki: <your question here>. Cite the wiki pages you use.
Do not go to raw sources — use only compiled wiki knowledge.
```

### Audit
```
Run a full wiki audit following OP-3 from SKILL.md. Output the full audit
report with counts for each category. Then list the top 3 actions I should
take to improve wiki health.
```

### Generate Q&A
```
Generate Q&A for <topic> following OP-4 from SKILL.md.
Cover all concept pages tagged [<topic>]. Include L1, L2, and L3 questions
with common follow-ups. Save to wiki/qa/<topic>-qa.md.
```


### Mock Quiz
```
Quiz me on <topic>. Use wiki/qa/<topic>-qa.md.
Ask one question at a time. Wait for my answer. Give feedback citing
specific wiki pages. Start with an L2 question.
```

### Study Plan
```
Generate a <N>-day study plan for the topic area "<topic>".
Map each day to specific wiki pages and Q&A sets.
```

---

## Milestone Checklist

| Milestone | Target | Done |
|---|---|---|
| Phase 0: Repo bootstrapped | Day 1 | [ ] |
| Phase 1: 15 sources ingested | End of Week 1 | [ ] |
| Phase 1: Core concept pages current | End of Week 1 | [ ] |
| Phase 2: All Q&A files generated | End of Week 2 | [ ] |
| Phase 2: Weak areas flagged | End of Week 2 | [ ] |
| Phase 3: Weekly audit habit | Ongoing | [ ] |
| Wiki size: 30+ concept pages | Month 1 | [ ] |
| Wiki size: 50+ concept pages | Month 2 | [ ] |
| Wiki size: 100+ concept pages | Month 3 | [ ] |

---

## Troubleshooting

**Claude Code creates content without reading SKILL.md**
→ Start every session with: *"First read SKILL.md in the repo root, then proceed."*

**Wiki pages are too shallow**
→ Provide richer source material. The wiki is only as deep as what you feed it.
  Paste full paper text, not just titles.

**Claude Code hallucinates content not in sources**
→ Add to your prompt: *"Do not add any information not present in the source.
  Stubs are acceptable. Accuracy over completeness."*

**Log.md is getting too long**
→ Archive old entries: *"Archive all log entries older than 30 days to
  wiki/log-archive-YYYY-MM.md. Keep only the last 30 days in log.md."*

**Obsidian shows broken links**
→ Run: *"Scan the wiki for broken [[wiki-links]] where the target page doesn't
  exist. List them all. Then create stubs for any that are high-priority."*

**Context window too large for audit**
→ Split the audit: *"Audit only wiki/concepts/ today."* Then run again for
  other subdirectories.

---

*BUILD_PLAN.md v1.0 — AI Engineer Wiki*
*Use alongside SKILL.md. Start at Phase 0 and work forward.*

---

# Appendix — Interview Agent: `agent.py` Dispatch Map & Extension Points

> Added by interview-agent-spec.md Phase 0 (recon). Documents how operations are
> routed today and where INTERVIEW / ASSESS / MAINTAIN plug in.

## How operation dispatch works today

There is **no code-level operation dispatch**. Operations (INGEST, QUERY, AUDIT,
GENERATE, CHEATSHEET) are *prompt-routed*: the LLM reads the operation specs and
decides which tools to call. The flow in `agent/agent.py`:

1. **System prompt construction** — `_build_system_prompt()` (`agent/agent.py:51`)
   runs **at import time** and embeds:
   - `wiki/index.md` (full text)
   - `skill.md` (first 24,000 chars — note the truncation at `agent/agent.py:85`)
   - A hardcoded operation list (OP-1…OP-5) and mandatory rules.
2. **Agent loop** — `run_agent(user_query, history)` (`agent/agent.py:123`) is a
   standard OpenAI tool-calling loop: call model → if `finish_reason ==
   "tool_calls"`, execute tools and loop; else return text.
3. **Tool dispatch** — `_dispatch_tool_call(name, payload)` (`agent/agent.py:102`)
   is the only real dispatch: an if-chain mapping 8 tool names to `wiki_tool.py`
   functions. `TOOLS` (`agent/agent.py:90`) is the parallel list of OpenAI schemas.
4. **Path safety** — lives entirely in `agent/wiki_tool.py` (writes restricted to
   `wiki/`, raw reads only via `read_raw_source`).
5. **Frontends** — `agent/cli.py` (REPL) and `agent/app.py` (Streamlit) both just
   call `run_agent()` with an accumulated history list.

## Extension points for INTERVIEW / ASSESS / MAINTAIN

| # | Extension point | Location | What to change |
|---|---|---|---|
| 1 | System prompt op list | `_build_system_prompt()` in `agent/agent.py:55-86` | Add OP-6 INTERVIEW, OP-7 ASSESS, OP-8 MAINTAIN to the hardcoded "Supported operations" block. The skill.md excerpt is truncated at 24k chars — verify new op specs in skill.md survive the cut or raise the limit. |
| 2 | Tool schema registry | `TOOLS` list, `agent/agent.py:90-99` | Append new tool schemas (e.g., `read_state_file`, `write_state_file`, `write_transcript`, `append_assessment_log`). |
| 3 | Tool dispatch | `_dispatch_tool_call()`, `agent/agent.py:102-120` | Add if-branches for new tools. Consider refactoring the if-chain to a dict registry so `interview.py` / `assess.py` / `maintain.py` can register tools without editing `agent.py` (single shared-file edit, then never again — avoids parallel-agent clobbering, spec §8). |
| 4 | Path safety | `agent/wiki_tool.py` | Current rules block writes outside `wiki/` and any write to `raw/`. New modules need carve-outs: `state/*` (ratings/queue/log), `raw/interviews/` + `raw/auto/` (transcripts and fetched sources are agent-*written* raw material). Add dedicated, narrowly-scoped tools rather than loosening `write_wiki_file`. |
| 5 | Session-loop ownership | `run_agent()` `agent/agent.py:123` | INTERVIEW is multi-turn and stateful (Elo picker, dedup, timer) — implement as its own session manager in `agent/interview.py` that *uses* the OpenAI client + prompts directly, rather than forcing it through the single-turn `run_agent()` loop. ASSESS and MAINTAIN are batch, headless-friendly: same pattern (`agent/assess.py`, `agent/maintain.py`), thin dispatch from `cli.py`. |
| 6 | CLI entry | `agent/cli.py` | Add `interview` / `assess` / `maintain` commands that route to the new modules instead of the generic chat loop. |
| 7 | Streamlit | `agent/app.py` | New Interview + History tabs (spec §3.4, Phase 4). |
| 8 | Import-time prompt staleness | `SYSTEM_PROMPT` built at import (`agent/agent.py:89`) | Long-running Streamlit sessions won't see index/skill updates mid-process; MAINTAIN regenerates pages, so it must rebuild prompts per run, not rely on import-time state. |

## Interview-agent phase progress

- **Phase 0 (recon & scaffolding)** — ✅ done 2026-06-10. Dirs + state seeds,
  pytest scaffolding, CI (`.github/workflows/test.yml`), skill.md OP-6/7/8 specs.
- **Phase 1 (INTERVIEW)** — ✅ done 2026-06-11. `agent/interview.py` (scope
  resolution, ±150 Elo-band picker, QA-bank parser, in-session dedup,
  transcript writer), `agent/prompts/interviewer_system.md`, `cli.py`
  `interview` subcommand + REPL detection, OP-6 note in `agent.py` system
  prompt, `tests/test_interview.py` (27 tests). Done-when verified: live
  5-question kv-cache drill via CLI produced
  `raw/interviews/2026-06-11-kv-cache-drill.md` with valid frontmatter and
  level-2→3 escalation after a correct answer.
- **Phase 2 (ASSESS)** — ✅ done 2026-06-11. `agent/assess.py` (transcript
  parser, page-as-rubric LLM grader with strict-JSON + fence stripping, Elo
  updates with K=32→16 schedule, report writer with 7-day study plan, queue
  writer, JSONL log, assessed-flag flip), `agent/prompts/grader_system.md`,
  `cli.py` `assess` subcommand + REPL detection, `tests/test_assess.py`
  (18 tests incl. Elo golden cases). Done-when exceeded: live ASSESS of the
  Phase 1 transcript wrote `wiki/reports/2026-06-11-kv-cache.md` with
  heading-cited gaps, updated 4 concept ratings, and queued 4 generate_qa
  tasks. Also fixed: `.env` now loads in `interview.py` so the `interview`/
  `assess` subcommands get the API key without importing `agent.py`.
- **Phase 3 (MAINTAIN)** — not started.
- **Phase 4 (UI/polish)** — not started.

## New state & directories (created in Phase 0)

- `state/skill_ratings.json` — Elo ratings, ASSESS-owned (seeded empty: defaults = rating 1200).
- `state/maintenance_queue.json` — ASSESS → MAINTAIN task queue (seeded `{"tasks": []}`).
- `state/assessment_log.jsonl` — append-only per-session results (seeded empty).
- `wiki/reports/` — human-readable weakness reports from ASSESS.
- `raw/interviews/` — INGEST-compatible interview transcripts.
- `raw/auto/` — watchlist-fetched sources for MAINTAIN.

See `state/README.md` for schemas and invariants.