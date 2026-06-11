---
name: ai-engineer-wiki
description: >
  Build and maintain a living, compounding AI Engineer knowledge base
  using the LLM Wiki pattern. Use this skill whenever the user wants to ingest
  study material (papers, notes, blog posts, transcripts), update wiki pages,
  query the knowledge base, audit for contradictions, or run any operation on
  the AI Engineer Wiki. Triggers on phrases like "add to wiki",
  "update the wiki", "ingest this", "what does the wiki say about", "audit the
  wiki", "add a source", "compile notes", "wiki prep", or any mention
  of the ai-engineer-wiki project.
---

# AI Engineer Wiki

A **living, compounding knowledge base** for AI Engineer / Senior Data Scientist
 preparation — built on Andrej Karpathy's LLM Wiki pattern.

Instead of re-reading raw sources at query time (RAG-style), you compile them
**once** into structured, interlinked Markdown pages. The wiki grows richer with
every source you add. Query time is fast because synthesis has already happened.

---

## 1. Philosophy

| RAG (what most tools do) | LLM Wiki (this project) |
|---|---|
| Re-derives knowledge every query | Compiles knowledge once, reuses forever |
| Stateless — nothing accumulates | Stateful — compounds with each source |
| Retrieves chunks | Synthesizes pages |
| Fast to set up | Fast to query at scale |

**Your role:** choose sources, ask good questions, guide ingestion.  
**The LLM's role:** read, extract, integrate, cross-link, flag contradictions.

---

## 2. Repository Layout

```
ai-engineer-wiki/
│
├── raw/                          # Immutable source material (never edited)
│   ├── transformers/
│   ├── rl-and-rlhf/
│   ├── mlops/
│   ├── system-design/
│   ├── coding-and-algos/
│   ├── statistics-and-ml/
│   ├── company-specific/
│   └── misc/
│
├── wiki/                         # Compiled knowledge — LLM-maintained
│   ├── index.md                  # Global TOC + entity registry
│   ├── log.md                    # Append-only operation log
│   │
│   ├── concepts/                 # Deep-dive concept pages
│   │   ├── attention-mechanism.md
│   │   ├── transformer-architecture.md
│   │   ├── positional-encoding.md
│   │   ├── kv-cache.md
│   │   ├── flash-attention.md
│   │   ├── lora-qlora.md
│   │   ├── rlhf.md
│   │   ├── grpo.md
│   │   ├── ppo.md
│   │   ├── dpo.md
│   │   ├── rag-systems.md
│   │   ├── vector-databases.md
│   │   ├── langgraph-agents.md
│   │   ├── mcp-protocol.md
│   │   └── ...
│   │
│   ├── system-design/            # System design patterns and templates
│   │   ├── rag-pipeline-design.md
│   │   ├── llm-serving-infra.md
│   │   ├── feature-store.md
│   │   ├── ml-platform.md
│   │   └── ...
│   │
│   ├── qa/                      # Synthesized Q&A pairs
│   │   ├── transformers-qa.md
│   │   ├── rl-qa.md
│   │   ├── mlops-qa.md
│   │   ├── system-design-qa.md
│   │   └── behavioral-qa.md
│   │
│   └── cheatsheets/              # Quick-reference summaries
│       ├── math-and-notation.md
│       ├── complexity-guide.md
│       └── acronyms.md
│
├── SKILL.md                      # This file — build instructions for Claude Code
├── README.md                     # Human-readable project overview
└── .gitignore
```

---

## 3. Wiki Page Schema

Every page in `wiki/concepts/` and `wiki/system-design/`
MUST follow this YAML frontmatter schema:

```yaml
---
title: "Attention Mechanism"
aliases: ["self-attention", "scaled dot-product attention"]
tags: [transformers, architecture, core-concept]
related: ["[[transformer-architecture]]", "[[kv-cache]]", "[[flash-attention]]"]
sources: ["raw/transformers/attention-is-all-you-need.md", "raw/transformers/karpathy-gpt-lecture.md"]
relevance: high                    # high | medium | low
last_updated: 2026-04-16
status: current                    # current | outdated | stub
---
```

Fields are **mandatory**. The LLM must populate all fields when creating or
updating a page. `status: stub` is allowed for newly created placeholder pages.

---

## 4. Core Operations

When the user gives an instruction, identify which operation applies and follow
the corresponding steps precisely.

---

### OP-1: INGEST — Add a new source

**Trigger phrases:** "add this", "ingest", "I just read", "here's a new paper",
"add to wiki", "compile this"

**Steps:**

1. **Receive source.** The user provides one of:
   - A file path in `raw/` (already placed there)
   - Pasted text / URL / transcript
   - A paper title (fetch if possible)

2. **Stage the source.** If not already in `raw/`, save it to the appropriate
   `raw/<topic>/YYYY-MM-DD-short-title.md`. Never modify files in `raw/`.

3. **Read and extract.** Thoroughly read the source. Extract:
   - Core concepts defined or explained
   - Key claims, formulas, algorithms
   - Comparisons to other methods
   - Practical insights (anything that answers "why", "when", "tradeoffs")

4. **Identify affected wiki pages.** Check `wiki/index.md` for existing pages
   that overlap with the source content.

5. **Update existing pages.** For each affected page:
   - Add new information under the relevant section
   - Strengthen or update existing claims if the source provides better evidence
   - Flag contradictions with a `> ⚠️ CONTRADICTION:` blockquote if the source
     disagrees with existing content
   - Add the source to the frontmatter `sources` list
   - Update `last_updated` and `status`

6. **Create new pages.** For concepts not yet in the wiki:
   - Create `wiki/concepts/<slug>.md` with full frontmatter
   - Write an encyclopedia-style page: definition → intuition → math/detail →
     variants → tradeoffs → practical applications
   - Add wiki-links `[[like-this]]` to all related concepts

7. **Update index.** Add new pages to `wiki/index.md` under the correct section.

8. **Log the operation.** Append to `wiki/log.md`:
   ```
   ## [INGEST] YYYY-MM-DD HH:MM
   - Source: raw/topic/filename.md
   - Pages updated: [[page-a]], [[page-b]]
   - Pages created: [[new-page]]
   - Contradictions flagged: N
   ```

---

### OP-2: QUERY — Answer a question from the wiki

**Trigger phrases:** "what does the wiki say about", "explain X", "quiz me on",
"what are the tradeoffs of", "how does X work", "compare X and Y"

**Steps:**

1. Check `wiki/index.md` for relevant pages.
2. Read the relevant wiki pages (NOT the raw sources — the wiki is the truth).
3. Synthesize a response grounded in wiki content.
4. Cite page names: *"According to [[kv-cache]]..."*
5. If the wiki doesn't cover the topic, say so and offer to ingest a source.
6. If the answer requires combining 3+ pages, note this — it's a sign the wiki
   needs a new synthesis page.

---

### OP-3: AUDIT — Check wiki health

**Trigger phrases:** "audit the wiki", "check for contradictions", "find gaps",
"what's missing", "wiki health check"

**Steps:**

1. **Contradiction scan.** Read all pages. List any `⚠️ CONTRADICTION` flags
   still unresolved, with the conflicting claims.

2. **Orphan detection.** List pages with no inbound `[[wiki-links]]`.

3. **Stub scan.** List pages with `status: stub` — these need fleshing out.

4. **Gap detection.** Scan all pages for concepts mentioned but not yet having
   their own page. Output a "Missing Pages" list.

5. **Staleness check.** List pages where `last_updated` is > 60 days old and
   `relevance: high` — these may need refreshing.

6. **Output a report:**
   ```
   ## Wiki Audit Report — YYYY-MM-DD
   ### Unresolved Contradictions: N
   ### Orphan Pages: N
   ### Stub Pages: N
   ### Missing Pages (referenced but absent): N
   ### Stale High-Priority Pages: N
   ```

---

### OP-4: GENERATE — Create  Q&A

**Trigger phrases:** "generate questions", "make a Q&A for", "create flashcards",
"give me technical questions on", "prep me for X"

**Steps:**

1. Read the relevant wiki concept pages.
2. Generate Q&A pairs at three levels:
   - **L1 — Conceptual:** "What is X? Explain to a non-expert."
   - **L2 — Technical:** "Walk me through the math/algorithm of X."
   - **L3 — Applied:** "How would you use X in a production system? What are
     the tradeoffs?"
3. Save output to `wiki/qa/<topic>-qa.md`.
4. Tag each question with `[L1]`, `[L2]`, or `[L3]`.
5. Include a "Common Follow-ups" section per question.

---

### OP-5: CHEATSHEET — Generate quick-reference summaries

**Trigger phrases:** "make a cheatsheet", "quick reference for", "summarize X
in one page", "give me a cheatsheet"

**Steps:**

1. Read relevant wiki pages.
2. Compress to essential facts: formulas, key distinctions, 1-line definitions.
3. Save to `wiki/cheatsheets/<topic>.md`.
4. Format as tables and bullet points — optimized for scanning, not reading.

---

### OP-6: INTERVIEW — Adaptive mock technical interview

**Trigger phrases:** "interview me on X", "interview me on my weakest topics",
"run a system design interview", "mock interview", "drill me on X"

Conduct an adaptive mock technical interview against wiki content.
(Spec: `interview-agent-spec.md` §3.1.)

**Steps:**

1. **Session setup.** Resolve scope:
   - `topic=X` → load `wiki/concepts/` pages tagged/linked to X plus `wiki/qa/`
     pages for X.
   - `company=Y` → additionally load `wiki/companies/Y.md` (when present) and
     bias question style/topics to that page's interview-format notes.
   - `weakest` → read `state/skill_ratings.json`, pick the N lowest-rated concepts.
2. **Question selection.** Pull from `wiki/qa/` first; if coverage is thin,
   generate novel questions grounded in the loaded concept pages (tag them
   `generated:true` so ASSESS can later persist good ones back into `wiki/qa/`).
3. **Adaptive difficulty.** Maintain a per-concept Elo-style rating
   (see the Elo rules under OP-7 below):
   - Correct + confident answer → escalate (e.g., KV cache basics → GQA/MQA
     tradeoffs → paged attention / continuous batching).
   - Wrong/partial → de-escalate one level and probe the prerequisite concept.
   - Question difficulty levels: 1 (recall) … 5 (open-ended design under
     constraints).
4. **Interview styles** (flag `style=`):
   - `drill` — rapid-fire Q&A, short answers.
   - `deep` — one question, multiple follow-up probes ("why", "what breaks if...").
   - `system-design` — single scenario, 20–40 min, whiteboard-style with
     checkpoints.
   - `behavioral` — STAR-format, sourced from `wiki/companies/` behavioral notes.
5. **Session output.** Write a full transcript to
   `raw/interviews/YYYY-MM-DD-<topic>-<style>.md` with this frontmatter
   (INGEST-compatible by design):

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

**Anti-leak rule:** During the session, never reveal rubric content or wiki
citations until ASSESS. The interviewer asks and probes; it does not teach
mid-session (unless `mode=tutor` is explicitly set).

---

### OP-7: ASSESS — Grade a completed interview session

**Trigger phrases:** "assess my interview", "grade my session", "score the
transcript", "how did I do"

Grade a completed interview session against the wiki.
(Spec: `interview-agent-spec.md` §3.2.)

**Steps:**

1. Load the transcript + every concept page referenced by the session's
   questions. The **concept page is the rubric** (full-page router pattern —
   load whole pages as context, never chunk).
2. For each Q/A pair, produce:
   - `score` 0–4 (0 = no answer, 4 = senior-level complete with tradeoffs)
   - `gaps`: bullet list of missing key points, each citing the wiki page +
     heading
   - `misconceptions`: anything stated that contradicts the wiki (these are
     gold — flag them prominently)
   - `wiki_gap`: boolean — set true when the answer was reasonable but the wiki
     page lacked the depth to grade it properly (**this is the self-maintenance
     hook**)
3. Update `state/skill_ratings.json` (Elo update per concept). Elo rules:
   - Start at 1200. Question difficulty maps to opponent rating
     (level 1 = 1000 … level 5 = 1800).
   - K-factor 32 for a concept's first 5 sessions, then 16.
   - Score 0–4 maps to Elo outcome: 0–1 → loss, 2 → draw, 3–4 → win.
   - Difficulty selection rule: pick questions whose level rating is within
     ±150 of the concept's current rating (productive struggle zone).
   - **Only ASSESS may modify `state/skill_ratings.json`.**
4. Write `wiki/reports/YYYY-MM-DD-<topic>.md` — human-readable weakness
   report: summary scores, top 3 weaknesses, top 3 strengths, misconception
   list, and a **7-day micro study plan** with links to existing wiki pages.
5. Append a machine-readable block to `state/assessment_log.jsonl`:

   ```json
   {"date": "2026-06-10", "topic": "transformers", "style": "deep",
    "overall": 2.7, "per_concept": {"kv-cache": 3, "rope": 1},
    "misconceptions": ["claimed RoPE is applied to V"],
    "report": "wiki/reports/2026-06-10-transformers.md"}
   ```

6. **Trigger follow-ups automatically:**
   - For each concept scoring ≤ 2 → queue a `GENERATE` task (harder Q&A for
     that topic).
   - For each `wiki_gap=true` → queue an `AUDIT`-style stub flag on that
     concept page.
   - Queue file: `state/maintenance_queue.json` (consumed by OP-8 MAINTAIN).
7. Flip `assessed: true` in the transcript frontmatter and append the
   operation to `wiki/log.md`.

---

### OP-8: MAINTAIN — Autonomous weekly updater

**Trigger phrases:** "run maintenance", "process the maintenance queue",
"weekly update", headless invocation via GitHub Actions cron.

Runs headless via GitHub Actions (cron) or manually via CLI.
(Spec: `interview-agent-spec.md` §3.3.)

**Steps:**

1. **Consume the queue.** Process `state/maintenance_queue.json`:
   - `generate_qa` tasks → run OP-4 GENERATE for the topic at the requested
     difficulty.
   - `wiki_gap` tasks → expand the flagged concept page section using `raw/`
     sources; if no source exists, add to the watchlist below.
2. **Monitor sources.** Use the watchlist (`agent/watchlist.yaml`): arXiv
   categories (cs.CL, cs.LG), Anthropic/OpenAI/Meta release blogs, HF blog.
   Fetch new items since last run into `raw/auto/`.
3. **Relevance filter.** For each new item, score relevance against
   `wiki/index.md` topics + current weakness ratings. Ingest only items above
   threshold; weakness-related topics get priority boost.
4. **Draft in the wiki's house style.** INGEST relevant items → draft/update
   concept pages. Style guide: mirror existing pages (bottom-up explanations,
   shape-annotated tensor walkthroughs before intuition, explicit
   prerequisites section, Q&A at bottom).
5. **Open a PR, never push to main.** Branch `maintain/YYYY-MM-DD`, one PR per
   run, PR body = changelog table (page, action, source, reason) + diff stats.
   The human reviews and merges. New/changed pages automatically become
   interviewable.
6. Append run summary to `wiki/log.md`.

**Safety rails:**
- Never delete pages.
- Never modify `state/skill_ratings.json` (that is ASSESS-only).
- Cap pages touched per run at 12.
- PRs must pass `scripts/validate_wiki.py`.

**Queue task schema (`state/maintenance_queue.json`):**

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

---

## 5. Wiki Page Template

Use this template when creating new concept pages:

```markdown
---
title: ""
aliases: []
tags: []
related: []
sources: []
relevance: medium
last_updated: YYYY-MM-DD
status: stub
---

# <Title>

## TL;DR
One-sentence definition. One-sentence "why it matters."

## Intuition
Plain-language explanation. No jargon. Use an analogy if helpful.

## Technical Detail
Math, algorithm steps, pseudocode — whatever is needed for depth.

## Variants & Extensions
How this concept evolved or branches (e.g., MHA → MQA → GQA).

## Tradeoffs
| Advantage | Disadvantage |
|---|---|
| ... | ... |

## Practical Applications
- Common use cases and when to apply
- Common follow-up questions
- Gotchas / misconceptions to avoid

## Connections
- [[related-concept-1]] — how they connect
- [[related-concept-2]] — how they connect

## Sources
- [[raw/path/to/source.md]]
```

---

## 6. Topic Coverage Map

The wiki covers the following domains. Use this map to route ingested sources
to the right `raw/` subdirectory and wiki section.

| Domain | raw/ folder | Key concepts to cover |
|---|---|---|
| Transformer Architecture | `raw/transformers/` | Attention, MHA, PE, LayerNorm, FFN, KV cache, Flash Attention, GQA |
| Fine-tuning & Alignment | `raw/rl-and-rlhf/` | RLHF, PPO, DPO, GRPO, LoRA, QLoRA, SFT, ORPO |
| Inference & Serving | `raw/inference-serving/` | Quantization, speculative decoding, continuous batching, paged attention, TP/PP |
| RAG & Retrieval | `raw/rag-retrieval/` | RAG, reranking, hybrid search, vector DBs, chunking, HyDE, embedding models |
| Agents & Orchestration | `raw/agents/` | LangGraph, ReAct, tool use, MCP, memory, multi-agent, agentic RAG |
| Evaluation | `raw/evaluation/` | LLM eval, RAG eval, agent eval, embedding eval, offline vs online |
| Production AI Systems | `raw/production-ai/` | Prompt injection, guardrails, cost optimization, observability, model routing |
| System Design | `raw/system-design/` | ML platform, feature store, data pipeline, A/B testing, shadow mode |
| Statistics & Classic ML | `raw/statistics-and-ml/` | Bias-variance, regularization, ensembles, Bayesian inference, causal inference |
| Coding & Algorithms | `raw/coding-and-algos/` | LeetCode patterns, complexity, Python idioms, numpy/pandas |

---

## 7. Bootstrapping the Wiki (First Run)

When starting from a fresh repo, execute this sequence:

```
Step 1: Create directory structure
  mkdir -p raw/{transformers,rl-and-rlhf,inference-serving,rag-retrieval,agents,evaluation,production-ai,system-design,statistics-and-ml,coding-and-algos,misc}
  mkdir -p wiki/{concepts,system-design,qa,cheatsheets}

Step 2: Create wiki/index.md
  Use the Index Template (see Section 8).

Step 3: Create wiki/log.md
  Header: "# Operation Log\nAppend-only. Most recent entry at top.\n"

Step 4: Seed with starter pages (stubs)
  Create stub pages for all concepts in the Topic Coverage Map above.
  Use the page template. Set status: stub.

Step 5: Run first INGEST
  Ask the user: "What's the first source you'd like to add?"
```

---

## 8. Index Template (`wiki/index.md`)

```markdown
# AI Engineer Wiki — Index

> Last audited: YYYY-MM-DD | Pages: N | Sources: N | Stubs: N

## Transformer Architecture
- [[attention-mechanism]] — Self-attention, scaled dot-product
- [[transformer-architecture]] — Full GPT/BERT architecture
- [[positional-encoding]] — Sinusoidal, RoPE, ALiBi
- [[kv-cache]] — Inference optimization
- [[flash-attention]] — Memory-efficient attention

## Fine-tuning & Alignment
- [[lora-qlora]] — Parameter-efficient fine-tuning
- [[rlhf]] — Reinforcement Learning from Human Feedback
- [[ppo]] — Proximal Policy Optimization
- [[dpo]] — Direct Preference Optimization
- [[grpo]] — Group Relative Policy Optimization

## RAG & Retrieval
- [[rag-systems]] — Architecture, chunking, eval
- [[vector-databases]] — Pinecone, Weaviate, pgvector, FAISS
- [[reranking]] — Cross-encoders, ColBERT

## Agents & Orchestration
- [[langgraph-agents]] — State machines, async, production patterns
- [[mcp-protocol]] — Model Context Protocol

## System Design
- [[rag-pipeline-design]]
- [[llm-serving-infra]]
- [[ml-platform]]

## Q&A
- [[transformers-qa]]
- [[rl-qa]]
- [[mlops-qa]]
- [[system-design-qa]]
- [[behavioral-qa]]

## Cheatsheets
- [[math-and-notation]]
- [[complexity-guide]]
- [[acronyms]]
```

---

## 9. Quality Rules

The LLM MUST follow these rules on every operation:

1. **Never edit `raw/`.** Raw sources are immutable.
2. **Always update frontmatter** (`last_updated`, `sources`, `status`) when
   editing a wiki page.
3. **Always log every operation** to `wiki/log.md`.
4. **Use wiki-links `[[page-name]]`** for all cross-references — never bare text.
5. **Flag contradictions** with `> ⚠️ CONTRADICTION:` — never silently overwrite.
6. **Practical framing.** Every concept page must have a "Practical Applications"
   section. Knowledge without practical grounding is incomplete.
7. **Cite sources** in frontmatter AND in page body where specific claims come
   from a specific source.
8. **Don't hallucinate.** If a concept isn't in the wiki or the provided source,
   say so. Don't invent content.

---

## 10. Example Workflows

### "I just watched Karpathy's GPT lecture — add it to the wiki"
```
1. Save transcript/notes to raw/transformers/2026-04-16-karpathy-gpt-lecture.md
2. Extract: GPT architecture, attention implementation, training loop, BPE
3. Update: [[transformer-architecture]], [[attention-mechanism]], [[kv-cache]]
4. Create new page if needed: [[byte-pair-encoding]]
5. Generate Q&A: wiki/qa/transformers-qa.md
6. Log the operation
```

### "Quiz me on attention mechanisms"
```
1. Read wiki/concepts/attention-mechanism.md
2. Pull L1/L2/L3 questions from wiki/qa/transformers-qa.md
3. Present questions one at a time, wait for answer, give feedback
4. Cite wiki pages in feedback: "According to [[kv-cache]], the reason is..."
```


---

## 11. README Template

When initializing the repo, create `README.md` with:

```markdown
# AI Engineer Wiki

A living, compounding knowledge base for AI engineering — built on Andrej Karpathy's LLM Wiki pattern.

## How it works
Raw sources go in `raw/`. An LLM agent compiles them into structured,
interlinked Markdown pages in `wiki/`. Knowledge compounds with every
source added. Query the wiki instead of re-reading raw sources.

## Quick Start (Claude Code)
1. Clone this repo
2. Open Claude Code in the repo root
3. Say: "Read SKILL.md and bootstrap the wiki"
4. Drop a source in `raw/` and say: "Ingest raw/<path>"

## Operations
- **INGEST:** Add a new source → "ingest raw/transformers/paper.md"
- **QUERY:** Ask a question → "what does the wiki say about KV cache?"
- **AUDIT:** Health check → "audit the wiki"
- **GENERATE:** Make Q&A → "generate questions on LoRA"
- **CHEATSHEET:** Quick ref → "make a cheatsheet for positional encoding"

## Stack
- Storage: Plain Markdown (portable, future-proof)
- Agent: Claude Code
- Viewer: Obsidian (optional, for graph view)
- Version control: Git

## Topics Covered
Transformers · RLHF/DPO/GRPO · RAG & Retrieval · LLM Agents · Inference & Serving ·
Evaluation · Production AI · System Design · Statistics · Algorithms
```

---

*SKILL.md version 1.0 — AI Engineer Wiki*  
*Designed for use with Claude Code. Compatible with Cursor and other agent frameworks.*