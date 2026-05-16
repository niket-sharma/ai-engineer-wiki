# AI Engineer Wiki

A living, compounding knowledge base for AI engineering — transformers, alignment, RAG, agents, MLOps, system design, and classic ML. Compiled from primary sources into interlinked concept pages. Knowledge compounds; never re-derive from scratch.

## How It Works
- Put source material in `raw/`.
- The agent compiles and maintains structured markdown in `wiki/`.
- Query the compiled wiki for answers, not raw chunks.

## Agent Capabilities
The agent supports the full operation set from `skill.md`:
- `INGEST`: read `raw/...` source, update/create wiki pages, update index, append log
- `QUERY`: answer from wiki pages with page citations
- `AUDIT`: contradictions, orphans, stubs, missing pages, stale high-priority pages
- `GENERATE`: create/update topic Q&A pages
- `CHEATSHEET`: create/update quick-reference summaries

## Repo Layout
```text
wiki/
  concepts/
  system-design/
  cheatsheets/
  qa/
  index.md
  log.md
raw/
  transformers/
  rl-and-rlhf/
  inference-serving/
  rag-retrieval/
  agents/
  evaluation/
  production-ai/
  system-design/
  statistics-and-ml/
  coding-and-algos/
agent/
  agent.py
  wiki_tool.py
  cli.py
  app.py
  fetch_sources.py
  requirements.txt
```

## Setup
```bash
cd agent
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set your OpenAI key:
```bash
export OPENAI_API_KEY="your_key_here"

# Optional overrides
export OPENAI_MODEL="gpt-4o"      # default
export AGENT_MAX_TOKENS="4096"    # default
```

## Run
### Terminal chat
```bash
cd agent && python cli.py
```

### Streamlit UI
```bash
cd agent && streamlit run app.py
```

### Fetch starter sources
```bash
cd agent && python fetch_sources.py --list
cd agent && python fetch_sources.py --only attention-is-all-you-need
```

## Example Prompts
- `Ingest raw/transformers/attention-is-all-you-need.md`
- `What does the wiki say about KV cache tradeoffs?`
- `Run a full wiki audit`
- `Generate questions on LoRA and save to wiki/qa/rl-qa.md`
- `Make a cheatsheet for positional encoding`

## Topics Covered
Transformers · RLHF/DPO/GRPO · RAG & Retrieval · LLM Agents · Inference & Serving · Evaluation · Production AI · System Design · Statistics · Algorithms
