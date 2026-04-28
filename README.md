# AI Engineer Wiki

A living, compounding knowledge base for AI engineering across transformers, RLHF/DPO/GRPO, RAG, agents, MLOps, system design, and company prep.

## How It Works
- Put source material in `raw/`.
- The agent compiles and maintains structured markdown in `wiki/`.
- Query the compiled wiki for answers, not raw chunks.

## Agent Capabilities
The agent now supports the full operation set from `skill.md`:
- `INGEST`: read `raw/...` source, update/create wiki pages, update index, append log
- `QUERY`: answer from wiki pages with page citations
- `AUDIT`: contradictions, orphans, stubs, missing pages, stale high-priority pages
- `GENERATE`: create/update topic Q&A pages
- `COMPANY`: create/update company prep pages
- `CHEATSHEET`: create/update quick-reference summaries

## Repo Layout
```text
wiki/
  concepts/
  companies/
  system-design/
  cheatsheets/
  qa/
  index.md
  log.md
raw/
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
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

Set your Anthropic key:
```bash
# PowerShell
$env:ANTHROPIC_API_KEY="your_key_here"

# Optional overrides
$env:ANTHROPIC_MODEL="claude-sonnet-4-6"
$env:AGENT_MAX_TOKENS="4096"
```

## Run
### Terminal chat
```bash
cd agent
python cli.py
```

### Streamlit UI
```bash
cd agent
streamlit run app.py
```

### Fetch starter sources
```bash
cd agent
python fetch_sources.py --list
python fetch_sources.py --only attention-is-all-you-need
```

## Example Prompts
- `Ingest raw/transformers/attention-is-all-you-need.md`
- `What does the wiki say about KV cache tradeoffs?`
- `Run a full wiki audit`
- `Generate questions on LoRA and save to wiki/qa/rl-qa.md`
- `Update company prep for Capital One`
- `Make a cheatsheet for positional encoding`
