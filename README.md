# AI Engineer Wiki

A living, compounding knowledge base for AI Engineer and Senior Data Scientist — built on Andrej Karpathy's LLM Wiki pattern.

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
- **GENERATE:** Make Q&A → "generate interview questions on LoRA"
- **COMPANY:** Company prep → "prep me for Capital One"
- **CHEATSHEET:** Quick ref → "make a cheatsheet for positional encoding"

## Stack
- Storage: Plain Markdown (portable, future-proof)
- Agent: Claude Code
- Viewer: Obsidian (optional, for graph view)
- Version control: Git

## Topics Covered
Transformers · RLHF/DPO/GRPO · RAG & Retrieval · LLM Agents · MLOps ·
System Design · Statistics · Algorithms · Company-Specific Prep
