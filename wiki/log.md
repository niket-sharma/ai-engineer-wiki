# Operation Log
Append-only. Most recent entry at top.

---

## [BUILD] 2026-04-22 — Full Wiki Bootstrap from Training Knowledge

### Phase 2 — Q&A Generation
- Pages created: [[transformers-qa]], [[rl-qa]], [[rag-qa]], [[system-design-qa]], [[mlops-qa]], [[agents-qa]], [[behavioral-qa]]
- All Q&A files have L1/L2/L3 questions with follow-ups
- System design Q&A includes full design walkthroughs (RAG pipeline, LLM serving, feature store, A/B testing)
- Behavioral Q&A includes STAR frameworks and finance-specific angles

### Phase 1 — Concept Population
- All 15 concept pages populated with full technical content from training knowledge
- Pages updated to `status: current`:
  - [[attention-mechanism]] — MHA/MQA/GQA, complexity,  angles
  - [[transformer-architecture]] — GPT/BERT/T5, parameter counts, modern improvements
  - [[positional-encoding]] — Sinusoidal/RoPE/ALiBi/YaRN, length extrapolation
  - [[kv-cache]] — Memory math, GQA savings, paged attention, prefix caching
  - [[flash-attention]] — IO-aware tiling, online softmax, FA1/FA2/FA3
  - [[lora-qlora]] — LoRA math, QLoRA NF4, variants (DoRA, AdaLoRA, LoRA+)
  - [[rlhf]] — Three stages, RM training, PPO objective, reward hacking
  - [[ppo]] — Clipped objective, GAE, RLHF adaptation, comparison to GRPO
  - [[dpo]] — Derivation from RLHF, loss formula, IPO/KTO/ORPO variants
  - [[grpo]] — Group normalization, no value model, DeepSeek-R1 connection
  - [[rag-systems]] — Full pipeline, chunking, hybrid search, HyDE, RAGAS eval
  - [[vector-databases]] — HNSW/IVF-PQ, filtering, comparison table
  - [[reranking]] — Bi-encoder vs cross-encoder, ColBERT, two-stage pipeline
  - [[langgraph-agents]] — ReAct, StateGraph, memory types, multi-agent patterns
  - [[mcp-protocol]] — Primitives (tools/resources/prompts), transport, ecosystem
- System design pages: [[rag-pipeline-design]], [[llm-serving-infra]] fully populated
- Company pages: [[capital-one]], [[massmutual]], [[fidelity]], [[exxon]] populated with domain knowledge, likely questions, red flags
- Cheatsheets: [[math-and-notation]], [[complexity-guide]], [[acronyms]] fully populated

### Phase 0 — Bootstrap
- Created full directory structure
- SKILL.md already present (user-provided)
- Created README.md, .gitignore, wiki/index.md, wiki/log.md
- Created 24 stub pages (15 concepts + 4 system-design + 4 companies + 3 cheatsheets)

### Summary
- Pages created: 34
- Pages at `status: current`: 32 (ml-platform and feature-store remain stubs)
- Q&A files: 7
- Sources ingested: training knowledge (knowledge base bootstrapped without raw sources)
- Contradictions flagged: 0
- Next action: add raw sources to `raw/` subdirectories and run INGEST to deepen pages

---

## Next Steps (Phase 1 continuation)
Drop any of the following into `raw/` and say "Ingest raw/<path>":
- Attention Is All You Need paper notes → `raw/transformers/`
- Karpathy GPT lecture notes → `raw/transformers/`
- DeepSeek-R1 paper notes → `raw/rl-and-rlhf/`
- LangGraph documentation → `raw/mlops/`
- Company JDs → `raw/company-specific/`
