# AI Engineer  Wiki — Index

> Last audited: 2026-04-22 | Pages: 34 | Sources: training-knowledge | Stubs: 0

## Transformer Architecture
- [[attention-mechanism]] — Self-attention, MHA/MQA/GQA, scaled dot-product ✅
- [[transformer-architecture]] — Full GPT/BERT/T5 architecture, parameter counts ✅
- [[positional-encoding]] — Sinusoidal, RoPE, ALiBi, YaRN, length extrapolation ✅
- [[kv-cache]] — Inference optimization, paged attention, GQA memory savings ✅
- [[flash-attention]] — IO-aware SRAM tiling, FA1/FA2/FA3, online softmax ✅

## Fine-tuning & Alignment
- [[lora-qlora]] — LoRA math, QLoRA, DoRA, NF4 quantization ✅
- [[rlhf]] — Three stages (SFT→RM→PPO), KL penalty, reward hacking ✅
- [[ppo]] — PPO-clip objective, GAE, RLHF adaptation ✅
- [[dpo]] — DPO loss derivation, IPO/KTO/ORPO variants ✅
- [[grpo]] — Group baseline, no value model, DeepSeek-R1 ✅

## RAG & Retrieval
- [[rag-systems]] — Pipeline, chunking, hybrid search, HyDE, evaluation (RAGAS) ✅
- [[vector-databases]] — HNSW, IVF-PQ, comparison table (Pinecone/Qdrant/pgvector) ✅
- [[reranking]] — Bi-encoder vs cross-encoder, ColBERT, two-stage retrieval ✅

## Agents & Orchestration
- [[langgraph-agents]] — ReAct, StateGraph, multi-agent patterns, memory types ✅
- [[mcp-protocol]] — MCP primitives, transport, comparison to function calling ✅

## System Design
- [[rag-pipeline-design]] — Full ingestion + query pipeline with component deep-dives ✅
- [[llm-serving-infra]] — Continuous batching, paged attention, parallelism, capacity planning ✅
- [[ml-platform]] — Feature store, experiment tracking, serving, monitoring (stub) 
- [[feature-store]] — Online/offline stores, point-in-time correctness (stub)

## Companies
- [[capital-one]] — Fraud detection, credit risk, NLP, AWS stack, likely questions ✅
- [[massmutual]] — Actuarial AI, survival analysis, insurance pricing ✅
- [[fidelity]] — Financial NLP, RAG for advice, portfolio ML ✅
- [[exxon]] — Predictive maintenance, process optimization, industrial AI ✅

##  Q&A
- [[transformers-qa]] — L1/L2/L3 questions on attention, KV cache, Flash Attention ✅
- [[rl-qa]] — RLHF stages, DPO derivation, GRPO, QLoRA deep dives ✅
- [[rag-qa]] — RAG pipeline, hybrid search, HyDE, evaluation, system design ✅
- [[system-design-qa]] — LLM serving, RAG design, feature store, A/B testing ✅
- [[mlops-qa]] — Quantization, drift detection, paged attention, monitoring ✅
- [[agents-qa]] — ReAct, LangGraph, multi-agent design, compliance guardrails ✅
- [[behavioral-qa]] — STAR stories, influence, conflict, failure, staying current ✅

## Cheatsheets
- [[math-and-notation]] — Attention formula, parameter counts, KV cache math, LoRA, RLHF ✅
- [[complexity-guide]] — Big O, data structures, sorting, graph algorithms, ML ops ✅
- [[acronyms]] — Full glossary: transformers, alignment, RAG, MLOps, agents, eval ✅

---

## Obsidian Dataview Dashboards

### Stubs to fill
```dataview
TABLE relevance, last_updated
FROM "wiki/concepts"
WHERE status = "stub"
SORT relevance DESC
```

### High-priority pages not updated recently
```dataview
TABLE last_updated
FROM "wiki"
WHERE relevance = "high" AND last_updated < date(today) - dur(30 days)
SORT last_updated ASC
```
