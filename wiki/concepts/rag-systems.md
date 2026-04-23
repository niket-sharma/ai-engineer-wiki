---
title: "RAG Systems"
aliases: ["Retrieval-Augmented Generation", "RAG", "naive RAG", "advanced RAG", "modular RAG"]
tags: [rag, retrieval, agents]
related: ["[[vector-databases]]", "[[reranking]]", "[[langgraph-agents]]", "[[rag-pipeline-design]]"]
sources: ["training-knowledge"]
interview_relevance: high
last_updated: 2026-04-22
status: current
---

# RAG Systems

## TL;DR
RAG grounds LLM responses in retrieved documents, reducing hallucination and enabling knowledge updates without retraining. The core pattern: embed query → retrieve relevant chunks → prepend to context → generate. Real production systems add chunking strategy, reranking, hybrid search, and evaluation layers that are the meat of interview questions.

## Intuition
LLMs have knowledge frozen at training time and hallucinate when asked about specific facts, recent events, or private data. Instead of retraining (expensive), RAG retrieves relevant documents at query time and inserts them into the context window. The LLM now has access to fresh, specific knowledge for each query.

The catch: garbage in, garbage out. The quality of retrieval determines the quality of generation. Most RAG failures are retrieval failures, not generation failures.

## Technical Detail

**Core RAG Pipeline:**
```
Query → Embed query → ANN search in vector DB → (optional rerank) → Insert into prompt → LLM → Response
```

**Ingestion Pipeline:**
```
Documents → Chunk → Embed chunks → Store (vector DB + metadata store)
```

**Chunking strategies:**
| Strategy | When to use |
|---|---|
| Fixed-size (512 tokens, 50% overlap) | Simple baseline, most common |
| Sentence-level | Better semantic boundaries |
| Semantic/paragraph | Best quality, slower |
| Hierarchical (parent-child) | Retrieve small, send large context |
| Document-level | Short docs, need full context |

**Retrieval types:**
- **Dense retrieval**: Embed query with bi-encoder, ANN search in vector DB (cosine/dot)
- **Sparse retrieval (BM25)**: Keyword-based TF-IDF, handles exact match well
- **Hybrid**: Combine dense + sparse scores (RRF: Reciprocal Rank Fusion)
- **Hybrid is usually best** for production — dense handles semantic, sparse handles keyword

**Evaluation metrics (RAGAS framework):**
| Metric | What it measures |
|---|---|
| Faithfulness | Is the answer grounded in the retrieved context? |
| Answer Relevancy | Does the answer address the question? |
| Context Precision | Are retrieved chunks relevant to the question? |
| Context Recall | Were all needed facts retrieved? |

## Variants & Extensions

**Naive RAG → Advanced RAG → Modular RAG evolution:**

| Stage | Techniques |
|---|---|
| Naive RAG | Fixed chunking + single dense retrieval + direct generation |
| Advanced RAG | HyDE, query rewriting, reranking, hybrid search, sliding window chunks |
| Modular RAG | Iterative retrieval, self-RAG, corrective RAG, agentic RAG |

**HyDE (Hypothetical Document Embedding):**
Generate a hypothetical answer to the query, embed that, then retrieve using the generated answer embedding. Works because hypothetical answers are stylistically closer to actual answers in embedding space.

**Self-RAG:** Model decides when to retrieve (not every query needs retrieval), generates critique tokens to assess relevance.

**Corrective RAG (CRAG):** Evaluates retrieval quality; if poor, falls back to web search.

**Agentic RAG:** Multi-step reasoning with iterative retrieval — the agent retrieves, reasons, decides to retrieve more, etc.

## Tradeoffs
| Advantage | Disadvantage |
|---|---|
| No retraining needed for knowledge updates | Retrieval quality ceiling limits generation quality |
| Reduces hallucination on factual queries | Latency: retrieval adds 50–200ms per query |
| Works with private/proprietary data | Chunking is lossy — context boundaries cause failures |
| Transparent — can cite sources | Hard to handle multi-hop reasoning |
| Cost-effective vs. fine-tuning for knowledge | Embedding models may miss semantic nuances |

## Interview Angles

**What interviewers are really testing (especially at finance companies):**
- Can you design a RAG pipeline end-to-end?
- Do you know chunking strategies and their tradeoffs?
- Do you understand hybrid search and why it beats dense-only?
- Can you evaluate RAG quality? Do you know RAGAS?
- What happens when retrieval fails?

**Common follow-up questions:**
- "Walk me through a RAG pipeline from document ingestion to response."
- "How would you chunk a 100-page financial document for RAG?"
- "When does dense retrieval fail? How does hybrid search help?"
- "What metrics would you use to evaluate a RAG system?"
- "How would you handle a query that requires combining info from 3 documents?"
- "What's HyDE and when would you use it?"
- "How do you handle hallucinations in a RAG system?"

**Gotchas / misconceptions:**
- Most RAG failures are RETRIEVAL failures, not generation failures — debug retrieval first
- "Chunk size" and "overlap" are the most impactful knobs to tune in naive RAG
- Dense retrieval alone fails on exact-match queries (product codes, names, dates) — always consider hybrid
- Increasing context window size is NOT a solution to RAG — it degrades "lost in the middle" performance

## Connections
- [[vector-databases]] — the storage and ANN search layer for RAG
- [[reranking]] — post-retrieval quality improvement step
- [[langgraph-agents]] — agents can orchestrate multi-step/agentic RAG
- [[rag-pipeline-design]] — system design for production RAG deployment

## Sources
- Training knowledge (Lewis et al. 2020 "RAG paper"; Gao et al. 2024 "Modular RAG"; Es et al. 2023 "RAGAS")
