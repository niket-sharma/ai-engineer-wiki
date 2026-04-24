---
title: "RAG Pipeline Design"
aliases: ["RAG system design", "production RAG"]
tags: [system-design, rag]
related: ["[[rag-systems]]", "[[vector-databases]]", "[[reranking]]", "[[llm-serving-infra]]"]
sources: ["training-knowledge"]
interview_relevance: high
last_updated: 2026-04-22
status: current
---

# RAG Pipeline Design

## TL;DR
Production RAG consists of two pipelines: an offline ingestion pipeline (documents → vectors) and an online query pipeline (question → answer). Most failures are in retrieval, not generation. Design for retrieval quality first.

## Problem Statement
Build a system that answers questions grounded in a private document corpus, with citations, at low latency and high accuracy.

**Scope questions to ask the interviewer:**
- How many documents? (10k vs 10M changes everything)
- Document types? (PDFs, HTML, structured data?)
- Query volume? (10 QPS vs 10,000 QPS)
- Latency SLA? (500ms vs 3s)
- Accuracy requirements? (FAQ bot vs compliance advisor)

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│  INGESTION PIPELINE (offline, batch + streaming)        │
│                                                         │
│  Raw Docs → Parse → Chunk → Embed → Vector DB           │
│                              ↓                          │
│                         BM25 Index                     │
│                         Metadata Store (SQL)            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  QUERY PIPELINE (online, real-time)                     │
│                                                         │
│  User Query                                             │
│    → [Optional] Query rewrite / HyDE                   │
│    → Hybrid retrieval: Dense ANN + BM25                 │
│    → RRF fusion → top-100 candidates                    │
│    → Cross-encoder reranker → top-5                    │
│    → Prompt assembly                                    │
│    → LLM generation                                     │
│    → Response + citations                               │
└─────────────────────────────────────────────────────────┘
```

## Component Deep-Dives

### Parsing
- **PDF**: PyMuPDF (fast) or Adobe PDF Extract (accurate, costly). Extract text + tables separately.
- **HTML**: BeautifulSoup + boilerplate removal (trafilatura)
- **Word/PowerPoint**: python-docx, python-pptx
- **Tables**: convert to markdown format — don't mix with prose

### Chunking

| Strategy | Chunk size | When |
|---|---|---|
| Fixed-size + overlap | 512 tokens, 50 overlap | Baseline, unstructured docs |
| Sentence-level | Variable | Better semantic boundaries |
| Section/paragraph | Variable | Structured docs (reports, wikis) |
| Hierarchical | Parent + child | When you need both precision and context |

**Hierarchical chunking (recommended for production):**
- Store full section as "parent" document
- Index paragraph-level "child" chunks for retrieval precision
- At query time: retrieve child chunks → return parent sections to LLM for full context

### Embedding Models

| Model | Dimensions | Multilingual | Use case |
|---|---|---|---|
| text-embedding-3-large | 3072 | No | Best English quality, managed |
| BGE-large-en | 1024 | No | Best open-source English |
| BGE-M3 | 1024 | Yes | Multilingual, hybrid-capable |
| E5-mistral-7b | 4096 | Yes | Best quality, expensive |

### Vector Database Selection

| Scale | Recommendation |
|---|---|
| < 1M vectors, dev/internal | Chroma or pgvector |
| 1M–100M, production | Qdrant or Weaviate (self-hosted) |
| 100M+, cost-managed | Pinecone or Milvus |
| Existing Postgres stack | pgvector with HNSW |

### Retrieval

**Always use hybrid search:**
```python
# Dense retrieval (semantic)
dense_results = vector_db.search(embed(query), top_k=50)

# Sparse retrieval (keyword/BM25)
sparse_results = es_client.search(query, top_k=50)

# Reciprocal Rank Fusion
fused = rrf_merge(dense_results, sparse_results, k=60)[:100]
```

### Reranking
Cross-encoder reranker on top-100 → top-5 for LLM context.
- **Open source**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast), `BAAI/bge-reranker-v2-m3` (quality)
- **Managed**: Cohere Rerank API (~50ms, high quality)
- Budget: can skip if latency < 200ms budget is tight AND using strong embeddings

### LLM Generation

**System prompt structure:**
```
You are a [domain] assistant. Answer questions using ONLY the provided context.
If the answer is not in the context, say "I don't have enough information."
Always cite your sources using [1], [2] format.
Context:
{retrieved_chunks_with_source_labels}
```

**Model selection:**
- Quality-critical (compliance, legal): Claude Opus / GPT-4
- General internal use: Claude Sonnet / GPT-4o
- High volume, cost-sensitive: Claude Haiku / GPT-4o-mini

## Scale & Reliability

| Metric | Target | Alert threshold |
|---|---|---|
| End-to-end latency p95 | < 2s | > 3s |
| Retrieval latency (p95) | < 200ms | > 500ms |
| LLM generation latency | < 1.5s first token | > 2s |
| Context precision | > 0.7 | < 0.5 |
| Faithfulness | > 0.85 | < 0.7 |

**Reliability:**
- Retrieval fallback: if vector DB is unavailable, fall back to BM25-only
- LLM fallback: if primary model fails, fall back to cheaper model with degraded quality notice
- Circuit breaker: don't hammer failing dependencies — fail fast with user-visible error

## Tradeoffs

| Decision | Option A | Option B | Recommendation |
|---|---|---|---|
| Embedding | OpenAI managed | Self-hosted BGE | Managed for speed-to-prod; self-hosted for cost at scale |
| Vector DB | Pinecone (managed) | Qdrant (self-hosted) | Pinecone for MVP; Qdrant for cost control |
| Retrieval | Dense-only | Hybrid | Always hybrid for production |
| Reranking | Skip | Cross-encoder | Include if precision matters; skip if latency < 1s |
| LLM | GPT-4 | Claude Sonnet | Claude Sonnet best cost/quality for RAG |

##  Angles

**Draw the architecture on a whiteboard:**
Start with the two pipelines clearly separated. Explain each component. Then discuss what breaks at scale.

**Top follow-up questions:**
- "How would you handle documents that update frequently?" → Delta ingestion, tombstone old chunks
- "What if a query needs info from 3 different documents?" → Multi-hop retrieval, or expand top-K and let LLM synthesize
- "How do you prevent hallucination?" → System prompt constraints, faithfulness monitoring, "I don't know" threshold

## Connections
- [[rag-systems]] — conceptual foundation for this design
- [[vector-databases]] — storage and ANN search layer
- [[reranking]] — quality improvement post-retrieval
- [[llm-serving-infra]] — the generation component of this system
