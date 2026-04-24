---
title: "RAG & Retrieval  Q&A"
tags: [-qa, rag, retrieval]
related: ["[[rag-systems]]", "[[vector-databases]]", "[[reranking]]"]
last_updated: 2026-04-22
---

# RAG & Retrieval  Q&A

---

## L1 — Conceptual

### Q1. What is RAG and why is it used instead of just fine-tuning?

**A:** RAG (Retrieval-Augmented Generation) retrieves relevant documents at query time and injects them into the LLM's context. It's preferred over fine-tuning for knowledge tasks because:
- **No retraining**: knowledge updates (new documents) don't require retraining
- **Citability**: sources can be surfaced to users
- **Private data**: works with data that can't be included in pretraining
- **Freshness**: handles real-time or frequently updated information

Fine-tuning is better when you need to change the model's *behavior* or *style*, not its *knowledge*. For knowledge-intensive tasks, RAG is almost always superior to fine-tuning and much cheaper.

---

### Q2. What are the main failure modes of RAG?

**A:**
1. **Retrieval failure**: Right document isn't in the top-K results (bad chunking, semantic gap, wrong embedding model)
2. **Lost in the middle**: LLM ignores relevant context that appears in the middle of a long context window
3. **Hallucination despite retrieval**: LLM ignores retrieved context and makes up an answer anyway
4. **Irrelevant context**: Retrieval returns plausible-sounding but wrong chunks, which confuses generation
5. **Chunking artifacts**: The answer spans chunk boundaries — neither chunk contains the full answer
6. **Exact match failure**: Dense retrieval misses exact strings (product codes, names) that BM25 would catch

**Rule of thumb**: 80% of RAG failures are retrieval failures. Debug retrieval before debugging generation.

---

### Q3. What is hybrid search and why is it better than dense-only retrieval?

**A:** Hybrid search combines dense (embedding-based ANN) and sparse (BM25 keyword) retrieval. Dense retrieval captures semantic meaning but fails on exact matches (e.g., "SB-7721 compliance code"). Sparse retrieval handles exact matches but misses paraphrasing. Hybrid catches both.

Standard combination: Reciprocal Rank Fusion (RRF):
```
score(d) = Σ_r  1 / (k + rank_r(d))    # k=60 standard
```
Rank-based fusion avoids score normalization issues. In practice, hybrid search improves retrieval recall by 5–20% over dense-only, especially for domain-specific or technical documents with precise terminology.

---

### Q4. What chunking strategy would you use for a 100-page financial annual report?

**A:** Annual reports have clear structure — use structure-aware chunking:
1. **Section-level split** by headers (10-K sections: Risk Factors, MD&A, Financial Statements)
2. Within each section, use **paragraph-level chunks** (not fixed-size) — financial paragraphs have natural semantic boundaries
3. **Overlap**: ~1 paragraph overlap between chunks to avoid context boundary artifacts
4. **Metadata**: attach section header, page number, company, fiscal year to each chunk — enables metadata filtering
5. **Tables**: extract separately as structured text or convert to markdown — tables break embedding models if mixed with prose
6. **Hierarchical**: store section-level summaries as "parent chunks" for context, paragraph-level as "child chunks" for retrieval precision

Avoid fixed 512-token chunks with 50% overlap — this breaks across sentence boundaries and destroys financial table structure.

---

## L2 — Technical

### Q5. Walk me through the full RAG pipeline from document ingestion to response generation.

**A:**

**Ingestion pipeline (offline):**
```
Raw docs (PDF, HTML, Word)
→ Parse & clean (remove headers, footers, extract tables)
→ Chunk (strategy depends on doc type)
→ Embed chunks (e.g., text-embedding-3-large, BGE-M3)
→ Store: vectors → vector DB (Pinecone/Qdrant), metadata → SQL/Postgres
→ (optional) BM25 index for hybrid search
```

**Query pipeline (online):**
```
User query
→ (optional) query rewriting / HyDE / expansion
→ Embed query
→ Retrieve: dense ANN + sparse BM25 → RRF → top-100 candidates
→ Rerank: cross-encoder → top-5
→ Prompt construction: [system] + [retrieved chunks + sources] + [query]
→ LLM generation
→ Response + citations
```

**Evaluation:**
- Retrieval: context precision, context recall (RAGAS)
- Generation: faithfulness, answer relevancy (RAGAS)
- End-to-end: win rate vs. baseline (human eval or LLM judge)

---

### Q6. What is HyDE and when does it improve RAG?

**A:** HyDE (Hypothetical Document Embedding): instead of embedding the raw query, generate a hypothetical answer document with the LLM, then embed that.

```
Query → LLM → "Hypothetical answer" → embed → ANN search
```

**Why it helps**: Queries and documents have different styles in embedding space (queries are short interrogatives; documents are declarative prose). A hypothetical answer is stylistically closer to real answers, so it retrieves better neighbors.

**When it helps most**:
- Short or ambiguous queries (e.g., "RLHF disadvantages")
- Domain shift between query style and document style
- Multilingual or cross-lingual retrieval

**When it hurts**:
- When the LLM generates a hallucinated hypothetical that diverges from real content
- Latency-sensitive systems (adds one extra LLM call)
- When the query is already precise (exact match terms)

---

### Q7. How do you evaluate a RAG system in production?

**A:**

**Offline evaluation (RAGAS framework):**
| Metric | How computed | What it catches |
|---|---|---|
| Context Precision | % retrieved chunks actually relevant | Noisy retrieval |
| Context Recall | % of needed facts that were retrieved | Retrieval gaps |
| Faithfulness | Are answer claims grounded in context? | Hallucination |
| Answer Relevancy | Does answer address the question? | Off-topic answers |

**Online evaluation:**
- Thumbs up/down feedback from users
- Click-through rate on cited sources
- Follow-up question rate (proxy for answer completeness)
- Escalation rate (user gives up or asks to speak to human)

**Golden dataset approach**: curate 200–500 (query, expected_answer, relevant_doc_ids) examples, run weekly regression tests against this dataset.

**LLM-as-judge**: Use GPT-4 to score faithfulness and relevancy on random samples. Cheaper than human eval, surprisingly well-correlated with human judgment at scale.

---

### Q8. Explain the difference between a bi-encoder and a cross-encoder in retrieval.

**A:**
**Bi-encoder (used for embedding/retrieval):**
- Encode query and document independently: `e_q = Encoder(q)`, `e_d = Encoder(d)`
- Score: `cosine(e_q, e_d)` — cheap at query time
- Document embeddings precomputed offline → ANN search
- Can't model interaction between query and document tokens
- Recall-oriented: good at finding candidates

**Cross-encoder (used for reranking):**
- Concatenate query + document: `score = BERT([CLS] q [SEP] d)`
- Full attention between all query and document tokens
- Can model "query asks for X but doc says NOT X" — handles negation, nuance
- Must be run at query time for each candidate — too slow for full corpus
- Precision-oriented: good at scoring a small candidate set

**Two-stage pipeline (production standard):**
```
Bi-encoder ANN → top-100 candidates → Cross-encoder → top-5 → LLM
```

---

## L3 — Applied

### Q9. Design a RAG system for a financial services company (like Capital One) that needs to answer compliance questions from a corpus of 10,000 regulatory documents.

**A:**

**Requirements analysis:**
- High precision (wrong compliance advice = regulatory risk)
- Citation mandatory
- Documents: PDFs with tables, regulatory structure
- Users: compliance officers (not consumers) — can handle complex answers
- SLA: <3 seconds

**Architecture:**

*Ingestion:*
- Parse PDFs with structure-aware parser (PyMuPDF + table extraction)
- Chunk by regulatory section, not fixed-size
- Embed with `BAAI/bge-large-en` (strong domain performance)
- Store: Qdrant (dense) + Elasticsearch (BM25) — hybrid search is critical for reg codes
- Metadata: document name, section, effective date, jurisdiction

*Query:*
- Query classification: is this about a specific regulation, or general policy?
- For specific reg queries: BM25 gets higher weight (exact code matching)
- Retrieve top-50 → Rerank with cross-encoder → top-5
- Prompt: system prompt enforces "only answer from provided context, cite source"

*Generation:*
- GPT-4 or Claude Opus — best faithfulness scores
- Response format: answer + numbered citations [1][2]

*Evaluation:*
- Golden dataset: 300 compliance Q&A pairs written by compliance team
- RAGAS faithfulness > 0.9 required (any hallucination is dangerous)
- Weekly regression: flag any answer where faithfulness < 0.85

*Safety:*
- "I don't know" threshold: if max context precision < 0.6, return "insufficient context found"
- All responses prefixed: "Based on retrieved regulatory documents as of [date]"
- Log all queries and responses for audit trail

---

### Q10. Your RAG system has high retrieval recall but users report answers are wrong. What do you investigate?

**A:** High recall but poor answers → generation or precision problem. Investigation order:

1. **Context precision**: Are the retrieved chunks actually relevant, or are they noisy? Low precision = LLM gets confused by irrelevant context.

2. **Lost in the middle**: Is the right chunk in position 1-2 or position 30-40 in the context? LLMs attend less to middle context. Rerank more aggressively, keep top-3 not top-10.

3. **Faithfulness**: Is the LLM ignoring context and hallucinating? Test by asking "is your answer in the provided context?" with the context. If faithfulness is low, strengthen the system prompt ("ONLY use the provided context").

4. **Chunk quality**: Does each chunk contain a complete thought? If chunks are mid-sentence or mid-table, the LLM can't make sense of them.

5. **Answer vs question mismatch**: Is the LLM answering the question asked? Check answer relevancy in RAGAS.

6. **Model temperature**: High temperature → less faithful. For Q&A, use temperature=0.

---
