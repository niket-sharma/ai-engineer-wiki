---
title: "Reranking"
aliases: ["cross-encoder reranking", "ColBERT", "reranker", "two-stage retrieval"]
tags: [rag, retrieval]
related: ["[[rag-systems]]", "[[vector-databases]]"]
sources: ["training-knowledge"]
interview_relevance: medium
last_updated: 2026-04-22
status: current
---

# Reranking

## TL;DR
Reranking is a second-stage retrieval step where a more powerful but slower model re-scores the top-K candidates from initial retrieval. Bi-encoders (used in ANN search) embed query and document independently — fast but miss fine-grained interactions. Cross-encoders attend to both query and document jointly — slower but much more accurate. Two-stage retrieval is the standard architecture for production RAG.

## Intuition
**Stage 1 (Bi-encoder / ANN):** "Give me the 100 most relevant documents, fast."
Embed query once, search vector index — O(log n). Fast. But bi-encoders can't model the interaction between query and document deeply.

**Stage 2 (Cross-encoder / Reranker):** "From those 100, give me the top 5, accurately."
Feed (query, document) pairs to a cross-encoder and score each jointly. The model can attend to relationships between query and document tokens. Much more accurate. But O(K) forward passes — you can only afford it for a small K.

The key insight: you get the recall of fast retrieval with the precision of slow reranking.

## Technical Detail

**Bi-encoder:**
```
score(q, d) = embed(q) · embed(d)    # cosine/dot similarity
```
- Query and document are encoded independently
- Can precompute document embeddings offline
- Fast at query time: one embedding + ANN search
- Misses: "query asks for X but document says NOT X" — can't model negation

**Cross-encoder:**
```
score(q, d) = BERT([CLS] q [SEP] d [SEP]) → linear → scalar
```
- Query and document are concatenated, run through full transformer
- All attention layers see both query and document simultaneously
- 10–100× slower than bi-encoder but significantly more accurate
- Can't precompute — must run at query time per candidate

**Typical pipeline:**
```
Query → Bi-encoder ANN search → top-100 candidates
      → Cross-encoder reranker → top-5 results
      → LLM generation with top-5 in context
```

**ColBERT (Late Interaction):**
A middle ground between bi-encoder and cross-encoder:
```
score(q, d) = Σ_i max_j (q_i · d_j)
```
- Encode query and document independently INTO sequences of vectors (not single vectors)
- Compute max similarity between each query token and all document tokens
- Much more expressive than bi-encoder, much faster than cross-encoder
- Requires storing per-token document embeddings (~100× more storage)
- Used in: RAGatouille library, production retrieval at scale

**Reranker models:**
- `cross-encoder/ms-marco-MiniLM-L-6-v2` — fast, small, decent quality
- `BAAI/bge-reranker-v2-m3` — multilingual, high quality
- Cohere Rerank API — managed, high quality, latency ~50–100ms
- Jina Reranker — open source alternative

## Tradeoffs
| Advantage | Disadvantage |
|---|---|
| Significantly improves precision of top results | Adds latency (50–200ms for 100 candidates) |
| Catches cases where bi-encoder retrieval misordered | Must limit to top-K candidates from stage 1 |
| Can model query-document interaction deeply | Cross-encoder can't precompute → no offline speedup |
| ColBERT: good tradeoff (mid-speed, mid-accuracy) | ColBERT requires storing token-level embeddings |

##  Angles

**What interviewers are really testing:**
- Do you understand the bi-encoder / cross-encoder distinction?
- Can you explain why two-stage retrieval is the standard?
- Do you know ColBERT and when to use it?
- Can you estimate the latency cost of reranking?

**Common follow-up questions:**
- "What's the difference between a bi-encoder and a cross-encoder?"
- "Why not just use a cross-encoder for all retrieval?"
- "How does reranking affect the end-to-end latency of a RAG system?"
- "What is ColBERT and where does it fit between bi-encoder and cross-encoder?"
- "How would you evaluate whether your reranker is actually helping?"

**Gotchas / misconceptions:**
- The cross-encoder is NOT an embedding model — it outputs a scalar, not a vector
- Reranking only helps if the initial retrieval has recall — if the right doc isn't in top-100, reranking can't find it
- LLM-as-reranker (using GPT to score candidates) works but is expensive and slow

## Connections
- [[rag-systems]] — reranking is a standard post-retrieval quality improvement layer
- [[vector-databases]] — reranking re-scores the candidates returned by ANN search

## Sources
- Training knowledge (Nogueira & Cho 2019 cross-encoder for reranking; Khattab & Zaharia 2020 "ColBERT")
