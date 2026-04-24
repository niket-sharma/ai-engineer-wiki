# Reranking in Retrieval: Cross-Encoders, ColBERT, and Two-Stage Pipelines

**Sources:**
- Nogueira & Cho (2019) "Passage Re-ranking with BERT" — original cross-encoder reranking paper
- Khattab & Zaharia (2020) "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction" — arXiv 2004.12832
- Cohere Rerank API documentation

---

## Why Reranking Exists: The Bi-Encoder Problem

**Bi-encoders (used in ANN retrieval):**
```
score(q, d) = embed(q) · embed(d)    # pre-computed dot product
```

Each document is encoded independently once, stored in the index. At query time: embed query → ANN search → fast.

**The fundamental problem:** Bi-encoders can't model the *interaction* between query and document tokens. The query "What are the side effects of aspirin?" and the document "Aspirin is effective for pain relief" will be close in embedding space because they're both about aspirin — even though the document doesn't answer the question about side effects.

**Cross-encoders (used in reranking):**
```
score(q, d) = BERT([CLS] q [SEP] d [SEP]) → linear → scalar
```

All transformer layers see both query and document simultaneously. Full attention over the concatenated sequence. The model can distinguish: "this document is *about* aspirin but doesn't discuss *side effects*."

**The catch:** Cross-encoders require a forward pass per (query, document) pair — can't precompute. Too slow for full-corpus retrieval, but fast enough for a small candidate set.

---

## Two-Stage Retrieval Architecture

```
Stage 1 — Recall: Bi-encoder ANN search
  Query → embed → HNSW search → top-100 candidates
  Speed: ~10ms
  Goal: don't miss anything relevant (high recall)

Stage 2 — Precision: Cross-encoder reranking  
  100 candidates → cross-encoder score each → top-5
  Speed: ~100–300ms
  Goal: correctly order the best candidates (high precision)

Generation: top-5 candidates → LLM → answer
```

**Why the split?**
- Stage 1 is fast but imprecise: designed for recall (don't miss anything)
- Stage 2 is accurate but slow: designed for precision (get the ordering right)
- Together: recall of fast retrieval + precision of slow scoring

---

## Cross-Encoder Architecture and Training

### Architecture

```
Input: "[CLS] question tokens [SEP] passage tokens [SEP]"
       → BERT-base/large → CLS embedding → linear(768, 1) → relevance score
```

The model outputs a single scalar relevance score for each (query, doc) pair.

### Training Data

MS MARCO Passage Ranking: 1M pairs (query, relevant_passage) + BM25 hard negatives.

Training signal:
- In-batch negatives: pairs from the same batch that aren't labeled as relevant
- Hard negatives: BM25 top results that are NOT relevant (harder to distinguish from relevant passages)

Loss: Binary cross-entropy on relevant/non-relevant labels.

### Popular Cross-Encoder Models

| Model | Size | Speed | Quality | When |
|---|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | Fast | Good | Production latency-constrained |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 33M | Medium | Better | General production |
| `BAAI/bge-reranker-large` | 560M | Slow | Best open | Accuracy-critical |
| `BAAI/bge-reranker-v2-m3` | varies | Medium | Multilingual | Non-English |
| Cohere Rerank API | managed | ~50ms | Very high | Managed service |
| Jina Reranker | managed | varies | High | Open source alternative |

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Score (query, passage) pairs
scores = model.predict([
    ("What are aspirin's side effects?", passage_1),
    ("What are aspirin's side effects?", passage_2),
    ...
])

# Rerank
ranked = sorted(zip(scores, passages), reverse=True)
top_5 = [passage for score, passage in ranked[:5]]
```

---

## ColBERT: Late Interaction

**Paper:** Khattab & Zaharia, SIGIR 2020 — arXiv 2004.12832

### The Middle Ground

```
Bi-encoder:       single vector per document  → fast, imprecise
Cross-encoder:    full cross-attention         → slow, precise  
ColBERT:          token-level vectors          → middle ground
```

### Architecture

```python
# Query encoding: output ONE vector per query token
q_embeddings = bert(query)  # [n_q, 128] — n_q tokens, 128 dims each

# Document encoding: output ONE vector per document token
d_embeddings = bert(document)  # [n_d, 128] — n_d tokens, 128 dims each
```

### Late Interaction Scoring

```
score(q, d) = Σ_{i=1}^{n_q}  max_{j=1}^{n_d}  (q_i · d_j)
```

For each query token q_i, find the document token d_j with the highest similarity. Sum these maximum similarities across all query tokens.

**Why this works:**
- "What are the side effects?" has tokens: [What, are, the, side, effects, ?]
- The "side" query token will maximally match "side effects" in the document
- The "effects" query token will match "effects" in the document
- Aggregating: document must contain representations matching each query concept

**vs. Bi-encoder:** The document "aspirin is effective for pain" won't match "side effects" well because it has no "side" or "effects" tokens.

**vs. Cross-encoder:** No joint attention — queries and documents are encoded independently. Can precompute document token embeddings offline.

### ColBERT Index Structure

Documents are encoded to token-level embeddings stored in a "PLAID" index:
- 1M passages × avg 50 tokens × 128 dims × 2 bytes = ~12 GB per 1M passages
- 100× more storage than bi-encoder (which stores 1 vector per document)

At query time: only the query is encoded live. MaxSim computation is vectorized with FAISS for efficiency.

### When to Use ColBERT vs Cross-Encoder

| | ColBERT | Cross-Encoder |
|---|---|---|
| **Precomputable** | Yes (document token embeddings) | No |
| **Latency at 100 candidates** | ~5ms | ~100ms |
| **Latency at 10k candidates** | ~50ms | Too slow |
| **Quality vs bi-encoder** | +5–10% recall | +10–20% precision |
| **Storage** | 100× bi-encoder | Minimal (just model) |
| **When** | High-scale, latency-sensitive | Smaller candidate set |

**Practical guidance:**
- For < 1000 candidates: cross-encoder is better quality and not that slow
- For 1000–100k candidates: ColBERT
- For 100k+ candidates: bi-encoder only (or ColBERT with aggressive pruning)

---

## LLM-as-Reranker

Use an LLM to score relevance:

```python
prompt = """
Score the relevance of the following passage to the query on a scale of 0-10.

Query: {query}
Passage: {passage}

Return only a number from 0 to 10.
"""

for passage in candidates:
    score = int(llm.invoke(prompt.format(query=query, passage=passage)).content)
```

**Pros:** Highest quality (LLM understands nuance, can reason about relevance)
**Cons:** Very expensive (1 LLM call per candidate), high latency

**When:** Only for offline evaluation, or when > 1 second latency is acceptable and quality is critical. Not practical for real-time RAG.

---

## Latency Budget Planning

Typical RAG pipeline latency breakdown:

| Stage | Time | Notes |
|---|---|---|
| Query embedding | 10ms | GPU inference, or ~50ms on CPU |
| ANN search (HNSW, 1M vectors) | 5ms | In-memory |
| Cross-encoder rerank (100 → 5) | 100–300ms | Biggest variable |
| LLM generation (first token) | 200–500ms | Depends on model and serving |
| **Total (target)** | **<800ms** | For p95 interactive SLA |

**If latency is tight:** Use MiniLM cross-encoder (22M params, fast) or skip reranking for initial deployment.

**If quality is critical:** Use BGE-reranker-large or Cohere Rerank, accept ~300ms reranking latency.

---

## Common Interview Questions

- "What's the difference between a bi-encoder and a cross-encoder?"
- "Why use a two-stage retrieval pipeline? Why not just use a cross-encoder for everything?"
- "What is ColBERT and how does it differ from both bi-encoders and cross-encoders?"
- "How do you estimate the latency of adding a reranker to your RAG pipeline?"
- "When would you use an LLM as a reranker?"
- "How do you evaluate whether your reranker is actually improving end-to-end quality?"
