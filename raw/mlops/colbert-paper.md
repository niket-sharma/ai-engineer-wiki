# ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction

**Paper:** "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction"
**Authors:** Omar Khattab, Matei Zaharia (Stanford)
**Published:** 2020-04-24
**arXiv ID:** 2004.12832
**Venue:** SIGIR 2020

Also covers: ColBERT v2 (2021, arXiv 2112.01488) and PLAID indexing (2022)

---

## The Retrieval Trade-off Problem

```
Bi-encoder (DPR):
  - Single vector per document → precomputable offline
  - Fast ANN search at query time
  - Can't model token-level query-document interaction
  - Quality: moderate

Cross-encoder (BM25+BERT reranker):
  - Full attention over concatenated [CLS] Q [SEP] D [SEP]
  - Highest quality — sees full interaction
  - Must score each (Q,D) pair → O(N) at query time → unusable for full corpus
  - Only feasible for reranking top-100 candidates

ColBERT: The Middle Ground
  - Token-level vectors per document → precomputable offline
  - Late interaction: query tokens meet document tokens at query time
  - Sublinear effective search via MaxSim
  - Quality: better than bi-encoder, close to cross-encoder
```

---

## ColBERT Architecture

### Encoding

Both query and document use the **same BERT model** (shared weights) but with different prepended tokens:

```
Query:    [Q] [CLS] query tokens [MASK] [MASK] ... [MASK]
Document: [D] [CLS] document tokens
```

The `[Q]` / `[D]` tokens are learned embeddings that signal the mode to the model.

**Query padding:** Queries are padded with `[MASK]` tokens to a fixed length (e.g., 32 tokens). The `[MASK]` tokens learn to act as "augmented query terms" — soft expansion of the query.

**Output embeddings:**
```python
# Each token gets its own embedding (not just [CLS])
q_embeddings = bert("[Q] " + query)          # shape: [n_q, dim]
d_embeddings = bert("[D] " + document)       # shape: [n_d, dim]

# Dimensionality reduction via linear layer
q_embeddings = linear(q_embeddings)          # shape: [n_q, 128]
d_embeddings = linear(d_embeddings)          # shape: [n_d, 128]

# L2 normalize
q_embeddings = normalize(q_embeddings)       # each vector has unit norm
d_embeddings = normalize(d_embeddings)       # each vector has unit norm
```

---

## Late Interaction Scoring: MaxSim

The ColBERT relevance score:

```
score(q, d) = Σ_{i=1}^{n_q}  max_{j=1}^{n_d}  (q_i · d_j)
```

For each query token embedding q_i, find the document token embedding d_j that is most similar to it (max dot product). Sum these maximum similarities over all query tokens.

**Why this captures meaning:**
- Query: "What are the side effects of aspirin?"
  - Token "side": maximally matches document token "side" in "side effects"
  - Token "effects": maximally matches "effects"
  - Token "aspirin": matches "aspirin"
  - Token "What": likely matches common words (low signal, low score)

- Document A: "Aspirin is effective for pain relief"
  - "side" → best match might be "effective" (low similarity, no "side")
  - Score: low

- Document B: "Aspirin's side effects include stomach bleeding"
  - "side" → matches "side" (high similarity)
  - "effects" → matches "effects" (high similarity)
  - Score: high

This granularity is impossible with a single bi-encoder vector.

---

## Computing MaxSim Efficiently

Naive MaxSim: O(n_q × n_d) per pair — still linear in document length.

**PLAID (2022):** Production ColBERT uses an approximate retrieval pipeline:

```
1. Centroid lookup: all document token embeddings assigned to one of 2^16 centroids
   At query time: find centroids closest to each query token (fast ANN)

2. Candidate generation: collect all documents that have tokens in the top-K centroids

3. Decompression: decompress candidate document embeddings (stored in compressed form)

4. MaxSim refinement: compute exact MaxSim scores for candidates

5. Return top-k documents
```

Storage: document token embeddings are stored as:
- Centroid IDs (2 bytes per token for 2^16 centroids)
- Residuals (4 bytes per token, compressed)

vs naive: 128 dims × 4 bytes = 512 bytes per token

---

## Index Size

| Representation | Size per 1M passages (avg 50 tokens) |
|---|---|
| BM25 inverted index | ~1 GB |
| Bi-encoder (single vector) | 512 dims × 4B × 1M = 2 GB |
| ColBERT (PLAID compressed) | ~10–20 GB |
| ColBERT (uncompressed) | 50 tokens × 128 dims × 4B × 1M = 25.6 GB |
| Cross-encoder | model weights only (no precomputed index) |

ColBERT requires ~10× more storage than bi-encoder, but enables much better quality.

---

## ColBERT v2 (2021)

The original ColBERT was trained with in-batch negatives only (easy negatives). ColBERT v2 adds:

1. **Hard negatives from cross-encoder:** Use a cross-encoder to re-rank candidates and collect hard negatives (documents that look similar to relevant ones but aren't)

2. **Residual compression:** Instead of storing full token embeddings, store centroid index + small residual (better compression, less quality loss)

3. **Results:** 10–15% quality improvement over ColBERT v1 with smaller index

---

## Training ColBERT

```python
from colbert import Trainer

trainer = Trainer(
    triples="data/triples.train.small.tsv",  # (query, positive, negative) triples
    queries="data/queries.train.tsv",
    collection="data/collection.tsv",
    config=ColBERTConfig(
        doc_maxlen=180,       # max document tokens
        query_maxlen=32,      # fixed query length (padded with MASK)
        dim=128,              # embedding dimension after projection
        nway=64,              # in-batch negatives per query
        lr=3e-6,
        bsize=32,
    )
)

trainer.train(checkpoint="colbert-ir/colbertv2.0")
```

---

## ColBERT vs Cross-Encoder Practical Comparison

| | Cross-Encoder | ColBERT |
|---|---|---|
| Offline precompute | Not possible | Full document embeddings |
| Query-time cost | O(n_docs) × BERT | O(n_q × n_d) MaxSim (fast) |
| Feasible corpus size | Top-100 reranking | Full corpus (millions) |
| Quality vs bi-encoder | +10–20% | +5–10% |
| Storage overhead | None | ~10× bi-encoder |
| Common usage | Stage 2 reranker | Stage 1 retriever or combined |

---

## DSPy and ColBERT

ColBERT is deeply integrated with **DSPy** (also by Khattab):
```python
import dspy
from dspy.retrieve.colbert_rm import ColBERTv2RM

colbert = ColBERTv2RM(url="http://my-colbert-server:8893")
dspy.settings.configure(rm=colbert)

# DSPy automatically uses ColBERT for retrieval in pipelines
```

---

## Common Interview Questions

- "Explain ColBERT's late interaction mechanism."
- "What is the MaxSim scoring formula and why does it capture meaning better than dot product of single vectors?"
- "How does ColBERT differ from a bi-encoder and a cross-encoder?"
- "What is the PLAID index and why is it needed for production ColBERT?"
- "When would you use ColBERT vs a cross-encoder reranker?"
- "Why does ColBERT pad query tokens with [MASK]?"
