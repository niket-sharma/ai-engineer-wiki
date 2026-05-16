# Vector Database Comparison Notes

**Compiled from:** FAISS paper (Johnson et al. 2017), HNSW paper (Malkov & Yashunin 2018), ANN-Benchmarks, vendor documentation, and production experience notes.

---

## The Problem: Approximate Nearest Neighbor (ANN) Search

**Exact k-NN search:** Given query vector q and database of N vectors, find the k vectors most similar to q.
- Brute force: O(N·d) per query — 100ms for 1M vectors × 1024 dims = too slow
- Exact solutions scale poorly: O(N) regardless of index structure

**ANN:** Trade a small amount of recall for orders-of-magnitude speed improvement.

**Key metrics:**
- **Recall@k**: what fraction of true top-k are in the returned top-k (target: > 0.95 for most apps)
- **Queries per second (QPS)**: throughput
- **Index build time**: how long to build the index
- **Memory**: how much RAM/GPU memory the index uses

---

## ANN Index Types

### HNSW (Hierarchical Navigable Small World)

**Paper:** Malkov & Yashunin (2016/2018)

**Data structure:** Multi-layer proximity graph
- Layer 0: all vectors, each connected to M neighbors
- Layer 1: subset of vectors, each connected to M neighbors
- Layer 2, 3, ...: increasingly sparse layers (top layer has ~1 vector)

**Search algorithm:**
```
1. Enter at a node in the topmost layer
2. Greedily navigate to the query's nearest neighbor in this layer
3. Drop to the next layer, repeat from where you ended
4. In layer 0: explore ef_search candidates, return top-k
```

**Key hyperparameters:**
- `M`: number of connections per node (controls index quality/size). Default: 16–64
- `ef_construction`: search breadth during index building. Higher = better index, slower build. Default: 100–200
- `ef_search` (runtime): search breadth during query. Higher = better recall, slower. Default: 64

**Complexity:**
- Build: O(N · M · log N) — nodes are inserted one at a time
- Search: O(log N) — navigating the hierarchy is log N
- Memory: O(N · M · d · bytes_per_float) — stores graph + vectors

**When to use HNSW:**
- High recall required (> 0.99)
- Frequent inserts (HNSW supports incremental insertion)
- Memory available (stores full vectors + graph links)
- Used in: Qdrant, Weaviate, pgvector, Elasticsearch kNN

**Trade-off:** Memory-hungry. 1M vectors × 1024 dims × 4 bytes + graph ≈ 4 GB + 1.5 GB for graph ≈ 5.5 GB

---

### IVF (Inverted File Index)

**Concept:** Partition vectors into nlist clusters via k-means. At query time, search only the nprobe nearest clusters.

**Build:**
1. k-means clustering: assign each of N vectors to one of nlist centroids
2. Build an "inverted list" for each centroid: which vectors belong to it

**Search:**
1. Find nprobe nearest centroids to the query
2. Search all vectors in those clusters
3. Return top-k from those candidates

**Key hyperparameters:**
- `nlist`: number of clusters. Rule of thumb: sqrt(N) to 4·sqrt(N)
- `nprobe`: number of clusters searched at query time. More → higher recall, slower

**Complexity:**
- Build: O(N · d · iterations) for k-means (typically 20 iterations)
- Search: O(nprobe · (N/nlist) · d) — search within clusters
- Memory: O(N · d) — just vectors + inverted lists (no graph)

**IVF-PQ:** Add Product Quantization to compress vectors:
- Split each d-dim vector into M sub-vectors of d/M dims
- Quantize each sub-vector to one of 256 centroids (8-bit code)
- Storage: M bytes per vector instead of d×4 bytes
- 1024-dim FP32 = 4096 bytes → IVF-PQ (M=64) = 64 bytes = 64× compression
- At the cost of ~1–2% recall loss

**When to use IVF-PQ:**
- Very large scale (100M–1B+ vectors) where HNSW memory is prohibitive
- Memory constrained (GPU or RAM limited)
- Can tolerate slightly lower recall (0.95 vs 0.99)
- Used in: FAISS (the primary GPU-accelerated option)

---

### Flat (Brute Force)

- Exact search: guaranteed 100% recall
- O(N·d) per query
- Only use for: < 100k vectors, or as ground truth for benchmarking
- FAISS IndexFlatL2 / IndexFlatIP

---

## Vector Database Comparison

### FAISS (Facebook AI Similarity Search)

- **Type:** Library (not a database — no server, no persistence, no API)
- **Indexes:** Flat, IVF, IVF-PQ, HNSW, ScaNN, and combinations
- **GPU support:** Yes — IVF on GPU is 5–10× faster than CPU
- **Language:** C++ with Python bindings
- **Scale:** Billions of vectors with IVF-PQ on GPU
- **Filtering:** None — you do it yourself in application code
- **When:** Research, custom pipelines, highest performance, own infrastructure

```python
import faiss
import numpy as np

d = 1024  # dimension
index = faiss.IndexHNSWFlat(d, 32)  # M=32
index.hnsw.efConstruction = 200
index.add(vectors.astype('float32'))  # N × d array

# Search
index.hnsw.efSearch = 64
D, I = index.search(query.astype('float32'), k=10)  # D=distances, I=indices
```

---

### Pinecone

- **Type:** Managed cloud service (no self-hosting option)
- **Index:** Proprietary (HNSW-based with custom optimizations)
- **Scale:** Billions of vectors
- **Filtering:** Metadata filtering (sparse + dense hybrid in their Hybrid endpoint)
- **Latency:** ~10–50ms for most queries
- **Price:** Starts free; $70/month for ~1M vectors hosted
- **When:** Startup, prototype, don't want to manage infrastructure, fast setup

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="...")
pc.create_index("my-index", dimension=1024, metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index("my-index")

# Upsert
index.upsert(vectors=[("id1", embedding, {"source": "doc.pdf"})])

# Query with filter
results = index.query(vector=query_embedding, top_k=10,
                      filter={"source": {"$eq": "doc.pdf"}})
```

---

### Qdrant

- **Type:** Self-hosted (Rust) or managed cloud
- **Index:** HNSW
- **Scale:** 100M+ vectors (tested), billions with sharding
- **Filtering:** Rich payload filtering (JSON schema)
- **Performance:** Best throughput/latency among self-hosted options (Rust implementation)
- **Features:** Named vectors (multiple embeddings per point), sparse vector support, scalar quantization
- **When:** Production self-hosted, cost-conscious, high performance needed

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient("localhost", port=6333)
client.create_collection("docs", vectors_config=VectorParams(size=1024, distance=Distance.COSINE))

# Upsert with payload
client.upsert("docs", points=[PointStruct(id=1, vector=embedding,
              payload={"source": "annual_report.pdf", "year": 2024})])

# Search with filter
results = client.search("docs", query_vector=query_embedding, limit=10,
                        query_filter=Filter(must=[FieldCondition(key="year", match=MatchValue(value=2024))]))
```

---

### Weaviate

- **Type:** Self-hosted or managed
- **Index:** HNSW (default), FLAT (for small collections)
- **Hybrid search:** Built-in BM25 + dense (with native RRF)
- **Schema:** GraphQL-based schema, class-property model
- **Features:** Generative modules (OpenAI, Cohere), question-answering modules built-in
- **When:** Need built-in hybrid search, GraphQL API preferred

---

### pgvector (PostgreSQL Extension)

- **Type:** Postgres extension — runs in existing Postgres cluster
- **Index:** HNSW (pgvector 0.5+), IVF (older versions)
- **Scale:** Best for < 5M vectors; degrades at larger scale vs dedicated DBs
- **Filtering:** Full SQL — arbitrary complex filters, joins, etc.
- **Consistency:** Full ACID (unlike dedicated vector DBs)
- **When:** Already on Postgres, small-medium scale, need SQL joins, transactional consistency

```sql
CREATE EXTENSION vector;
CREATE TABLE documents (id BIGSERIAL, embedding vector(1024), content TEXT);
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Search
SELECT id, content, embedding <=> '[0.1, 0.2, ...]'::vector AS distance
FROM documents
ORDER BY distance LIMIT 10;
```

---

### Chroma

- **Type:** Open-source, self-hosted, embedded or client-server
- **Index:** HNSW (via hnswlib)
- **Scale:** < 1M vectors in practice
- **Filtering:** Metadata filtering
- **When:** Development, local prototyping, small-scale production

---

## Decision Matrix

| | FAISS | Pinecone | Qdrant | Weaviate | pgvector | Chroma |
|---|---|---|---|---|---|---|
| **Ease of setup** | ● | ●●● | ●● | ●● | ●●● | ●●● |
| **Scale** | ●●●● | ●●●● | ●●● | ●●● | ●● | ● |
| **Performance** | ●●●● | ●●● | ●●●● | ●●● | ●● | ●● |
| **Filtering** | ● (DIY) | ●●● | ●●●● | ●●● | ●●●● | ●● |
| **Cost** | Free | $$$$ | Free/$ | Free/$ | Free | Free |
| **Hybrid search** | ● (DIY) | ●●● | ●●● | ●●●● | ●● | ● |

**Recommended by use case:**
- MVP/startup: **Pinecone** (zero ops) or **Chroma** (local dev)
- Production self-hosted, < 100M vectors: **Qdrant**
- Production, need hybrid search: **Weaviate** or **Qdrant**
- Existing Postgres, < 5M vectors: **pgvector**
- Research, custom, billions of vectors: **FAISS**
- Enterprise, managed, billions: **Pinecone** or **Milvus**

---

## Common  Questions

- "Walk me through how HNSW works."
- "When would you use IVF-PQ instead of HNSW?"
- "How do you handle metadata filtering in a vector database?"
- "Compare Pinecone, Qdrant, and pgvector. When would you use each?"
- "How does hybrid search work in vector databases? What is RRF?"
- "You have 500M product embeddings — which vector database and index type?"
