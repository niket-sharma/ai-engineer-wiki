---
title: "Vector Databases"
aliases: ["vector DB", "vector store", "ANN", "FAISS", "Pinecone", "Weaviate", "pgvector", "HNSW"]
tags: [rag, retrieval, infrastructure]
related: ["[[rag-systems]]", "[[reranking]]"]
sources: ["training-knowledge"]
interview_relevance: high
last_updated: 2026-04-22
status: current
---

# Vector Databases

## TL;DR
Vector databases store high-dimensional embeddings and enable approximate nearest-neighbor (ANN) search — find the K vectors most similar to a query vector in milliseconds across millions of documents. They're the retrieval backbone for RAG, semantic search, and recommendation systems. Key  topics: HNSW vs IVF-PQ, filtering, hybrid search, managed vs self-hosted.

## Intuition
Embedding models map text/images/etc. to dense vectors where semantic similarity ≈ geometric proximity. Finding exact nearest neighbors in high-dimensional space is O(n·d) per query — too slow for large datasets. ANN algorithms trade a small amount of accuracy for orders-of-magnitude speed improvement.

## Technical Detail

**Key ANN Algorithms:**

**HNSW (Hierarchical Navigable Small World):**
- Builds a multi-layer graph where each node is connected to its approximate neighbors
- Search: start at top layer (few nodes), greedily navigate to query, descend layers
- Complexity: O(log n) search, O(n log n) build
- High accuracy, fast queries, but high memory (stores the graph)
- Default in: Weaviate, Qdrant, pgvector (partial)

**IVF-PQ (Inverted File + Product Quantization):**
- IVF: cluster vectors into nlist centroids, search only nearby clusters (nprobe of them)
- PQ: compress each vector to a short code using sub-quantizers — reduces memory 4–32×
- Complexity: O(nprobe · cluster_size) search
- Good for very large datasets where memory is tight (billions of vectors)
- Default in FAISS

**Flat (Brute Force):**
- Exact search, O(n·d) — only feasible for small datasets (<100k vectors)
- Used as the ground truth for ANN benchmarks

**Distance metrics:**
| Metric | Formula | Use when |
|---|---|---|
| Cosine similarity | (a·b)/(‖a‖‖b‖) | Embeddings (magnitude-normalized) |
| Dot product | a·b | When vectors are normalized (= cosine) |
| Euclidean (L2) | ‖a-b‖ | Image embeddings, some multi-modal |

For text embeddings, cosine ≈ normalized dot product. Most libraries use dot product internally after L2 normalization.

**Filtering:**
Pre-filtering: apply metadata filter before ANN search → reduces candidate set → may miss good results
Post-filtering: ANN search first, then filter → fast but may return fewer than K results
Hybrid: partition index by filter values (e.g., separate index per tenant) → best for high-cardinality filters

**Hybrid Search (BM25 + Dense):**
Combine sparse (BM25 keyword) + dense (vector) retrieval scores:
```
RRF(d) = Σ_r 1 / (k + rank_r(d))    # Reciprocal Rank Fusion
```
RRF is rank-based — doesn't require score normalization. k=60 is standard.

## Vector DB Comparison

| System | Type | Index | Filtering | Scale | Best for |
|---|---|---|---|---|---|
| FAISS | Library (no server) | IVF-PQ, HNSW, Flat | Manual | 100M+ | Research, custom pipelines |
| Pinecone | Managed | HNSW-like | Metadata | Billions | Startup/fast setup |
| Weaviate | Self-hosted/managed | HNSW | Rich schema | 100M+ | Hybrid search, GraphQL |
| Qdrant | Self-hosted/managed | HNSW | Payload filter | 100M+ | Performance, Rust-based |
| Chroma | Self-hosted, local | HNSW | Metadata | <10M | Dev/prototyping |
| pgvector | Postgres extension | HNSW/IVF | SQL | <10M | Existing Postgres stack |
| Milvus | Self-hosted | HNSW, IVF-PQ | Scalar | Billions | Large-scale enterprise |

## Tradeoffs
| Advantage | Disadvantage |
|---|---|
| Sub-millisecond ANN search at millions of vectors | ANN is approximate — small recall loss |
| Combines storage + search + metadata filtering | HNSW memory-intensive for very large datasets |
| Hybrid search (dense + sparse) in one system | Consistency/ACID not a priority (vs RDBMS) |
| Cloud-managed options for easy scaling | Filtering at scale is still complex |

##  Angles

**What interviewers are really testing:**
- Do you understand HNSW vs IVF-PQ tradeoffs?
- Can you explain hybrid search and RRF?
- Do you know when to use a managed service vs FAISS?
- Do you understand the filtering problem in vector DBs?

**Common follow-up questions:**
- "Walk me through how HNSW works."
- "When would you use IVF-PQ instead of HNSW?"
- "How does RRF combine dense and sparse scores?"
- "If I have 1 billion product embeddings, which vector DB and index would you recommend?"
- "What's the filtering problem in ANN search and how do you solve it?"
- "How would you handle multi-tenant isolation in a vector database?"

**Gotchas / misconceptions:**
- FAISS is a library, not a database — no persistence, no server, no replication
- Cosine similarity and normalized dot product are mathematically identical — choose by API convention
- pgvector HNSW has much lower recall than dedicated vector DBs at large scale
- "nprobe" in IVF controls the recall-speed tradeoff — always tune it, not just nlist

## Connections
- [[rag-systems]] — vector DB is the retrieval backbone for RAG
- [[reranking]] — operates on the candidates returned by vector DB search

## Sources
- Training knowledge (Johnson et al. 2017 "FAISS"; Malkov & Yashunin 2018 "HNSW"; ANN-Benchmarks)
