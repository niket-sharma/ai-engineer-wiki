---
title: Cohere AI Engineering
aliases: [Cohere, Command, Embed, Rerank, RAG]
tags: [company, llm, enterprise-ai, rag, embeddings, reranking]
related: [rag-systems, colbert-paper, transformer-architecture, llm-serving-infra]
sources: [training-knowledge, cohere-blog, cohere-docs]
relevance: 7
last_updated: 2025-01-15
status: current
---

# Cohere AI Engineering

## Company Context

Cohere ($5B valuation) is an enterprise-focused LLM company founded by Aidan Gomez (co-author of "Attention Is All You Need"), Ivan Zhang, and Nick Frosst (ex-Google Brain). Unlike OpenAI/Anthropic (consumer + enterprise), Cohere is **purely enterprise** — API access for businesses, cloud deployment, and on-prem deployment for regulated industries (finance, healthcare, defense).

**Key products:** Command R/R+ (enterprise LLM, RAG-optimized), Embed (multilingual embedding model), Rerank (cross-encoder re-ranking), Cohere Platform (cloud + on-prem deployment), Aya (multilingual model, 101 languages), Toolkit (RAG framework).

**Differentiation:** On-premises deployment capability (runs in customer's VPC or air-gapped environment), strong multilingual support, enterprise security (HIPAA, SOC2), RAG-native model design.

---

## What Cohere Engineers Work On

### 1. Command R: RAG-Optimized LLM

Command R is specifically designed for retrieval-augmented generation:

```python
import cohere

co = cohere.Client(api_key=COHERE_API_KEY)

# RAG with grounded generation (Command R's key feature)
documents = [
    {"title": "Q4 Earnings", "text": "Revenue grew 23% YoY to $4.2B..."},
    {"title": "Product Roadmap", "text": "H1 2025 will see launch of..."},
]

response = co.chat(
    model="command-r-plus",
    message="What was our revenue growth last quarter?",
    documents=documents,  # grounding documents
)

# Command R returns citations with source tracking
print(response.text)
print(response.citations)  # [{start, end, text, document_ids}]

# Grounded generation forces model to cite specific passages
# Reduces hallucination in enterprise settings (auditable outputs)
```

**Command R RAG-specific features:**
- **Grounded generation:** Model constrained to cite from provided documents
- **Citation tracking:** Each claim linked to specific document passage
- **Tool use:** Supports multi-step reasoning with external tool calls
- **Multi-hop RAG:** Can chain multiple retrievals within one response

### 2. Embed: Multilingual Embedding Model

Cohere Embed v3 is one of the best production embedding models:

```python
import cohere
import numpy as np

co = cohere.Client(api_key=COHERE_API_KEY)

# Embed documents (supports 100+ languages)
doc_embeddings = co.embed(
    texts=["The company reported strong earnings", "Die Firma berichtete starke Gewinne"],
    model="embed-multilingual-v3.0",
    input_type="search_document",  # optimized for storage/retrieval
).embeddings

# Embed queries (different input_type — matters for asymmetric retrieval)
query_embedding = co.embed(
    texts=["What were the earnings?"],
    model="embed-multilingual-v3.0",
    input_type="search_query",  # optimized for similarity computation
).embeddings[0]

# Similarity search
doc_embeddings_np = np.array(doc_embeddings)
query_np = np.array(query_embedding)
similarities = doc_embeddings_np @ query_np  # cosine similarity (embeddings L2-normalized)

# input_type matters: asymmetric retrieval (query ≠ document distribution)
# "search_query": short, keyword-like
# "search_document": longer, complete sentences
# "classification": for downstream classification tasks
# "clustering": for unsupervised clustering
```

**Embed v3 technical details:**
- Matryoshka representation learning: can truncate to 256/512/1024 dims without re-embedding
- Int8 quantization support: 4× storage reduction with minimal quality loss
- 1024-dimensional output (default)
- Trained with hard negatives from web-scale data

### 3. Rerank: Cross-Encoder Re-ranking

Rerank is a separate product for re-scoring retrieved documents:

```python
# Two-stage retrieval pattern (standard production RAG):
# Stage 1: ANN search (fast, approximate) — top-100 candidates
# Stage 2: Rerank (cross-encoder, slower but accurate) — top-5 for LLM

co = cohere.Client(api_key=COHERE_API_KEY)

# After initial vector search returns 100 documents:
results = co.rerank(
    model="rerank-english-v3.0",
    query="What is the company's revenue growth?",
    documents=initial_100_docs,
    top_n=5,  # keep top 5 for LLM context
    return_documents=True,
)

for result in results.results:
    print(f"Score: {result.relevance_score:.3f} | {result.document['text'][:100]}")

# Why reranking works better than pure embedding similarity:
# - Cross-encoder reads both query AND document together
# - Captures semantic nuance, negation, entity-level matching
# - Trade-off: O(n×d) computation vs O(d) for bi-encoder
```

**Bi-encoder vs Cross-encoder:**

| | Bi-encoder (Embed) | Cross-encoder (Rerank) |
|---|---|---|
| Architecture | Query + doc encoded separately | Query+doc encoded together |
| Index | Pre-computed, cached | On-the-fly per query |
| Speed | O(d) per query | O(n×d) per query |
| Accuracy | Good (approximate) | Better (exact interaction) |
| Use case | Stage 1: retrieve 100 | Stage 2: rerank to 5 |

### 4. On-Premises Deployment (Enterprise Differentiator)

Cohere's biggest enterprise differentiator is running in customer infrastructure:

```
Cohere deployment options:

1. Cohere Cloud (SaaS): api.cohere.com
   - Managed, easiest, lowest cost
   - Data processed by Cohere

2. Private Cloud (VPC): AWS/Azure/GCP private deployment
   - Runs in customer's cloud account
   - Cohere's software, customer's infrastructure
   - Data never leaves customer's cloud

3. On-premises: Air-gapped deployment
   - Customer's physical servers
   - Required by: defense, banking regulators, healthcare (HIPAA)
   - Cohere ships Docker containers + Kubernetes manifests

Engineering challenge: maintain feature parity across 3 deployment modes
while meeting strict version compatibility requirements.

Stack (estimated):
- Models: ONNX Runtime or TensorRT for inference
- API: FastAPI + uvicorn
- Kubernetes: Helm charts for k8s deployment
- Monitoring: Prometheus metrics (pluggable, no data to Cohere)
```

### 5. RAG Pipeline Architecture

Cohere's Toolkit is their reference RAG architecture:

```python
# Cohere Toolkit pattern (production RAG)
from typing import Any

class CohereRAGPipeline:
    def __init__(self):
        self.co = cohere.Client(COHERE_API_KEY)
        self.vector_store = initialize_vector_store()  # Weaviate, Qdrant, etc.

    def retrieve(self, query: str, top_k: int = 100) -> list[dict]:
        # Stage 1: Dense retrieval
        query_embedding = self.co.embed(
            texts=[query],
            model="embed-multilingual-v3.0",
            input_type="search_query",
        ).embeddings[0]
        
        candidates = self.vector_store.query(query_embedding, top_k=top_k)
        
        # Stage 2: Rerank
        reranked = self.co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=[c["text"] for c in candidates],
            top_n=5,
        )
        
        return [candidates[r.index] for r in reranked.results]

    def generate(self, query: str, docs: list[dict]) -> str:
        response = self.co.chat(
            model="command-r-plus",
            message=query,
            documents=[{"text": d["text"], "title": d.get("title", "")} for d in docs],
        )
        return response.text, response.citations
```

---

## Key Questions

**RAG / Retrieval:**
- "Design a production RAG system. Where does Embed fit vs Rerank?"
- "What is the difference between bi-encoder and cross-encoder? When do you use each?"
- "How do you evaluate retrieval quality? What metrics matter?"
- "How would you handle multilingual RAG with mixed-language documents?"
- "Design an enterprise RAG system with audit trails and citation tracking"

**ML Depth:**
- "How do you train an embedding model with contrastive learning?"
- "What is Matryoshka Representation Learning? How does it work?"
- "How does Command R's grounded generation differ from standard RAG?"
- "Explain asymmetric retrieval — why use different input_types for query vs document?"

**Enterprise/Systems:**
- "How would you design a model serving system that supports both cloud and on-prem?"
- "How do you ensure data isolation between enterprise customers sharing infrastructure?"
- "Design a RAG system for a regulated industry (finance or healthcare)"

---

## 7-Day Learning Path

| Day | Focus |
|---|---|
| 1 | Embedding models: contrastive learning, hard negatives, bi-encoders |
| 2 | Re-ranking: cross-encoders, BERT-based rerankers, late interaction (ColBERT) |
| 3 | RAG pipeline: two-stage retrieval, chunking strategies, context window management |
| 4 | Command R: grounded generation, citation extraction, tool use |
| 5 | Multilingual models: multilingual pre-training, cross-lingual transfer |
| 6 | Enterprise deployment: VPC, air-gap, model serving without data egress |
| 7 | System design: enterprise RAG with compliance, audit trails, permissions |
