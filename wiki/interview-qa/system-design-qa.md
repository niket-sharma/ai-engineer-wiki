---
title: "System Design Interview Q&A"
tags: [interview-qa, system-design]
related: ["[[rag-pipeline-design]]", "[[llm-serving-infra]]", "[[ml-platform]]", "[[feature-store]]"]
last_updated: 2026-04-22
---

# System Design Interview Q&A

---

## L1 — Conceptual

### Q1. What are the key components of an LLM serving system?

**A:**
```
Load Balancer → Serving Engine (vLLM/TGI) → Model (GPU cluster)
                    ↕
              KV Cache Manager
                    ↕
              Request Queue
```

Key components:
- **Load balancer**: routes requests, handles autoscaling signals
- **Serving engine**: continuous batching (don't wait for requests, batch dynamically), paged attention
- **KV cache manager**: allocates/frees KV cache pages per request
- **Request queue**: handles burst traffic, priority queuing
- **Monitoring**: TTFT (time to first token), TPS (tokens per second), GPU utilization, KV cache hit rate

---

### Q2. What is continuous batching and why does it matter?

**A:** Naive batching: wait for N requests, run batch, return all. Problem: requests finish at different times — GPU idles waiting for the slowest.

Continuous batching (aka in-flight batching): as soon as one request in the batch finishes (hits EOS token), immediately insert a new request from the queue into that batch slot. GPU is always doing useful work. Result: 3–10× higher throughput vs naive batching at the same latency.

Implemented in: vLLM, TGI, TensorRT-LLM. This is the most important serving optimization after paged attention.

---

### Q3. What is the difference between latency-optimized and throughput-optimized LLM serving?

**A:**
| Goal | Batch size | KV cache usage | Parallelism |
|---|---|---|---|
| Latency (low TTFT) | Small (1–4) | Less pressure | Tensor parallelism |
| Throughput (max tokens/sec/GPU) | Large | Fill KV cache | Pipeline parallelism |

For customer-facing products: optimize TTFT (< 500ms) and time-per-output-token (< 50ms).
For batch processing (document analysis, embeddings): optimize tokens/sec/GPU — larger batches.

Tensor parallelism (split model across GPUs within a node) reduces latency by cutting per-layer compute. Pipeline parallelism (different layers on different GPUs) increases throughput for batch but adds pipeline bubble latency.

---

## L2 — Technical

### Q4. Design an LLM serving system that handles 10,000 requests per minute with p95 latency < 2 seconds.

**A:**

**Back of envelope:**
- 10,000 req/min = ~167 req/sec
- Average prompt: 500 tokens, response: 200 tokens
- At 50 tok/sec per stream: ~4 sec per request
- With continuous batching, effective throughput: ~1000 tok/sec/GPU (A100)
- Total output tokens/sec: 167 × 200 / 60 = ~556 tok/sec → ~1 A100 for throughput
- But latency < 2s means: prefill < 1s (500 tokens → fast), decode < 1s (200 tokens at 200 tok/s)

**Architecture:**

*Serving layer:*
- vLLM with paged attention + continuous batching
- Model: run on 2×A100 (tensor parallelism for latency)
- Prefix cache for shared system prompts (high cache hit rate if system prompt is fixed)

*Request routing:*
- Priority queue: short prompts (< 200 tokens) fast lane, long prompts slow lane
- Timeout: kill requests > 10 seconds, return error

*Scaling:*
- Kubernetes HPA on GPU utilization (> 80% → scale out)
- 3 replicas minimum (one per availability zone)

*Monitoring:*
- Alert: p95 TTFT > 1s, p95 total latency > 3s, GPU utilization < 50% (underprovisioned) or > 90% (overloaded)

---

### Q5. Design a RAG pipeline for a company's internal knowledge base (50k documents, 1k daily users).

**A:**

**Scale assessment:** 50k docs × ~10 chunks/doc = 500k vectors — trivially fits in memory. 1k DAU = not high load. This is a mid-size internal tool.

**Architecture:**

*Ingestion:*
```
Documents (S3/SharePoint) 
→ Airflow DAG (daily/on-upload trigger)
→ Parse (Unstructured.io for mixed formats)
→ Chunk (section-aware for structured docs, paragraph for unstructured)
→ Embed (OpenAI text-embedding-3-large or local BGE)
→ Upsert to Qdrant (dense) + Elasticsearch (sparse/BM25)
→ Store metadata in Postgres
```

*Query:*
```
User query (Slack/internal webapp)
→ Query rewrite (optional, for short queries)
→ Hybrid retrieval: Qdrant + Elasticsearch → RRF → top-50
→ Rerank: Cohere Rerank API → top-5
→ Claude Sonnet (good quality, fast, cost-efficient)
→ Response with cited document names and links
```

*Storage:*
- Qdrant (self-hosted, 500k vectors fits easily in 4GB RAM)
- Elasticsearch single-node for BM25
- Postgres for chunk metadata (doc_id, chunk_id, source_url, last_modified)

*Access control:*
- Pre-filter by user's document permissions before retrieval
- Store permission groups in Postgres, filter on chunk metadata

*Monitoring:*
- Thumbs up/down feedback on each answer
- Weekly automated eval against 50-question golden set

---

### Q6. Design a feature store for a fraud detection ML system.

**A:**

**Requirements:**
- Online features: <10ms latency for real-time fraud scoring during transactions
- Offline features: batch training data with point-in-time correctness
- Feature freshness: account velocity features must be < 1 minute stale

**Architecture:**

*Online store (low-latency lookups):*
- Redis or DynamoDB for key-value feature serving
- Key: user_id or card_id → latest feature vector
- Features: `transaction_velocity_1h`, `avg_txn_amount_30d`, `merchant_category_risk_score`
- P99 read latency target: < 5ms

*Offline store (batch training):*
- Parquet files in S3, partitioned by date
- Point-in-time correct: `AS OF TIMESTAMP` joins — features available at time of each training label
- Spark/Athena for feature computation and joining

*Feature computation:*
- Streaming: Kafka → Flink → computes sliding window features → writes to Redis + Kafka Compacted Topic
- Batch: Airflow DAG daily → Spark on EMR → writes to S3

*Train-serve consistency:*
- Same feature definitions in a feature registry (Feast or in-house)
- CI tests: compare online vs offline feature values for 100 random keys

*Feature freshness monitoring:*
- Alert if any feature's `last_updated` > 2 minutes (velocity features) or > 1 day (static features)

---

## L3 — Applied

### Q7. Walk me through how you'd design an ML platform for a team of 50 data scientists.

**A:**

**Core components:**

*Data layer:*
- Feature store (online: Redis, offline: S3/Parquet)
- Data catalog (Amundsen/DataHub) — discoverability of datasets
- Data versioning (DVC or Delta Lake)

*Experimentation:*
- MLflow for experiment tracking (metrics, params, artifacts)
- Jupyter Hub or VS Code server for notebooks
- GPU cluster (Kubernetes + GPU node pools) for training jobs

*Orchestration:*
- Airflow for scheduled training pipelines
- Kubeflow Pipelines for ML workflow DAGs (data → train → eval → register)

*Model registry:*
- MLflow Model Registry with staging → production promotion
- Versioned artifacts, lineage tracking

*Serving:*
- Seldon Core or KServe on Kubernetes for online serving
- Batch inference via Spark or Ray
- Shadow mode: run new model in parallel, compare offline before promoting

*Monitoring:*
- Evidently AI or Arize for feature/prediction drift detection
- Grafana dashboards: model latency, error rate, drift metrics
- Automated retraining trigger when drift exceeds threshold

*Self-service:*
- Internal CLI: `mlplatform submit-training-job --config job.yaml`
- SDK: Python library wrapping feature store, model registry, serving APIs

**What teams care about most:** reproducibility (can I retrain and get the same model?), discoverability (can I find good features without asking around?), deployment speed (how long from model to production?).

---

### Q8. How would you implement A/B testing for an LLM-powered product?

**A:**

**Challenges vs. classic A/B testing:**
- LLM responses are long-form — hard to define "conversion" 
- Quality is subjective — need proxy metrics
- High cost to generate responses — can't run unlimited experiments

**Framework:**

*Assignment:*
- Hash user_id + experiment_id → deterministic group assignment
- Sticky assignment: same user always sees same model variant
- Percentage rollout: start 5% → 20% → 50% → 100%

*Metrics:*
- Primary: task completion rate (did user get their answer?), user satisfaction (thumbs up/down)
- Secondary: session length, follow-up question rate (↑ = answer was incomplete), escalation rate
- Guardrail: error rate, latency (don't degrade these)
- Cost: tokens consumed per session (new model must not cost > 20% more)

*Evaluation:*
- Log all (prompt, response, metadata) for both variants
- Run LLM judge nightly: GPT-4 scores both variants on random 500 samples → win rate
- Statistical significance: two-proportion z-test on thumbs up/down rate
- Minimum detectable effect: 2% absolute improvement in satisfaction rate

*Shadow mode:*
Before live A/B: run new model on all traffic, don't show users, evaluate offline. Only run live A/B if shadow mode shows ≥ 2% win rate on offline metrics.

---
