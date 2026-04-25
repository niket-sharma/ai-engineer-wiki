---
title: Google / DeepMind AI Engineering
aliases: [Google, DeepMind, Google Brain, Alphabet AI]
tags: [company, faang, llm, tpu, search, recommendation]
related: [transformer-architecture, flash-attention, rag-systems, vector-databases]
sources: [training-knowledge, google-ai-blog, deepmind-blog]
relevance: 10
last_updated: 2025-01-15
status: current
---

# Google / DeepMind AI Engineering

## Company Context

Google is the origin of the Transformer ("Attention Is All You Need"), BERT, T5, PaLM, Gemini, and AlphaFold. DeepMind (merged with Google Brain in 2023 → Google DeepMind) produced AlphaGo, AlphaCode, Gemini Ultra, and reinforcement learning breakthroughs. Google runs the world's largest ML infrastructure at scale.

**Key AI products:** Search (ranking, featured snippets), YouTube recommendations, Google Ads, Google Assistant, Bard/Gemini, Translate, Photos, Gmail Smart Compose, Workspace AI.

**Research contributions that dominate the field:** Transformer (2017), BERT (2018), T5 (2019), GQA (2023), Flash Attention (Google helped productionize), Word2Vec (2013), ResNet (2015), TPU hardware.

---

## What Google AI Engineers Work On

### 1. Search and Ranking

- **Multitask Unified Model (MUM):** 1000× more powerful than BERT for complex search queries. Multimodal (text + images).
- **Neural matching:** Understand intent behind queries, not just keyword overlap
- **LLM-augmented search:** Featured snippets, "Search Generative Experience" (SGE)
- **Ranking:** Learning-to-rank with Gradient Boosting + neural models at planet scale

**Key tech:** TPUs, TF-Serving, Spanner for feature storage, massive distributed training

### 2. Recommendations (YouTube, Play Store, Maps)

- **Two-tower retrieval:** Candidate generation at billions-of-items scale
- **Deep Neural Ranking (DNR):** Multi-task ranking with user engagement signals
- **Real-time features:** Streaming feature pipelines via Dataflow/Pub/Sub
- **Exploration-exploitation:** Contextual bandits for recommendation diversity

### 3. LLM Infrastructure

- **TPU v4/v5:** Custom AI accelerator, 275 TFLOPS per chip, pods of 4096+ chips
- **Pathways:** System for training single large models that can do many tasks
- **Gemini 1.5:** 1M+ context window, multimodal
- **Distributed training at scale:** Megatron-style tensor parallelism, pipeline parallelism

### 4. ML Platform (Vertex AI / internal)

- **Borg → Kubernetes:** Google invented container orchestration for ML
- **TFX (TensorFlow Extended):** ML pipeline framework open-sourced from internal
- **Vizier:** Bayesian optimization for hyperparameter tuning (open-sourced)
- **Feature Store:** Internal real-time feature serving at massive scale

---

## Technical Deep-Dives (What They Ask)

### TPU Architecture

```
TPU v4 specs:
- 275 TFLOPS BF16
- 32 GB HBM per chip
- 600 GB/s memory bandwidth
- Interconnect: 1.2 Tbps per chip (inter-chip)

vs A100:
- 312 TFLOPS BF16
- 80 GB HBM
- 2 TB/s memory bandwidth
- NVLink: 600 GB/s

TPU advantage: custom interconnect topology (torus), designed for LLM training
```

**Why TPUs for Transformers:** TPUs use systolic arrays optimized for matrix multiply. The attention mechanism is almost entirely matrix ops. TPUs also eliminate the overhead of general GPU programmability.

### Distributed Training at Scale

```python
# Google's approach: data parallelism + model parallelism + pipeline parallelism

# Data parallel: each TPU sees different batch, averages gradients
strategy = tf.distribute.TPUStrategy(tpu)

# For 540B PaLM: 6144 TPU v4 chips
# Model parallel: each chip handles a shard of the model
# Pipeline parallel: layers split across groups of chips

# Key: Pathways allows sparse activation — only parts of the model needed
# for a given input are activated (MoE-like)
```

### Recommendation Systems Architecture

```
User request
     ↓
Candidate Generation (two-tower, ~100M items → ~1000 candidates)
     ↓ 
Scoring (neural ranking model, 1000 → 100)
     ↓
Re-ranking (diversity, freshness, policy filters, 100 → 10)
     ↓
User sees results
```

**Two-tower model:**
```python
# Query tower: encode user context
query_embedding = user_tower(user_features, context_features)  # (d,)

# Item tower: encode content (precomputed offline)
item_embeddings = item_tower(item_features)  # (N, d) for all N items

# Retrieval: ANN search (ScaNN = Google's HNSW alternative)
scores = query_embedding @ item_embeddings.T
top_k_items = top_k(scores, k=1000)
```

---

## Key Questions

**System Design:**
- "Design Google Search's ranking system"
- "Design YouTube's recommendation system end-to-end"
- "How would you train and serve a model at 10 billion QPS?"
- "Design a real-time feature store for YouTube recommendations"
- "How does Google scale BERT for production search latency?"

**ML Depth:**
- "Explain the Transformer architecture — why was it better than LSTMs?"
- "What is GQA and how does it reduce KV cache memory?" (Google invented it)
- "How does distillation work? How would you distill Gemini Ultra to Gemini Nano?"
- "What is the Mixture of Experts (MoE) architecture and what are the trade-offs?"
- "How do TPUs differ from GPUs for training Transformers?"

**Coding:**
- Implement attention from scratch (NumPy or PyTorch)
- Implement a two-tower retrieval model
- Graph algorithms — Google loves graph problems
- System design + back-of-envelope (latency, throughput, memory)

---

## Red Flags at Google

- **Not knowing the Transformer paper:** You must know "Attention Is All You Need" deeply. Google invented it.
- **Ignoring scale:** Always think in terms of billions of items, millions of QPS. "It depends" without specifics is weak.
- **Not mentioning trade-offs:** Google engineers are expected to reason about trade-offs explicitly.
- **Weak on distributed systems:** Google = massive scale. Understand sharding, replication, consistency.

---

## 7-Day Learning Path

| Day | Focus |
|---|---|
| 1 | Transformer architecture deep dive (attention, MHA, GQA, Flash Attention) |
| 2 | BERT, T5, PaLM, Gemini architecture differences |
| 3 | Recommendation systems (two-tower, ranking, exploration-exploitation) |
| 4 | Distributed training at scale (tensor/pipeline/data parallelism) |
| 5 | System design: Google Search, YouTube recs |
| 6 | TPU vs GPU, ML infrastructure (TFX, Vertex AI) |
| 7 | Coding: implement attention, two-tower; LeetCode graphs (Hard level) |
