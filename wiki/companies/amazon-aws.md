---
title: Amazon / AWS AI Engineering
aliases: [Amazon, AWS, Amazon AI, Alexa AI]
tags: [company, faang, llm, recommendation, alexa, sagemaker]
related: [rag-systems, llm-serving-infra, ml-platform, feature-store]
sources: [training-knowledge, aws-ai-blog, amazon-science]
relevance: 10
last_updated: 2025-01-15
status: current
---

# Amazon / AWS AI Engineering

## Company Context

Amazon AI spans two major businesses: **Amazon** (consumer: recommendations, search, Alexa, logistics optimization) and **AWS** (cloud: SageMaker, Bedrock, Trainium/Inferentia, managed AI services). The culture is LP (Leadership Principles) driven — every role has behavioral and technical components framed around LPs.

**Key AI products:** Product recommendations (35% of revenue), Alexa, Amazon Search, Fraud detection, AWS SageMaker, Amazon Bedrock (managed LLM API), Kendra (enterprise search), Rekognition, Transcribe, Comprehend.

**Scale:** Amazon.com serves 300M+ customers, 350M+ product catalog. Recommendations at this scale are a defining engineering challenge.

---

## What Amazon AI Engineers Work On

### 1. Recommendations and Personalization

Amazon's recommendation system is legendary. Key components:

**Collaborative Filtering at Scale:**
- Item-to-item CF: "Customers who bought X also bought Y"
- User-to-item: personalized recommendations based on history
- Real-time + batch hybrid: immediate signal from current session + long-term preferences

**Deep Learning for Recommendations:**
- DLRM (Deep Learning Recommendation Model): Embedding tables for categorical features + dense layers
- Session-based recommendations: Transformers over browsing sessions

```python
# Amazon-style DLRM sketch
class DLRM(nn.Module):
    def __init__(self, embedding_dim, dense_features, sparse_features):
        super().__init__()
        # Embedding tables (one per sparse feature: product_id, category, brand)
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(cardinality, embedding_dim)
            for feat, cardinality in sparse_features.items()
        })
        # Bottom MLP: processes dense features (price, CTR, etc.)
        self.bottom_mlp = nn.Sequential(nn.Linear(dense_features, 256), nn.ReLU(),
                                         nn.Linear(256, embedding_dim))
        # Top MLP: combines all features
        n_interactions = len(sparse_features) + 1  # +1 for dense
        self.top_mlp = nn.Sequential(nn.Linear(n_interactions * embedding_dim, 128),
                                      nn.ReLU(), nn.Linear(128, 1))
    
    def forward(self, dense, sparse):
        dense_emb = self.bottom_mlp(dense)
        sparse_embs = [self.embeddings[f](sparse[f]) for f in sparse]
        all_embs = torch.stack([dense_emb] + sparse_embs, dim=1)  # (B, n, d)
        # Dot-product interactions
        interactions = torch.bmm(all_embs, all_embs.transpose(1, 2)).flatten(1)
        return self.top_mlp(interactions)
```

### 2. Amazon Search

- **Hybrid retrieval:** BM25 + dense (bi-encoder) → reranking with cross-encoder
- **Query understanding:** NER, intent classification, query expansion
- **Product ranking:** Learn-to-rank with 100s of signals (relevance, sales rank, price, reviews)
- **Ads integration:** Sponsored products interleaved with organic results

### 3. Alexa (Conversational AI)

- **NLU pipeline:** ASR → NLU (intent classification + slot filling) → Dialog Management → NLG → TTS
- **End-to-end models:** Moving from modular pipeline to single transformer
- **Multi-turn conversation:** Session state management, entity resolution
- **On-device inference:** Alexa on small devices requires extreme quantization/distillation

### 4. AWS SageMaker / Bedrock

- **SageMaker:** End-to-end ML platform (data labeling, training, deployment, monitoring)
- **Bedrock:** Managed API for foundation models (Claude, Llama, Titan, Mistral)
- **Trainium/Inferentia:** Custom ML chips for training and inference (competing with Nvidia)

---

## Amazon Leadership Principles

Every answer should implicitly or explicitly touch an LP:

| LP | How it shows up in ML assessments |
|---|---|
| **Customer Obsession** | Why does this metric matter for customers? Not just model accuracy |
| **Dive Deep** | Don't hand-wave. Know the math, the algorithm internals |
| **Invent and Simplify** | Propose a simpler solution first, then add complexity if needed |
| **Bias for Action** | Ship something now (A/B test) vs perfect solution later |
| **Data-Driven** | Everything is an experiment. How do you measure success? |
| **Think Big** | Design for 10×, not 10% improvement |

---

## Key Questions

**System Design:**
- "Design Amazon's product recommendation system"
- "Design a real-time fraud detection system for Amazon Pay"
- "Design the ML pipeline for Amazon search ranking"
- "How would you build a feature store for Amazon's 350M product catalog?"
- "Design an A/B testing platform for ML models at Amazon scale"

**ML Depth:**
- "How does collaborative filtering work? What are its limitations?"
- "Explain embedding tables in recommendation models. How do you handle cold-start?"
- "How do you prevent training-serving skew in a large-scale recommendation system?"
- "What is RLHF? How would you apply it to improve Alexa?"
- "How do you detect and handle concept drift in a product ranking model?"

**Coding (Amazon loves these):**
- OOP design (e.g., design a shopping cart, a parking lot)
- Trees and graphs (Binary tree problems, graph traversal)
- Dynamic programming
- Sliding window / two pointers

**Behavioral (STAR format, mapped to LPs):**
- "Tell me about a time you disagreed with a team decision about an ML model" (Disagree and Commit)
- "Tell me about the most impactful ML project you've worked on" (Think Big + Deliver Results)
- "Tell me about a time you dived deep into data to find an unexpected insight" (Dive Deep)

---

## Amazon-Specific Technical Details

### Fraud Detection at Amazon

```
Features used:
- Transaction features: amount, time, merchant category
- User behavior: velocity (tx count in 1h, 24h), device fingerprint
- Account features: age, verification status, previous chargebacks
- Graph features: connected accounts, shared devices/IPs

Model: Gradient Boosting (LightGBM) + deep learning for sequential patterns
Latency: < 50ms (synchronous in payment flow)
Monitoring: Concept drift on feature distributions, chargeback feedback loop (delayed label)
```

### Cold Start Problem in Recommendations

```python
# New product with no purchase history:
# 1. Content-based: embed product description, images → find similar products
# 2. Category-based: use average embeddings of the category
# 3. Popularity-based: recommend trending items in category
# 4. Exploration: epsilon-greedy or Thompson Sampling to explore new items

# New user (new account):
# 1. Session-based: use items viewed in current session
# 2. Geographic/demographic: location-based popular items
# 3. Context: time of day, referral source
```

---

## Red Flags at Amazon

- **Not thinking about cost:** Amazon is obsessed with cost efficiency. Always mention cost vs accuracy trade-offs.
- **No behavioral prep:** Technical-only preparation will fail. You need 5+ STAR stories mapped to LPs.
- **Ignoring latency:** E-commerce ML must be fast. "We'll optimize later" is not acceptable.
- **Not measuring business impact:** Tie model improvements to revenue, conversion rate, customer satisfaction.

---

## 7-Day Learning Path

| Day | Focus |
|---|---|
| 1 | Recommendation systems: collaborative filtering, DLRM, two-tower |
| 2 | Feature stores, training-serving skew, real-time ML |
| 3 | Fraud detection: imbalanced classification, streaming features, concept drift |
| 4 | NLP pipeline for search: BM25, dense retrieval, reranking |
| 5 | System design: recommendation system, fraud detection |
| 6 | Behavioral stories: 6 STAR stories mapped to 6 LPs |
| 7 | Coding: OOP design, trees/graphs, DP (LeetCode Medium-Hard) |
