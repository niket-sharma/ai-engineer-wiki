---
title: Meta AI Engineering
aliases: [Meta, Facebook AI, FAIR, Meta AI Research]
tags: [company, faang, llm, llama, recommendation, ads, pytorch]
related: [transformer-architecture, rlhf, vector-databases, llm-serving-infra]
sources: [training-knowledge, meta-ai-blog, fair-research]
relevance: 10
last_updated: 2025-01-15
status: current
---

# Meta AI Engineering

## Company Context

Meta (Facebook) is the open-source AI powerhouse. FAIR (Fundamental AI Research) produced LLaMA, PyTorch, FAISS, RoBERTa, OPT, and dozens of foundational papers. Meta's business depends entirely on AI: Feed and Reels ranking drive engagement → ads revenue → $130B+ annual revenue.

**Key AI products:** Feed/Reels/Stories ranking, Ads targeting and bidding, Instagram recommendations, WhatsApp (message suggestions), Meta AI assistant (LLaMA-based), Content moderation, Reality Labs (VR/AR).

**Open-source contributions:** PyTorch, LLaMA 1/2/3, FAISS, RoBERTa, BART, wav2vec 2.0, Segment Anything Model (SAM), Dino, ImageBind.

---

## What Meta AI Engineers Work On

### 1. Feed and Reels Ranking (Core Business)

Meta's ranking system processes billions of posts per day. The architecture:

```
Thousands of candidate posts (from social graph)
      ↓
Integrity filtering (remove policy-violating content)
      ↓
Value model (1000 → 500 candidates): fast model, many features
      ↓
Ranking model (500 → 50): deep model, user interaction features
      ↓
Contextual adjustment (diversity, reranking policies)
      ↓
Final feed
```

**Key models:**
- **GBDT for early-stage ranking:** Fast, handles sparse features well
- **Deep learning for final ranking:** DLRM-style with embedding tables for users, posts, authors
- **Multi-task learning:** Predict likes, comments, shares, hide-rate, negative signals simultaneously

### 2. Ads (Meta's Revenue Engine)

```
Advertiser: "Show my ad to users likely to buy running shoes"
      ↓
Audience targeting: user embedding similarity to advertiser's target
      ↓
Auction: bid × predicted value (CTR × CVR × bid)
      ↓
Ad ranking: relevance + business value
      ↓
Attribution: did ad exposure cause conversion? (causal inference challenge)
```

**Key technical challenges:**
- **Delayed feedback:** Conversion signal arrives days/weeks after ad impression
- **Auction theory:** Vickrey-Clarke-Groves (VCG) mechanism for truthful bidding
- **Budget pacing:** Distribute ad spend evenly over time, not front-loaded
- **Privacy (post-ATT):** Apple's App Tracking Transparency broke Meta's cross-app tracking → forced investment in on-device learning and differential privacy

### 3. LLaMA and Open-Source AI

Meta's bet: open-source wins long-term by creating ecosystem dependency on Meta's hardware and services.

```
LLaMA 3.1 405B architecture:
- 126 transformer layers
- 16,384 hidden dim
- 128 attention heads, 8 KV heads (GQA)
- 128K context window with RoPE
- SwiGLU activation
- RMSNorm (not LayerNorm)
- Trained on 15 trillion tokens
```

**LLaMA fine-tuning pipeline (what Meta engineers build):**
1. Pre-train on web-scale data
2. SFT on curated instruction-following data
3. RLHF: reward model training + PPO OR DPO
4. Red-teaming and safety evaluation
5. Quantize (GGUF format for community, INT4/INT8)

### 4. PyTorch and AI Infrastructure

Meta created PyTorch (now a Linux Foundation project). Internal AI infra:
- **FSDP (Fully Sharded Data Parallel):** Meta's answer to ZeRO
- **Triton:** GPU kernel language (co-created with OpenAI)
- **TorchServe:** Model serving framework
- **Prophet:** Time-series forecasting (open-sourced)

---

## Key Questions

**System Design:**
- "Design Facebook Feed ranking system"
- "Design Meta's ad targeting and auction system"
- "Design a content moderation system for 3 billion users"
- "How would you scale LLaMA inference to serve 500M users?"
- "Design an A/B testing platform for ranking models"

**ML Depth:**
- "Explain the DLRM architecture. Why use embedding tables for categorical features?"
- "How does LLaMA 3 differ from LLaMA 2 architecturally?" (GQA, context length, training data)
- "What is the challenge of multi-task learning in feed ranking? How do you handle conflicting objectives?"
- "How do you handle the delayed label problem in ads conversion prediction?"
- "What is differential privacy and how would you apply it to ad targeting post-ATT?"

**Coding:**
- Meta is among the hardest companies for coding assessments
- Expect LeetCode Hard
- Focus: graphs, trees, DP, sliding window, backtracking
- Also: system design coding (implement a simplified version of a component)

---

## Multi-Task Ranking: Handling Conflicting Objectives

A critical Meta-specific concept:

```python
# Single model predicts multiple signals simultaneously
class MultiTaskRanker(nn.Module):
    def __init__(self, shared_dim):
        super().__init__()
        self.shared = TransformerEncoder(shared_dim)
        # Separate heads for each task
        self.like_head = nn.Linear(shared_dim, 1)
        self.comment_head = nn.Linear(shared_dim, 1)
        self.share_head = nn.Linear(shared_dim, 1)
        self.hide_head = nn.Linear(shared_dim, 1)     # negative signal
        self.report_head = nn.Linear(shared_dim, 1)   # negative signal
    
    def forward(self, features):
        shared = self.shared(features)
        return {
            'like': self.like_head(shared),
            'comment': self.comment_head(shared),
            'share': self.share_head(shared),
            'hide': self.hide_head(shared),
            'report': self.report_head(shared)
        }

# Final ranking score: weighted combination
def rank_score(predictions, weights):
    # weights are tuned to maximize long-term user engagement
    return (weights['like'] * sigmoid(predictions['like'])
          + weights['comment'] * sigmoid(predictions['comment'])
          + weights['share'] * sigmoid(predictions['share'])
          - weights['hide'] * sigmoid(predictions['hide'])
          - weights['report'] * sigmoid(predictions['report']))
```

**Challenge:** Weight tuning is a policy decision, not a pure ML decision. Heavy engagement (share-bait) vs meaningful interactions trade-off.

---

## Red Flags at Meta

- **Weak coding:** Meta is famous for hard coding assessments. You need LeetCode Hard fluency.
- **Not knowing PyTorch:** Everything at Meta is PyTorch. TensorFlow experience doesn't transfer well.
- **Ignoring engagement metrics:** "Model accuracy" is not the goal — user time-well-spent, engagement, and business metrics are.
- **No sense of scale:** Meta = billions of users. Design must work at that scale.

---

## 7-Day Learning Path

| Day | Focus |
|---|---|
| 1 | Recommendation systems: DLRM, two-tower, multi-task learning |
| 2 | LLaMA architecture: GQA, RoPE, SwiGLU, RMSNorm, training pipeline |
| 3 | Ads ML: auction theory, CTR/CVR prediction, delayed labels |
| 4 | RLHF and DPO for LLaMA fine-tuning |
| 5 | System design: Feed ranking, ads platform |
| 6 | FSDP, distributed training, PyTorch internals |
| 7 | Coding: LeetCode Hard — graphs, DP, trees, backtracking |
