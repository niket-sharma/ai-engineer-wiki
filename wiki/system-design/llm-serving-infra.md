---
title: "LLM Serving Infrastructure"
aliases: ["LLM serving", "model serving", "inference infrastructure", "vLLM", "TGI"]
tags: [system-design, mlops, inference]
related: ["[[kv-cache]]", "[[flash-attention]]", "[[rag-pipeline-design]]", "[[ml-platform]]"]
sources: ["training-knowledge"]
interview_relevance: high
last_updated: 2026-04-22
status: current
---

# LLM Serving Infrastructure

## TL;DR
LLM serving is fundamentally a memory-bandwidth-bound problem at small batch sizes and a compute-bound problem at large batches. The critical optimizations are: continuous batching (fill GPU every cycle), paged attention (no memory waste), and tensor parallelism (cut per-token latency). KV cache management is the central engineering challenge.

## Problem Statement
Serve an LLM to many concurrent users at low latency (TTFT < 500ms) and high throughput (maximize tokens/sec/GPU-dollar).

**Key metrics:**
- **TTFT** (Time to First Token): latency to first streamed token — drives perceived responsiveness
- **TPOT** (Time Per Output Token): latency between tokens — drives streaming smoothness
- **Throughput**: tokens generated per second per GPU
- **GPU utilization**: how efficiently are you using expensive hardware (target: > 80%)

## High-Level Architecture

```
                    ┌─────────────────────────────────┐
Internet → Load     │  Serving Engine (vLLM / TRT-LLM) │
           Balancer │                                  │
                    │  ┌──────────┐  ┌─────────────┐  │
                    │  │ Request  │  │ KV Cache    │  │
                    │  │ Queue    │  │ Manager     │  │
                    │  └──────────┘  │ (Paged Attn)│  │
                    │       ↓        └─────────────┘  │
                    │  ┌──────────────────────────┐   │
                    │  │  GPU Cluster (A100/H100)  │   │
                    │  │  Tensor Parallelism       │   │
                    │  └──────────────────────────┘   │
                    └─────────────────────────────────┘
```

## Component Deep-Dives

### Continuous Batching

**Naive batching problem:** Wait for N requests → run batch → return all. Requests that finish early force the batch to wait for the slowest request → GPU idles.

**Continuous batching:** Process new requests immediately when a batch slot opens:
```
Step 1: [req_A, req_B, req_C, req_D]  → generate one token each
Step 2: req_B finishes (hit EOS)
        [req_A, req_E, req_C, req_D]  → req_E enters the batch immediately
```
Result: GPU always has maximum batch size running. 3–10× throughput improvement.

### Paged Attention (KV Cache Management)

Naive allocation: reserve max_seq_len * KV_size per request upfront → massive waste.

Paged attention: allocate KV in 16-token pages on demand:
- Block table: maps logical page numbers to physical GPU memory pages
- Freed immediately when request completes
- Enables prefix caching: shared system prompt → shared physical pages

### Parallelism Strategies

| Strategy | How | Latency | Throughput | When |
|---|---|---|---|---|
| Tensor Parallelism | Split weight matrices across GPUs | ↓ (all GPUs work each step) | = | Latency optimization |
| Pipeline Parallelism | Different layers on different GPUs | ↑ (pipeline bubble) | ↑ | Large models, throughput |
| Data Parallelism | Copy of model, different requests | = | ↑ | Scale out |

**Practical:** For 7B model on 2×A100 → tensor parallelism (TP=2). For 70B model → TP=4 or 8 within a node, pipeline across nodes.

### Speculative Decoding

**Problem:** Autoregressive generation is sequential — can't parallelize token generation.
**Solution:** Use a small draft model to propose k tokens in parallel, then verify with the large model in one forward pass.
```
Draft model: [token_1, token_2, token_3, token_4, token_5]  (fast)
Large model: verify all 5 at once → accept [token_1, token_2, token_3] → reject rest
```
Result: ~2–3× speedup for tasks where draft model's acceptance rate is high (code, structured output). Less useful for creative tasks.

### Quantization for Serving

| Format | Throughput vs FP16 | Quality | Use case |
|---|---|---|---|
| FP16 | 1× | Baseline | Default |
| BF16 | 1× | ≈ FP16 | Better range for LLMs |
| INT8 (SmoothQuant) | 1.5× | -0.5% | Standard production |
| FP8 (H100) | 2× | ≈ FP16 | Best on H100 |
| INT4 (GPTQ/AWQ) | 2–3× | -1–3% | Memory constrained |

## Scale & Reliability

**Capacity planning back-of-envelope:**
- 70B model weights: 140 GB in FP16 → needs 2×A100-80GB minimum
- At TP=2: each A100 holds 70 GB weights, leaving 10 GB for KV cache
- KV cache at 4096 tokens: ~67 MB per request
- ~150 concurrent requests max before KV memory exhausts
- At 100 tokens/sec/request: 15,000 tokens/sec total throughput

**Autoscaling triggers:**
- Scale out when: GPU utilization > 85% for > 2 min, OR request queue depth > 50
- Scale in when: GPU utilization < 30% for > 10 min
- Kubernetes HPA on custom GPU utilization metric

**Health checks:**
- Liveness: `/health` returns 200
- Readiness: `/generate` with 10-token prompt returns in < 200ms
- Startup: wait for model to load before accepting traffic (can take 2–5 minutes for 70B)

## Tradeoffs

| Decision | Option A | Option B | Recommendation |
|---|---|---|---|
| Serving framework | vLLM | TRT-LLM | vLLM for flexibility; TRT-LLM for max throughput on NVIDIA |
| Parallelism | TP | PP | TP for latency; PP for very large models across nodes |
| Quantization | FP16 | INT8 | INT8 in production — 50% memory, minimal quality loss |
| Speculative decoding | Yes | No | Yes for structured output tasks; skip for open-ended |

##  Angles

**Common questions:**
- "What is continuous batching and why does it matter?"
- "Walk me through how paged attention works."
- "How would you serve a 70B model at <500ms TTFT?"
- "What metrics would you monitor for an LLM serving system?"
- "How do you handle memory pressure when KV cache fills up?"

**System design prompt:** "Design an LLM serving system that handles 10,000 concurrent users."
→ Walk through: load balancing, serving engine, GPU cluster sizing, KV cache capacity, autoscaling, monitoring.

## Connections
- [[kv-cache]] — the central resource managed by the serving infrastructure
- [[flash-attention]] — the compute kernel that runs inside each layer during serving
- [[rag-pipeline-design]] — RAG system calls the LLM serving layer for generation
