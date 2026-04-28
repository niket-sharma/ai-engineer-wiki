---
title: "MLOps Q&A"
tags: [qa, mlops]
related: ["[[llm-serving-infra]]", "[[ml-platform]]", "[[feature-store]]"]
last_updated: 2026-04-22
---

# MLOps Q&A

---

## L1 — Conceptual

### Q1. What is the difference between model latency and throughput, and when do you optimize for each?

**A:**
- **Latency**: time from request to first/last token (TTFT: time to first token; E2E: total response time)
- **Throughput**: tokens generated per second per GPU (or requests/sec)

They're in tension: smaller batches → lower latency but lower throughput; larger batches → higher throughput but higher latency.

**Optimize for latency** when: user is waiting (interactive chat, real-time fraud scoring, live recommendation)
**Optimize for throughput** when: batch jobs (bulk document processing, overnight report generation, dataset annotation)

For LLMs: target TTFT < 500ms for interactive use. For batch: maximize tokens/sec/dollar.

---

### Q2. What is quantization and what are the tradeoffs?

**A:** Quantization reduces model weight precision from float32/float16 to lower-bit representations:

| Format | Bits | Memory vs FP16 | Quality loss | Use case |
|---|---|---|---|---|
| FP16 / BF16 | 16 | 1× (baseline) | None | Training, high-quality inference |
| INT8 | 8 | 0.5× | Minimal (<1%) | Standard serving |
| INT4 / NF4 | 4 | 0.25× | Small (1–3%) | Edge devices, QLoRA |
| INT2 | 2 | 0.125× | Significant | Experimental |

**Post-training quantization (PTQ)**: quantize after training — fast but quality loss. Methods: GPTQ, AWQ (weight-only), SmoothQuant (weight + activation).
**Quantization-aware training (QAT)**: simulate quantization during training — better quality, expensive.

For LLM serving: INT8 is the safe default. FP8 is gaining adoption on H100s (near FP16 quality, 2× throughput). INT4 (GPTQ/AWQ) for memory-constrained deployments.

---

### Q3. What is model drift and how do you detect it?

**A:** Model drift = degradation in model performance over time due to changes in the real-world data distribution.

Types:
- **Data drift**: input distribution changed (feature values shifted)
- **Concept drift**: the relationship between inputs and outputs changed (e.g., fraud patterns changed)
- **Prediction drift**: model output distribution shifted (proxy for performance change)

Detection:
- **Statistical tests**: KS test, PSI (Population Stability Index) on feature distributions
- **Prediction monitoring**: track prediction distribution daily, alert on significant shift
- **Performance monitoring**: track actual metrics (if labels are available) — AUC, precision, recall
- **LLM-specific**: embed responses, track embedding drift; track user satisfaction rate

Response to drift: investigate root cause → retrain on recent data → A/B test new model → deploy.

---

## L2 — Technical

### Q4. Walk me through how vLLM implements paged attention and why it improves throughput.

**A:**

**Problem with naive KV cache:** Each request needs a contiguous memory block for its KV cache, allocated upfront for the maximum sequence length. Problems:
1. Internal fragmentation: 90% of memory may be allocated but unused (request finished at 100 tokens, reserved 2048)
2. External fragmentation: can't fit new requests because free memory is non-contiguous
3. Can't share KV pages between requests that share a prefix (e.g., system prompt)

**Paged attention solution:**
- KV cache is divided into fixed-size pages (e.g., 16 tokens per page)
- Each request gets a **block table**: a mapping from logical page number to physical page
- Pages are allocated on demand as generation proceeds — no upfront reservation
- When a request finishes, pages are freed immediately

**Memory efficiency:**
- Near-zero internal fragmentation (last page wastes < 1 page on average)
- Zero external fragmentation (any free page can serve any request)
- Prefix sharing: requests with same system prompt share the same physical KV pages — one copy for N requests

**Throughput improvement:** More requests can fit in GPU memory simultaneously → larger effective batch size → higher GPU utilization → 2–4× higher throughput vs naive allocation.

---

### Q5. Explain ONNX and TensorRT — when would you use each?

**A:**

**ONNX (Open Neural Network Exchange):**
- Open format for representing ML models as a computation graph
- Framework-agnostic: export from PyTorch → run in any ONNX-compatible runtime
- Use when: deploying to diverse hardware (CPU servers, mobile, ARM), or using non-NVIDIA hardware
- Runtime: ONNX Runtime (Microsoft) — optimized for CPU and some GPU ops

**TensorRT:**
- NVIDIA's high-performance inference optimizer for NVIDIA GPUs
- Takes ONNX (or PyTorch) model → applies fusions, kernel selection, quantization → generates GPU-optimized engine
- Key optimizations: layer fusion (Conv+BN+ReLU → one kernel), FP16/INT8 quantization, kernel auto-tuning
- Use when: NVIDIA GPU is the deployment target, latency is critical, throughput matters
- Limitation: NVIDIA-only, compiled engine is hardware-specific (A100 engine ≠ V100 engine)

**For LLMs specifically:**
- Pure TensorRT is complex for dynamic shapes (variable sequence lengths)
- TensorRT-LLM: NVIDIA's library combining TensorRT with LLM-specific optimizations (paged attention, inflight batching)
- vLLM uses custom CUDA kernels (similar goals to TRT-LLM, more Pythonic)

**Decision rule:** NVIDIA GPU + latency critical → TensorRT-LLM or vLLM. Multi-hardware/cloud-agnostic → ONNX Runtime.

---

### Q6. How do you implement shadow mode deployment for an ML model?

**A:** Shadow mode: run the new model in parallel on all production traffic, don't show its results to users, compare offline.

**Implementation:**

```python
# Request handler
def handle_request(request):
    # Primary: serves the user
    response = primary_model.predict(request)
    
    # Shadow: async, not on critical path
    asyncio.create_task(shadow_predict(request))
    
    return response

async def shadow_predict(request):
    shadow_response = new_model.predict(request)
    log_to_warehouse({
        "request": request,
        "primary_response": cached_primary_response,
        "shadow_response": shadow_response,
        "timestamp": now()
    })
```

**Key requirements:**
- Shadow inference must be async — never add to user latency
- Log primary + shadow responses with same request ID for comparison
- Handle shadow failures gracefully (don't fail primary)
- Cost: running two models on all traffic is expensive — consider sampling (10% of traffic)

**Evaluation:**
- Compare prediction distributions: KL divergence between primary and shadow outputs
- For LLMs: LLM-as-judge win rate (does new model's response beat primary?)
- For classifiers: compare confusion matrices offline
- Gate for promotion: new model must match or beat primary on 95% of sampled queries

---

## L3 — Applied

### Q7. You just deployed a new LLM to production and p99 latency spiked from 1.5s to 4s. Walk me through your debugging process.

**A:**

**Triage (first 5 minutes):**
1. Check if spike is uniform or on specific request types (long prompts? certain endpoints?)
2. Check GPU utilization — if < 50%, might be queuing/batching issue; if > 95%, model is saturated
3. Compare model size to previous — is the new model larger? More layers?

**Hypotheses and checks:**

| Hypothesis | How to check | Fix |
|---|---|---|
| New model is larger (more layers) | Model config diff | Quantize to INT8, add GPU |
| KV cache pressure (larger context window) | vLLM memory logs | Reduce max_model_len, add KV quantization |
| Batch size too small (underutilizing GPU) | GPU utilization metrics | Increase max_num_seqs |
| Continuous batching misconfigured | Queue depth metrics | Tune batch_wait_timeout |
| Tensor parallelism degree wrong | Per-GPU utilization | Increase TP degree |
| Prompt preprocessing bottleneck | CPU metrics | Async tokenization |

**Typical culprit for LLM latency spikes:** KV cache pressure — new model has larger context window configured, filling GPU memory faster, forcing smaller batch sizes, reducing throughput which manifests as higher latency under load.

**Resolution:** If KV cache is the issue → quantize KV cache to INT8 (halves memory, minimal quality impact) → restore batch size → recheck p99.

---

### Q8. How would you set up monitoring for an LLM deployed in a financial services company?

**A:**

**Infrastructure monitoring (standard):**
- GPU utilization, memory, temperature (Prometheus + Grafana)
- Request rate, error rate, latency (p50/p95/p99) per endpoint
- Alert: p95 TTFT > 1s, error rate > 0.1%, GPU > 95% for > 5 min

**LLM-specific monitoring:**

*Quality monitoring:*
- User satisfaction (thumbs up/down rate) — alert if drops > 5% week-over-week
- Response length distribution — sudden shortening = model may have degraded
- Refusal rate — sudden increase = model is being overly cautious

*Safety monitoring (critical for finance):*
- PII detection in inputs and outputs (names, SSNs, account numbers)
- Toxicity/bias scoring on random sample of outputs
- Hallucination rate: automated faithfulness scoring via LLM judge
- Regulatory keyword monitoring: flag outputs containing regulatory claims that need review

*Drift monitoring:*
- Embed all inputs with a fixed encoder, track embedding distribution shift
- PSI score on input embedding dimensions — alert if > 0.2
- Track topic distribution (clustering) — alert if new topics emerge

*Audit logging (regulatory requirement):*
- Log all (user_id, session_id, prompt, response, timestamp, model_version) with 7-year retention
- Immutable append-only log (S3 with object lock)
- Searchable by user, time range, session for compliance investigations

---
