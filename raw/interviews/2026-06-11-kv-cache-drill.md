---
type: interview-transcript
date: 2026-06-11
topic: kv-cache
style: drill
duration_min: 1
questions: 5
concepts_touched: [kv-cache, attention-mechanism, flash-attention, paged-attention]
assessed: true
---

# Interview Transcript — kv-cache (drill)

## Q1 (concept: kv-cache, level: 2, source: wiki/qa/transformers-qa.md)
**Interviewer:** What is a KV cache and why is it important?

**Candidate:** KV cache stores K and V projections of already-generated tokens so each decode step only computes projections for the new token, turning O(n^2) total work into O(n). It trades GPU memory for compute and is the main memory bottleneck at long context.

## Q2 (concept: attention-mechanism, level: 2, source: wiki/qa/transformers-qa.md)
**Interviewer:** What is the attention mechanism and why does it matter?

**Candidate:** Attention lets each token gather information from others via softmax(QK^T/sqrt(d_k))V, giving direct long-range dependencies and full parallelism during training.

## Q3 (concept: flash-attention, level: 3, source: wiki/qa/transformers-qa.md)
**Interviewer:** What is Flash Attention and what problem does it solve?

**Candidate:** Encoder-only models like BERT use bidirectional attention for understanding tasks; decoder-only models like GPT use causal attention for generation; encoder-decoder like T5 pairs both for seq2seq.

## Q4 (concept: paged-attention, level: 3, source: wiki/qa/mlops-qa.md)
**Interviewer:** Walk me through how vLLM implements paged attention and why it improves throughput.

**Candidate:** Flash Attention fuses the attention computation into SRAM tiles so the n^2 attention matrix is never materialized in HBM, reducing memory to O(n) and speeding up training.

## Q5 (concept: kv-cache, level: 3, source: wiki/qa/transformers-qa.md)
**Interviewer:** Walk me through the memory cost of a KV cache for Llama 3 8B at a 4096-token context.

**Candidate:** vLLM allocates KV cache in fixed-size pages like OS virtual memory, removing fragmentation and enabling prefix sharing, which raises GPU utilization and throughput.
