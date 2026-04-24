# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**Authors:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
**Published:** 2022-05-27
**arXiv ID:** 2205.14135
**URL:** https://arxiv.org/abs/2205.14135
**Venue:** NeurIPS 2022

---

## Abstract Summary

Transformers are slow and memory-hungry on long sequences because self-attention is quadratic in sequence length. Existing methods approximate attention to reduce compute but often lack wall-clock speed-ups and don't reduce memory footprint. FlashAttention is an IO-aware exact attention algorithm that uses tiling to reduce the number of HBM read/writes. It runs 2–4× faster than baseline attention in PyTorch with 5–20× less memory — without approximation.

---

## Core Problem: GPU Memory Hierarchy

Modern GPUs have two memory tiers that differ drastically in speed:

| Memory | Size (A100) | Bandwidth |
|---|---|---|
| HBM (High Bandwidth Memory, DRAM) | 40–80 GB | ~2 TB/s |
| SRAM (on-chip, shared memory) | ~20 MB total | ~19 TB/s |

**The bottleneck is NOT compute**: attention has the same FLOPs whether implemented naively or with Flash. The bottleneck is **HBM reads/writes**. Standard attention materializes the n×n attention matrix in HBM — that's a lot of data to move.

**Standard attention HBM traffic:**
1. Load Q, K → write QK^T to HBM (n×n matrix, O(n²) writes)
2. Load QK^T → apply softmax → write S to HBM (O(n²) reads + writes)
3. Load S, V → write output to HBM (O(n²) reads)
4. Total: O(n²) HBM accesses

---

## FlashAttention Algorithm

**Key insight**: if the n×n matrix never fits in SRAM, tile the computation to fit piece by piece. Use the **online softmax algorithm** to compute exact softmax without materializing the full row.

### Online Softmax Algorithm

Standard softmax requires two passes over the row (first to find max, then to compute exp):
```
m = max(x_1, ..., x_n)
l = sum(exp(x_i - m))
softmax(x_i) = exp(x_i - m) / l
```

Online softmax maintains running statistics:
```
For each block B_j of K, V:
    # Update running max
    m_new = max(m_prev, max(scores in B_j))
    
    # Update running sum (rescale previous accumulation)
    l_new = exp(m_prev - m_new) * l_prev + sum(exp(scores_j - m_new))
    
    # Update output accumulation (rescale previous, add new)
    O_new = diag(exp(m_prev - m_new)) * O_prev + exp(scores_j - m_new) * V_j
```

After processing all blocks: `O_final = diag(1/l_new) * O_new`

This gives exact softmax with one pass, entirely in SRAM.

### Tiling Strategy

```
for Q_block in split(Q, block_size_r):
    for K_block, V_block in zip(split(K, block_size_c), split(V, block_size_c)):
        # Load Q_block, K_block, V_block into SRAM
        # Compute attention scores for this block
        # Update running O, m, l using online softmax
        # Write nothing to HBM — all intermediate stays in SRAM
    # Write final O_block to HBM once
```

Block sizes chosen so Q_block, K_block, V_block fit in SRAM.

### HBM Traffic Reduction

| Implementation | HBM reads/writes | Memory |
|---|---|---|
| Standard | O(n² + nd) | O(n²) |
| FlashAttention | O(n²d / M) where M = SRAM size | O(n) |

The n²d/M factor: for A100 with M=20MB, d=128, n=4096:
```
Ratio = n*d / M = 4096*128*2 / 20,000,000 ≈ 0.05
→ FA does ~20× fewer HBM ops than standard
```

### Backward Pass

Standard attention backward: needs to store the n×n attention matrix for gradient computation (or recompute it). FA backward: recomputes attention weights on the fly from Q, K, V (uses saved m, l statistics). Trades compute for memory → O(n) memory during backward pass.

---

## FlashAttention v2 (2023)

**Paper:** https://arxiv.org/abs/2307.08691

Key improvements over FA1:
1. **Parallelism over sequence length**: FA1 parallelized over batch × heads. FA2 also parallelizes over the query sequence length — better GPU utilization at long sequences.
2. **Work partitioning**: reduces non-matmul FLOPs (which have lower throughput than matmul)
3. **Better warp-level parallelism**: avoids cross-warp communication for the softmax reduction
4. **Speed**: 2× faster than FA1. Achieves 50–73% of theoretical peak FLOPs on A100.

---

## FlashAttention v3 (2024)

Key improvements on H100:
- Exploits **async warpgroups** (H100 feature): overlap WGMMA (matrix multiply) and softmax
- **FP8 support**: 2× higher throughput than BF16 with same precision for most workloads
- **Pipelined GEMM + softmax**: producer-consumer pipelining in SRAM
- Speed: ~750 TFLOPS on H100 FP16 (vs ~350 for FA2)

---

## Results

**Training speed:**
| Model | Context | FA1 speedup | FA2 speedup |
|---|---|---|---|
| BERT-large | 512 | 2.4× | — |
| GPT-2 | 1024 | 3.1× | — |
| Any model | 4096+ | 4–8× | 2× over FA1 |

**Memory:**
- At seq_len=4096: FA uses ~68 MB vs 17 GB for standard attention (250× reduction in attention memory)
- Enables sequences that were previously impossible: 64k tokens on A100

**Quality:** Mathematically exact. Results are bit-identical to standard attention (accounting for floating point non-associativity).

---

## Integration

FA is now the default in:
- PyTorch 2.0+ (`F.scaled_dot_product_attention` — uses FA when possible)
- Hugging Face Transformers (via `attn_implementation="flash_attention_2"`)
- vLLM, TGI (inference serving)
- All major training frameworks (FSDP, DeepSpeed, Megatron-LM)

Installation:
```bash
pip install flash-attn --no-build-isolation
```

---

## -Relevant Insights

**The single most important point:** FlashAttention does NOT change the mathematical output of attention. It's a systems-level optimization (kernel fusion + IO awareness), not an algorithmic approximation.

**Why this matters for interviews:**
- Interviewers distinguish: "What is Flash Attention?" (IO-aware tiling, reduce HBM traffic) from "What is approximate attention?" (Linformer, Performer — these change the math)
- FA is why you can train with 32k+ context windows on modern GPUs

**Counterintuitive fact:** FA is slower at very short sequences (< 512 tokens) because the tiling overhead exceeds the benefit. PyTorch's dispatcher handles this automatically.

**Connection to KV cache:** FA optimizes the per-step attention computation. KV cache reduces HOW MANY steps you need. They're complementary.

**Connection to LLM serving:** FlashDecoding (a variant of FA) handles the inference case where batch_size is large but sequence length for the current step is 1. Regular FA is optimized for training (long sequences, small batch).

---

## Common  Questions From This Paper

- "What problem does Flash Attention solve? Is it about compute or memory?"
- "How does the online softmax trick work? Why doesn't it need to store the full attention matrix?"
- "Flash Attention has the same FLOPs as standard attention — so why is it faster?"
- "What does 'IO-aware' mean in the context of Flash Attention?"
- "Can Flash Attention be used for inference as well as training?"
