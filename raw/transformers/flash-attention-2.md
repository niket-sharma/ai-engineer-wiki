# Flash Attention 2: Faster Attention with Better Parallelism

**Paper:** "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
**Authors:** Tri Dao (Princeton / Together AI)
**Published:** 2023-07-17
**arXiv ID:** 2307.08691
**Context:** Follow-up to FA1 (Dao et al. 2022). Achieves 2× speedup over FA1, 5-9× over standard attention.

---

## What FA2 Fixes in FA1

FA1 had suboptimal GPU utilization due to poor work partitioning across warps (groups of 32 CUDA threads that execute in lockstep).

### FA1 Problem: Sequential Warp Execution

In FA1's forward pass, warps shared outputs and needed synchronization via shared memory:
```
Warp 1: reads keys K_1, K_2    → computes partial scores
Warp 2: reads values V_1, V_2  → accumulates partial output
→ Both warps write to shared memory → need __syncthreads()
```

Synchronization = idle time.

### FA2 Fix: Better Work Partitioning

FA2 restructures the computation to minimize inter-warp communication:

1. **Outer loop over sequence Q blocks, inner loop over K/V blocks** (swapped vs FA1)
2. Each warp handles a full slice of Q rows independently
3. No synchronization between warps needed in the inner loop

---

## Algorithm Comparison

### Standard Attention
```python
# Materialized N×N attention matrix — O(N²) memory
S = Q @ K.T / sqrt(d)    # [N, N]
P = softmax(S, dim=-1)   # [N, N] — must store full matrix
O = P @ V                # [N, d]
```

### FA1 (2022)
- Tiled computation over SRAM blocks
- Outer loop: K/V blocks; Inner loop: Q blocks
- Online softmax to avoid materializing full S
- Memory: O(N) instead of O(N²)

### FA2 (2023)
- **Swapped loop order:** Outer loop over Q blocks; Inner loop over K/V blocks
- Each thread block handles one Q tile and iterates over all K/V tiles
- Result: each thread block accumulates O for its Q tile without needing to sync with other thread blocks

**Why this matters:** Thread blocks in GPUs run independently without communication cost. Warps within a thread block can synchronize, but it's expensive. FA2 moves work to the no-communication level.

---

## FA2 Forward Pass Algorithm

```
For each Q block Q_i (parallel across thread blocks):
    Initialize: m_i = -inf, l_i = 0, O_i = 0
    
    For each K/V block (K_j, V_j):
        S_ij = Q_i @ K_j.T / sqrt(d)      # [Br, Bc] tile of attention
        
        # Online softmax update
        m_ij = max(S_ij, dim=-1)           # [Br] row maxima of this tile
        m_new = max(m_i, m_ij)             # updated running max
        
        P_ij = exp(S_ij - m_ij)            # [Br, Bc] unnormalized
        l_new = exp(m_i - m_new) * l_i + exp(m_ij - m_new) * rowsum(P_ij)
        
        O_i = (exp(m_i - m_new) * l_i * O_i + P_ij @ V_j) / l_new
        m_i, l_i = m_new, l_new
    
    Write O_i, l_i, m_i to HBM
```

This loop is independent for each Q block — perfect parallelism.

---

## FA2 Backward Pass

The backward pass is more complex because we need dQ, dK, dV.

**Key insight:** FA2 stores only O, l (logsumexp), and m from the forward pass — not the full attention matrix P. In the backward pass, it recomputes attention tiles from Q, K on-the-fly (trading compute for memory).

**Memory for backward:** O(N) — same as forward. No need to store the N×N P matrix.

---

## Speedup Sources

| Source | FA1 → FA2 gain |
|---|---|
| Better warp partitioning | 1.5–2× |
| Reduced shared memory reads/writes | 20% |
| Supports head dim 256 (FA1 max 128) | enables larger models |
| Causal masking optimization | avoids computing masked blocks |

**Total: FA2 is 2× faster than FA1 in practice.**

---

## Multi-Head Attention Parallelism in FA2

FA2 also parallelizes across attention heads. For MHA with H heads and batch size B:

**Total thread blocks:** `B × H × (N / Br)` — each block handles one (batch, head, Q-tile) combination.

For causal (autoregressive) models, this saturates GPU utilization when `B × H` is large. For small batch (e.g., single user inference), FA2 introduces **sequence-length parallelism:** split the sequence across thread blocks even within a single head.

---

## FA2 Performance Numbers

From the paper (A100 80GB, BF16):

| Sequence length | Standard Attention (TFLOPs/s) | FA2 (TFLOPs/s) | Speedup |
|---|---|---|---|
| 512 | 25 | 155 | 6.2× |
| 1024 | 30 | 175 | 5.8× |
| 2048 | 35 | 185 | 5.3× |
| 4096 | 35 | 195 | 5.6× |
| 8192 | OOM | 195 | — |
| 16384 | OOM | 200 | — |

Peak theoretical A100 BF16: 312 TFLOPs/s. FA2 reaches ~65% MFU (model FLOP utilization) for attention.

---

## FA3 (Preview, 2024)

Flash Attention 3 (Dao & Shah, 2024) targets Hopper (H100) architecture:
- **TMA (Tensor Memory Accelerator):** Async HBM→SRAM loads while compute runs
- **Warp specialization:** Separate warps for compute vs memory, running concurrently
- **FP8:** Hardware FP8 support on H100 for 2× more compute
- Performance: ~750 TFLOPs/s BF16 on H100, ~1200 TFLOPs/s FP8

---

## Using FA2 in Practice

```python
# PyTorch 2.0+ integrates FA2 via SDPA (scaled_dot_product_attention)
import torch

# Automatic FA2 backend selection
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,       # use FA2 if possible
    enable_math=False,       # disable naive implementation
    enable_mem_efficient=False
):
    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value,
        attn_mask=None,      # None → assumes causal if is_causal=True
        dropout_p=0.0,
        is_causal=True       # enables causal masking optimization
    )

# Or via HuggingFace Transformers
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8b-hf",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
)
```

**Supported dtypes:** float16, bfloat16 (not float32 — too slow to be worthwhile)
**Minimum head dim:** 16; **Maximum head dim:** 256 (FA2), 128 (FA1)

---

## Common  Questions

- "What specifically did FA2 improve over FA1?"
- "How does FA2 avoid the N×N memory problem in standard attention?"
- "What does 'better warp partitioning' mean in the context of FA2?"
- "Why does FA2 need to recompute attention scores in the backward pass?"
- "How does FA2 enable longer context lengths than standard attention?"
- "How do you enable Flash Attention in a HuggingFace model?"
