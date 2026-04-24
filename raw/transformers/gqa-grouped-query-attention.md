# GQA: Grouped-Query Attention

**Paper:** "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
**Authors:** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zelenski, Slav Petrov, Sumit Sanghai (Google Research)
**Published:** 2023-05-22
**arXiv ID:** 2305.13245
**Context:** Addresses KV cache memory bottleneck in inference; adopted by Llama 2/3, Mistral, Gemma, Falcon

---

## The Problem: KV Cache Memory at Scale

During autoregressive inference, the KV cache stores:
```
KV cache = 2 × n_layers × n_heads × d_head × seq_len × bytes_per_element
```

For Llama 2 70B (MHA baseline):
- 80 layers × 64 heads × 128 d_head × 4096 seq × 2 bytes = **~68 GB** for KV cache alone

This limits:
1. **Batch size:** KV cache occupies memory that could serve more concurrent requests
2. **Maximum sequence length:** Long contexts exhaust GPU VRAM
3. **Inference throughput:** Memory bandwidth (moving KV cache to/from HBM) is the bottleneck at decode time

---

## Attention Variants: MHA → MQA → GQA

### Multi-Head Attention (MHA) — Original Transformer (Vaswani 2017)

```
Q: [n_heads, seq, d_head]    # one query head per attention head
K: [n_heads, seq, d_head]    # one key head per attention head  
V: [n_heads, seq, d_head]    # one value head per attention head
```

Each query head has its own dedicated key and value heads.

### Multi-Query Attention (MQA) — Noam Shazeer 2019

```
Q: [n_heads, seq, d_head]    # n_heads query heads (unchanged)
K: [1, seq, d_head]          # single shared key head
V: [1, seq, d_head]          # single shared value head
```

All query heads share a **single** K and V projection. KV cache reduced by n_heads×.

**Memory savings:** 64× smaller KV cache (for 64-head model)
**Quality loss:** Notable degradation on many tasks (sharing one K/V across 64 heads is too aggressive)
**Used in:** PaLM, Falcon (some variants)

### Grouped-Query Attention (GQA) — Google 2023

```
Q: [n_heads, seq, d_head]          # n_heads query heads
K: [n_kv_heads, seq, d_head]       # n_kv_heads << n_heads shared key heads
V: [n_kv_heads, seq, d_head]       # same n_kv_heads value heads
```

Query heads are divided into **G groups**. Each group shares one K/V head.

```
n_kv_heads = n_heads / G     # G = group size (how many Q heads per KV head)
```

**Example:** n_heads=32, G=8 → n_kv_heads=4. Four KV heads, 8 query heads per KV head.

---

## GQA Attention Computation

```python
import torch
import torch.nn.functional as F
import math

def grouped_query_attention(Q, K, V, n_heads, n_kv_heads):
    """
    Q: [batch, seq, n_heads * d_head]
    K: [batch, seq, n_kv_heads * d_head]  
    V: [batch, seq, n_kv_heads * d_head]
    """
    batch, seq, _ = Q.shape
    d_head = Q.shape[-1] // n_heads
    group_size = n_heads // n_kv_heads
    
    # Reshape
    Q = Q.view(batch, seq, n_heads, d_head)
    K = K.view(batch, seq, n_kv_heads, d_head)
    V = V.view(batch, seq, n_kv_heads, d_head)
    
    # Expand K, V to match Q (repeat each KV head group_size times)
    K = K.repeat_interleave(group_size, dim=2)  # [batch, seq, n_heads, d_head]
    V = V.repeat_interleave(group_size, dim=2)  # [batch, seq, n_heads, d_head]
    
    # Transpose for batch matmul
    Q = Q.transpose(1, 2)  # [batch, n_heads, seq, d_head]
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)
    
    # Standard attention
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_head)
    weights = F.softmax(scores, dim=-1)
    out = torch.matmul(weights, V)
    
    return out.transpose(1, 2).contiguous().view(batch, seq, n_heads * d_head)
```

In practice, most frameworks use `torch.nn.functional.scaled_dot_product_attention` with Flash Attention 2, which handles GQA natively.

---

## KV Cache Memory Comparison

| Model config | n_heads | n_kv_heads | KV cache (4K ctx, FP16) | vs MHA |
|---|---|---|---|---|
| Llama 2 7B (MHA) | 32 | 32 | 2 GB | baseline |
| Llama 2 70B (GQA) | 64 | 8 | 4.9 GB | 8× smaller than MHA-equivalent |
| Llama 3 8B (GQA) | 32 | 8 | 0.5 GB | 4× smaller |
| Llama 3 70B (GQA) | 64 | 8 | 4.9 GB | 8× smaller |
| Mistral 7B (GQA) | 32 | 8 | 0.5 GB | 4× smaller |
| MQA extreme | 32 | 1 | 62 MB | 32× smaller, quality loss |

**Formula:**
```
KV cache = 2 × n_layers × n_kv_heads × d_head × seq_len × bytes
```

---

## Quality vs Efficiency Trade-off

From the paper's evaluation on WMT 2014 EN-DE translation and language modeling:

| Method | Relative quality | KV cache size | Inference speed |
|---|---|---|---|
| MHA | 100% (baseline) | 100% | Baseline |
| MQA | ~97–99% | 1/n_heads | 1.7× faster decode |
| GQA (G=8) | ~99.5% | 1/(n_heads/8) | 1.5× faster decode |
| GQA (G=2) | ~100% | 1/(n_heads/2) | 1.2× faster decode |

**Sweet spot:** G=8 (8 query heads per KV head) delivers MHA-quality with major memory savings. This is what Llama 2/3 and Mistral use.

---

## Converting MHA Checkpoints to GQA

The paper's main contribution is a **recipe for converting existing MHA models to GQA without retraining from scratch:**

1. **Mean pooling:** For each KV head group (e.g., 8 consecutive MHA KV heads), take the mean of their weight matrices
2. **Continue pretraining:** Uptrain for 5% of the original tokens with GQA

This allows reusing existing MHA checkpoints (e.g., GPT-3, T5) and getting GQA inference benefits with minimal training cost.

```python
def convert_mha_to_gqa(K_proj_weights, n_kv_heads):
    """
    K_proj_weights: [n_heads, d_head, d_model]  (MHA)
    Returns: [n_kv_heads, d_head, d_model]  (GQA)
    """
    n_heads = K_proj_weights.shape[0]
    group_size = n_heads // n_kv_heads
    
    # Group heads and mean-pool
    grouped = K_proj_weights.view(n_kv_heads, group_size, *K_proj_weights.shape[1:])
    return grouped.mean(dim=1)
```

---

## Why GQA Speeds Up Inference

At decode step (autoregressive generation, batch size B, sequence length T):
- Load KV cache: `2 × n_kv_heads × d_head × T × B × bytes` from HBM
- Memory bandwidth is the bottleneck at small batch sizes (not compute)
- Fewer KV heads = less HBM bandwidth needed = faster decode

**Throughput gain:** GQA with 8× fewer KV heads = 8× less KV cache bandwidth = significant throughput improvement at the same quality.

---

## GQA in Popular Models

| Model | n_heads | n_kv_heads | Group size |
|---|---|---|---|
| Llama 2 7B | 32 | 32 | 1 (= MHA) |
| Llama 2 13B | 40 | 40 | 1 (= MHA) |
| Llama 2 70B | 64 | 8 | 8 |
| Llama 3 8B | 32 | 8 | 4 |
| Llama 3 70B | 64 | 8 | 8 |
| Mistral 7B | 32 | 8 | 4 |
| Gemma 7B | 16 | 16 | 1 (= MHA) |
| Gemma 2 9B | 8 | 4 | 2 |
| Falcon 7B | 71 | 1 | 71 (= MQA) |

---

## Common Interview Questions

- "What is GQA and why was it introduced?"
- "How does GQA reduce KV cache memory? Give a concrete example with Llama 3."
- "What's the difference between MHA, MQA, and GQA?"
- "Why does reducing KV cache size speed up inference?"
- "At what group size does GQA quality match MHA? What does the paper show?"
- "How do you convert an MHA checkpoint to GQA?"
