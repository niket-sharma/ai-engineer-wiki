# RoFormer: Enhanced Transformer with Rotary Position Embedding

**Authors:** Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu
**Published:** 2021-04-20
**arXiv ID:** 2104.09864
**URL:** https://arxiv.org/abs/2104.09864

Also includes notes on:
- ALiBi (Press et al. 2022, arXiv 2108.12409)
- YaRN (Peng et al. 2023, arXiv 2309.00071)

---

## The Problem: Why Absolute PE Fails

The original sinusoidal and learned absolute positional encodings:
1. **Added to embeddings** before the transformer layers
2. Models trained at 512 tokens can't generalize to 513+ tokens
3. Relative position information is implicit and has to be learned indirectly

What we really want: attention dot products that **naturally encode relative position** (token A attends to token B based on how far apart they are, not their absolute positions).

---

## RoPE: Rotary Position Embedding

### Core Idea

Apply a position-dependent rotation to Q and K vectors **before** computing attention. The rotation is designed so the dot product `q_m · k_n` depends only on the relative position `(m-n)`, not on absolute positions `m` and `n` separately.

### Mathematical Formulation

For a single 2D case (pair of dimensions):
```
R(θ, m) = [[cos(mθ)  -sin(mθ)]
            [sin(mθ)   cos(mθ)]]
```

Applied to a query vector at position m:
```
q_m = R(θ, m) · q
```

Applied to a key vector at position n:
```
k_n = R(θ, n) · k
```

The dot product:
```
q_m^T · k_n = q^T · R(θ, m)^T · R(θ, n) · k
            = q^T · R(θ, n-m) · k
```

**Key property:** The dot product depends only on `(n-m)` — the relative position. This gives us relative position encoding for free, without ever computing pairwise distances explicitly.

### For d-dimensional vectors

RoPE pairs up dimensions and applies 2D rotations to each pair:
```
q = [q_0, q_1, q_2, q_3, ..., q_{d-2}, q_{d-1}]
     ↓      ↓        ↓           ↓
  rotate   rotate  rotate ...  rotate
  with θ_0  θ_1    θ_2    ...  θ_{d/2-1}
```

Each pair has a different base frequency:
```
θ_i = 1 / (10000^(2i/d))     for i = 0, 1, ..., d/2-1
```

Low-frequency pairs rotate slowly (encode long-range structure).
High-frequency pairs rotate quickly (encode local structure).

### Implementation

```python
def apply_rope(x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    # x: [batch, seq_len, n_heads, head_dim]
    # Build rotation matrices
    dim = x.shape[-1]
    theta = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    angles = torch.outer(position_ids.float(), theta)  # [seq_len, dim/2]
    cos = torch.cos(angles)  # [seq_len, dim/2]
    sin = torch.sin(angles)  # [seq_len, dim/2]
    
    # Rotate pairs of dimensions
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rotated_1 = x1 * cos - x2 * sin
    x_rotated_2 = x1 * sin + x2 * cos
    return torch.stack([x_rotated_1, x_rotated_2], dim=-1).flatten(-2)
```

### Key Properties

1. **No extra parameters**: rotation matrices are computed deterministically from position — nothing to learn
2. **Applied inside attention**: not added to embeddings, but applied to Q and K before their dot product
3. **Relative position**: dot product naturally encodes relative distance
4. **Long-range decay**: by design, the inner product between distant positions decays — models locality
5. **Extensible**: can extend beyond training length with scaling tricks

---

## Context Length Extension with RoPE

### The Problem

A model trained with max_pos=2048 has never seen position embeddings for position 2049. When you try to serve at 4096 tokens, the rotation angles are "out of distribution".

### Linear RoPE Scaling

Divide all position indices by a scale factor s:
```
position_id_scaled = position_id / s
```
With s=4 and training at 2048: effective positions become 0–512, which are all in-distribution. Then 8192 positions map to 0–2048 range → length extension of 4×.

**Simple but imperfect**: low-frequency dimensions (large θ) are handled well; high-frequency dimensions (small θ) are compressed too much.

### YaRN (Yet Another RoPE extensioN)

Better approach: treat different frequency dimensions differently.

- **Low-frequency dimensions** (rotate slowly): scale by s (same as linear scaling)
- **High-frequency dimensions** (rotate quickly): no scaling (leave as-is — they're already short-range)
- **Mid-frequency dimensions**: interpolate between the two

Plus a temperature correction to the attention logits to compensate for distribution shift.

**Result:** Extension from 4k → 128k with only ~0.1% perplexity increase after fine-tuning on ~1B tokens of long documents. Used in Llama 3.1 (128k context).

---

## ALiBi: Attention with Linear Biases

**Paper:** "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
**arXiv:** 2108.12409

### Core Idea

Instead of modifying embeddings OR Q/K vectors, add a **linear bias** directly to the attention scores before softmax:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + m · M) · V
```

Where `M[i,j] = -|i-j|` (negative distance between positions) and `m` is a head-specific slope.

### Slope Assignment

With h attention heads, slopes are geometric sequence:
```
m_1 = 2^(-8/h),  m_2 = 2^(-16/h), ..., m_h = 2^(-8)
```
For h=8: slopes = {1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256}

Different heads have different slopes → different "attention horizons."

### Properties

- **No position-dependent input modification**: no PE added to embeddings or applied to Q/K
- **Excellent length extrapolation**: trained at 1024, works at 4096+ with minimal degradation
- **Inductive bias toward locality**: closer tokens get higher attention → good for language
- **Simple implementation**: just add a bias matrix to attention logits

### ALiBi vs RoPE

| Property | RoPE | ALiBi |
|---|---|---|
| Length extrapolation | Good (with YaRN) | Excellent (no fine-tuning needed) |
| Quality at training length | Best | Slightly lower |
| Implementation | Applied inside attention to Q/K | Added bias matrix |
| Memory overhead | Zero | Zero |
| Used in | Llama, Mistral, Falcon | MPT, BLOOM |

**Verdict:** RoPE (with YaRN for extension) is now more popular. ALiBi is simpler and extrapolates better without fine-tuning, but slightly weaker at the training length. Most frontier labs chose RoPE.

---

## Interview-Relevant Insights

**Why RoPE became the standard:**
1. Relative position is inherent to the formulation (better inductive bias than absolute PE)
2. Compatible with Flash Attention (no changes to the attention kernel needed)
3. Extensible with YaRN scaling to very long contexts
4. Works with KV cache (no interaction issues)

**The key fact to remember:** RoPE is applied to Q and K inside attention, NOT to input embeddings. This is the most common misconception.

**When asked "how does Llama 3 handle 128k context?"**
→ RoPE with YaRN scaling, fine-tuned on long-context data (~1–5% of training), and careful evaluation with RULER benchmark. The model weights themselves are trained on longer sequences — it's not just inference-time scaling.

---

## Common Interview Questions From This Paper

- "What is RoPE? How does it encode relative position?"
- "Why does RoPE encode relative and not absolute position?"
- "How do you extend a model trained at 4k context to 128k? What changes?"
- "What is ALiBi? When would you use it instead of RoPE?"
- "How does YaRN improve on simple linear RoPE scaling?"
