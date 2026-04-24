# Attention Is All You Need

**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
**Published:** 2017-06-12
**arXiv ID:** 1706.03762
**URL:** https://arxiv.org/abs/1706.03762
**Venue:** NeurIPS 2017

---

## Abstract (verbatim summary)

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and decoder. The best performing models also connect the encoder and decoder through an attention mechanism. The authors propose a new simple network architecture, the Transformer, based **solely on attention mechanisms**, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.

---

## Key Contributions

1. **Transformer architecture** — encoder-decoder model using only attention (no RNN/CNN)
2. **Scaled dot-product attention** — the core attention formulation still used today
3. **Multi-head attention** — parallel attention heads operating over different subspaces
4. **Positional encoding** — sinusoidal PE to inject sequence order
5. **Pre/post-norm residual structure** — LayerNorm + residual connections in each sublayer

---

## Architecture Details

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V
```

- Q ∈ R^(n × d_k), K ∈ R^(n × d_k), V ∈ R^(n × d_v)
- Scaling by `sqrt(d_k)`: prevents softmax saturation when d_k is large.
  For random vectors with variance 1, their dot product has variance d_k → scale to variance 1.
- Compared to additive attention (Bahdanau): dot-product is faster and more space-efficient.
  Additive wins at small d_k; scaled dot-product dominates at large d_k.

### Multi-Head Attention

```
MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
head_i = Attention(Q·W^Q_i, K·W^K_i, V·W^V_i)
```

- h=8 heads in the paper, d_k = d_v = d_model/h = 64
- Each head projects into a 64-dimensional subspace
- Total compute = same as single head at d_model=512
- **Why multiple heads?** Different heads learn to attend to different positions and relation types (syntactic, semantic, long-range vs local)

### Encoder

- N=6 identical layers
- Each layer: Multi-Head Self-Attention → Feed-Forward Network
- Residual connection + LayerNorm after each sublayer
- **Post-norm** (LayerNorm after residual): original paper uses post-norm.
  Note: modern models use **pre-norm** (more stable training)

### Decoder

- N=6 identical layers
- Three sublayers per layer:
  1. **Masked** multi-head self-attention (causal — can't see future tokens)
  2. **Cross-attention**: Q from decoder, K/V from encoder output
  3. Feed-forward network
- Residual + LayerNorm after each

### Feed-Forward Network

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

- Inner dimension: d_ff = 2048 (4× d_model = 512)
- Applied identically to each position — position-wise
- Interpreted as: attention gathers info across positions; FFN processes each position independently
- Modern note: SwiGLU replaces ReLU in LLaMA etc.

### Positional Encoding (Sinusoidal)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

- Fixed (not learned), added to input embeddings
- Each dimension has a different frequency (exponentially spaced)
- PE(pos+k) can be expressed as a linear function of PE(pos) — allows attending by relative offset
- **Limitation**: poor extrapolation beyond training length → superseded by RoPE, ALiBi

---

## Training Details

- Dataset: WMT 2014 English-German (4.5M sentence pairs), English-French (36M pairs)
- Optimizer: Adam with custom LR schedule: `lr = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))`
  - Warmup: 4000 steps (increase lr), then inverse sqrt decay
- Regularization: dropout (P_drop = 0.1), label smoothing (ε_ls = 0.1)
- 8 P100 GPUs for 3.5 days (base model), 12 hours (big model)

---

## Results

| Model | EN-DE BLEU | EN-FR BLEU | Training FLOPs |
|---|---|---|---|
| Previous SOTA (ensemble) | 26.4 | 41.0 | — |
| Transformer (base) | 27.3 | 38.1 | 3.3×10^18 |
| Transformer (big) | 28.4 | **41.0** | 2.3×10^19 |

**28.4 BLEU on EN-DE**: SOTA at the time, beating all previous models including ensembles.

---

## Ablations (Important for Interviews)

| Variation | EN-DE BLEU | Key finding |
|---|---|---|
| Single head (d_k=512) | 23.3 | Multi-head is essential |
| h=16 heads | 25.5 | Too many heads → worse (subspaces too small) |
| h=32 heads | 25.7 | Diminishing returns |
| No dropout | 26.8 | Dropout helps |
| Replace sine PE with learned | 27.3 | Same — PE type doesn't matter much for original task |
| Remove positional encoding | major drop | PE is essential |

**Key ablation insight for interviews:** Single-head at the same total dimension performs much worse than multi-head → the multi-head structure (not just dimension) is what matters.

---

## What Changed Since 2017 (Modern Adaptations)

| Original Paper | Modern LLMs (2024-2025) |
|---|---|
| Post-norm (norm after residual) | Pre-norm (norm before sublayer) |
| Sinusoidal PE | RoPE (applied inside attention) |
| Multi-Head Attention (MHA) | Grouped-Query Attention (GQA) |
| ReLU in FFN | SwiGLU |
| Encoder-Decoder (for MT) | Decoder-only (GPT-style) |
| Adam with warmup | AdamW with cosine LR |
| No bias in attention | Still no bias (confirmed good) |
| d_ff = 4×d_model | d_ff = ~8/3×d_model (SwiGLU variant) |

---

## Interview-Relevant Insights

**Why did the Transformer replace RNNs?**
- **Parallelization**: RNNs process tokens sequentially (each step depends on previous). Transformers process all positions simultaneously → much faster training.
- **Long-range dependencies**: RNNs lose information over long sequences (even LSTMs). Attention directly connects any two positions in O(1) layers.
- **No gradient vanishing across positions**: attention bypasses the sequential bottleneck.

**The paper's most impactful contribution (in hindsight):** Decoder-only pre-training for language modeling. GPT (2018) took the decoder half and used it with next-token prediction → the LLM paradigm. This wasn't GPT's architectural innovation — GPT just simplified the Transformer.

**Scaling**: Transformers scale exceptionally well (Chinchilla, GPT-3, scaling laws). RNNs don't scale as cleanly → this is why transformers won.

**What the paper got wrong (corrected by later work):**
- Post-norm → pre-norm is better for deep models
- Learned PE ≈ sinusoidal PE at fixed length, but sinusoidal fails to extrapolate → RoPE
- MHA → GQA (KV cache reduction is critical at inference)

---

## Common Interview Questions Sourced From This Paper

- "What is scaled dot-product attention? Why divide by sqrt(d_k)?"
- "What is multi-head attention? Why not just use one large head?"
- "What is the architecture difference between encoder and decoder in the original Transformer?"
- "What is cross-attention? How does it connect encoder and decoder?"
- "Why did Transformers replace RNNs for NLP tasks?"
