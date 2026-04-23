---
title: "Transformers Interview Q&A"
tags: [interview-qa, transformers]
related: ["[[attention-mechanism]]", "[[transformer-architecture]]", "[[positional-encoding]]", "[[kv-cache]]", "[[flash-attention]]"]
last_updated: 2026-04-22
---

# Transformers Interview Q&A

---

## L1 — Conceptual

### Q1. What is the attention mechanism and why does it matter?
**A:** Attention lets each token in a sequence selectively gather information from every other token by computing weighted sums of value vectors. Given query Q, key K, and value V matrices, it computes `softmax(QK^T / sqrt(d_k)) · V`. The result: every token can directly attend to any other token in O(1) layers (vs. O(n) for RNNs). This gives transformers their ability to model long-range dependencies and parallelize training.

**Common follow-ups:**
- Why divide by sqrt(d_k)?
- What's the difference between self-attention and cross-attention?

---

### Q2. What is the difference between encoder-only, decoder-only, and encoder-decoder transformers?

**A:**
| Type | Attention | Pretraining | Examples | Use case |
|---|---|---|---|---|
| Encoder-only | Bidirectional | MLM (predict masked tokens) | BERT, RoBERTa | Classification, embeddings |
| Decoder-only | Causal (left-to-right) | Next-token prediction | GPT, Llama | Generation |
| Encoder-Decoder | Bidirectional encoder + causal decoder | Seq2seq | T5, BART | Translation, summarization |

Decoder-only (GPT-style) has become dominant for LLMs because next-token prediction scales better and the architecture is simpler.

---

### Q3. What is a KV cache and why is it important?

**A:** During autoregressive generation, the K and V projections for already-generated tokens don't change. KV cache stores them so they don't get recomputed each step. Without it: generating token N costs O(N²) total compute. With it: O(N). The tradeoff is GPU memory — KV cache grows linearly with sequence length and is the primary memory bottleneck for long-context inference.

---

### Q4. Why do transformers need positional encoding?

**A:** Attention is permutation-invariant — shuffling input tokens gives the same attention scores. PE injects position information so the model knows token order. Modern LLMs use RoPE (Rotary Position Embedding): instead of adding position vectors to embeddings, RoPE rotates Q and K by position-dependent angles, so the dot product encodes relative position (n-m) naturally. This is better than absolute PE for length generalization.

---

## L2 — Technical

### Q5. Derive the scaled dot-product attention formula and explain each component.

**A:**
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V
```
- **QK^T**: dot product similarity between all query-key pairs. Entry (i,j) = how much token i should attend to token j.
- **/ sqrt(d_k)**: scaling to prevent dot products from growing large as d_k increases (which would push softmax into saturation, killing gradients). For d_k-dim vectors sampled from N(0,1), the variance of their dot product is d_k — dividing by sqrt(d_k) normalizes variance to 1.
- **softmax(...)**: converts raw scores to attention weights summing to 1 (probability distribution over tokens)
- **· V**: weighted sum of value vectors — the output is a mixture of values, weighted by attention.

Complexity: O(n² · d) time, O(n²) space (for the attention matrix). Flash Attention reduces space to O(n) by fusing computation in SRAM tiles.

---

### Q6. What is Multi-Head Attention (MHA) and why use multiple heads?

**A:**
```
MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)
```
Each head operates on a d_model/h dimensional subspace. Total compute is the same as single-head attention, but multiple heads allow the model to simultaneously attend to:
- Different positional offsets
- Syntactic vs semantic relationships
- Local vs global context

Empirically essential — single-head attention underperforms significantly.

---

### Q7. Walk me through the memory cost of a KV cache for Llama 3 8B at a 4096-token context.

**A:**
```
KV cache bytes = 2 × n_layers × n_kv_heads × d_head × seq_len × bytes_per_element
               = 2 × 32 × 8 × 128 × 4096 × 2  (bfloat16)
               = 536 MB
```
Llama 3 8B uses GQA with 8 KV heads (vs 32 Q heads) — this is already 4× smaller than if it used MHA. At 128k context, this becomes ~17 GB — competitive with the model weights themselves (16 GB in bfloat16). This is why KV cache management (paged attention, quantization) is critical for long-context serving.

---

### Q8. What is Flash Attention and what problem does it solve?

**A:** Flash Attention is an IO-aware CUDA kernel for standard scaled dot-product attention. The bottleneck in vanilla attention is NOT FLOPs — it's HBM (GPU DRAM) bandwidth. Standard attention materializes the n×n attention matrix in HBM three times (write QK^T, read for softmax, read for V multiply). Flash Attention tiles Q, K, V into SRAM blocks and computes running softmax using the online softmax algorithm — never materializing the full n×n matrix in HBM. Result: same mathematical output, 2–4× wall-clock speedup, O(n) memory vs O(n²). Bit-identical to standard attention.

---

### Q9. Explain the difference between MHA, MQA, and GQA.

**A:**
| Variant | KV heads | KV cache vs MHA | Quality |
|---|---|---|---|
| MHA | = Q heads (32) | 1× | Best |
| MQA | 1 | 32× smaller | Some degradation |
| GQA | groups g (e.g. 8) | n_heads/g smaller | Near-MHA |

GQA groups Q heads to share KV heads. With h=32 Q heads and g=8 KV groups, each KV head serves 4 Q heads. Memory savings: 4× vs MHA with minimal quality loss. Used in Llama 3, Mistral, Gemma — essentially all modern open-source LLMs.

---

## L3 — Applied / System Design

### Q10. You're serving a 70B model with 128k context for 1000 concurrent users. What are the main bottlenecks and how do you address them?

**A:** 
**Bottlenecks:**
1. **KV cache memory**: 128k context × KV cache formula = ~200 GB just for KV cache at full batch — exceeds GPU memory
2. **Prefill compute**: First token generation (prefill) for long prompts is compute-bound and blocks GPU
3. **Decode throughput**: Sequential decoding is memory-bandwidth-bound

**Solutions:**
- Paged attention (vLLM): non-contiguous KV pages, no memory fragmentation
- Prefix caching: share KV pages for common system prompt across requests
- KV quantization (int8/fp8): halve KV cache memory at minimal quality cost
- Chunked prefill: split long prefills into chunks to avoid blocking decode
- Tensor parallelism across GPUs for the 70B model weights
- Speculative decoding: draft small model to propose tokens, large model to verify

---

### Q11. Describe how you would extend a Llama 3 8B model from 8k to 128k context length.

**A:**
RoPE was trained at 8k but we need 128k (16× extension). Options:
1. **RoPE scaling (linear)**: Divide all RoPE frequencies by scale factor 16. Preserves relative position info but degrades on positions never seen.
2. **YaRN (Yet Another RoPE extensioN)**: Splits RoPE dimensions into low-freq (scale up), high-freq (no scaling). More principled, better quality. Fine-tune on a small amount of long-context data.
3. **Continue pre-training**: Extend context in training with long documents (Llama 3.1's approach — trained at 128k natively).

In practice: use YaRN + fine-tuning on 1-5% long-context data. Monitor "lost in the middle" degradation with RULER benchmark. Also need to scale KV cache budget accordingly.

---

### Q12. Why does Flash Attention enable training on longer sequences but not actually increase the maximum context length by itself?

**A:** FA reduces the **memory** required for attention from O(n²) to O(n), which removes one bottleneck for long sequences. But:
1. **KV cache** still grows O(n) with context — this is a different memory pool
2. **Positional encoding** — the model wasn't trained at longer positions, so it doesn't generalize (separate issue requiring RoPE scaling)
3. **The FFN and other layers** also consume memory that scales with sequence length
4. **Compute** is still O(n²) FLOPs — FA just hides the memory cost, not the compute cost

FA is necessary but not sufficient for long context. You also need: RoPE extension + KV cache management + sufficient GPU memory for activations.

---
