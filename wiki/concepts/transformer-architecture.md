---
title: "Transformer Architecture"
aliases: ["transformer", "GPT architecture", "BERT architecture", "decoder-only", "encoder-only"]
tags: [transformers, architecture, core-concept]
related: ["[[attention-mechanism]]", "[[positional-encoding]]", "[[kv-cache]]", "[[flash-attention]]", "[[lora-qlora]]"]
sources: ["training-knowledge"]
interview_relevance: high
last_updated: 2026-04-22
status: current
---

# Transformer Architecture

## TL;DR
The transformer is a sequence model built entirely on attention (no recurrence, no convolution). It comes in three flavors: encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5). Decoder-only is the dominant architecture for modern LLMs.

## Intuition
The original transformer was designed for machine translation. The encoder reads the source sentence (attending to everything bidirectionally); the decoder generates the target word-by-word, attending to what it has generated so far (causal self-attention) and to the encoder output (cross-attention).

GPT simplified this to decoder-only: predict the next token autoregressively. BERT simplified it to encoder-only: predict masked tokens bidirectionally. Both are widely used; decoder-only won for generative tasks.

## Technical Detail

**Single Transformer Layer (decoder-only):**
```
x = x + Attention(LayerNorm(x))        # residual + attention
x = x + FFN(LayerNorm(x))              # residual + feed-forward
```
- **Pre-norm** (above) is more stable than post-norm (original paper)
- FFN is two linear layers with a nonlinearity: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
- FFN intermediate dim is typically 4× model dim (d_ff = 4 · d_model)

**Full GPT-style model:**
```
Input → Token Embedding + Positional Encoding
      → N × [LayerNorm → MHA → Residual → LayerNorm → FFN → Residual]
      → LayerNorm → Linear → Softmax → Next-token probabilities
```

**Key hyperparameters:**
| Symbol | Meaning | GPT-2 small | Llama 3 8B |
|---|---|---|---|
| d_model | Model dimension | 768 | 4096 |
| n_layers | Number of layers | 12 | 32 |
| n_heads | Attention heads | 12 | 32 |
| d_ff | FFN intermediate dim | 3072 | 14336 |
| vocab_size | Vocabulary | 50257 | 128256 |

**Parameter count estimate (decoder-only):**
- Embedding: vocab_size × d_model
- Per layer: ~12 · d_model² (attention W_Q, W_K, W_V, W_O + FFN W₁, W₂)
- Total ≈ 12 · n_layers · d_model²

## Variants & Extensions

| Architecture | Type | Key Difference | Use Case |
|---|---|---|---|
| BERT | Encoder-only | Bidirectional attention, MLM pretraining | Classification, retrieval |
| GPT | Decoder-only | Causal attention, next-token prediction | Generation |
| T5 | Encoder-Decoder | Cross-attention between encoder and decoder | Translation, summarization |
| Llama 3 | Decoder-only | GQA, RoPE, SwiGLU FFN, no bias | SOTA open-source LLM |
| Mistral | Decoder-only | Sliding window attention + GQA | Efficient inference |

**Modern improvements over original transformer:**
- Pre-norm instead of post-norm (training stability)
- RoPE instead of sinusoidal PE (better length generalization)
- GQA instead of MHA (smaller KV cache)
- SwiGLU instead of ReLU in FFN (better empirical performance)
- No bias terms in attention layers (slightly better at scale)

## Tradeoffs
| Advantage | Disadvantage |
|---|---|
| Global receptive field from layer 1 | O(n²) attention cost — expensive for long context |
| Highly parallelizable (unlike RNNs) | No recurrence → context is bounded by window |
| Scales well with data and compute | Autoregressive generation is sequential |
| Transfer learning via pretraining | Inference KV cache grows with context length |

##  Angles

**What interviewers are really testing:**
- Can you describe a transformer layer component-by-component?
- Do you know the difference between encoder-only, decoder-only, encoder-decoder?
- Can you estimate parameter counts and explain what drives them?
- Do you understand why pre-norm became standard?

**Common follow-up questions:**
- "Walk me through what happens at inference time for a single forward pass."
- "How does the FFN relate to attention? What does each one learn?"
- "Why is d_ff = 4 · d_model? What would happen if you changed it?"
- "Llama 3 uses SwiGLU — what is that and why is it better?"
- "What's the difference in pretraining objectives between GPT and BERT?"

**Gotchas / misconceptions:**
- The FFN is often underappreciated — research shows it acts as a key-value memory
- Residual connections are critical: without them, gradients vanish in deep networks
- "Pre-norm" means LayerNorm is applied BEFORE attention/FFN, not after
- Modern LLMs drop bias terms in linear layers but keep them in LayerNorm

## Connections
- [[attention-mechanism]] — core building block inside each transformer layer
- [[positional-encoding]] — applied to token embeddings before the first layer
- [[kv-cache]] — caches K/V from each layer to speed up autoregressive generation
- [[flash-attention]] — memory-efficient attention implementation
- [[lora-qlora]] — PEFT adapters inserted into transformer weight matrices

## Sources
- Training knowledge (Vaswani et al. 2017; Brown et al. 2020 GPT-3; Touvron et al. 2023 Llama)
