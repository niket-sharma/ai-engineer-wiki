# Andrej Karpathy — "Let's build GPT: from scratch, in code, spelled out"

**Source:** https://www.youtube.com/watch?v=kCc8FmEb1nY
**Duration:** ~2 hours
**Date:** January 2023
**Companion code:** https://github.com/karpathy/nanoGPT

---

## Overview

Karpathy builds a GPT-style language model from scratch in ~2 hours, starting from a bigram model and incrementally adding all the components of GPT-2 (small). The video is the gold standard for building intuition about transformer internals.

---

## Part 1: The Bigram Model (Baseline)

### Setup
- Dataset: Shakespeare plays (~1M chars)
- Tokenization: character-level (vocab_size=65: all unique chars)
- Task: given a character, predict the next character

### Bigram Model (No Attention)
```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly predicts the next token logits
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C)
        ...
```
- `logits[b, t, :]` is the prediction for what follows token `t` in batch `b`
- No context — each position only sees itself
- Training: minimize cross-entropy loss

**Why this is a useful baseline:** Shows the training loop, loss function, and generation loop without attention complexity.

---

## Part 2: Self-Attention From Scratch

### Incremental Build-Up

**Version 1: Average past context (no learning)**
```python
# For each position, average all previous token embeddings
xbow = torch.zeros(B, T, C)
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]  # (t, C)
        xbow[b, t] = xprev.mean(0)
```
Problem: slow (O(T²) naive), and uniform average loses position information.

**Version 2: Matrix multiplication trick**
```python
# Lower triangular matrix (causal mask)
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros(T, T)
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)  # uniform over past
xbow2 = wei @ x  # (T, T) @ (B, T, C) → (B, T, C)
```
Key insight: `wei @ x` computes a weighted average of past token embeddings. With uniform weights, this is the same as the loop above but vectorized.

**Version 3: Self-Attention (the real thing)**
```python
head_size = 16
key   = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)    # (B, T, head_size)
q = query(x)  # (B, T, head_size)

# Compute attention weights
wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)  # (B, T, T)
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)  # (B, T, T)

v = value(x)  # (B, T, head_size)
out = wei @ v  # (B, T, head_size)
```

**Karpathy's framing — "attention is communication":**
> "Every token emits a query ('what am I looking for?') and a key ('what do I contain?'). The dot product of query × key gives the affinity. Then I aggregate the values based on those affinities."

**The critical point about masking:**
- `tril == 0` positions get `-inf` before softmax
- After softmax: `-inf` → 0 (future tokens get zero weight)
- This makes attention **causal** — each position only attends to itself and the past

**Why scale by `head_size ** -0.5`:**
- Without scaling: for head_size=16, `q @ k^T` has variance ~16
- After softmax: the distribution becomes very sharp (peaky) → gradients vanish
- Scaling makes variance ~1 → softer distribution → better gradient flow

---

## Part 3: Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embd)  # W_O
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)
```

**Karpathy's intuition:** Multiple heads = multiple "communication channels." One head might learn to look for nouns, another for verbs, another for recent history, etc.

---

## Part 4: Feed-Forward Network

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)
```

**Karpathy's intuition:** Attention is "communication" — tokens talk to each other. FFN is "computation" — each token processes what it gathered independently. These alternate: communicate → compute → communicate → compute.

Note: d_ff = 4 × d_model in the original paper. This 4× ratio is conventional and appears in essentially all transformers (though LLaMA uses 8/3× with SwiGLU).

---

## Part 5: Transformer Block

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # pre-norm residual
        x = x + self.ffwd(self.ln2(x)) # pre-norm residual
        return x
```

**Pre-norm vs post-norm:**
Karpathy uses **pre-norm** (LayerNorm before attention/FFN). The original "Attention Is All You Need" paper used post-norm. Pre-norm is more stable for deep models and became standard.

**Residual connections (`x + ...`):**
- Allow gradients to flow directly back to earlier layers
- Without them: deep transformers are very hard to train (gradient vanishing)
- Also allow the model to learn "identity" — just pass x through if nothing useful to add

---

## Part 6: Full GPT Model

```python
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)       # (B, T, C)
        pos_emb = self.position_embedding_table(        # (T, C)
            torch.arange(T, device=device))
        x = tok_emb + pos_emb                          # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                       # (B, T, vocab_size)
        ...
```

**Hyperparameters (nanoGPT "GPT-2 small" equivalent):**
```python
n_embd  = 768   # d_model
n_head  = 12    # attention heads
n_layer = 12    # transformer blocks
block_size = 1024  # context length
vocab_size = 50257 # BPE tokenizer
dropout = 0.2
```
~117M parameters.

---

## Part 7: Training Loop

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for steps in range(max_iters):
    # Sample batch
    xb, yb = get_batch('train')
    
    # Forward + loss
    logits, loss = model(xb, yb)
    
    # Backward
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

**Key choices:**
- **AdamW not Adam**: weight decay decoupled from gradient → better regularization at scale
- `set_to_none=True`: more memory efficient than zeroing (skips memory write)
- No gradient clipping in the simple version (GPT-2 used it at scale: `nn.utils.clip_grad_norm_`)

---

## Key Intuitions (Karpathy's Own Words)

**On attention:**
> "Attention is a communication mechanism. Nodes in a graph are communicating with each other. Each node has a query: 'what am I looking for?' and a key: 'what do I contain?' The dot product tells you how much each node wants to attend to every other."

**On residual connections:**
> "Think of the residual stream as a highway. The transformer blocks are the on-ramps and off-ramps. Information flows along the highway; blocks read from it (via LayerNorm → attention/FFN) and write back to it (via the residual add)."

**On why transformers work:**
> "The attention mechanism allows any token to look at any other token in the context and gather information from it. That direct path is the key difference from RNNs where information has to flow sequentially."

**On the FFN:**
> "After tokens have talked to each other (attention), each token thinks independently (FFN). The FFN is where the actual 'computation' happens — think of the keys and values in the FFN as a lookup table of facts the model has learned."

---

## BPE Tokenization (Brief)

Real GPT uses Byte Pair Encoding (BPE) not character-level:
- Start with characters as tokens
- Iteratively merge the most frequent adjacent pair into a new token
- vocab_size=50257 for GPT-2

BPE balances: single characters (handle any input) + common words/subwords (efficiency).

---

## Interview-Relevant Insights From This Lecture

1. **Attention = soft lookup in a graph** — Karpathy's framing is the clearest explanation anywhere
2. **`-inf` mask before softmax → causal attention**: know how to implement this
3. **Pre-norm vs post-norm**: know the code difference, know pre-norm is current standard
4. **Residual stream**: think of it as a shared information highway — blocks read and write to it
5. **FFN as key-value memory**: per-position computation after cross-position communication

**Building from scratch means you can answer:**
- "Write the self-attention equations"
- "How do you implement causal masking in code?"
- "Why AdamW over Adam?"

---

## Common Interview Questions Sourced From This Lecture

- "Walk me through implementing scaled dot-product attention in PyTorch."
- "Why do we use pre-norm instead of post-norm in modern transformers?"
- "What is the role of the feed-forward network in a transformer block?"
- "What is the residual connection doing in a transformer?"
- "How would you implement causal masking?"
