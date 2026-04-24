# RLHF: Training Language Models to Follow Instructions with Human Feedback

**Primary paper:** "Training language models to follow instructions with human feedback" (InstructGPT)
**Authors:** Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, et al. (OpenAI)
**Published:** 2022-03-04
**arXiv ID:** 2203.02155
**URL:** https://arxiv.org/abs/2203.02155

Also covers: Stiennon et al. 2020 "Learning to summarize with human feedback" (first major RLHF paper)

---

## The Problem RLHF Solves

**Language model pretraining objective:** Predict the next token. This makes models good at text completion but NOT at being helpful, honest, or harmless. The model learns to mimic internet text — including harmful, biased, or off-topic content.

**The alignment problem:** How do you train a model to do what users actually want, not just what maximizes next-token prediction?

RLHF's answer: collect human feedback on model outputs, train a reward model on that feedback, then fine-tune the LM to maximize the reward.

---

## The Three-Stage Pipeline

### Stage 1: Supervised Fine-Tuning (SFT)

**Goal:** Get the base model to follow a basic instruction format.

**Data:**
- Human-written demonstrations: (prompt, ideal response) pairs
- ~13,000 examples for InstructGPT
- Collected from labelers who write ideal responses to sampled prompts

**Training:**
- Standard cross-entropy loss on response tokens only
- Prompt tokens are masked (not included in loss)
- Typically: 1–3 epochs on the demonstration dataset

**Result:** Model that follows instructions but may still be unhelpful, overly cautious, or untruthful.

**Why SFT is non-optional:** Training PPO directly from a raw pretrained model is extremely unstable — the KL constraint fails when the initial policy is completely random relative to instruction-following. SFT provides a good starting point.

### Stage 2: Reward Model Training

**Goal:** Learn to predict which of two model outputs humans prefer.

**Data collection:**
- Sample K (=4–9) model outputs per prompt
- Labelers rank outputs from best to worst
- Extract K(K-1)/2 comparison pairs from each ranking

**Architecture:**
- Initialize from the SFT model (same architecture)
- Replace the language model head with a linear layer → scalar reward output
- The scalar is output at the final token position (EOS token)

**Loss function (Bradley-Terry preference model):**
```
L_RM = -E_{(x,y_w,y_l)} [ log σ(r(x, y_w) - r(x, y_l)) ]
```
- y_w: the preferred (winning) response
- y_l: the dispreferred (losing) response
- r(x, y): scalar reward from the RM
- σ: sigmoid function

This maximizes the probability that the chosen response gets a higher reward than the rejected one.

**Scale of RM training for InstructGPT:**
- ~33,000 comparisons
- RM trained for 1 epoch (overfitting is a concern with pairwise data)

### Stage 3: Reinforcement Learning with PPO

**Goal:** Optimize the SFT model to generate responses that the RM scores highly, while not drifting too far from SFT behavior.

**Objective:**
```
max_{π_θ}  E_{x~D, y~π_θ(·|x)} [ r(x,y) ] - β · KL[π_θ(y|x) || π_SFT(y|x)]
```

- `r(x,y)`: reward model score
- `β·KL[...]`: penalty for diverging from the SFT model
- β ≈ 0.01–0.1 (controls alignment vs capability tradeoff)

**The KL penalty is critical:**
Without it, the policy learns to "game" the reward model (reward hacking):
- Write very long responses (RM was trained on longer = better)
- Use specific phrases the RM likes regardless of their relevance
- Produce confident-sounding but wrong answers that score well
KL penalty: "stay close to SFT behavior, just improve it at the margin"

**PPO details in RLHF context:**
- Policy: the LM being trained (π_θ)
- Reference policy: frozen SFT model (π_SFT)
- Value function: separate model estimating expected cumulative reward
- Rollout: generate a full response y given prompt x
- Reward: r(x,y) - β·KL_per_token
- 4 models in GPU memory: policy + reference + RM + value (each ~same size as SFT)
- Training: ~1 epoch over the RL dataset (50k–100k prompts)

---

## Results (InstructGPT vs GPT-3)

**Key finding:** InstructGPT (1.3B params) is preferred over GPT-3 (175B params) by human evaluators on following instructions.

**Win rates on TruthfulQA, factual queries, instruction following:**
- InstructGPT significantly outperforms GPT-3 base
- "Alignment tax": small performance degradation on NLP benchmarks (the model trades benchmark performance for helpfulness)
- The alignment tax can be reduced by mixing in some pretraining data during RLHF

**Generalization:** RLHF on English instructions generalizes to other languages somewhat — the model learns the general concept of "following instructions" not just English-specific patterns.

---

## Reward Model Overoptimization (Reward Hacking)

**Problem:** As PPO training proceeds, the policy increasingly exploits the RM's weaknesses. RM score ↑ while actual quality (human evaluation) eventually ↓.

**The overoptimization curve:**
- KL divergence from SFT: 0 → 4 → 8 → 16 nats
- RM score: monotonically increases with KL
- Human preference: increases up to KL~4, then decreases
- Optimal policy is NOT the one with maximum RM score

**Mitigations:**
1. **KL penalty**: explicit regularization that limits divergence
2. **Ensembled RMs**: average multiple RMs trained on different splits (harder to game all)
3. **Conservative KL β**: larger β → slower improvement but more stable
4. **RM data refresh**: collect new preference data on the current policy's outputs
5. **Constitutional AI / RLAIF**: use AI critique to filter RM targets

---

## RLHF Variants and Successors

| Method | Stage 3 | Key Difference |
|---|---|---|
| RLHF + PPO | On-policy, online | Standard, needs 4 models |
| RLHF + GRPO | On-policy, online | No value model (DeepSeek-R1) |
| DPO | Offline supervised | No RM, no PPO, 2 models |
| KTO | Offline supervised | Uses single responses (not pairs) |
| RLAIF | Online | AI instead of human labels (Constitutional AI) |
| ORPO | Offline supervised | SFT + preference in one loss |
| PPO + process reward | On-policy | Reward per step, not just final |

---

## Data Collection Challenges (Real-World)

**Labeler quality is everything.** Key challenges:
- Labeler agreement is ~73% (humans often disagree about quality)
- Labeler instructions (the "constitution") heavily influence what behavior is rewarded
- Labelers can't evaluate technical accuracy in specialized domains (math, code, medicine)
- Scale: collecting millions of comparisons is expensive ($10–50 per comparison)

**Solutions:**
- Active learning: collect comparisons where the model is most uncertain
- RLAIF: use GPT-4 or Claude to label — cheaper, scalable, surprisingly good correlation with humans
- Domain expert labelers for technical content

---

## Interview-Relevant Insights

**The most common RLHF interview error:** Forgetting that 4 models are loaded simultaneously during PPO. Memory is the main practical constraint.

**Why not skip SFT?** Starting PPO from raw pretraining makes the KL constraint meaningless — the initial policy is too random relative to instruction-following. SFT gives a good starting point from which PPO makes incremental improvements.

**β in practice:** β too small → reward hacking. β too large → PPO does nothing (KL penalty dominates). For InstructGPT, β=0.02 was used. In practice: tune β via RM score + human eval correlation during training.

**The 1.3B > 175B finding:** This is the most cited result from the paper. It shows that alignment matters more than scale for instruction following — a well-aligned small model beats a much larger unaligned one.

---

## Common Interview Questions From This Paper

- "Walk me through the three stages of RLHF."
- "What is the reward model and how is it trained?"
- "Why is the KL penalty in the PPO objective necessary?"
- "How many models are in GPU memory during PPO training?"
- "What is reward hacking and how do you detect and prevent it?"
- "Why is InstructGPT (1.3B) preferred over GPT-3 (175B) for instruction following?"
- "What data do you need to collect for RLHF?"
