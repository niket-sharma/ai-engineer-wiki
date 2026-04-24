---
title: "RL & Alignment  Q&A"
tags: [-qa, alignment, fine-tuning, rl]
related: ["[[rlhf]]", "[[ppo]]", "[[dpo]]", "[[grpo]]", "[[lora-qlora]]"]
last_updated: 2026-04-22
---

# RL & Alignment  Q&A

---

## L1 — Conceptual

### Q1. What is RLHF and what problem does it solve?

**A:** RLHF (Reinforcement Learning from Human Feedback) aligns a pretrained language model with human preferences. Pretraining on text makes a model good at predicting distributions, not at being helpful or harmless. RLHF adds three stages: (1) supervised fine-tuning on demonstrations, (2) training a reward model on human preference pairs, (3) optimizing the LM with PPO to maximize reward while staying close to the SFT model. Used to build GPT-4, Claude, Llama-Chat.

**Common follow-ups:** Why is the KL penalty needed? Why 3 stages and not just stage 3?

---

### Q2. What is LoRA and why is it useful?

**A:** LoRA (Low-Rank Adaptation) fine-tunes a pretrained model by adding low-rank decomposition matrices to frozen weight matrices. Instead of training the full d×d weight update ΔW, you train ΔW = B·A where B ∈ R^(d×r), A ∈ R^(r×d), r << d. For Llama 3 8B with r=16, this reduces trainable parameters from 8B to ~16M (0.2%). After training, B·A is merged into W — zero inference latency. Essential for fine-tuning large models on consumer/research hardware.

---

### Q3. What is the difference between DPO and RLHF?

**A:** Both optimize for human preferences, but differently:
- **RLHF**: 3-stage pipeline — SFT → reward model → PPO. Requires 4 models simultaneously, online sampling, complex.
- **DPO**: Reformulates RLHF as supervised learning. Given (prompt, chosen, rejected) pairs, directly optimizes a loss that increases the probability gap between chosen and rejected relative to a reference model. No reward model, no PPO, 2 models.

DPO is simpler and cheaper; RLHF with online RL remains better for complex reasoning tasks where online exploration matters.

---

### Q4. What is reward hacking?

**A:** Reward hacking (reward model overoptimization) is when the policy learns to exploit flaws in the reward model rather than truly improving. Example: the RM was trained on human preferences for concise responses, so the policy generates very short responses that score highly but are unhelpful. Detected by measuring RM score vs human evaluation — they diverge. Mitigated by: KL penalty (don't drift too far from SFT), RM ensembles, RLAIF with AI critique, regenerating RM data as the policy improves.

---

## L2 — Technical

### Q5. Walk me through the three stages of RLHF with technical detail.

**A:**

**Stage 1 — SFT:**
- Fine-tune pretrained base LM on (prompt, response) demonstration pairs
- Standard cross-entropy loss on response tokens only
- Produces a model that follows instructions

**Stage 2 — Reward Model:**
- Dataset: (prompt, y_chosen, y_rejected) — human labeled A > B
- Architecture: SFT model + linear head projecting final token embedding to scalar
- Loss: `L_RM = -log σ(r(y_chosen) - r(y_rejected))` — Bradley-Terry model
- RM is typically SFT-initialized (same architecture, fine-tuned for scoring)

**Stage 3 — PPO:**
- Objective: `max E[r(x,y)] - β · KL[π_θ(y|x) || π_SFT(y|x)]`
- β controls how much the model can deviate from SFT
- PPO clips the policy ratio to prevent large updates
- Four models in memory: policy (trained), reference/SFT (frozen), RM (frozen), value (trained)

---

### Q6. Derive the DPO loss and explain what it optimizes.

**A:**
Starting from the RLHF objective: `max_π E[r(x,y)] - β KL[π||π_ref]`

The optimal solution is: `π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)`

Rearranging: `r(x,y) = β log[π*(y|x)/π_ref(y|x)] + β log Z(x)`

Substituting into the Bradley-Terry preference model P(y_w > y_l):
```
L_DPO = -E[ log σ(β·log[π_θ(y_w|x)/π_ref(y_w|x)] - β·log[π_θ(y_l|x)/π_ref(y_l|x)]) ]
```

This loss increases the log-ratio of chosen response to reference, relative to the log-ratio of rejected. The model is pushed to be more likely to generate chosen responses (relative to the reference) than rejected ones. No explicit reward model is trained — the reward is implicit in the policy ratio.

---

### Q7. Why does GRPO eliminate the value model and how does it still compute advantages?

**A:** PPO's value model V(s) estimates "how good is this state" — needed to compute advantages (actual return - expected return). Training V requires a full copy of the model architecture.

GRPO replaces V with group statistics:
1. For each prompt x, sample G responses: {y_1, ..., y_G}
2. Score each with the reward model: {r_1, ..., r_G}
3. Normalize: `A_i = (r_i - mean(r)) / std(r)`

Group mean serves as the baseline (like V(s)) without needing to learn it. Works because:
- Within a group, all responses share the same prompt — mean reward is a reasonable baseline
- With G=8–16 samples, statistics are stable enough

Memory savings: removes one full model copy (~25% of total memory in PPO RLHF setup).

---

### Q8. Explain QLoRA and how it enables fine-tuning 65B models on a single GPU.

**A:** QLoRA stacks three techniques:

1. **4-bit NF4 quantization of base model**: The frozen pretrained weights are stored in NF4 (Normal Float 4), a 4-bit datatype with quantization bins optimized for normally-distributed values (LLM weights ≈ normal). A 65B model in fp16 = 130 GB. In NF4 = 32.5 GB.

2. **Double quantization**: The quantization constants themselves are quantized, saving another ~0.3 bytes/parameter.

3. **Paged optimizers**: Adam optimizer states (2× fp32 per parameter) are stored in CPU RAM and paged to GPU as needed. This handles the optimizer memory spike.

The LoRA adapters are always in bfloat16 (small, ~16M params for r=16). Computation: dequantize NF4 → bfloat16 on the fly → compute → don't store dequantized weights.

Result: 65B model fit on 48GB A100 with full LoRA fine-tuning quality.

---

## L3 — Applied

### Q9. You're building a preference optimization pipeline for a financial document summarization model. Walk me through your choices: RLHF vs DPO, LoRA vs full fine-tune, reward signal.

**A:**

**Base model**: Start with Llama 3 8B Instruct (already SFT'd — skips stage 1).

**DPO vs RLHF**: Choose DPO. Reasons: (a) financial summarization has clear "better/worse" with offline expert data — no need for online exploration; (b) 2 models vs 4 is much cheaper; (c) DPO is simpler to debug. Would reconsider PPO/GRPO if we had verifiable accuracy metrics (e.g., financial figures must match).

**LoRA vs full fine-tune**: QLoRA with r=64 on all linear layers. Financial domain shift doesn't require updating all weights — QLoRA can reach ~95% of full fine-tune quality at 5% of the cost.

**Reward signal for DPO**: Collect (prompt, doc) → generate candidate summaries → have financial analysts label preferred/rejected. Key quality dimensions: factual accuracy (no hallucinated figures), completeness (all key metrics covered), conciseness.

**Evaluation**: ROUGE is insufficient. Use: LLM-as-judge (GPT-4) for preference win rate, factual accuracy on named entities/figures, human evaluation on 200-item test set.

---

### Q10. When would you choose GRPO over DPO for training a reasoning model?

**A:** GRPO is preferred when:

1. **Verifiable rewards exist**: Math, code, formal logic — correct/incorrect is deterministic. GRPO can use ground-truth as reward without a learned RM (no reward hacking).

2. **Online exploration matters**: DPO is offline — it only improves on existing preference pairs. GRPO samples new solutions each iteration, discovering reasoning paths the data didn't cover.

3. **You want emergent behavior**: DeepSeek-R1 showed GRPO can induce chain-of-thought reasoning without labeled CoT data — the model discovers reasoning strategies through trial and error.

4. **Distribution shift is a concern**: Offline DPO can overfit to the fixed preference dataset. GRPO's online rollouts stay on-policy.

Use DPO when: offline preference data is plentiful, task is instruction-following (not reasoning), compute budget is tight (DPO is cheaper), and you don't have verifiable rewards.

---
