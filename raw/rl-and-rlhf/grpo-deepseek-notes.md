# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

**Authors:** DeepSeek-AI (DeepSeek Team)
**Published:** 2025-01-22
**arXiv ID:** 2501.12948
**URL:** https://arxiv.org/abs/2501.12948

Also covers: DeepSeekMath (Shao et al. 2024, arXiv 2402.03300) — original GRPO paper

---

## What DeepSeek-R1 Is

DeepSeek-R1 is a family of reasoning models (7B–671B parameters) trained using large-scale reinforcement learning on verifiable reasoning tasks. The key contribution: **GRPO (Group Relative Policy Optimization)**, an RL algorithm that eliminates the value/critic model from PPO while maintaining training stability.

The paper showed that sophisticated reasoning capabilities (chain-of-thought, self-verification, backtracking) can **emerge** from RL training without explicit chain-of-thought supervision.

---

## GRPO: Group Relative Policy Optimization

### Problem with PPO for LLM RLHF

PPO requires a **value function** V(s) to estimate the expected cumulative reward from state s. In practice:
- V is a full neural network (same size as the policy model)
- Training V alongside the policy adds ~50% more compute
- At scale: 4 models in memory (policy + ref + RM + value)
- V is hard to train well for long-horizon text generation

### GRPO's Solution

For each prompt x, generate G responses:
```
{y_1, y_2, ..., y_G} ~ π_θ_old(·|x)
```

Score each with the reward model:
```
{r_1, r_2, ..., r_G} = {r(x, y_1), r(x, y_2), ..., r(x, y_G)}
```

Normalize to get advantages (the baseline is the **group mean**):
```
A_i = (r_i - mean(r_1,...,r_G)) / std(r_1,...,r_G)
```

**Why this works as a baseline:**
- Within a group, all responses are from the same prompt x → they share the same "state value"
- Group mean approximates E[r(x,y)] = V(x) without training a separate model
- With G=8 or G=16 responses, statistics are stable

### GRPO Objective

```
L_GRPO(θ) = E_{x, {y_i}~π_old} [
    (1/G) Σ_i min(
        r_i(θ) · A_i,
        clip(r_i(θ), 1-ε, 1+ε) · A_i
    )
] - β · KL[π_θ || π_ref]
```

Where:
- `r_i(θ) = π_θ(y_i|x) / π_old(y_i|x)` — probability ratio
- `ε = 0.2` — PPO clip range
- `β` — KL coefficient
- KL term: computed per-token between current policy and reference model

**Memory savings:**
- PPO: policy + ref + RM + value model = ~4 × model size in GPU memory
- GRPO: policy + ref + RM = ~3 × model size
- Savings: ~25% GPU memory, significant at 70B+ scale

**Compute tradeoff:**
- GRPO needs G forward passes per prompt (to get G rollouts)
- PPO needs 1 rollout per prompt + value model forward pass
- GRPO with G=8: 8× more generation compute per update step

---

## DeepSeek-R1 Training Recipe

### Phase 1: Cold Start (SFT on Chain-of-Thought)

**Problem:** Pure RL from a base model produces incoherent reasoning traces (the model hasn't learned the "format" of chain-of-thought).

**Solution:** Cold start — collect a small dataset (~thousands) of long-form CoT examples in a specific format:

```
<think>
Let me work through this step by step.
...intermediate reasoning...
Wait, I made an error. Let me reconsider.
...correction...
The answer is X.
</think>
<answer>X</answer>
```

**Key format elements:**
- `<think>` block: extended reasoning (can be very long — thousands of tokens)
- `<answer>` block: final answer
- Allowed behaviors: backtracking, self-correction, considering multiple approaches

The cold start SFT ensures the model generates coherent reasoning traces before RL.

### Phase 2: GRPO on Verifiable Rewards

**Reward signal (rule-based, not learned RM):**
- Math: final answer in `<answer>` block matches ground truth → reward = 1
- Code: output passes unit tests → reward = 1
- Format: `<think>` before `<answer>` is required → format reward component
- No partial credit in early training (binary correct/incorrect)

**Why verifiable rewards are key:**
- No reward model needed (no reward hacking possible)
- Unambiguous training signal
- Scales to arbitrarily hard problems (just need a verifier)
- No human labeling required

**Training at scale:**
- DeepSeek-R1 uses 671B MoE base model
- GRPO with G=8 rollouts per prompt
- Math (AIME, AMC) and code (Codeforces) datasets
- Multi-stage: warmup → main RL → rejection sampling for SFT data

### Phase 3: Rejection Sampling + SFT Mixture

- Use the RL-trained model to generate many solutions per problem
- Keep only correct solutions (rejection sampling)
- Fine-tune the model on these correct solutions + original SFT data
- Produces a model that's both good at reasoning and good at general tasks

### Phase 4: Secondary GRPO (Alignment)

- Add helpfulness + safety reward model alongside math/code verifiers
- Ensures the final model is helpful and safe, not just accurate on math

---

## The "Aha Moment" Phenomenon

**One of the most cited results from the paper:**

During RL training (without explicit CoT supervision), the model spontaneously developed:
1. **Self-correction**: "Wait, I think I made an error. Let me reconsider..."
2. **Extended thinking**: very long reasoning chains (1000–5000 tokens)
3. **Multiple approaches**: trying different solution strategies before committing
4. **Explicit uncertainty**: acknowledging when a problem is hard

This was NOT in the training data — it emerged from RL on math/code tasks.

**Why this matters:** Suggests that complex reasoning behaviors are emergent from the RL objective (maximize accuracy), not from imitation learning on curated CoT data.

---

## Results

**DeepSeek-R1 vs OpenAI o1 on reasoning benchmarks:**

| Benchmark | GPT-4o | OpenAI o1 | DeepSeek-R1 |
|---|---|---|---|
| AIME 2024 | 9.3% | 74.3% | **79.8%** |
| MATH-500 | 76.6% | 96.4% | **97.3%** |
| Codeforces (percentile) | 23.6% | 89.3% | **96.3%** |
| MMLU | 87.2% | 91.8% | **90.8%** |

**Key result:** DeepSeek-R1 matches or exceeds o1 (OpenAI's best reasoning model) at a fraction of the cost, using an open training methodology.

---

## Distillation: R1 → Small Models

DeepSeek also distilled R1 reasoning into smaller dense models:

```
R1-Distill-Qwen-7B:   83.9% on MATH-500  (vs 87.4% Qwen-7B full DPO)
R1-Distill-Llama-8B:  89.1% on MATH-500
R1-Distill-Llama-70B: 94.5% on MATH-500
```

**Method:** Generate long-form reasoning traces from R1-671B, fine-tune smaller models on those traces. The smaller models learn the thinking format without RL.

This shows: **CoT reasoning can be distilled** — you don't need to run RL on every model size.

---

## GRPO vs PPO vs DPO: Summary Table

| | PPO | GRPO | DPO |
|---|---|---|---|
| **Online/Offline** | Online | Online | Offline |
| **Value model** | Yes | No | No |
| **Reward model** | Learned | Rule-based or learned | No (implicit) |
| **New data each iter** | Yes | Yes | No |
| **Memory** | 4× model | 3× model | 2× model |
| **Generate rollouts** | 1 per prompt | G per prompt | No |
| **Best for** | General RLHF | Verifiable reasoning | Offline preference data |
| **Reward hacking** | Possible | Not with verifiable rewards | N/A |

---

## -Relevant Insights

**The key GRPO question:** "What does GRPO use instead of a value model?"
→ Group statistics (mean and std of rewards within a prompt's rollout group). The group mean serves as the baseline, analogous to V(x) in PPO.

**Why verifiable rewards make GRPO especially effective:**
→ Binary correct/incorrect signal is unambiguous. The model can't "hack" a deterministic verifier. The reward is inherently calibrated (0 or 1). Perfect for GRPO's group normalization.

**The "aha moment" as an  talking point:**
This is a great example of emergent behavior from RL. The model learned to be uncertain, to backtrack, to use extended thinking — none of which was in the training signal. Shows the power of online RL vs. behavioral cloning.

**Why DeepSeek-R1 matters for the field:**
- First major paper to match o1-level reasoning with open methodology
- GRPO made large-scale reasoning RL practical without prohibitive memory requirements
- Spawned dozens of derivative works (DAPO, Dr. GRPO, etc.)

---

## Common  Questions From This Paper

- "What is GRPO and how does it differ from PPO?"
- "Why does removing the value model save memory?"
- "What is a verifiable reward and why does GRPO benefit from it?"
- "What is the 'aha moment' in DeepSeek-R1 training?"
- "How would you train a reasoning model using GRPO on math problems?"
- "Compare GRPO, DPO, and PPO. When would you use each?"
