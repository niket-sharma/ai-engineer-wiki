# Direct Preference Optimization: Your Language Model is Secretly a Reward Model

**Authors:** Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn
**Published:** 2023-05-29
**arXiv ID:** 2305.18290
**URL:** https://arxiv.org/abs/2305.18290
**Venue:** NeurIPS 2023 (Outstanding Paper)

---

## Abstract Summary

While RLHF has proven effective at aligning LLMs with human preferences, it is complex and unstable: it requires a separately trained reward model and fine-tuning with reinforcement learning. DPO is a simple algorithm that implicitly optimizes the same objective as RLHF without the need for a reward model or RL. The key insight: the optimal RLHF policy can be expressed analytically in terms of the log probability ratio between the policy and the reference model — so you can directly optimize the policy for human preferences without an explicit reward model.

---

## The Core Derivation (The Math  Questions Are About)

### RLHF Objective (Review)

```
max_π  E_{x~D, y~π} [ r(x,y) ] - β · KL[π(y|x) || π_ref(y|x)]
```

This is maximized by the **optimal policy:**

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y) / β)
```

Where `Z(x) = Σ_y π_ref(y|x) · exp(r(x,y) / β)` is the partition function (normalization).

### Rearranging to Express Reward in Terms of Policy

Rearrange the optimal policy equation:
```
r(x,y) = β · log[π*(y|x) / π_ref(y|x)] + β · log Z(x)
```

**Key insight:** The reward is determined by the log ratio of the optimal policy to the reference policy (plus a normalizing constant). This means: given a policy, you can read off the implied reward.

### Substituting Into the Bradley-Terry Loss

The preference modeling loss (from RM training):
```
L_RM = -E[log σ(r(x, y_w) - r(x, y_l))]
```

Substitute the reparametrized reward:
```
r(x, y_w) - r(x, y_l) 
= β·log[π*(y_w|x)/π_ref(y_w|x)] - β·log[π*(y_l|x)/π_ref(y_l|x)]
  + β·log Z(x) - β·log Z(x)  ← cancels!
```

The `Z(x)` terms cancel (same x for both). This gives the **DPO loss:**

```
L_DPO(π_θ; π_ref) = -E_{(x,y_w,y_l)~D} [
    log σ(
        β · log[π_θ(y_w|x) / π_ref(y_w|x)] 
      - β · log[π_θ(y_l|x) / π_ref(y_l|x)]
    )
]
```

No explicit reward model. No RL. Just binary cross-entropy on log-probability ratios.

---

## What DPO Optimizes (Intuition)

The loss increases when:
- `π_θ(y_w|x) / π_ref(y_w|x)` increases → model assigns MORE probability to chosen, relative to reference
- `π_θ(y_l|x) / π_ref(y_l|x)` decreases → model assigns LESS probability to rejected, relative to reference

It's a contrastive loss: push chosen responses up in probability, push rejected responses down — both measured relative to the reference model's baseline.

**Why measure relative to reference?**
Without the reference denominator, the model would just make all responses have high (or low) probability, ignoring preferences. The reference normalizes: "how much better/worse than what the pretrained model already does?"

---

## β Parameter: The Key Hyperparameter

```
L_DPO = -E[ log σ(β · (log_ratio_chosen - log_ratio_rejected)) ]
```

- **β → 0**: sigmoid input → 0 → loss → -log(0.5) for all pairs. No gradient. No learning.
- **β → ∞**: sigmoid saturates to 0 or 1. Hard labels. Maximizes the log-ratio difference at any cost — tends to collapse `π_ref(y_l)` to 0, which can harm general capabilities.
- **β ≈ 0.1–0.5**: the sweet spot. Allows meaningful gradients while staying close to the reference.

**β in RLHF vs DPO:** β plays the same conceptual role (KL penalty strength) but the optimal value differs. RLHF β ≈ 0.01–0.1. DPO β ≈ 0.1–0.5 (larger values work better in DPO's offline setting).

---

## Algorithm

```
Input: preference dataset D = {(x_i, y_w_i, y_l_i)}, 
       reference model π_ref (frozen SFT checkpoint)

1. For each batch of (x, y_w, y_l):
2.   Compute log π_θ(y_w|x) and log π_θ(y_l|x)  ← current policy
3.   Compute log π_ref(y_w|x) and log π_ref(y_l|x)  ← reference (frozen)
4.   Compute log_ratio_w = log π_θ(y_w|x) - log π_ref(y_w|x)
5.   Compute log_ratio_l = log π_θ(y_l|x) - log π_ref(y_l|x)
6.   loss = -log σ(β · (log_ratio_w - log_ratio_l))
7.   Backprop through π_θ only (π_ref is frozen)
```

**Implementation note:** Computing `log π(y|x)` requires a forward pass through the LM and summing log-probabilities of the response tokens. Batch the reference and policy model forward passes together for efficiency.

---

## Practical Implementation

```python
def dpo_loss(policy_logps_chosen, policy_logps_rejected,
             ref_logps_chosen, ref_logps_rejected, beta=0.1):
    
    # Log probability ratios
    pi_logratios = policy_logps_chosen - policy_logps_rejected
    ref_logratios = ref_logps_chosen - ref_logps_rejected
    
    # DPO loss
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()
    
    # Metrics for monitoring
    chosen_rewards = beta * (policy_logps_chosen - ref_logps_chosen).detach()
    rejected_rewards = beta * (policy_logps_rejected - ref_logps_rejected).detach()
    
    return loss, chosen_rewards.mean(), rejected_rewards.mean()
```

**Key diagnostic metrics to monitor during training:**
- `chosen_rewards`: should increase (model learns to prefer chosen)
- `rejected_rewards`: should decrease (model learns to disprefer rejected)
- `chosen_rewards - rejected_rewards` (reward margin): should increase

---

## Experimental Results

**Compared to RLHF + PPO on:**

| Task | Method | Result |
|---|---|---|
| Reddit TL;DR summarization | RLHF+PPO | Win rate: 56% vs SFT |
| Reddit TL;DR summarization | DPO | Win rate: 57% vs SFT (statistically tied) |
| Anthropic HH (helpfulness) | RLHF+PPO | 54% win rate |
| Anthropic HH (helpfulness) | DPO | 61% win rate |

**DPO matches or exceeds PPO RLHF** on these benchmarks with:
- No reward model training
- No PPO training loop
- 2 models instead of 4
- Much simpler implementation

---

## Limitations and When DPO Fails

**Offline limitation:**
- DPO trains on a fixed dataset of (prompt, chosen, rejected) pairs
- The chosen/rejected responses were generated by an earlier model or humans
- As training progresses, the distribution of π_θ diverges from the data distribution
- DPO can't generate new data to fix its mistakes (unlike PPO's online rollouts)

**Distribution shift problem:**
If the preference data was collected from a weak model, and DPO trains a stronger model, the training distribution becomes stale. PPO avoids this: it generates fresh rollouts every iteration using the current policy.

**Task-specific performance:**
- Instruction following: DPO ≈ PPO
- Complex reasoning (math, code): PPO > DPO (online exploration matters here)
- Creative writing: DPO ≈ PPO
- Tool use with verifiable rewards: GRPO > DPO (online RL essential)

---

## Variants and Extensions

| Method | Key Difference | When Better |
|---|---|---|
| **IPO** (Azar et al. 2023) | Squared loss instead of log-sigmoid | Avoids overconfidence on easy pairs |
| **KTO** (Ethayarajh et al. 2023) | Single-response feedback (good/bad flag) | When you don't have paired comparisons |
| **ORPO** (Hong et al. 2024) | No reference model — fused into SFT | Simpler pipeline, faster |
| **SimPO** (Meng et al. 2024) | Normalize by response length | Better calibration |
| **Online DPO** | Generate pairs with current policy | Closes gap to PPO |
| **RSO** (Liu et al. 2023) | Statistical rejection sampling from DPO | Better sampling efficiency |

---

## -Relevant Insights

**"DPO's language model is secretly a reward model":**
The paper's title refers to: after training, you can extract an implicit reward from DPO's policy: `r_DPO(x,y) = β · log[π_θ(y|x)/π_ref(y|x)]`. This implicit RM corresponds exactly to the optimal RM for the RLHF objective.

**Why the partition function Z(x) cancels:**
This is the key mathematical trick. Z(x) depends only on x (not on y_w or y_l), so it cancels when you compute the difference `r(x,y_w) - r(x,y_l)`. This cancellation is only valid because the reference model is the same for both responses — which is satisfied by construction.

**Common misconception:** "DPO doesn't need a reference model." — False. DPO requires `π_ref` for training (you compute log probabilities under π_ref for both chosen and rejected responses). It's true that no *explicit* reward model is trained.

---

## Common  Questions From This Paper

- "How does DPO avoid training a reward model? Walk me through the derivation."
- "What does the β parameter control in DPO?"
- "Why is DPO offline? What problems does that cause?"
- "When would you choose PPO over DPO? When would you choose DPO over PPO?"
- "What is the partition function Z(x) and why does it cancel in the DPO derivation?"
- "What is IPO and why was it proposed as an improvement to DPO?"
