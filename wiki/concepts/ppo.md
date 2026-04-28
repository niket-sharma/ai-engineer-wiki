---
title: "PPO"
aliases: ["Proximal Policy Optimization", "clipped PPO", "PPO-clip"]
tags: [rl, alignment, fine-tuning]
related: ["[[rlhf]]", "[[grpo]]", "[[dpo]]"]
sources: ["training-knowledge"]
relevance: high
last_updated: 2026-04-22
status: current
---

# PPO

## TL;DR
PPO (Proximal Policy Optimization) is the on-policy reinforcement learning algorithm used in the RLHF training stage. It addresses a fundamental RL problem: how do you update a policy using gradient ascent without taking steps so large that you destroy the current policy? PPO clips the policy ratio to stay within a trust region, giving stable training without the complexity of TRPO.

## Intuition
In RL, you collect trajectories under the current policy, estimate the advantage (how much better an action was than expected), and update the policy to increase the probability of good actions. The problem: if your update step is too large, you overshoot and get a bad policy — and since the new policy generates new data, bad policies compound.

PPO solves this by clipping: if an action's probability ratio (new/old) goes too far from 1.0, stop increasing it. This prevents the policy from changing too much in a single step.

## Technical Detail

**PPO-Clip Objective:**
```
L_CLIP(θ) = E_t[ min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t) ]
```
Where:
- r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) — probability ratio new/old
- A_t — advantage estimate (GAE)
- ε — clip range, typically 0.1–0.2

If A_t > 0 (good action): increase π_θ(a_t), but not past (1+ε) × old probability
If A_t < 0 (bad action): decrease π_θ(a_t), but not past (1-ε) × old probability

**Generalized Advantage Estimation (GAE):**
```
A_t = Σ_{k=0}^{∞} (γλ)^k δ_{t+k}    where δ_t = r_t + γV(s_{t+1}) - V(s_t)
```
- γ: discount factor (usually 1.0 for LLM RLHF — episode is a single response)
- λ: GAE smoothing (0 = TD(0), 1 = Monte Carlo)
- Requires a **value/critic model** V(s_t) — separate from the policy

**PPO in RLHF context:**
- State s_t: prefix of generated tokens so far
- Action a_t: next token to generate
- Reward r_t: sparse — reward model score given only at end of response
- Episode: a single (prompt, response) pair
- KL penalty term: r(x,y) - β·KL[π_θ || π_ref] is the effective reward

**Full training loop:**
1. Sample batch of prompts
2. Generate responses with current policy π_θ
3. Score responses with frozen RM
4. Compute KL penalty vs reference model
5. Compute advantages using value model
6. Update policy with PPO-clip loss
7. Update value model
8. Repeat

## Variants & Extensions

| Variant | Change | Benefit |
|---|---|---|
| PPO-clip | Clipping instead of KL constraint | Simpler than TRPO |
| PPO-KL | Adaptive KL penalty instead of clip | Explicit KL control |
| GRPO | No value model; group-based baseline | 50% memory reduction |
| REINFORCE | Monte Carlo, no value model | Simple but high variance |

## Tradeoffs
| Advantage | Disadvantage |
|---|---|
| Stable on-policy updates | Requires 4 models (policy, ref, RM, value) |
| Works with sparse rewards (end-of-sequence) | Memory-intensive — often needs model parallelism |
| Clipping prevents catastrophic updates | Sample-inefficient — needs fresh rollouts each iteration |
| Well-understood, widely used | Sensitive to hyperparameters (ε, β, learning rate) |

##  Angles

**What to understand deeply:**
- Can you explain the clipping objective and why it prevents large updates?
- Do you understand GAE and why a value model is needed?
- Do you know how PPO is adapted for the text generation (LLM) setting?
- Do you understand the tradeoff between PPO and GRPO (value model vs group baseline)?

**Common follow-up questions:**
- "Why does PPO use clipping? What problem does it solve?"
- "What is GAE and what does the λ parameter control?"
- "How does PPO handle the fact that rewards in RLHF are sparse (only at end of response)?"
- "GRPO removes the value model — how does it still compute advantages?"
- "Why is on-policy RL like PPO sample-inefficient?"

**Gotchas / misconceptions:**
- PPO does NOT guarantee staying in a trust region — the clip just discourages large deviations
- The value model must be updated alongside the policy — it's not frozen
- In RLHF, the "reward" at each token is typically 0 except at the last token (where RM score is given)
- PPO-clip and PPO-KL are two variants; PPO usually refers to PPO-clip in ML contexts

## Connections
- [[rlhf]] — PPO is the RL optimizer in stage 3 of the RLHF pipeline
- [[grpo]] — GRPO replaces the value model with group-based normalization, reducing memory
- [[dpo]] — DPO entirely bypasses the need for PPO by reformulating as supervised learning

## Sources
- Training knowledge (Schulman et al. 2017 "Proximal Policy Optimization Algorithms")
