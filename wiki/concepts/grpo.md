---
title: "GRPO"
aliases: ["Group Relative Policy Optimization", "DeepSeek-R1 training", "GRPO"]
tags: [rl, alignment, fine-tuning]
related: ["[[ppo]]", "[[rlhf]]", "[[dpo]]"]
sources: ["training-knowledge"]
interview_relevance: high
last_updated: 2026-04-22
status: current
---

# GRPO

## TL;DR
GRPO (Group Relative Policy Optimization) is the RL algorithm used to train DeepSeek-R1 reasoning models. It replaces PPO's learned value function with a group-based reward baseline: sample G outputs per prompt, use the group mean as the baseline, normalize advantages by group std. This eliminates the need for a separate critic model (saving ~50% memory) while maintaining stable training.

## Intuition
PPO needs a value model V(s) to estimate how good a state is (the baseline for advantage computation). Training this requires a whole extra model, doubling memory overhead.

GRPO's insight: for LLM RLHF, you don't need a learned value function. Instead, for each prompt x, generate G responses {y_1, ..., y_G}, score each with the reward model, and use the group statistics as the baseline:
```
Advantage_i = (r_i - mean(r_1..G)) / std(r_1..G)
```
This is essentially REINFORCE with a group baseline — simple, effective, no critic needed.

## Technical Detail

**GRPO Objective:**
```
L_GRPO(θ) = -E_{x, {y_i}~π_old} [ (1/G) Σ_i min(r_i(θ) · A_i, clip(r_i(θ), 1-ε, 1+ε) · A_i) - β · KL[π_θ || π_ref] ]
```
Where:
- G: group size (number of samples per prompt, e.g., 8–16)
- r_i(θ) = π_θ(y_i|x) / π_old(y_i|x) — probability ratio
- A_i = (reward_i - mean_reward) / std_reward — normalized advantage
- β: KL coefficient for reference model penalty

**Comparison to PPO:**
| Component | PPO | GRPO |
|---|---|---|
| Advantage estimation | GAE with learned value model V | Group statistics (mean/std) |
| Critic model | Separate neural net (same size as policy) | None |
| Memory overhead | 4 models | 3 models |
| Variance | Lower (GAE smoothing) | Higher (but manageable at scale) |
| Implementation | Complex | Simple |

**Why it works for reasoning tasks:**
- DeepSeek-R1 uses verifiable rewards (math/code have ground-truth answers)
- G=8 or G=16 samples provide a decent reward distribution to normalize against
- Group normalization is a natural fit for tasks with binary correct/incorrect rewards
- No risk of value model becoming a bottleneck or overfitting

**GRPO in DeepSeek-R1:**
- Used to train Chain-of-Thought reasoning without supervised CoT data
- Reward: correctness of final answer (verifiable) + format compliance
- The model learned to self-improve by sampling many reasoning traces and reinforcing correct ones
- Demonstrated "aha moment" emergence — model learned to revisit and correct reasoning

## Variants & Extensions

- **REINFORCE**: Single-sample version, no clipping — GRPO is more stable multi-sample REINFORCE
- **Dr. GRPO**: Removes potential bias in GRPO normalization
- **DAPO**: Dynamic sampling based on difficulty

## Tradeoffs
| Advantage | Disadvantage |
|---|---|
| No critic model — 25–50% memory savings | Higher variance than GAE-based advantages |
| Simple implementation | Requires G rollouts per prompt — G× more generation compute |
| Works well with verifiable rewards | Less suitable for complex reward signals without clear grouping |
| Enabled SOTA reasoning (DeepSeek-R1) | G is a new hyperparameter to tune |

##  Angles

**What interviewers are really testing:**
- Do you know what GRPO is and why it's relevant (DeepSeek-R1)?
- Can you explain how it differs from PPO structurally?
- Do you understand what the group normalization baseline does?
- Can you explain why it works especially well for verifiable reward tasks?

**Common follow-up questions:**
- "What is GRPO and how does it differ from PPO?"
- "Why does removing the value model reduce memory by 25–50%?"
- "DeepSeek-R1 used GRPO — what made it effective for reasoning?"
- "What is a verifiable reward and why does GRPO benefit from it?"
- "How would GRPO compare to DPO for training a reasoning model?"

**Gotchas / misconceptions:**
- GRPO still uses PPO-style clipping — it's not purely REINFORCE
- Group normalization is done per-prompt per-batch, not globally
- GRPO requires G × more generation compute than PPO per update step
- DeepSeek-R1 used GRPO from scratch (cold start RL), not from SFT — unusual and notable

## Connections
- [[ppo]] — GRPO replaces PPO's value model with group-based normalization
- [[rlhf]] — GRPO is an alternative RL algorithm for the optimization stage of RLHF
- [[dpo]] — both are PPO alternatives; GRPO is online, DPO is offline

## Sources
- Training knowledge (DeepSeek-AI 2025 "DeepSeek-R1"; Shao et al. 2024 "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models")
