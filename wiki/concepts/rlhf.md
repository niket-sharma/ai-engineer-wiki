---
title: "RLHF"
aliases: ["Reinforcement Learning from Human Feedback", "InstructGPT", "reward model", "RLAIF"]
tags: [alignment, fine-tuning, rl]
related: ["[[ppo]]", "[[dpo]]", "[[grpo]]", "[[lora-qlora]]", "[[sft]]"]
sources: ["training-knowledge"]
interview_relevance: high
last_updated: 2026-04-22
status: current
---

# RLHF

## TL;DR
RLHF is the three-stage pipeline that turns a pretrained language model into a helpful, harmless assistant: (1) supervised fine-tuning on demonstrations, (2) train a reward model on human preference pairs, (3) optimize the LM with PPO against the reward model. It's how GPT-4, Claude, and Llama Chat are aligned. DPO has largely replaced stage 3 for offline settings, but RLHF with online RL remains SOTA for frontier models.

## Intuition
A pretrained LM is good at predicting text distributions but not at being helpful. RLHF uses human feedback as a training signal:
1. Show humans pairs of model outputs → collect preference labels (A > B)
2. Train a reward model (RM) to predict these preferences
3. Fine-tune the LM to generate text the RM scores highly, with a KL penalty to prevent drifting too far from the original model

The KL penalty is critical: without it, the LM learns to "game" the RM by producing degenerate high-scoring text (reward hacking).

## Technical Detail

**Stage 1 — Supervised Fine-Tuning (SFT):**
- Start from pretrained base LM
- Fine-tune on human-written demonstrations (prompt → ideal response pairs)
- Standard cross-entropy loss on response tokens
- Result: SFT model — follows instructions but may not be optimally helpful/harmless

**Stage 2 — Reward Model (RM) Training:**
- Dataset: (prompt, chosen_response, rejected_response) triplets
- RM architecture: LM with final embedding projected to scalar reward
- Bradley-Terry model: P(A > B) = σ(r(A) - r(B))
- Loss: L_RM = -log σ(r(chosen) - r(rejected))
- RM is typically initialized from the SFT model

**Stage 3 — PPO Fine-Tuning:**
```
Objective: maximize E[r(x, y)] - β · KL[π_θ(y|x) || π_ref(y|x)]
```
- π_θ: the model being trained
- π_ref: the frozen SFT model (reference policy)
- β: KL coefficient (typically 0.01–0.1)
- r(x, y): reward model score
- PPO's clipped surrogate objective prevents overly large policy updates

**Four models in memory during PPO:**
1. Policy model (trained) — π_θ
2. Reference model (frozen) — π_ref
3. Reward model (frozen) — RM
4. Value/critic model (trained) — estimates expected return

This is why PPO is memory-expensive: 4× model memory at minimum.

## Variants & Extensions

| Method | Stage 3 Alternative | Key Difference |
|---|---|---|
| RLHF + PPO | Standard | 4 models, online sampling, memory-intensive |
| DPO | Offline, no RM needed | Reformulates RLHF as supervised |
| GRPO | Online RL, no value model | Group-based normalization replaces critic |
| RLAIF | AI feedback instead of human | Constitutional AI, scalable feedback |
| ORPO | No separate RM or RL stage | Preference optimization fused into SFT |
| KTO | Single-response feedback (not pairwise) | Works with unpaired preference data |

## Tradeoffs
| Advantage | Disadvantage |
|---|---|
| Directly optimizes for human preference | Requires 4 models — high memory/compute cost |
| SOTA alignment quality for frontier models | RM can be hacked (reward model overoptimization) |
| Online exploration generates new data | Complex training pipeline |
| Works with any differentiable reward signal | Reward model can have its own biases/errors |

##  Angles

**What interviewers are really testing:**
- Can you describe all three RLHF stages and what each one does?
- Do you understand why the KL penalty is necessary?
- Do you know the difference between RLHF and DPO?
- Can you explain reward hacking and how to mitigate it?

**Common follow-up questions:**
- "Walk me through the three stages of RLHF."
- "Why is a KL penalty added to the PPO objective in RLHF?"
- "How many models are active during PPO training and what are they?"
- "What is reward hacking? How do you detect and prevent it?"
- "Why did DPO become popular? What does it avoid that RLHF needs?"
- "What is RLAIF and when would you use it instead of human feedback?"

**Gotchas / misconceptions:**
- The SFT stage is NOT optional — it's critical for stable PPO training (PPO from raw pretrain is unstable)
- The KL penalty compares to the SFT model, NOT the pretrained base — β controls alignment vs capability tradeoff
- Reward hacking is real and measured: win rate goes up but quality eventually degrades
- DPO doesn't fully replace RLHF — online RL (PPO, GRPO) is still used for frontier models

## Connections
- [[ppo]] — the RL algorithm used in stage 3 of RLHF
- [[dpo]] — offline alternative that skips the RL stage entirely
- [[grpo]] — online RL alternative to PPO that removes the value model
- [[lora-qlora]] — LoRA is typically used to fine-tune policy model in stage 1 and 3

## Sources
- Training knowledge (Ouyang et al. 2022 "InstructGPT"; Stiennon et al. 2020 RLHF for summarization)
