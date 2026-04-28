---
title: "DPO"
aliases: ["Direct Preference Optimization", "IPO", "KTO", "ORPO"]
tags: [alignment, fine-tuning]
related: ["[[rlhf]]", "[[ppo]]", "[[grpo]]", "[[lora-qlora]]"]
sources: ["training-knowledge"]
relevance: high
last_updated: 2026-04-22
status: current
---

# DPO

## TL;DR
DPO reformulates the RLHF objective as a supervised learning problem — no reward model, no PPO, no online sampling. Given (prompt, chosen, rejected) triplets, DPO directly optimizes the policy to increase the log probability ratio of chosen over rejected responses, relative to a reference model. It's simpler, cheaper, and often competitive with PPO RLHF for offline settings.

## Intuition
RLHF stage 3 is complex: train a reward model, then run PPO to maximize it. Can we skip the reward model entirely?

DPO derives the optimal RLHF solution analytically:
```
r*(x, y) = β · log [π*(y|x) / π_ref(y|x)] + β · log Z(x)
```
The optimal reward is implicitly defined by the optimal policy. We can reparametrize the RLHF objective in terms of the policy directly, without ever training an explicit RM. The result is a binary cross-entropy loss on preference pairs.

## Technical Detail

**DPO Loss:**
```
L_DPO(π_θ) = -E_{(x,y_w,y_l)} [ log σ(β · log [π_θ(y_w|x)/π_ref(y_w|x)] - β · log [π_θ(y_l|x)/π_ref(y_l|x)]) ]
```
Where:
- y_w: the chosen (winning) response
- y_l: the rejected (losing) response
- π_ref: the reference/SFT model (frozen)
- β: temperature controlling deviation from reference (typically 0.1–0.5)
- σ: sigmoid

**Intuition of the loss:**
- Increase log π_θ(y_w|x) (make the good response more likely)
- Decrease log π_θ(y_l|x) (make the bad response less likely)
- Both are measured relative to π_ref — we're increasing the *gap* between chosen and rejected, normalized by what the reference model already prefers

**Requirements:**
- Offline dataset of (prompt, chosen, rejected) triplets
- Frozen reference model (SFT checkpoint)
- No RM training needed
- No PPO rollouts — purely supervised batches

**DPO vs RLHF memory:**
- DPO: 2 models (policy + frozen ref)
- PPO RLHF: 4 models (policy, ref, RM, value)

## Variants & Extensions

| Method | Key Change | Motivation |
|---|---|---|
| DPO | Baseline: pairwise preference, offline | Simplicity |
| IPO (Identity PO) | Replaces log σ with a squared loss | Avoids overfitting to easy pairs |
| KTO | Uses single-response feedback (good/bad) | No pairwise data needed |
| ORPO | Fuses preference optimization into SFT | No reference model needed |
| SimPO | Normalizes reward by sequence length | Better calibration |
| Online DPO | Generates new pairs during training | Closes distribution gap vs PPO |

## Tradeoffs
| Advantage | Disadvantage |
|---|---|
| Simple: just supervised learning | Offline — no new data generation, can overfit to fixed dataset |
| 2 models vs 4 in PPO — much cheaper | Reference model is fixed — distribution mismatch can accumulate |
| No reward model hacking possible | β tuning is sensitive |
| Easy to implement with standard SFT code | Online DPO (best of both) adds back complexity |
| Competitive quality on instruction following | PPO still preferred for complex reasoning/RL tasks |

##  Angles

**What to understand deeply:**
- Can you derive or explain the DPO loss at a conceptual level?
- Do you understand what the β parameter controls?
- Do you know when to use DPO vs RLHF/PPO?
- Are you aware of the limitations (offline, distribution shift)?

**Common follow-up questions:**
- "How does DPO avoid training a reward model?"
- "What does β control in DPO? What happens if you set it too high or too low?"
- "DPO is offline — what problems does that cause?"
- "When would you choose PPO over DPO?"
- "What is KTO and why would you prefer it over DPO?"
- "What is online DPO and how does it compare to standard PPO RLHF?"

**Gotchas / misconceptions:**
- DPO still requires a reference model (the SFT checkpoint) — it's not truly reference-free
- ORPO is reference-free by fusing SFT + preference into one loss, which DPO is not
- β in DPO is NOT the same as β in PPO RLHF (though both relate to KL divergence)
- DPO doesn't eliminate the need for data collection — you still need (chosen, rejected) pairs

## Connections
- [[rlhf]] — DPO is derived from the RLHF objective and replaces PPO stage
- [[ppo]] — DPO avoids PPO by reformulating as supervised learning
- [[grpo]] — both are PPO alternatives; GRPO is online, DPO is offline
- [[lora-qlora]] — DPO is commonly run with LoRA adapters

## Sources
- Training knowledge (Rafailov et al. 2023 "Direct Preference Optimization: Your Language Model is Secretly a Reward Model")
