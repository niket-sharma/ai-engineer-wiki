# Proximal Policy Optimization Algorithms

**Authors:** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (OpenAI)
**Published:** 2017-07-20
**arXiv ID:** 1707.06347
**URL:** https://arxiv.org/abs/1707.06347

Also covers: PPO applied to RLHF (Ziegler et al. 2019, arXiv 1909.08593)

---

## Context: The Policy Gradient Problem

**Policy gradient (REINFORCE):**
```
∇_θ J(θ) = E_{τ~π_θ}[ Σ_t ∇_θ log π_θ(a_t|s_t) · A_t ]
```

**Problem:** If you take a large gradient step, the new policy might be very different from the policy that collected the data. Since you're using old data to estimate the new policy's performance, the estimate becomes wrong → catastrophic policy collapse.

**TRPO (Trust Region Policy Optimization, 2015):**
The principled solution: constrain each update to stay within a KL-divergence trust region.
```
max_θ E[r_t(θ) · A_t]    subject to E[KL[π_old || π_θ]] ≤ δ
```
**Problem with TRPO:** Requires computing the constraint's Fisher information matrix → expensive second-order optimization.

**PPO:** Achieves TRPO-like stability with first-order optimization by simply clipping the objective.

---

## PPO: Two Variants

### PPO-Clip (the standard one)

```
L_CLIP(θ) = E_t[ min(r_t(θ) · A_t,  clip(r_t(θ), 1-ε, 1+ε) · A_t) ]
```

Where:
- `r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)` — probability ratio
- `A_t` — advantage estimate
- `ε` — clip range (paper: ε=0.2 works well; RLHF: often 0.1–0.2)

**How clipping works:**
- If `A_t > 0` (good action): increasing `π_θ(a_t|s_t)` helps → but clip at `(1+ε)×π_old` — don't increase too much
- If `A_t < 0` (bad action): decreasing `π_θ(a_t|s_t)` helps → but clip at `(1-ε)×π_old` — don't decrease too much

The `min` ensures the objective can only be pessimistically wrong — we take the worse of the clipped and unclipped objective, so we never gain from going outside the clip range.

**Visualization:**
```
When A_t > 0:
L = r_t · A_t                if r_t ≤ (1+ε)    → normal policy gradient
L = (1+ε) · A_t              if r_t > (1+ε)     → clipped, no gradient signal

When A_t < 0:
L = r_t · A_t                if r_t ≥ (1-ε)    → normal policy gradient  
L = (1-ε) · A_t              if r_t < (1-ε)     → clipped, no gradient signal
```

### PPO-KL (Adaptive KL Penalty)

```
L_KL(θ) = E_t[r_t(θ) · A_t] - β · KL[π_old || π_θ]
```

Adaptively adjust β: if KL too high → increase β. If KL too low → decrease β. More interpretable but harder to tune.

---

## Advantage Estimation: GAE

**Why not just use the Monte Carlo return?**
Return = sum of future rewards. For LLM RLHF: the only reward is at the end of the episode (RM score). So every token in the response has the same advantage. Very high variance — small changes to any token might as well be random.

**Temporal Difference (TD) estimates:**
Use a learned value function V(s_t) to estimate expected future rewards:
```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)    # TD error
```

**Generalized Advantage Estimation (GAE):**
```
A_t^GAE(γ,λ) = Σ_{k=0}^{∞} (γλ)^k · δ_{t+k}
```

- **λ=0**: A_t = δ_t = r_t + γV(s_{t+1}) - V(s_t) → low variance, high bias (trusts V too much)
- **λ=1**: A_t = Σ_k γ^k r_{t+k} - V(s_t) → Monte Carlo return, low bias, high variance
- **λ=0.95**: typical value, balances the two

**In RLHF context:**
- γ = 1 (discount factor = 1 for episodic tasks where episode = one response)
- reward is sparse: r_t = 0 for all tokens except the last (EOS), where r_EOS = RM_score - β·KL_at_EOS
- Sometimes: distribute the KL penalty per-token to make advantage estimates smoother

---

## The Value Function in RLHF

**Architecture:** Typically the same LM architecture as the policy, with the LM head replaced by a single linear layer projecting to a scalar:
```
V(s_t) = linear(transformer_output[s_t])
```

**Training:** Minimize mean-squared error between V(s_t) and the observed return:
```
L_VF(φ) = E_t[(V_φ(s_t) - R_t)^2]
```

Often: clip the value update similar to policy clipping:
```
L_VF_clip(φ) = E_t[ max((V_φ - R_t)^2, (clip(V_φ, V_old-ε_vf, V_old+ε_vf) - R_t)^2) ]
```

**The value model problem in RLHF:**
- Training V requires a forward pass through the full LM
- V needs to be accurate: if V is wrong, advantages are wrong → policy updates in wrong direction
- V lags behind the policy during training → GAE estimates can be stale

**This is a key motivation for GRPO** (group-based baseline instead of learned V).

---

## PPO Full Algorithm (RLHF Version)

```
Initialize: π_θ (policy, from SFT), π_ref (frozen SFT), RM (frozen), V_φ (value)

for iteration = 1, 2, ...:
    # 1. Collect rollouts with current policy
    for prompt x in batch:
        y = sample(π_θ, x)                    # generate full response
        r_t = 0 for t < T; r_T = RM(x, y)    # sparse reward
        KL_t = log π_θ(y_t|...) - log π_ref(y_t|...)  # per-token KL
        r_t_adjusted = r_t - β · KL_t          # KL-penalized reward
    
    # 2. Compute advantages
    A_t = GAE(r_t_adjusted, V_φ)
    
    # 3. Update policy with PPO-clip loss (K epochs on this batch)
    for epoch in range(K=4):
        L_π = -mean(min(r(θ)·A, clip(r(θ),1-ε,1+ε)·A))
        L_V = mean((V_φ(s_t) - returns)^2)
        L_entropy = -mean(Σ_a π_θ(a|s) log π_θ(a|s))  # optional entropy bonus
        
        loss = L_π + c_v · L_V - c_e · L_entropy
        loss.backward(); optimizer.step()
    
    # 4. Update old policy
    π_old ← π_θ
```

**Key hyperparameters (RLHF practical values):**
| Param | Value | What it controls |
|---|---|---|
| ε | 0.1–0.2 | Policy clip range |
| β | 0.01–0.1 | KL penalty strength |
| γ | 1.0 | Discount (episodic, so 1) |
| λ | 0.95 | GAE smoothing |
| K (PPO epochs) | 4 | How many gradient steps per batch |
| c_v | 0.5 | Value loss coefficient |

---

## PPO vs Other RL Algorithms for RLHF

| Algorithm | Compute | Memory | Stability | When |
|---|---|---|---|---|
| REINFORCE | Low | Low | Low (high variance) | Simple tasks |
| Actor-Critic | Medium | Medium | Medium | General RL |
| PPO | High | High (4 models) | High | Standard RLHF |
| GRPO | High (G rollouts) | Medium (3 models) | High | Verifiable rewards, scale |
| DPO | Low | Low (2 models) | High | Offline preference data |

---

## -Relevant Insights

**"Why does PPO clip instead of constrain (like TRPO)?"**
Clipping is a first-order approximation to the KL constraint. TRPO's constraint requires computing the Fisher information matrix — O(params²) in memory, requires conjugate gradient. PPO achieves similar stability with just one hyperparameter (ε) and first-order gradients.

**"Why does the value model lag behind the policy?"**
The value model is trained on the same data as the policy (batch collected by π_old). As the policy changes, the value estimates for the new policy are computed from a V that was trained on old policy's experience. This is fundamental to on-policy RL — a known weakness.

**"PPO-clip guarantees the policy stays close. Does it guarantee this?"**
No — it only provides a soft constraint. If the advantage is very large, the clipped objective still pushes the policy, just not as strongly. The clip prevents *gaining* from going outside the trust region, but doesn't enforce a hard KL bound.

---

## Common  Questions From This Paper

- "How does PPO-clip work? Why clip instead of using a KL constraint?"
- "What is GAE and what does λ control?"
- "Why is the value model needed in PPO and what does it estimate?"
- "Walk me through the PPO training loop for RLHF."
- "What is the advantage of GRPO over PPO for LLM training?"
- "What hyperparameters are most important to tune in PPO?"
