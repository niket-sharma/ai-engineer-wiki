# Probability and Statistics for ML Interviews

**Sources:** "All of Statistics" (Wasserman), "Probability Theory" (Jaynes), standard ML  prep

---

## Probability Fundamentals

### Bayes' Theorem

```
P(A|B) = P(B|A) · P(A) / P(B)
```

- **Prior:** P(A) — belief before seeing data
- **Likelihood:** P(B|A) — probability of data given hypothesis
- **Evidence:** P(B) = Σ_A P(B|A)·P(A) — normalizing constant
- **Posterior:** P(A|B) — updated belief after seeing data

**Classic  example:** A test for a disease is 99% accurate (sensitivity = specificity = 0.99). Disease prevalence = 1 in 1000. You test positive. What's the probability you have the disease?

```
P(disease | positive) = P(positive | disease) · P(disease) / P(positive)

P(positive) = P(positive|disease)·P(disease) + P(positive|no disease)·P(no disease)
            = 0.99 × 0.001 + 0.01 × 0.999
            = 0.00099 + 0.00999 = 0.01098

P(disease | positive) = 0.00099 / 0.01098 ≈ 0.09 = 9%
```

Despite 99% accuracy, only 9% of positive tests are true positives. The **base rate** (prevalence) dominates.

### Common Distributions

| Distribution | PMF/PDF | Mean | Variance | When |
|---|---|---|---|---|
| Bernoulli(p) | p^x (1-p)^{1-x} | p | p(1-p) | Single binary trial |
| Binomial(n,p) | C(n,k)p^k(1-p)^{n-k} | np | np(1-p) | k successes in n trials |
| Poisson(λ) | e^{-λ}λ^k/k! | λ | λ | Count events in fixed interval |
| Normal(μ,σ²) | 1/√(2πσ²) exp(-...) | μ | σ² | CLT, weights, errors |
| Exponential(λ) | λe^{-λx} | 1/λ | 1/λ² | Time until next event |
| Beta(α,β) | x^{α-1}(1-x)^{β-1}/B(α,β) | α/(α+β) | — | Prior for probabilities |
| Dirichlet(α) | Generalization of Beta | α_k/Σα | — | Prior for categorical |

### Expectation and Variance

```
E[X] = Σ x·P(X=x)         (discrete)
E[X] = ∫ x·f(x)dx          (continuous)

Var[X] = E[X²] - E[X]²
Var[aX + b] = a²·Var[X]
Var[X + Y] = Var[X] + Var[Y] + 2·Cov(X,Y)

# If X, Y independent:
Var[X + Y] = Var[X] + Var[Y]
```

---

## Statistical Tests

### Hypothesis Testing Framework

1. State H₀ (null) and H₁ (alternative)
2. Choose significance level α (usually 0.05)
3. Compute test statistic
4. Compute p-value = P(test statistic ≥ observed | H₀ is true)
5. Reject H₀ if p-value < α

**p-value ≠ P(H₀ is true).** It's the probability of seeing data at least this extreme if H₀ were true.

### Type I and Type II Errors

| | H₀ True | H₀ False |
|---|---|---|
| **Reject H₀** | Type I error (false positive, rate = α) | Correct (True Positive) |
| **Fail to reject H₀** | Correct (True Negative) | Type II error (false negative, rate = β) |

- **Power** = 1 - β = probability of correctly rejecting false H₀
- Increase power: larger sample size, larger effect size, higher α

### Common Tests

**t-test:** Compare means when variance is unknown
```python
from scipy import stats

# One-sample: is the mean different from μ₀?
t_stat, p_val = stats.ttest_1samp(data, popmean=0)

# Two-sample: are the means of two groups different?
t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)  # Welch's t-test

# Paired: before/after measurements on same subjects
t_stat, p_val = stats.ttest_rel(before, after)
```

**Chi-squared test:** Test independence of categorical variables
```python
# Contingency table test
chi2, p_val, dof, expected = stats.chi2_contingency(observed_table)
```

**Mann-Whitney U:** Non-parametric alternative to t-test (doesn't assume normality)
```python
stat, p_val = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
```

---

## A/B Testing Concepts

### Sample Size Calculation

```python
import numpy as np
from scipy import stats

def required_sample_size(p1, p2, alpha=0.05, power=0.80):
    """
    p1: baseline conversion rate
    p2: expected treatment conversion rate
    alpha: significance level (Type I error)
    power: 1 - beta (1 - Type II error)
    """
    z_alpha = stats.norm.ppf(1 - alpha/2)   # 1.96 for α=0.05 two-sided
    z_beta  = stats.norm.ppf(power)          # 0.84 for 80% power
    
    p_bar = (p1 + p2) / 2
    
    n = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
         z_beta  * np.sqrt(p1*(1-p1) + p2*(1-p2)))**2 / (p2 - p1)**2
    
    return int(np.ceil(n))

# Example: 5% baseline, want to detect 6% (1pp lift), 80% power, α=0.05
n = required_sample_size(0.05, 0.06)
print(f"Need {n} users per group")  # ~15,000
```

### Multiple Testing Problem

Running 20 independent tests at α=0.05: expected false positives = 20 × 0.05 = 1.

**Bonferroni correction:** α_adjusted = α / n_tests (conservative)

**Benjamini-Hochberg (FDR control):** Control the expected fraction of false positives among rejections
```python
from statsmodels.stats.multitest import multipletests

p_values = [0.03, 0.001, 0.08, 0.04, 0.11]
reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
# reject[i] = True → this test is significant after correction
```

**In ML context:** When doing hyperparameter search over 20 configurations, ~1 will look better than baseline by chance alone.

---

## Maximum Likelihood Estimation (MLE)

Find parameters θ that maximize the probability of observed data:

```
θ̂_MLE = argmax_θ  Π_i P(x_i | θ)
       = argmax_θ  Σ_i log P(x_i | θ)   (log-likelihood, numerically stable)
```

**Example: Gaussian MLE**
```
L(μ, σ²) = -n/2 · log(2πσ²) - 1/(2σ²) · Σ(x_i - μ)²

∂L/∂μ = 0 → μ̂ = (1/n)Σx_i = sample mean
∂L/∂σ² = 0 → σ̂² = (1/n)Σ(x_i - μ̂)² = biased sample variance
```

Note: MLE for variance is biased. Unbiased estimator uses 1/(n-1).

**Cross-entropy and MLE:** Minimizing cross-entropy loss (neural network training) = MLE for the conditional distribution P(y|x).

---

## Central Limit Theorem

For i.i.d. random variables X₁,...,Xₙ with mean μ and variance σ²:

```
(X̄ - μ) / (σ/√n)  →  N(0, 1)  as n → ∞
```

**Practical implications:**
- Sample means are approximately normally distributed for n ≥ 30 (rule of thumb)
- Standard error of the mean = σ/√n
- Confidence interval for the mean: X̄ ± z_{α/2} · σ/√n

**Why it matters for ML:**
- SGD mini-batch gradient estimates have variance ∝ 1/batch_size
- A/B test statistics converge to normal → use z-tests/t-tests
- Bootstrap distributions are approximately normal for large N

---

## Information Theory Basics

### Entropy

```
H(X) = -Σ_x P(x) · log₂ P(x)   [bits]
     = -Σ_x P(x) · log P(x)     [nats, more common in ML]
```

Maximum entropy: uniform distribution. H = log₂(|X|) bits.
Minimum entropy: deterministic. H = 0.

### Cross-Entropy and KL Divergence

```
H(p, q) = -Σ_x p(x) · log q(x)    # cross-entropy of q relative to p

KL(p || q) = H(p, q) - H(p)
           = Σ_x p(x) · log[p(x)/q(x)]
```

**KL divergence properties:**
- KL(p||q) ≥ 0 always
- KL(p||q) = 0 iff p = q
- Not symmetric: KL(p||q) ≠ KL(q||p)

**In training:** Minimizing cross-entropy H(p_true, q_model) = minimizing KL(p_true || q_model) since H(p_true) is constant.

**RLHF connection:** The KL penalty in PPO: E[log(π_θ/π_ref)] = KL(π_θ || π_ref). Prevents the policy from drifting too far from the reference.

---

## Common  Questions

- "A test has 99% sensitivity and 99% specificity. For a disease with 1% prevalence, what's the PPV?"
- "Explain Type I vs Type II error. How do you reduce each?"
- "What is p-value? What does p < 0.05 mean and what does it NOT mean?"
- "How do you calculate required sample size for an A/B test?"
- "What is the multiple testing problem and how do you handle it?"
- "Explain MLE. How does it connect to cross-entropy loss?"
- "What is the Central Limit Theorem and why does it matter for ML?"
