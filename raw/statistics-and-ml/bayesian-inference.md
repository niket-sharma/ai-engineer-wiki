# Bayesian Inference for ML Engineers

**Sources:** "Pattern Recognition and Machine Learning" (Bishop), "Bayesian Data Analysis" (Gelman et al.), "Probabilistic Programming and Bayesian Methods for Hackers" (Davidson-Pilon)

---

## The Bayesian Framework

**Core equation:**
```
P(θ | data) = P(data | θ) · P(θ) / P(data)

Posterior = Likelihood × Prior / Evidence
```

**Interpretation:**
- **Prior P(θ):** Belief about parameters before seeing data
- **Likelihood P(data | θ):** Probability of observed data given parameters
- **Evidence P(data) = ∫ P(data | θ) P(θ) dθ:** Normalizing constant (often intractable)
- **Posterior P(θ | data):** Updated belief after seeing data

**Bayesian vs Frequentist:**
- Frequentist: parameters are fixed; data is random. P-values, confidence intervals.
- Bayesian: parameters are random variables with distributions. Credible intervals, posterior distributions.

---

## Conjugate Priors

When the prior and posterior are in the same distribution family — makes Bayesian inference analytic.

### Beta-Binomial (Coin flips, click-through rates)

```python
# Model: X ~ Binomial(n, θ), Prior: θ ~ Beta(α, β)
# Posterior: θ | X ~ Beta(α + successes, β + failures)

import numpy as np
from scipy import stats

# Prior: Beta(1, 1) = uniform
alpha_prior, beta_prior = 1, 1

# Data: 30 heads out of 50 flips
heads, flips = 30, 50
tails = flips - heads

# Posterior: Beta(1 + 30, 1 + 20) = Beta(31, 21)
alpha_post = alpha_prior + heads
beta_post = beta_prior + tails

posterior = stats.beta(alpha_post, beta_post)
print(f"Posterior mean: {posterior.mean():.3f}")
print(f"95% credible interval: {posterior.ppf(0.025):.3f} – {posterior.ppf(0.975):.3f}")
```

**Application:** A/B testing. Prior based on historical CTR. Update with new data.

### Normal-Normal (Continuous measurements)

```
Prior: μ ~ N(μ₀, σ₀²)
Likelihood: x_i ~ N(μ, σ²)  (σ known)
Posterior: μ | x ~ N(μ_n, σ_n²)

μ_n = (μ₀/σ₀² + n·x̄/σ²) / (1/σ₀² + n/σ²)    # precision-weighted average
1/σ_n² = 1/σ₀² + n/σ²                           # precision adds up
```

As n → ∞: posterior concentrates at the MLE (data overwhelms prior).

### Dirichlet-Categorical (Topic modeling, LDA)

```
Prior: π ~ Dirichlet(α)
Likelihood: x_i ~ Categorical(π)
Posterior: π | x ~ Dirichlet(α + counts)
```

---

## Bayesian Inference Methods

### Markov Chain Monte Carlo (MCMC)

Sample from the posterior when analytic solution is intractable.

**Metropolis-Hastings:**
```python
def metropolis_hastings(log_posterior, initial_theta, n_samples, step_size):
    theta = initial_theta
    samples = [theta]
    
    for _ in range(n_samples):
        theta_proposed = theta + np.random.normal(0, step_size, size=theta.shape)
        
        log_ratio = log_posterior(theta_proposed) - log_posterior(theta)
        
        if np.log(np.random.uniform()) < log_ratio:
            theta = theta_proposed  # accept
        # else: reject, stay at theta
        
        samples.append(theta)
    
    return np.array(samples)
```

**Hamiltonian Monte Carlo (HMC) / NUTS:**
- Uses gradient information to make efficient proposals
- Much better than MH for high-dimensional parameters
- Used in Stan, PyMC

**In practice (PyMC):**
```python
import pymc as pm

with pm.Model() as model:
    # Prior
    mu = pm.Normal('mu', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)
    
    # Likelihood
    obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=data)
    
    # Sample
    trace = pm.sample(2000, tune=1000, return_inferencedata=True)

pm.plot_posterior(trace, var_names=['mu', 'sigma'])
```

### Variational Inference (VI)

Approximate the posterior with a simpler distribution Q(θ) that minimizes KL divergence:

```
Q*(θ) = argmin_Q KL(Q(θ) || P(θ | data))
       = argmax_Q ELBO(Q)

ELBO = E_Q[log P(data, θ)] - E_Q[log Q(θ)]
     = E_Q[log P(data | θ)] - KL(Q(θ) || P(θ))
```

**Mean-field VI:** Assume Q factors: Q(θ) = Π_i Q_i(θ_i) — each parameter independent in Q.

**Why VI instead of MCMC:**
- VI: faster, scales to large data, but approximate (biased toward overconfident posteriors)
- MCMC: exact (asymptotically), but slow, doesn't scale as well

**In neural networks:** Variational autoencoders (VAEs) use VI to learn a latent posterior.

---

## Bayesian Neural Networks

Replace point estimates for weights with distributions:

```python
# Regular NN: W = specific values
# Bayesian NN: W ~ N(μ_W, σ_W²) — distribution over weights

# Approximate inference via mean-field VI:
# q(W) = N(μ_W, σ_W²) where μ, log(σ) are learned parameters

import torch
import torch.nn as nn

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_log_sigma = nn.Parameter(torch.full((out_features,), -3.0))
    
    def forward(self, x):
        # Reparameterization trick
        weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_mu)
        return x @ weight.T + bias
```

**Uncertainty quantification:** Run multiple forward passes, observe variance in predictions. High variance → high uncertainty.

---

## Bayesian Optimization

Use a Gaussian Process to model the objective function and optimize acquisition function:

```python
from skopt import gp_minimize

# Define objective (black box)
def objective(params):
    lr, n_layers, dropout = params
    model = train_model(lr=lr, n_layers=n_layers, dropout=dropout)
    return -model.val_accuracy  # minimize negative = maximize accuracy

# Bayesian optimization over hyperparameters
result = gp_minimize(
    func=objective,
    dimensions=[
        (1e-5, 1e-1, 'log-uniform'),  # lr
        (1, 10),                       # n_layers
        (0.0, 0.5)                     # dropout
    ],
    n_calls=50,           # total function evaluations
    n_initial_points=10,  # random initial samples
    acq_func='EI'         # expected improvement acquisition
)
```

**Why BO outperforms random/grid search:** The GP model predicts where the objective is likely to be good based on previous evaluations. Acquisition function balances exploration (high uncertainty) vs exploitation (high expected value).

---

## Bayesian for ML Engineers in Practice

**Bayesian A/B testing:**
- Instead of p-value, compute P(variant_B_conversion > variant_A_conversion)
- Can stop early when probability exceeds threshold (95%)
- No multiple testing correction needed for sequential tests

```python
# Bayesian A/B test
control_data = {'successes': 500, 'trials': 10000}
treatment_data = {'successes': 550, 'trials': 10000}

# Beta posteriors (Beta(1,1) uniform prior)
control_posterior = stats.beta(1 + 500, 1 + 9500)
treatment_posterior = stats.beta(1 + 550, 1 + 9450)

# Monte Carlo estimate of P(treatment > control)
n_samples = 100000
control_samples = control_posterior.rvs(n_samples)
treatment_samples = treatment_posterior.rvs(n_samples)
p_treatment_better = (treatment_samples > control_samples).mean()
print(f"P(treatment > control) = {p_treatment_better:.3f}")
```

---

## Common Interview Questions

- "What is Bayes' theorem and how does it apply to ML?"
- "What is a conjugate prior and why is it useful?"
- "What is the difference between MCMC and variational inference?"
- "How do Bayesian neural networks provide uncertainty estimates?"
- "What is Bayesian optimization and when would you use it over grid search?"
- "How does Bayesian A/B testing differ from frequentist A/B testing?"
