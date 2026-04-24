# Bias-Variance Tradeoff and Regularization

**Sources:** ESL (Hastie, Tibshirani, Friedman) Chapters 2, 3, 7; PRML (Bishop) Chapter 3

---

## Bias-Variance Decomposition

For a regression model f̂(x) trained on dataset D, the expected prediction error decomposes as:

```
E[(y - f̂(x))²] = Bias²[f̂(x)] + Var[f̂(x)] + σ²
```

where:
- **Bias²** = (E[f̂(x)] - f(x))² — systematic error (model is wrong on average)
- **Var** = E[(f̂(x) - E[f̂(x)])²] — variance due to training data randomness
- **σ²** = irreducible noise in the true relationship

**Key insight:** Models with more capacity (more parameters, lower regularization) have lower bias but higher variance. Simpler models: higher bias, lower variance.

### Concrete Examples

| Model | Bias | Variance | When it fails |
|---|---|---|---|
| Linear regression on nonlinear data | High | Low | Underfitting |
| Deep neural net on small dataset | Low | High | Overfitting |
| Polynomial degree 1 (linear) | High | Low | Wrong function class |
| Polynomial degree 10 (overfit) | Low | High | Noisy data |
| Ensemble (bagging) | Same as base | Reduced | — |
| Ensemble (boosting) | Reduced | Slightly higher | — |

### For Classification

The decomposition is more complex but the intuition holds:
- High bias → model underfits the decision boundary
- High variance → decision boundary changes a lot with different training samples

---

## Regularization

### L2 Regularization (Ridge / Weight Decay)

```
L_ridge = MSE + λ Σ_i w_i²
```

**Effect:** Shrinks weights toward zero (but not exactly zero). All weights are penalized proportionally.

**Solution (closed form for linear regression):**
```
w* = (X^T X + λI)^{-1} X^T y
```

The λI term ensures the matrix is always invertible (conditioning improvement). This is why L2 is sometimes called "weight decay."

**Bayesian interpretation:** L2 regularization = MAP estimation with Gaussian prior N(0, 1/λ) on weights.

**In neural networks (PyTorch):**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# weight_decay=λ adds L2 penalty to loss before gradient computation
```

### L1 Regularization (Lasso)

```
L_lasso = MSE + λ Σ_i |w_i|
```

**Effect:** Produces **sparse solutions** — many weights become exactly zero. This is feature selection.

**Why L1 produces sparsity:** The L1 ball (diamond shape in 2D) has corners at the axes. The optimization is likely to land at a corner where some weights = 0. L2 ball (circle) has no corners → unlikely to land exactly on axis.

**No closed form:** Must use iterative methods (coordinate descent, subgradient).

**Bayesian interpretation:** MAP estimation with Laplace prior on weights.

### Elastic Net

```
L_elasticnet = MSE + λ₁ Σ|w_i| + λ₂ Σw_i²
```

Combines L1 (sparsity) and L2 (stability). Useful when features are correlated (L1 alone picks one of correlated features arbitrarily; Elastic Net tends to include them together).

### Dropout (Neural Networks)

During training, randomly zero out activations with probability p:
```python
h = dropout(relu(Wx + b), p=0.5)  # zeroes 50% of neurons each forward pass
```

**At inference:** scale activations by (1-p) or use inverted dropout (scale during training).

**Interpretation:** Ensemble of 2^n models sharing weights. Each training step trains a different subnetwork.

**Effective regularization effect:** Reduces co-adaptation of neurons (neurons can't rely on any specific other neuron being present).

---

## Cross-Validation

### K-Fold Cross-Validation

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model.fit(X_train, y_train)
    scores.append(model.score(X_val, y_val))

print(f"CV score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

**K=5 or K=10** is standard. **K=N** (Leave-One-Out) is unbiased but expensive.

**Use CV for:** hyperparameter selection (λ, C, depth), model comparison.

### Bias-Variance of Cross-Validation

- **5-fold CV** has higher bias (less training data per fold) but lower variance
- **LOO CV** is nearly unbiased but has high variance (models trained on nearly identical datasets)
- **10-fold** is the standard balance

---

## Learning Curves

Plot train error and val error vs training set size:

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# Underfitting diagnosis: both train and val error are high and close together
# Overfitting diagnosis: train error low, val error high, large gap
# Well-fit: both errors converge to a low value with more data
```

**Bias problem (underfitting):** Adding more data won't help → change model or features.
**Variance problem (overfitting):** Adding more data will help → or regularize.

---

## Bayesian Regularization Perspective

| Regularizer | Equivalent Prior |
|---|---|
| L2 (λ) | Gaussian: N(0, 1/λ) |
| L1 (λ) | Laplace: p(w) ∝ exp(-λ|w|) |
| L∞ | Uniform on a box |
| No regularization | Improper flat prior |
| Dropout | Implicit mixture-of-Bernoulli prior |

This framing is useful for Bayesian deep learning discussions.

---

## Common Interview Questions

- "Explain the bias-variance tradeoff."
- "What's the difference between L1 and L2 regularization? Why does L1 produce sparse solutions?"
- "How do you diagnose underfitting vs. overfitting from learning curves?"
- "What is dropout and how does it regularize neural networks?"
- "How does weight decay in Adam compare to L2 regularization? (Hint: they're not the same — AdamW fixes this.)"
- "When would you use Elastic Net over Lasso?"
