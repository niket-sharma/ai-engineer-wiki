# Ensemble Methods: Bagging, Boosting, Stacking

**Sources:** ESL Chapter 8, 10; Breiman (2001) "Random Forests"; Friedman (2001) "Greedy Function Approximation: A Gradient Boosting Machine"

---

## Why Ensembles Work

**Variance reduction:** If f̂₁, f̂₂, ..., f̂_M are uncorrelated models each with variance σ²:
```
Var[mean(f̂_i)] = σ²/M
```

Average M uncorrelated models → variance drops M-fold. Bias unchanged.

**Problem:** Real models are correlated (trained on the same data). The variance reduction is:
```
Var[ensemble] = ρσ² + (1-ρ)σ²/M
```
where ρ = average pairwise correlation between models. As M→∞: variance → ρσ².

**Strategy:** Reduce ρ (make models more different from each other) while keeping σ² low.

---

## Bagging (Bootstrap Aggregating)

**Breiman, 1996.** Reduce variance by training on different bootstrap samples.

```python
# Bagging algorithm
for m in range(M):
    D_m = bootstrap_sample(D)      # sample N examples with replacement
    f_m = train(model, D_m)        # train same model class on this sample

def predict(x):
    return mean([f_m(x) for f_m in models])  # average for regression
    # or majority vote for classification
```

**Key properties:**
- Each bootstrap sample contains ~63.2% unique training examples (the rest are duplicates)
- ~36.8% of examples are left out of each bag → "out-of-bag" (OOB) samples
- OOB samples can be used for free validation (no held-out set needed!)
- Works best with high-variance, low-bias base learners (deep trees)

### Random Forests (Breiman, 2001)

Random Forests = Bagging + random feature subsets at each split.

**At each node split:**
```python
# Instead of finding the best split over ALL features:
candidate_features = random.sample(all_features, k=sqrt(n_features))
best_split = find_best_split(X[:, candidate_features], y)
```

**Why this helps:** Bagged trees are still correlated because they all have the same strong features at the root. Random feature selection decorrelates the trees (reduces ρ).

**Hyperparameters:**
- `n_estimators`: number of trees (more is always better, diminishing returns after ~100)
- `max_features`: features per split (sqrt(n) for classification, n/3 for regression)
- `max_depth`: None (fully grown) is typical; depth limits can reduce variance
- `min_samples_leaf`: minimum samples per leaf (regularization)

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",   # sqrt(n_features) features per split
    max_depth=None,        # fully grown trees
    min_samples_leaf=1,
    oob_score=True,        # compute OOB validation score
    n_jobs=-1              # parallel
)
rf.fit(X_train, y_train)
print(f"OOB score: {rf.oob_score_:.3f}")
```

---

## Boosting

**Key idea:** Train models sequentially. Each new model focuses on the examples that previous models got wrong.

### AdaBoost (Freund & Schapire, 1996)

```python
# Initialize equal sample weights
w = ones(N) / N

for m in range(M):
    # Train weak learner on weighted data
    f_m = train(weak_learner, X, y, sample_weight=w)
    
    # Error rate (weighted)
    err_m = sum(w[f_m(X) != y]) / sum(w)
    
    # Learner weight (larger weight for lower error)
    alpha_m = 0.5 * log((1 - err_m) / err_m)
    
    # Update sample weights (misclassified examples get higher weight)
    w = w * exp(-alpha_m * y * f_m(X))
    w = w / sum(w)

def predict(x):
    return sign(sum(alpha_m * f_m(x) for alpha_m, f_m in zip(alphas, models)))
```

AdaBoost minimizes **exponential loss**: L(y, f) = exp(-y·f(x)).

### Gradient Boosting (Friedman, 2001)

Generalization of AdaBoost to arbitrary differentiable loss functions.

**Key insight:** Fit each new model to the **negative gradient** of the loss w.r.t. the current predictions (the "pseudo-residuals").

```python
# Initial prediction (constant)
F_0 = argmin_gamma sum(Loss(y_i, gamma))

for m in range(1, M+1):
    # Compute pseudo-residuals (negative gradient of loss)
    r_m = -[dL(y_i, F_{m-1}(x_i)) / dF for i in range(N)]
    
    # Fit a tree to the pseudo-residuals
    h_m = DecisionTreeRegressor(max_depth=3).fit(X, r_m)
    
    # Line search for optimal step size
    gamma_m = argmin_gamma sum(Loss(y_i, F_{m-1}(x_i) + gamma * h_m(x_i)))
    
    # Update ensemble
    F_m = F_{m-1} + learning_rate * gamma_m * h_m
```

**For MSE loss:** pseudo-residuals = actual residuals (y - F(x)). This is the familiar "fit to residuals" interpretation.

**For log-loss (classification):** pseudo-residuals are different and capture the gradient of the cross-entropy.

### XGBoost, LightGBM, CatBoost

Modern implementations add:
- **Regularization:** explicit L1/L2 on leaf weights and tree complexity (XGBoost)
- **Histogram-based splits:** LightGBM bins features into 255 bins, finds splits in O(n_bins) not O(n_samples)
- **Column subsampling:** Like random forests, sample features at each tree/node
- **Parallel tree construction:** GPU support
- **Categorical features:** CatBoost handles categoricals natively (no one-hot encoding needed)

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,    # shrinkage (lower = better but needs more trees)
    subsample=0.8,         # row subsampling
    colsample_bytree=0.8,  # column subsampling
    reg_lambda=1.0,        # L2 on leaf weights
    reg_alpha=0.0,         # L1 on leaf weights
    eval_metric="logloss",
    early_stopping_rounds=50
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
```

**LightGBM key difference:** Grows trees **leaf-wise** (best leaf) vs XGBoost's **level-wise** (full row). Leaf-wise can overfit on small datasets but is faster and better on large ones.

---

## Bagging vs Boosting Comparison

| | Bagging | Boosting |
|---|---|---|
| **Training order** | Parallel (independent) | Sequential (dependent) |
| **Primary effect** | Reduces variance | Reduces bias AND variance |
| **Overfitting risk** | Low | High (if too many rounds) |
| **Noise robustness** | High (averaging reduces outliers) | Low (focuses on hard examples, including noise) |
| **Base learner** | Strong, high-variance (deep trees) | Weak, low-variance (shallow trees) |
| **Interpretability** | Feature importance (mean impurity decrease) | Feature importance (gain) |
| **Best for** | High-variance problem, noisy labels | Clean data, bias is the issue |

---

## Stacking (Stacked Generalization)

Train a **meta-learner** on the out-of-fold predictions of base models.

```python
# Level 0: Base learners
base_models = [RandomForest(), GradientBoosting(), LinearSVC(), LogisticRegression()]

# Generate out-of-fold predictions (avoid leakage!)
oof_preds = zeros((N, len(base_models)))
for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
    for m_idx, model in enumerate(base_models):
        model.fit(X[train_idx], y[train_idx])
        oof_preds[val_idx, m_idx] = model.predict_proba(X[val_idx])[:, 1]

# Level 1: Meta-learner trained on OOF predictions
meta_model = LogisticRegression()
meta_model.fit(oof_preds, y)
```

**Key:** Use out-of-fold predictions to prevent leakage. Never train the meta-learner on data the base models have seen.

**Practical note:** Stacking rarely outperforms a well-tuned single model by more than 1-2%, but can provide meaningful gains in competitions. Production use is rare due to complexity.

---

## Common Interview Questions

- "Explain the difference between bagging and boosting."
- "Why do random forests subsample features at each split?"
- "How does gradient boosting relate to gradient descent?"
- "What is the difference between XGBoost and LightGBM?"
- "When would you choose random forest over gradient boosting?"
- "How do you prevent overfitting in gradient boosting?"
- "What is the bias-variance effect of ensembling?"
