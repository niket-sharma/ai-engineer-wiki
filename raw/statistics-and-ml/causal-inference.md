# Causal Inference for ML Engineers

**Sources:** "The Book of Why" (Pearl), "Causal Inference in Statistics: A Primer" (Pearl, Glymour, Jewell), Judea Pearl's work, Hernan & Robins "Causal Inference: What If"

---

## Why Causal Inference Matters for ML

**Correlation ≠ Causation** is a cliché, but in production ML:

- **A/B test vs observational:** Did our recommendation algorithm cause more purchases, or did it just show popular items to users who would have bought anyway?
- **Feature importance vs causal effect:** A "time_of_day" feature being important doesn't mean changing the time of day would change the outcome.
- **Selection bias:** Training on users who signed up vs the general population. Users who sign up are self-selected — your model won't generalize to users who haven't signed up yet.

---

## The Ladder of Causation (Pearl)

| Level | Query type | Example | Method |
|---|---|---|---|
| 1. Association | P(Y \| X) | If I see X, what is Y? | Regression, ML |
| 2. Intervention | P(Y \| do(X=x)) | If I set X to x, what is Y? | A/B test, do-calculus |
| 3. Counterfactual | P(Y_x \| X=x', Y=y') | If X had been x instead, what would Y have been? | Structural causal models |

Most ML models operate at Level 1 (association). Production causal questions require Level 2 or 3.

---

## Confounders and Directed Acyclic Graphs (DAGs)

A **confounder** is a variable that affects both the treatment (X) and the outcome (Y):

```
W (confounder)
↙           ↘
X (treatment) → Y (outcome)

Example:
Age (W)
↙      ↘
Exercise (X) → Health (Y)

Naive regression: Exercise correlates with health.
But age causes both more exercise tendency AND better health.
Naive estimate overstates the benefit of exercise.
```

**DAG rules:**
- Arrows indicate causal direction
- Blocking a path: conditioning on a variable on the path "blocks" it
- Backdoor path: a path from X to Y that goes "through the back" via confounders

---

## Randomized Controlled Trials (RCTs)

The gold standard. Randomly assign treatment — breaks all backdoor paths.

```python
# A/B test: randomly assign users to treatment/control
users = fetch_all_users()
np.random.shuffle(users)
treatment = users[:len(users)//2]
control = users[len(users)//2:]

# Now: E[Y | treatment] - E[Y | control] = causal effect
# No confounders because assignment was random
ate = treatment['conversion'].mean() - control['conversion'].mean()
```

**When RCTs aren't possible:**
- Ethical constraints (can't randomly give people diseases)
- Practical constraints (can't randomize a city's policy)
- Historical data analysis (can't run experiment in the past)

---

## Observational Causal Methods

### Propensity Score Matching

Estimate the probability of treatment given covariates (propensity score). Match treated and control units with similar propensity scores.

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Step 1: Estimate propensity score
ps_model = LogisticRegression()
ps_model.fit(X_confounders, treatment)
propensity_scores = ps_model.predict_proba(X_confounders)[:, 1]

# Step 2: Match treated to control units with similar propensity scores
from sklearn.neighbors import NearestNeighbors

treated_idx = np.where(treatment == 1)[0]
control_idx = np.where(treatment == 0)[0]

nn = NearestNeighbors(n_neighbors=1)
nn.fit(propensity_scores[control_idx].reshape(-1, 1))
matches = nn.kneighbors(propensity_scores[treated_idx].reshape(-1, 1))[1].flatten()

# Step 3: Estimate ATE on matched pairs
matched_control = outcomes[control_idx[matches]]
ate = outcomes[treated_idx].mean() - matched_control.mean()
```

**Assumption:** Conditional ignorability — no unmeasured confounders.

### Difference-in-Differences (DiD)

Compare trends before/after treatment between treated and control groups.

```python
# Classic 2x2 DiD
# Y_it = alpha + beta * Treated_i + gamma * Post_t + delta * (Treated_i * Post_t) + e_it

# delta = DiD estimate (causal effect of treatment)
# = (Y_treated_post - Y_treated_pre) - (Y_control_post - Y_control_pre)
```

**Assumption:** Parallel trends — treated and control groups would have followed the same trend absent treatment.

**Example:** You launch a new feature in one city but not another. DiD compares the trend difference between the two cities.

### Instrumental Variables (IV)

When there's unmeasured confounding: find an "instrument" Z that:
1. Affects X (relevant)
2. Only affects Y through X (exclusion restriction)
3. Is independent of confounders (independence)

```
Z → X → Y
↑
(unobservable)
W (confounder) → Y

LATE (Local Average Treatment Effect) estimate:
β_IV = Cov(Y, Z) / Cov(X, Z)
```

**Classic example:** Draft lottery number (Z) as instrument for military service (X) to estimate effect on wages (Y). Lottery is random → independent of confounders.

### Regression Discontinuity Design (RDD)

When treatment is assigned by crossing a threshold:

```python
# Example: Scholarship eligibility requires test score >= 70
# Students just above 70 are similar to students just below 70
# Compare outcomes in a small bandwidth around the threshold

import statsmodels.formula.api as smf

df_bandwidth = df[abs(df['score'] - 70) < 5]  # within 5 points of threshold
df_bandwidth['treated'] = (df_bandwidth['score'] >= 70).astype(int)

result = smf.ols('outcome ~ treated + score + treated:score', 
                  data=df_bandwidth).fit()
causal_effect = result.params['treated']
```

---

## Causal ML (Modern Methods)

### Double Machine Learning (Chernozhukov et al. 2018)

Handles high-dimensional confounders:

```python
from econml.dml import LinearDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

# Model: Y = theta(X) * T + g(W) + epsilon
# theta(X) = heterogeneous treatment effect (varies by features X)
dml = LinearDML(
    model_y=GradientBoostingRegressor(),    # predict Y from W
    model_t=GradientBoostingClassifier(),   # predict T from W
    featurizer=None                          # treatment effect linear in X
)

dml.fit(Y, T, X=X_features, W=W_confounders)
treatment_effects = dml.effect(X_test)
```

### Causal Forests (Wager & Athey 2018)

Heterogeneous treatment effect estimation — different users respond differently to treatment:

```python
from econml.grf import CausalForest

forest = CausalForest(n_estimators=1000, min_samples_leaf=5)
forest.fit(Y, T, X)

# Individual treatment effects
ite = forest.predict(X_test)
print(f"Average ITE: {ite.mean():.3f}")
print(f"Users with positive ITE: {(ite > 0).mean():.1%}")
```

---

## Simpson's Paradox

A trend appears in groups but reverses when groups are combined. Classic example:

```
University admissions (1973 Berkeley study):
  Overall: Men admitted at higher rate than women
  By department: Women admitted at higher rate in every department

Reason: Women applied to competitive departments at higher rates.
Confounder: department choice.
```

**In ML:** A model might show better performance overall for Group A vs Group B, but Group B is better in every subgroup (different base rates).

---

## Practical Guidance for ML Engineers

**When to use causal methods:**
1. You're making a policy decision ("should we change feature X?")
2. You have selection bias in training data (users who opted in vs general population)
3. You want to estimate heterogeneous treatment effects (who benefits most from a feature)
4. Historical observational data, can't run an A/B test

**Before claiming causation from ML:**
1. Draw the causal DAG — are there backdoor paths?
2. Identify potential confounders
3. Can you run an A/B test instead?
4. Do you need causal effect or just correlation (for prediction)?

**Common mistake:** Using SHAP values to claim causation. SHAP shows feature importance for prediction, not causal effect. A high SHAP value for "user_age" doesn't mean changing age changes the outcome.

---

## Common Interview Questions

- "What is confounding and how do you handle it?"
- "What's the difference between correlation and causation in the context of ML models?"
- "You can't run an A/B test. How do you estimate causal effects from observational data?"
- "What is Simpson's paradox? Give an example."
- "What is an instrumental variable and when would you use IV estimation?"
- "Why can't you use SHAP values to infer causal effects?"
