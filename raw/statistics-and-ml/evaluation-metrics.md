# ML Evaluation Metrics

**Sources:** Standard ML curriculum, scikit-learn docs, "Hands-On Machine Learning" (Géron), real-world practitioner notes

---

## Classification Metrics

### Confusion Matrix

```
               Predicted Positive  Predicted Negative
Actual Positive       TP                  FN
Actual Negative       FP                  TN
```

### Core Metrics

```
Accuracy   = (TP + TN) / (TP + FP + TN + FN)
Precision  = TP / (TP + FP)   ← "Of predicted positives, how many are correct?"
Recall     = TP / (TP + FN)   ← "Of actual positives, how many did we find?"
F1         = 2 · Precision · Recall / (Precision + Recall)  ← harmonic mean
Specificity = TN / (TN + FP)  ← "Of actual negatives, how many correct?"
```

**F-β score:** Weight recall β times more than precision:
```
F_β = (1 + β²) · Precision · Recall / (β² · Precision + Recall)
```
- β=2: recall twice as important (spam filter: FN cost is missing important email)
- β=0.5: precision twice as important (medical test: FP cost is unnecessary treatment)

### Precision-Recall Trade-off

Precision and recall are inversely related through the **decision threshold**:
- Lower threshold → more positives predicted → higher recall, lower precision
- Higher threshold → fewer positives predicted → lower recall, higher precision

```python
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

probs = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, probs)

plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
```

### AUC-ROC

ROC curve: TPR (recall) vs FPR (1 - specificity) at varying thresholds.

```python
from sklearn.metrics import roc_auc_score, roc_curve

auc = roc_auc_score(y_test, probs)
fpr, tpr, thresholds = roc_curve(y_test, probs)
```

**AUC interpretation:** Probability that a randomly chosen positive is ranked higher than a randomly chosen negative.

- AUC = 0.5: random classifier
- AUC = 1.0: perfect classifier
- AUC = 0.7–0.8: good, industry often acceptable
- AUC > 0.85: very good

**AUC-ROC vs AUC-PR:**
- **Use AUC-ROC** when: class balance is reasonable, both FP and FN costs matter equally
- **Use AUC-PR** when: severe class imbalance (fraud detection, cancer screening), care more about rare positive class

**Why AUC-ROC is misleading for imbalanced data:**
- 1% fraud rate: 99% accuracy classifier just predicts "not fraud" for everything
- AUC-ROC can still be high (0.9+) for a poor classifier on imbalanced data
- AUC-PR is more discriminating when positives are rare

### Matthews Correlation Coefficient (MCC)

```
MCC = (TP·TN - FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

- Single metric for imbalanced classification: -1 (worst) to +1 (perfect)
- More reliable than F1 for evaluating imbalanced classifiers
- Industry: increasingly preferred over F1 for imbalanced problems

---

## Multi-Class Classification

### Macro vs Micro vs Weighted Averaging

```python
from sklearn.metrics import f1_score

# Macro: average of per-class F1 (treats all classes equally)
f1_macro = f1_score(y_true, y_pred, average='macro')

# Micro: global TP, FP, FN across all classes (dominated by frequent classes)
f1_micro = f1_score(y_true, y_pred, average='micro')

# Weighted: weighted by class support (number of actual instances)
f1_weighted = f1_score(y_true, y_pred, average='weighted')
```

**When to use which:**
- Macro: care equally about all classes (e.g., disease subtype classification)
- Micro: care about overall correctness (dominated by majority class)
- Weighted: balance between macro and micro, most common for reports

---

## Regression Metrics

| Metric | Formula | Properties |
|---|---|---|
| MAE | (1/n)Σ\|y - ŷ\| | Robust to outliers, in original units |
| MSE | (1/n)Σ(y - ŷ)² | Penalizes large errors heavily, not in original units |
| RMSE | √MSE | In original units, outlier-sensitive |
| R² | 1 - SS_res/SS_tot | Proportion of variance explained (0 to 1, can be negative) |
| MAPE | (1/n)Σ\|(y-ŷ)/y\| | Percentage error, undefined for y=0 |
| SMAPE | (1/n)Σ\|y-ŷ\|/((|y|+|ŷ|)/2) | Symmetric MAPE, bounded [0, 200%] |

**R² = 1**: perfect prediction.
**R² = 0**: model does no better than predicting the mean.
**R² < 0**: model is worse than predicting the mean (possible!).

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
```

---

## Ranking Metrics (Retrieval / Recommendations)

### Precision@k and Recall@k

For a query with K returned results and R relevant documents:
```
Precision@k = (relevant in top k) / k
Recall@k    = (relevant in top k) / R
```

### Mean Average Precision (MAP)

```
AP = (1/R) Σ_{k} P@k · rel(k)    # rel(k)=1 if kth result is relevant
MAP = mean(AP) over all queries
```

### Normalized Discounted Cumulative Gain (nDCG)

```
DCG@k  = Σ_{i=1}^{k} rel_i / log₂(i+1)
IDCG@k = DCG of perfect ranking (ideal ordering)
nDCG@k = DCG@k / IDCG@k
```

nDCG handles **graded relevance** (relevant = 3, somewhat relevant = 1, irrelevant = 0).

```python
from sklearn.metrics import ndcg_score
import numpy as np

# y_true: true relevance scores for each doc
# y_score: model's predicted scores
nDCG = ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1), k=10)
```

**For RAG systems:** Context Precision and Context Recall (RAGAS) are essentially Precision@k and Recall@k at the chunk level.

---

## LLM-Specific Metrics

### Perplexity

```
PP(W) = P(w_1, w_2, ..., w_N)^{-1/N} = exp(-1/N · Σ log P(w_i | w_{1:i-1}))
```

- **Lower perplexity = better language model** (less "surprised" by the text)
- GPT-2 XL on WikiText-103: ~18 perplexity
- GPT-3 on PTB: ~20 perplexity
- **Limitation:** Perplexity is a language modeling metric, not a task performance metric

### BLEU (Machine Translation)

```
BLEU = BP · exp(Σ_n w_n · log p_n)

BP = min(1, exp(1 - reference_length/candidate_length))   # brevity penalty
p_n = (# n-gram matches) / (# n-grams in candidate)       # n-gram precision
```

Modified n-gram precision clips match counts at reference frequency.
- BLEU-4 (up to 4-grams) is standard
- **Limitations:** Doesn't capture meaning, sensitive to tokenization, not correlated with human quality

### ROUGE (Summarization)

```
ROUGE-N = (# overlapping N-grams between candidate and reference) / (# N-grams in reference)
```

- ROUGE-1: unigram overlap
- ROUGE-2: bigram overlap  
- ROUGE-L: longest common subsequence

**Standard for summarization evaluation** (CNN/DailyMail, XSum benchmarks).

### BERTScore

```
P = max_{r_j ∈ Reference} sim(c_i, r_j)    # for each candidate token
R = max_{c_i ∈ Candidate} sim(c_i, r_j)    # for each reference token
F = 2PR/(P+R)
```

Uses contextual embeddings (BERT) to compute semantic similarity. Better correlation with human judgments than BLEU/ROUGE.

---

## Calibration

A classifier is **calibrated** if P(y=1 | predicted probability = p) = p for all p.

```python
from sklearn.calibration import calibration_curve

fraction_of_positives, mean_predicted_value = calibration_curve(
    y_true, y_prob, n_bins=10
)

# Perfectly calibrated: fraction_of_positives == mean_predicted_value
# Overconfident: predicted probs > actual fractions
# Underconfident: predicted probs < actual fractions
```

**Expected Calibration Error (ECE):**
```
ECE = Σ_m (|B_m| / n) · |acc(B_m) - conf(B_m)|
```

**Why calibration matters for LLMs:** If an LLM says "I'm 90% confident this is correct," it should be right 90% of the time. Overconfident models are dangerous in production.

---

## Common  Questions

- "Explain precision and recall. When would you optimize for each?"
- "Why is accuracy a bad metric for fraud detection?"
- "What is AUC-ROC and how do you interpret it?"
- "When would you use AUC-PR instead of AUC-ROC?"
- "What is nDCG and how does it differ from MAP?"
- "What is BLEU score and what are its limitations?"
- "How do you evaluate a RAG system end-to-end?"
- "What does calibration mean and why does it matter?"
