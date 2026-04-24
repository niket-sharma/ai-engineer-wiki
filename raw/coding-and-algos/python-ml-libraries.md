# Python Libraries for ML Engineering Interviews

**Focus:** NumPy, Pandas, PyTorch — the three most tested in technical ML interviews

---

## NumPy

### Array Creation

```python
import numpy as np

np.zeros((3, 4))         # 3×4 all zeros
np.ones((3, 4))          # 3×4 all ones
np.eye(3)                # 3×3 identity
np.arange(0, 10, 2)      # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)     # [0, 0.25, 0.5, 0.75, 1.0]
np.random.randn(3, 4)    # N(0,1) samples
np.random.randint(0, 10, (3, 4))  # random integers
```

### Indexing

```python
a = np.array([[1, 2, 3], [4, 5, 6]])

a[0, 1]          # → 2
a[:, 1]          # → [2, 5] (all rows, col 1)
a[0, :]          # → [1, 2, 3] (row 0, all cols)
a[0, 1:3]        # → [2, 3] (row 0, cols 1-2)

# Boolean indexing
mask = a > 3
a[mask]          # → [4, 5, 6]

# Fancy indexing
a[[0, 1], [0, 2]]  # → [1, 6] (row 0 col 0, row 1 col 2)
```

### Broadcasting Rules

Arrays are broadcast together if dimensions are compatible:
- Rule: compare dims from right; dims are compatible if equal OR one of them is 1

```python
# Shape (3, 4) + (4,) → (3, 4): broadcasts (4,) as (1, 4) → (3, 4)
a = np.ones((3, 4))
b = np.array([1, 2, 3, 4])
result = a + b   # shape (3, 4) — adds b to each row

# Shape (3, 1) + (1, 4) → (3, 4): outer product-like
c = np.array([[1], [2], [3]])   # (3, 1)
d = np.array([[10, 20, 30, 40]])  # (1, 4)
result = c + d   # (3, 4)
```

### Key Operations

```python
a = np.random.randn(3, 4)

# Axis operations
a.sum()          # sum all elements
a.sum(axis=0)    # sum each column → shape (4,)
a.sum(axis=1)    # sum each row → shape (3,)
a.mean(axis=0)
a.std(axis=1)
a.max(); a.argmax()
a.min(axis=0)

# Matrix operations
a @ b            # matrix multiply (preferred over np.dot)
a.T              # transpose
np.linalg.inv(a) # matrix inverse
np.linalg.norm(a, axis=-1)  # L2 norm along last axis
np.einsum('ij,jk->ik', a, b)  # flexible einstein summation = matmul

# Reshape and concat
a.reshape(4, 3)
a.flatten()
np.concatenate([a, b], axis=0)
np.stack([a, b], axis=0)   # new axis
np.vstack([a, b])          # axis=0 concat
np.hstack([a, b])          # axis=1 concat

# Softmax (implement yourself in interviews)
def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)  # numerical stability
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)
```

---

## Pandas

### DataFrame Creation and I/O

```python
import pandas as pd

df = pd.read_csv("data.csv")
df = pd.read_parquet("data.parquet")
df.to_csv("output.csv", index=False)

# Create from dict
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [70000, 80000, 90000]
})
```

### Exploration

```python
df.shape         # (n_rows, n_cols)
df.dtypes        # column types
df.describe()    # mean, std, quartiles for numeric cols
df.info()        # non-null counts + dtypes
df.head(10)
df.value_counts('column')
df.isnull().sum()   # missing values per column
```

### Selection

```python
df['col']             # Series
df[['col1', 'col2']]  # DataFrame

# loc: label-based (inclusive end)
df.loc[0:5, 'age':'salary']
df.loc[df['age'] > 25]

# iloc: integer-based (exclusive end)
df.iloc[0:5, 1:3]
df.iloc[-1]           # last row
```

### Transformation

```python
# Apply function
df['salary_k'] = df['salary'] / 1000
df['age_group'] = df['age'].apply(lambda x: 'senior' if x >= 30 else 'junior')

# GroupBy
df.groupby('department')['salary'].mean()
df.groupby(['department', 'age_group']).agg({'salary': ['mean', 'std'], 'age': 'count'})

# Merge
pd.merge(df1, df2, on='user_id', how='left')
pd.merge(df1, df2, left_on='user', right_on='id', how='inner')

# Pivot
df.pivot_table(values='sales', index='month', columns='product', aggfunc='sum')

# Sort
df.sort_values('salary', ascending=False).head(10)

# Handling missing values
df.fillna(0)
df.fillna(df.mean())
df.dropna(subset=['required_col'])

# String operations
df['name'].str.lower()
df['name'].str.contains('alice', case=False)
df['name'].str.split(' ', expand=True)

# Datetime
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
```

### Common  Data Tasks

```python
# Moving average (time series)
df['rolling_avg'] = df['value'].rolling(window=7).mean()

# Cumulative sum
df['cumsum'] = df['value'].cumsum()

# Lag features
df['prev_value'] = df['value'].shift(1)
df['diff_from_prev'] = df['value'].diff(1)

# Percentile rank
df['rank'] = df['value'].rank(pct=True)

# Top N per group
df.groupby('category').apply(lambda x: x.nlargest(3, 'sales')).reset_index(drop=True)

# Explode list column
df['tags'] = df['tags'].str.split(',')
df = df.explode('tags')
```

---

## PyTorch

### Tensor Basics

```python
import torch

# Creation
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
x = torch.zeros(3, 4)
x = torch.randn(3, 4)
x = torch.arange(10, dtype=torch.float32)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)
# or: x = x.cuda()

# Dtype
x = x.float()   # float32
x = x.half()    # float16
x = x.long()    # int64
```

### Autograd

```python
# Requires gradient tracking
x = torch.randn(3, 4, requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)   # dy/dx = 2x

# Detach from computation graph
with torch.no_grad():
    output = model(x)  # no gradient tracking (inference)

x_detached = x.detach()  # detach single tensor
```

### Neural Network

```python
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

model = MLP(784, 256, 10).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for batch_x, batch_y in dataloader:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    optimizer.zero_grad()
    output = model(batch_x)
    loss = criterion(output, batch_y)
    loss.backward()
    optimizer.step()

# Inference
model.eval()
with torch.no_grad():
    preds = model(test_x)
    probs = torch.softmax(preds, dim=-1)
```

### Common Tensor Operations

```python
# Shape manipulation
x.reshape(3, 4)          # returns view if possible, else copy
x.view(3, 4)             # must be contiguous
x.permute(2, 0, 1)       # permute dimensions (no copy)
x.transpose(0, 1)        # swap two dims
x.unsqueeze(0)           # add dim at position 0
x.squeeze()              # remove all size-1 dims
x.expand(3, -1)          # broadcast without copying (-1 = keep)
x.repeat(3, 1)           # copy data

# Reductions
x.sum(); x.sum(dim=0, keepdim=True)
x.mean(); x.max(); x.min()
x.argmax(dim=-1)

# Matrix ops
x @ y                    # matmul
torch.bmm(x, y)          # batched matmul: (B, M, K) @ (B, K, N) → (B, M, N)
torch.einsum('bijk,bkl->bijl', x, y)

# Attention pattern
def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / d_k ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return weights @ V
```

---

## scikit-learn Patterns

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

# Pipeline (prevents data leakage)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', GradientBoostingClassifier())
])

# Grid search
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [3, 5, 7]
}
gs = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
gs.fit(X_train, y_train)

print(f"Best params: {gs.best_params_}")
print(f"Best CV score: {gs.best_score_:.3f}")

# Feature importance
model = gs.best_estimator_['clf']
feature_importance = model.feature_importances_
```

---

## Common  Coding Exercises

```python
# Implement attention from scratch
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    return torch.softmax(scores, dim=-1) @ V

# Implement batch normalization
class BatchNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    
    def forward(self, x):  # x: (B, D)
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        x_norm = (x - mean) / (std + self.eps)
        return self.gamma * x_norm + self.beta

# Implement cosine similarity search
def cosine_similarity_search(query, corpus):
    """
    query: (d,)
    corpus: (N, d)
    Returns: indices sorted by similarity (highest first)
    """
    query_norm = query / torch.norm(query)
    corpus_norm = corpus / torch.norm(corpus, dim=1, keepdim=True)
    similarities = corpus_norm @ query_norm
    return torch.argsort(similarities, descending=True)
```
