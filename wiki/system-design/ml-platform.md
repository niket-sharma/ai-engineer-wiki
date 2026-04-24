---
title: ML Platform Design
aliases: [ML Platform, MLOps Platform, Feature Platform]
tags: [system-design, mlops, feature-store, model-registry, training, serving]
related: [feature-store, llm-serving-infra, rag-pipeline-design]
sources: [training-knowledge, mlops-platform-patterns]
interview_relevance: 8
last_updated: 2025-01-15
status: current
---

# ML Platform Design

## What Interviewers Are Testing

"Design an ML platform for 50 data scientists and 10 ML engineers." Tests your understanding of:
- The full ML lifecycle (data → features → training → evaluation → serving → monitoring)
- Infrastructure choices (compute, storage, orchestration)
- Developer experience tradeoffs
- Common pitfalls (training-serving skew, reproducibility, model drift)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ML PLATFORM                                │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │ Data Layer   │  │ Feature Layer│  │ Experiment Layer      │ │
│  │              │  │              │  │                       │ │
│  │ Data Lake    │  │ Feature Store│  │ Experiment Tracking   │ │
│  │ (S3/GCS)     │  │ (Feast/      │  │ (MLflow/W&B)          │ │
│  │              │  │  Tecton)     │  │                       │ │
│  │ Data Catalog │  │ Offline      │  │ Notebook Environment  │ │
│  │ (Datahub)    │  │ Online Store │  │ (JupyterHub/VS Code)  │ │
│  └──────────────┘  └──────────────┘  └───────────────────────┘ │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │Training Layer│  │ Registry     │  │ Serving Layer         │ │
│  │              │  │              │  │                       │ │
│  │ Job Scheduler│  │ Model        │  │ Inference Server      │ │
│  │ (Airflow/    │  │ Registry     │  │ (Triton/TorchServe/   │ │
│  │  Kubeflow)   │  │ (MLflow)     │  │  BentoML)             │ │
│  │              │  │              │  │                       │ │
│  │ Compute      │  │ Dataset      │  │ A/B Testing           │ │
│  │ (k8s + GPU)  │  │ Versioning   │  │ Shadow Mode           │ │
│  └──────────────┘  └──────────────┘  └───────────────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Monitoring & Observability                               │   │
│  │ (Prometheus + Grafana + Evidently/WhyLogs)               │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Data Layer

### Data Lake

```
Raw → Bronze → Silver → Gold
  ↓       ↓         ↓       ↓
Files  Ingested  Cleaned  Feature-ready
(S3)   (Parquet) (joins)  (ML-ready)
```

**Technology choices:**
- Storage: S3 / GCS / ADLS (cloud-agnostic Parquet)
- Format: Parquet (columnar, efficient for ML) + Delta Lake / Iceberg for ACID
- Query: Spark (batch) + Trino/Presto (interactive SQL)

### Data Catalog

Every dataset needs:
- Owner, description, schema, lineage
- Data quality checks (Great Expectations / Deequ)
- Access controls (column-level encryption for PII)

---

## Layer 2: Feature Store

The most -questioned component. See [[feature-store]] for full detail.

**Key concept: training-serving skew prevention**

```python
# BAD: compute features differently in training vs serving
# training: pandas rolling window
# serving: SQL over last N rows (different computation)

# GOOD: shared feature computation via feature store
# training: feature_store.get_historical_features(entity_df, feature_refs)
# serving: feature_store.get_online_features(entity_rows, feature_refs)
```

---

## Layer 3: Experiment Tracking

Every training run should log:
- **Hyperparameters:** learning rate, batch size, model architecture
- **Metrics:** train/val loss, AUC, F1 at every epoch
- **Artifacts:** model checkpoint, confusion matrix, learning curve
- **Environment:** Python version, library versions, git commit hash
- **Data:** dataset version, train/val splits

```python
import mlflow

with mlflow.start_run():
    mlflow.log_params({"lr": 1e-3, "batch_size": 32, "n_layers": 6})
    mlflow.log_metrics({"val_auc": 0.85, "val_f1": 0.82})
    mlflow.pytorch.log_model(model, "model")
    mlflow.log_artifact("confusion_matrix.png")
```

---

## Layer 4: Training Infrastructure

### Compute Orchestration

**Options:** Kubeflow Pipelines, Airflow + Ray, SageMaker Pipelines, Vertex AI Pipelines

**Key requirements:**
- GPU scheduling (priority queues, gang scheduling for multi-GPU jobs)
- Spot/preemptible instances (80% cheaper, need checkpointing)
- Resource quotas per team
- Job monitoring (wandb / MLflow tracking)

**Reproducibility checklist:**
- Fixed random seeds (`torch.manual_seed`, `np.random.seed`)
- Pinned library versions (requirements.txt or conda env)
- Data versioning (DVC or feature store snapshot)
- Git commit hash logged with every run

---

## Layer 5: Model Registry

Model lifecycle states:
```
Staging → Validated → Production → Archived
```

**What to store per model version:**
- Model weights (format: ONNX for portability, or framework-native)
- Input/output schema (Pydantic model or JSON Schema)
- Performance metrics on validation set
- Training run reference (links to experiment tracker)
- Approval history (who approved production promotion)

```python
# MLflow model registry
mlflow.register_model("runs:/abc123/model", "FraudDetector")

client = MlflowClient()
client.transition_model_version_stage(
    name="FraudDetector", version=5, stage="Production"
)
```

---

## Layer 6: Model Serving

See [[llm-serving-infra]] for LLM-specific detail.

**Canary deployment pattern:**
1. Deploy new model to 1% of traffic
2. Compare metrics to production model (5 minute windows)
3. Auto-rollback if error rate spikes or latency > 2× baseline
4. Gradual ramp: 1% → 5% → 20% → 50% → 100%

---

## Layer 7: Monitoring

### Three Pillars

**1. System monitoring:** Latency (p50/p95/p99), throughput, error rate, GPU utilization

**2. Data monitoring (feature drift):**
- PSI (Population Stability Index) for numeric features: PSI < 0.1 no shift, > 0.25 significant
- KL divergence between training and serving distributions
- Missing value rates

**3. Prediction monitoring:**
- Prediction distribution shift (score histogram)
- Downstream business metrics (conversion rate, revenue)

```python
# PSI calculation for numeric feature drift
def psi(expected, actual, buckets=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    exp_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    epsilon = 1e-6
    return np.sum((act_pct - exp_pct) * np.log((act_pct + epsilon) / (exp_pct + epsilon)))
```

---

## Tradeoffs

| Decision | Option A | Option B |
|---|---|---|
| Orchestration | Kubeflow (k8s-native, complex) | Airflow (simpler, less ML-aware) |
| Experiment tracking | MLflow (self-hosted, free) | W&B (managed, $$$) |
| Feature store | Feast (open-source) | Tecton (managed, streaming) |
| Online store | Redis (fast, in-memory cost) | DynamoDB (scalable, slightly slower) |
| Serving | Triton (GPU optimized) | BentoML (easier, ML framework aware) |

---

##  Angles

- "How do you prevent training-serving skew?" → Feature store with shared computation
- "How do you handle model rollback?" → Blue-green deployment + model registry versioning
- "How do you detect data drift?" → PSI monitoring on input features, alert on PSI > 0.25
- "How do you support 50 data scientists without blocking each other?" → k8s resource quotas, shared feature store with PR-gated writes

## Connections
- [[feature-store]] — provides features to training pipelines
- [[llm-serving-infra]] — serving is one component of the full platform
- [[rag-pipeline-design]] — RAG systems are a specialized ML platform use case
