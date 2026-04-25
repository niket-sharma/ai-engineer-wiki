---
title: Databricks AI Engineering
aliases: [Databricks, MLflow, Delta Lake, Mosaic ML, DBRX, Spark]
tags: [company, mlops, mlflow, delta-lake, llm, spark, data-platform]
related: [ml-platform, feature-store, llm-serving-infra, rlhf]
sources: [training-knowledge, databricks-blog, mlflow-docs]
relevance: 8
last_updated: 2025-01-15
status: current
---

# Databricks AI Engineering

## Company Context

Databricks ($43B valuation, 2024) is the Data + AI platform company. Founded by the creators of Apache Spark and Delta Lake. Acquired Mosaic ML (2023, ~$1.3B) for LLM training capability. Databricks' product is the Lakehouse: unified platform for data engineering, analytics, ML, and LLM deployment. Engineers here build the infrastructure that thousands of companies use to train, track, serve, and monitor ML models.

**Key products:** Databricks Platform (unified Lakehouse), Delta Lake (open-source ACID table format), MLflow (experiment tracking + model registry, open-source), Unity Catalog (data governance), Mosaic AI (LLM training + fine-tuning platform), DBRX (open 132B MoE LLM), Feature Store, Model Serving.

**Open-source:** Apache Spark, Delta Lake, MLflow, DBRX, MosaicML Composer, LLM Foundry.

---

## What Databricks Engineers Work On

### 1. MLflow: The Experiment Tracking Standard

MLflow is Databricks' biggest open-source contribution to ML — used by virtually every serious ML team:

```python
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# Track experiments
mlflow.set_experiment("llm-fine-tuning")

with mlflow.start_run(run_name="qlora-llama3-8b"):
    # Log hyperparameters
    mlflow.log_params({
        "model": "meta-llama/Meta-Llama-3-8B",
        "lora_r": 64,
        "lora_alpha": 16,
        "learning_rate": 2e-4,
        "epochs": 3,
        "batch_size": 4,
    })

    # Log metrics per step
    for step, (train_loss, eval_loss) in enumerate(training_loop()):
        mlflow.log_metrics({
            "train_loss": train_loss,
            "eval_loss": eval_loss,
        }, step=step)

    # Log model with signature
    signature = mlflow.models.infer_signature(
        model_input=sample_input,
        model_output=sample_output
    )
    mlflow.pytorch.log_model(
        model,
        artifact_path="model",
        signature=signature,
        registered_model_name="llama3-8b-finetuned"
    )

# Promote model through stages
client = MlflowClient()
client.transition_model_version_stage(
    name="llama3-8b-finetuned",
    version=1,
    stage="Production"  # None → Staging → Production → Archived
)
```

**MLflow components:**
- **Tracking:** Experiments, runs, parameters, metrics, artifacts
- **Projects:** Reproducible ML code packaging (MLproject file)
- **Models:** Standard model format with `mlflow.pyfunc` wrapper
- **Registry:** Versioning + lifecycle staging
- **Deployments:** REST API serving, batch inference, Databricks Model Serving

### 2. Delta Lake: ACID Transactions on the Lakehouse

Delta Lake brings database guarantees to object storage (S3/GCS/ADLS):

```python
from delta import DeltaTable
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# Write Delta table
df.write.format("delta").mode("overwrite").save("s3://bucket/features/user_embeddings")

# ACID upsert (MERGE): critical for ML feature stores
delta_table = DeltaTable.forPath(spark, "s3://bucket/features/user_embeddings")

delta_table.alias("target").merge(
    source=new_embeddings.alias("source"),
    condition="target.user_id = source.user_id"
).whenMatchedUpdateAll() \
 .whenNotMatchedInsertAll() \
 .execute()

# Time travel: query historical data (critical for point-in-time correct training)
historical_features = spark.read \
    .format("delta") \
    .option("versionAsOf", 42) \
    .load("s3://bucket/features/user_embeddings")

# Or by timestamp (for training on data as it was at a specific point in time)
historical_features = spark.read \
    .format("delta") \
    .option("timestampAsOf", "2024-01-15 00:00:00") \
    .load("s3://bucket/features/user_embeddings")
```

**Why Delta Lake matters for ML:** Point-in-time correct feature retrieval prevents data leakage in training. Time travel lets you reproduce any historical training dataset exactly.

### 3. Mosaic AI: LLM Training Platform

After acquiring Mosaic ML, Databricks built a full LLM training stack:

```python
# MosaicML Composer: efficient LLM training
from composer import Trainer
from composer.models import HuggingFaceModel
from composer.callbacks import SpeedMonitor, LRMonitor
from composer.algorithms import GradientClipping, GradNorm

trainer = Trainer(
    model=HuggingFaceModel(model, tokenizer),
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    optimizers=optimizer,
    schedulers=scheduler,
    max_duration="3ep",
    # Databricks-developed optimizations:
    algorithms=[
        GradientClipping(clipping_type="norm", clipping_threshold=1.0),
    ],
    callbacks=[SpeedMonitor(window_size=10), LRMonitor()],
    precision="amp_bf16",  # BF16 mixed precision
    device_train_microbatch_size=4,
    # FSDP for multi-GPU
    fsdp_config={
        "sharding_strategy": "FULL_SHARD",
        "mixed_precision": "PURE",
        "activation_checkpointing": True,
    }
)
trainer.fit()
```

**LLM Fine-tuning on Databricks (production pattern):**

```python
# Fine-tuning API (high-level, no code required)
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Create fine-tuning run
run = w.fine_tuning.chat.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    training_data_path="dbfs:/training/conversations.jsonl",
    train_data_percentage=0.9,
    num_epochs=3,
    learning_rate=2e-4,
    # LoRA config
    lora_r=16,
    lora_alpha=32,
    registered_model_name="my-fine-tuned-llm"
)
```

### 4. DBRX: Databricks' Open LLM

DBRX (March 2024) is a 132B parameter MoE model — at release, best open-source LLM:

```
DBRX architecture:
- 132B total parameters, 36B active per forward pass (MoE)
- 16 experts per layer, top-4 routing
- 32 transformer layers
- 6144 hidden dimension
- Trained on 12T tokens
- Context: 32K tokens
- License: Databricks Open Model License

MoE routing mechanism:
- Each token is routed to top-4 of 16 experts by router network
- Expert capacity prevents load imbalance
- Auxiliary loss to encourage uniform expert utilization:
  L_aux = α × Σ_i (f_i × P_i)
  where f_i = fraction of tokens routed to expert i
        P_i = mean routing probability for expert i
```

### 5. Databricks Feature Store

The Databricks Feature Store solves training-serving skew:

```python
from databricks.feature_store import FeatureStoreClient
from databricks.feature_store.entities.feature_lookup import FeatureLookup

fs = FeatureStoreClient()

# Define and register features
@feature_table(
    name="user_activity_features",
    primary_keys=["user_id"],
    timestamp_keys=["event_timestamp"],  # for point-in-time lookups
    description="User activity aggregates"
)
def compute_user_features(df):
    return df.groupBy("user_id").agg(
        count("*").alias("event_count_30d"),
        avg("session_duration").alias("avg_session_duration"),
        max("event_timestamp").alias("last_active_ts")
    )

# Point-in-time correct training set creation
training_set = fs.create_training_set(
    df=labels_df,
    feature_lookups=[
        FeatureLookup(
            table_name="user_activity_features",
            feature_names=["event_count_30d", "avg_session_duration"],
            lookup_key="user_id",
            timestamp_lookup_key="label_timestamp"  # ← gets features as of this time
        )
    ],
    label="converted",
    exclude_columns=["label_timestamp"]
)

training_df = training_set.load_df()
```

---

## Key Questions

**Data Platform / Systems:**
- "How does Delta Lake implement ACID transactions on object storage?"
- "What is training-serving skew? How does a feature store prevent it?"
- "Design a point-in-time correct feature retrieval system for ML training"
- "How would you design a model registry with promotion workflows?"
- "What is Z-ordering in Delta Lake and when would you use it?"

**ML Platform:**
- "What are the key components of an ML platform? How do they connect?"
- "How does MLflow track experiments? Design your own experiment tracking system."
- "How do you handle model versioning and rollback in production?"
- "Design a system to detect data drift and trigger retraining"

**LLM / Mosaic:**
- "How does MoE routing work in DBRX? What prevents expert collapse?"
- "Explain Flash Attention and how it speeds up LLM training"
- "How would you fine-tune a 70B LLM on a 8×A100 cluster with limited VRAM?"
- "What is FSDP? How does it compare to ZeRO?"

**Coding:**
- Python + PySpark proficiency expected
- Know Delta Lake operations (merge, time travel, optimize)
- Distributed computing patterns (partitioning, broadcast joins, skew handling)

---

## Databricks-Specific Culture Notes

- **Open-source first:** Most products have OSS versions (Delta Lake, MLflow, DBRX). Engineers contribute to open-source and are judged partly by community impact.
- **Data engineering + ML combined:** Databricks sits at the intersection. Knowing Spark and ML is more important than knowing just one.
- **Customer obsession:** Most customers are enterprises; production reliability and SLAs matter enormously.
- **The Lakehouse thesis:** Belief that data warehouse + data lake will converge. All designs should support this unified architecture.

---

## Red Flags at Databricks

- **No Spark/distributed computing knowledge:** Databricks is Spark's home. Pandas-only ML experience falls short.
- **Not knowing Delta Lake:** ACID transactions on object storage, time travel, schema enforcement.
- **Ignoring MLOps concerns:** Model tracking, versioning, deployment, and monitoring are core, not afterthoughts.
- **Cloud-agnostic naivety:** Databricks runs on AWS, Azure, GCP — designs must be cloud-neutral or explicitly parameterized.

---

## 7-Day Learning Path

| Day | Focus |
|---|---|
| 1 | Delta Lake: ACID guarantees, time travel, merge/upsert, Z-ordering |
| 2 | MLflow: tracking API, model registry, deployment patterns |
| 3 | Feature Store: training-serving skew, point-in-time joins, materialization |
| 4 | LLM fine-tuning: QLoRA, FSDP, Mosaic Composer, MoE architecture |
| 5 | Distributed training: ZeRO vs FSDP, pipeline parallelism, gradient checkpointing |
| 6 | System design: ML platform architecture, feature store design |
| 7 | Coding: PySpark (groupby, join, window functions), Delta Lake operations |
