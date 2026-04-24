---
title: Feature Store Design
aliases: [Feature Store, Online Feature Store, Offline Feature Store]
tags: [system-design, mlops, feature-store, real-time-ml]
related: [ml-platform, rag-pipeline-design]
sources: [training-knowledge, feast-docs, tecton-blog]
interview_relevance: 7
last_updated: 2025-01-15
status: current
---

# Feature Store Design

## TL;DR

A feature store is a centralized system with two stores: an **offline store** (historical features for training, batch) and an **online store** (latest features for serving, < 10ms). The core value: same feature computation logic used at training and serving time → eliminates training-serving skew.

---

## Problem Statement

### Training-Serving Skew (The Core Problem)

Without a feature store, teams compute features separately for training and serving:

```python
# Training (data scientist, runs once on historical data)
df['user_avg_spend_30d'] = df.groupby('user_id')['amount'].transform(
    lambda x: x.rolling(30).mean()  # rolling 30 rows
)

# Serving (engineer, runs on each request)
def get_user_avg_spend(user_id: str) -> float:
    result = db.query("""
        SELECT AVG(amount) FROM transactions 
        WHERE user_id = %s AND date > NOW() - INTERVAL 30 DAY
    """, user_id)  # rolling 30 calendar days
    return result[0]
```

These compute differently (30 rows vs 30 calendar days). Model trained on one, served with the other.

### Feature Duplication

Five teams each write their own "user average spend" feature, slightly differently, wasting engineering time and creating maintenance burden.

---

## High-Level Architecture

```
                    Feature Pipeline
                    (Spark / Flink job)
                           │
                           ▼
Sources ──────────► Feature Store ──────────► Consumers
(events,           ┌────────────┐            Training job
 transactions,     │ Offline    │──────────► Batch scoring
 user profiles)    │ Store      │
                   │ (S3 +      │
                   │ Parquet)   │
                   └────────────┘
                   ┌────────────┐
                   │ Online     │──────────► Inference API
                   │ Store      │            (5ms p99)
                   │ (Redis /   │
                   │ DynamoDB)  │
                   └────────────┘
```

---

## Component Deep-Dives

### Offline Store

**Purpose:** Historical feature values for training and batch scoring.

**Critical feature: Point-in-time correct joins** — fetch feature value as of the label timestamp, not current:

```python
# BAD: naive join — data leakage (uses future feature values for past labels)
training_df = pd.merge(labels_df, features_df, on='user_id')

# GOOD: point-in-time join via feature store
from feast import FeatureStore

fs = FeatureStore(repo_path=".")
entity_df = pd.DataFrame({
    "user_id": labels_df['user_id'],
    "event_timestamp": labels_df['label_timestamp']  # fetch as of this time
})

training_df = fs.get_historical_features(
    entity_df=entity_df,
    features=["user_stats:avg_spend_30d", "user_stats:transaction_count_7d"]
).to_df()
```

**Storage:** S3/GCS Parquet partitioned by date. Hive-compatible metastore for SQL access.

### Online Store

**Purpose:** Latest feature values for real-time inference. Must be < 10ms.

```python
# At inference time
entity_rows = [{"user_id": "user_123"}]
features = fs.get_online_features(
    features=["user_stats:avg_spend_30d", "user_stats:transaction_count_7d"],
    entity_rows=entity_rows
).to_df()
prediction = model.predict(features)
```

**Storage options:**
- **Redis:** < 1ms, must fit in RAM, single-region
- **DynamoDB:** < 5ms, unlimited scale, multi-region
- **Bigtable/Cassandra:** Low-latency, high-throughput, wide rows

**Freshness:** Online store updated via materialization jobs.

```python
# Incremental materialization (hourly cron)
fs.materialize_incremental(end_date=datetime.now())
```

### Feature Definition (Feast)

```python
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

user = Entity(name="user_id", join_keys=["user_id"])

user_stats_fv = FeatureView(
    name="user_stats",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name="avg_spend_30d", dtype=Float32),
        Field(name="transaction_count_7d", dtype=Int64),
    ],
    source=FileSource(
        path="s3://bucket/user_transactions/",
        timestamp_field="event_timestamp"
    ),
    online=True
)
```

---

## Scale & Reliability

### Online Store Throughput

Fraud detection at 1000 TPS with 20 features per request:
- 1000 × 20 = 20,000 feature reads/second
- Redis: easily handles 100k+ reads/second per instance
- DynamoDB: auto-scales to millions of reads/second

### Batch Pipeline SLA

Training pipelines need yesterday's features by 6 AM:
- Spark job triggered at midnight
- Processes previous day's events
- Writes Parquet to offline store partitioned by date
- Materializes to online store by 4 AM
- 2-hour buffer before SLA

### Streaming Features (Near Real-Time)

For fraud detection needing transaction velocity in the last hour:

```python
# Kafka → Flink → Online Store
def process_transaction(event: dict):
    user_id = event['user_id']
    # Fetch current state from online store
    current = online_store.get(user_id)
    # Update streaming aggregate
    online_store.set(user_id, {
        'tx_count_1h': current['tx_count_1h'] + 1,
        'amount_sum_1h': current['amount_sum_1h'] + event['amount']
    })
```

---

## Tradeoffs

| Decision | Option A | Option B |
|---|---|---|
| Online store | Redis (< 1ms, RAM cost) | DynamoDB (< 5ms, unlimited scale) |
| Feature framework | Feast (open-source, DIY ops) | Tecton (managed, streaming first-class) |
| Freshness | Batch hourly (simpler) | Streaming (complex, near-real-time) |
| Offline store format | Parquet (fast reads) | Delta Lake (ACID, time-travel) |

---

##  Angles

- **"What is training-serving skew and how does a feature store prevent it?"**
  Training-serving skew: features computed differently at training vs serving time. Feature store prevents it by sharing the same transformation logic: `get_historical_features()` and `get_online_features()` both invoke the same feature definitions.

- **"What is point-in-time correct retrieval?"**
  When building training data, features must be fetched as of each label's timestamp, not current time. Prevents data leakage — using future feature values to predict past outcomes.

- **"How fresh do features need to be for real-time fraud detection?"**
  Transaction velocity features (tx count in last hour) need < 1 minute freshness — requires streaming materialization via Kafka/Flink, not batch jobs.

- **"When would you use Redis vs DynamoDB for online store?"**
  Redis for < 1ms latency with features fitting in RAM. DynamoDB for larger feature sets, multi-region availability, or when RAM cost for Redis is prohibitive (billions of users).

## Connections
- [[ml-platform]] — feature store is a core component of the ML platform
- [[rag-pipeline-design]] — RAG document stores have similar offline/online split patterns
