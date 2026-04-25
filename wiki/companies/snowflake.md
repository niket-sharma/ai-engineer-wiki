---
title: Snowflake AI Engineering
aliases: [Snowflake, Cortex AI, Arctic, Document AI, Snowpark]
tags: [company, data-platform, llm, sql-ai, vector-search, enterprise-data]
related: [rag-systems, feature-store, ml-platform, llm-serving-infra]
sources: [training-knowledge, snowflake-blog, snowflake-docs]
relevance: 7
last_updated: 2025-01-15
status: current
---

# Snowflake AI Engineering

## Company Context

Snowflake ($50B market cap) is the cloud data platform (data warehouse + data lake + data sharing). Snowflake's AI pivot (2023–2024) brings LLMs and ML directly into the data platform — no data movement, AI runs where the data lives. Key acquisitions: Streamlit (2022, $800M), Neeva (2023, enterprise search), TruEra (2024, ML observability).

**Key AI products:** Cortex AI (LLM API inside Snowflake), Arctic (open-source enterprise LLM), Cortex Search (vector search in Snowflake), Document AI (extract structured data from PDFs), Snowpark ML (train/serve ML models in Snowflake), Streamlit in Snowflake (ML apps), ML Observability (TruEra integration).

**Unique proposition:** AI without data egress — query LLMs, run ML, and store vectors all within Snowflake's security boundary. For regulated enterprises, this removes compliance blockers.

---

## What Snowflake AI Engineers Work On

### 1. Cortex AI: LLM Inside the Data Warehouse

Cortex AI lets SQL users call LLMs without leaving Snowflake:

```sql
-- Cortex AI functions in SQL (no Python, no data movement)

-- Text summarization
SELECT 
    customer_id,
    SNOWFLAKE.CORTEX.SUMMARIZE(support_ticket_text) AS summary
FROM support_tickets
WHERE created_at > DATEADD('day', -7, CURRENT_DATE());

-- Entity extraction
SELECT 
    document_id,
    SNOWFLAKE.CORTEX.COMPLETE(
        'llama3-70b',
        CONCAT('Extract the company name, date, and total amount from this invoice. 
               Return JSON only.\n\n', invoice_text)
    ) AS extracted_json
FROM invoices;

-- Sentiment classification
SELECT 
    review_id,
    SNOWFLAKE.CORTEX.SENTIMENT(review_text) AS sentiment_score  -- returns -1 to 1
FROM product_reviews;

-- Translation
SELECT SNOWFLAKE.CORTEX.TRANSLATE(text, 'en', 'de') AS german_text FROM docs;
```

**Available Cortex models:**
- `llama3-70b`, `llama3-8b` (Meta, open-source)
- `mistral-large`, `mistral-7b`
- `snowflake-arctic-instruct` (Snowflake's own)
- `jamba-instruct` (AI21 Labs)

**Engineering challenge:** Running LLM inference at SQL scale — queries may call CORTEX.COMPLETE on millions of rows. Requires batching, async execution, and cost governance.

### 2. Cortex Search: Vector Search in Snowflake

Cortex Search integrates vector similarity search directly into Snowflake tables:

```sql
-- Create a Cortex Search service (indexes a column for semantic search)
CREATE OR REPLACE CORTEX SEARCH SERVICE product_search
  ON description_text
  ATTRIBUTES product_id, price, category
  WAREHOUSE = compute_wh
  TARGET_LAG = '1 minute'  -- how fresh the index stays
  AS (
    SELECT product_id, description_text, price, category
    FROM products
    WHERE active = TRUE
  );
```

```python
# Python SDK for semantic search
from snowflake.cortex import CortexSearchService
import snowflake.connector

conn = snowflake.connector.connect(**connection_params)
search_service = CortexSearchService(conn, "DB.SCHEMA.PRODUCT_SEARCH")

results = search_service.search(
    query="wireless headphones with noise cancellation",
    columns=["product_id", "description_text", "price"],
    filter={"@eq": {"category": "Electronics"}},
    limit=10,
)

for r in results.results:
    print(r["product_id"], r["price"], r["description_text"][:80])
```

**Architecture (inferred):**
- Embedding model runs inside Snowflake compute (no external API call)
- Vector index stored in Snowflake storage (ANN index, likely HNSW-based)
- Hybrid search: BM25 keyword + dense vector similarity, RRF fusion
- Governed by Snowflake RBAC (same access control as tables)

### 3. Arctic: Snowflake's Enterprise LLM

Arctic (April 2024) is Snowflake's open-source MoE LLM optimized for enterprise tasks:

```
Arctic architecture:
- 480B total parameters, 17B active per token (MoE)
- Dense transformer + Residual MoE (novel hybrid)
- 128 experts, top-2 routing
- Trained on 3.5T tokens
- Training cost: ~$2M (claimed, vs $10M+ for comparable models)

What makes Arctic unique:
1. Dense + MoE hybrid: dense layers for basic language + MoE for specialization
2. Enterprise focus: optimized for SQL generation, coding, instruction following
3. Open weights: Apache 2.0 license (commercial use allowed)
4. Efficiency: 17B active params → fast inference despite large total capacity

SQL generation benchmark (Spider dataset):
- Arctic: 78.5% execution accuracy
- GPT-4 Turbo: 79.0% (marginal difference at 5% cost)
```

### 4. Document AI: PDF/Unstructured Data Extraction

Document AI extracts structured data from unstructured documents:

```python
import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col

# Document AI: build and use extraction models
# No-code: define schema, upload labeled examples, Snowflake trains classifier

# Step 1: Create document model (via Snowsight UI or SQL)
# Step 2: Run on documents

session = snowpark.Session.builder.configs(connection_params).create()

# Process uploaded PDFs
results = session.sql("""
    SELECT 
        relative_path AS filename,
        SNOWFLAKE.ML.DOCUMENT_AI!PREDICT(
            GET_PRESIGNED_URL('@invoice_stage', relative_path),
            1  -- model version
        ) AS extracted
    FROM DIRECTORY(@invoice_stage)
""").collect()

for row in results:
    import json
    extracted = json.loads(row["EXTRACTED"])
    vendor = extracted["vendor_name"]["value"]
    amount = extracted["total_amount"]["value"]
    confidence = extracted["total_amount"]["score"]  # confidence 0-1
    print(f"{vendor}: ${amount} (confidence: {confidence:.2f})")
```

### 5. Snowpark ML: Training Models in Snowflake

Snowpark ML brings scikit-learn-compatible training to Snowflake:

```python
from snowflake.ml.modeling.ensemble import RandomForestClassifier
from snowflake.ml.modeling.preprocessing import StandardScaler, OneHotEncoder
from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.registry import Registry

# All computation happens inside Snowflake — no data leaves
pipeline = Pipeline(steps=[
    ("encoder", OneHotEncoder(input_cols=["CATEGORY", "REGION"], output_cols=["CATEGORY_ENC", "REGION_ENC"])),
    ("scaler", StandardScaler(input_cols=["AMOUNT", "AGE"], output_cols=["AMOUNT_SCALED", "AGE_SCALED"])),
    ("model", RandomForestClassifier(
        input_cols=["CATEGORY_ENC", "REGION_ENC", "AMOUNT_SCALED", "AGE_SCALED"],
        label_cols=["CHURN"],
        n_estimators=100,
    )),
])

pipeline.fit(train_df)  # runs inside Snowflake, uses Snowflake warehouse compute

# Register model in Snowflake Model Registry
registry = Registry(session=session)
mv = registry.log_model(
    model=pipeline,
    model_name="churn_predictor",
    version_name="v1",
    metrics={"test_auc": 0.87},
)

# Deploy to Model Serving (inference endpoint inside Snowflake)
mv.deploy(
    deployment_name="churn_predictor_prod",
    platform=deploy_platforms.TargetPlatform.SNOWPARK_CONTAINER_SERVICES,
)
```

---

## Key Questions

**Data Platform / SQL AI:**
- "Design Cortex AI's architecture for running LLM inference on SQL query results at scale"
- "How would you implement vector search inside a data warehouse?"
- "What are the trade-offs of running LLM inference inside the data warehouse vs calling an external API?"
- "How do you handle cost governance when users can call CORTEX.COMPLETE on millions of rows?"

**ML Systems:**
- "How would you build a document extraction pipeline for PDFs at enterprise scale?"
- "Design a feature store that lives entirely within Snowflake"
- "How does Snowflake's Model Registry differ from MLflow?"

**Architecture:**
- "What is the MoE hybrid architecture in Arctic? Why dense + MoE?"
- "How does Cortex Search implement hybrid BM25 + vector search?"
- "How do you ensure that vector search respects Snowflake's row-level security?"

---

## Red Flags at Snowflake

- **Not thinking SQL-first:** Snowflake's customers are data engineers and analysts who live in SQL. Python-only ML mindset misses the core value proposition.
- **Ignoring governance:** Data governance (RBAC, column masking, row-level security) must extend to AI features. AI features that bypass security are a dealbreaker.
- **Not knowing the data stack:** Know Delta Lake, Iceberg, Parquet — Snowflake competes with Databricks, BigQuery, Redshift.
- **Underestimating scale:** Snowflake's customers run exabytes of queries. AI features must work at that scale.

---

## 7-Day Learning Path

| Day | Focus |
|---|---|
| 1 | Snowflake fundamentals: virtual warehouses, clustering keys, zero-copy cloning |
| 2 | Vector search: HNSW index, hybrid BM25+dense, recall vs latency trade-offs |
| 3 | LLM SQL integration: prompt engineering for SQL, text-to-SQL, entity extraction |
| 4 | MoE architecture: routing, expert capacity, load balancing loss |
| 5 | Document AI: document layout analysis, information extraction, few-shot labeling |
| 6 | ML in data platforms: Snowpark ML, feature stores, model registry |
| 7 | System design: enterprise AI pipeline entirely within Snowflake's trust boundary |
