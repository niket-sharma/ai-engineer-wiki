---
title: "Capital One"
aliases: ["Capital One AI", "Capital One interview prep"]
tags: [company, finance, fraud, nlp, aws]
related: ["[[rag-systems]]", "[[rag-pipeline-design]]", "[[vector-databases]]", "[[llm-serving-infra]]"]
sources: ["training-knowledge", "public-job-postings"]
interview_relevance: high
last_updated: 2026-04-22
status: current
---

# Capital One

## Role Focus
Capital One is one of the most ML-forward banks in the US. AI Engineer / ML Engineer roles focus on:
- **Fraud detection**: real-time transaction scoring, anomaly detection, graph-based fraud rings
- **Credit risk modeling**: approval models, limit setting, behavioral scoring
- **NLP on financial text**: customer communications, document analysis, internal knowledge bases
- **Conversational AI**: Eno (their AI assistant), internal chatbots built on LLMs
- **Decisioning infrastructure**: ML platform, real-time feature serving, A/B testing

Capital One is AWS-first — everything runs on AWS. Deep SageMaker usage. Strong MLOps culture.

## Tech Stack Signals
- **Cloud**: AWS (Capital One's "cloud-first" migration is famous)
- **ML training**: SageMaker, PyTorch
- **Serving**: SageMaker endpoints, custom serving on EKS
- **Data**: Redshift, S3, Glue, Spark on EMR
- **Feature store**: internal platform (likely Feast-based or custom)
- **LLMs**: internal deployments of GPT-4 (Azure OpenAI), some open-source via SageMaker
- **Fraud graph**: likely Neo4j or custom graph processing

## Interview Style
Based on public signals and typical Capital One interview structure:
- **Phone screen**: 45 min ML theory + coding (LeetCode medium)
- **Technical loop (virtual onsite)**: 
  - 2× coding (LeetCode medium-hard, focus on arrays/graphs/DP)
  - 1× ML design (fraud detection or RAG system design)
  - 1× behavioral (leadership principles style)
- **ML depth**: expects strong fraud/risk domain knowledge for ML roles
- **Coding**: LeetCode medium difficulty, occasional hard. Focus: sliding window, graph traversal, DP

## Domain-Specific Concepts

### Fraud Detection
- **Problem**: binary classification (fraud/not fraud) on highly imbalanced data (0.1% fraud rate)
- **Imbalanced data techniques**: SMOTE, class weights, precision-recall AUC instead of ROC-AUC
- **Features**: transaction velocity (how many txns in last hour), merchant category risk, geographic anomaly, device fingerprint
- **Real-time constraint**: decision must be made in < 100ms during authorization
- **Model types**: XGBoost/LightGBM for tabular, Graph Neural Networks for fraud rings
- **Key metric for fraud**: F1 at specific precision thresholds (e.g., F1 at 95% precision)

### Credit Risk
- **Scorecard models**: logistic regression with WoE (Weight of Evidence) encoding — regulatory requirement for explainability
- **Behavioral scoring**: revolvers vs. transactors, utilization patterns
- **Key metric**: Gini coefficient (= 2×AUC - 1), KS statistic

### LLM/NLP at Capital One
- **Internal chatbots**: Eno handles customer questions — RAG-based, strict guardrails
- **Document processing**: contract analysis, regulatory filings (Reg E complaints)
- **Content moderation**: customer service conversation analysis
- **Interest**: Capital One hosts "Capital One Tech" engineering blog — read it before interviews

## Likely Questions
1. "Design a real-time fraud detection system that can score 10,000 transactions per second."
2. "How would you handle a 1000:1 class imbalance in a fraud model?"
3. "Walk me through how you'd build a RAG system for Capital One's internal knowledge base."
4. "How would you ensure your credit model is fair across protected attributes?"
5. "What is a feature store and why is it critical for real-time fraud scoring?"
6. "You see a sudden drop in your fraud model's precision. Walk me through debugging it."
7. "How would you design an A/B test for a new credit approval model?"
8. "Explain the tradeoffs between model accuracy and explainability for a regulatory context."

## Red Flags to Avoid
- Don't propose uninterpretable black-box models for credit decisions without discussing explainability requirements (ECOA, FCRA compliance)
- Don't propose offline-only solutions for fraud — real-time is non-negotiable
- Don't forget data privacy considerations — financial data has strict regulatory requirements
- Don't underestimate the MLOps/infrastructure side — Capital One cares about production, not just notebooks

## Study Plan for Capital One Interview

**Day 1-2:** [[rag-systems]], [[vector-databases]] — design a RAG for financial Q&A
**Day 3:** Fraud ML: imbalanced classification, precision-recall, feature engineering for transactions
**Day 4:** [[rag-pipeline-design]] — practice the system design whiteboard
**Day 5:** [[feature-store]], [[llm-serving-infra]] — real-time serving architecture
**Day 6:** LeetCode: sliding window, BFS/DFS, DP (10 mediums)
**Day 7:** [[behavioral-qa]] + mock system design with timer

## Connections
- [[rag-systems]] — Eno and internal knowledge base use cases
- [[feature-store]] — real-time feature serving for fraud scoring
- [[rag-pipeline-design]] — system design for internal document search
