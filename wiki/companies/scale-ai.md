---
title: Scale AI Engineering
aliases: [Scale AI, Scale, Spellbook, RLHF data, data labeling]
tags: [company, data-labeling, rlhf, llm-evaluation, enterprise-ai]
related: [rlhf, constitutional-ai, evaluation-metrics, llm-serving-infra]
sources: [training-knowledge, scale-ai-blog, scale-research]
relevance: 7
last_updated: 2025-01-15
status: current
---

# Scale AI Engineering

## Company Context

Scale AI ($14B valuation) is the data infrastructure company for AI — they produce the training data and evaluation pipelines that power frontier models at OpenAI, Anthropic, Meta, Google, and the US government (DoD). Scale's core products are human-in-the-loop data annotation, RLHF preference collection, red-teaming services, and enterprise fine-tuning. In 2024, Scale launched Spellbook (LLM for enterprise) and SEAL (evaluation leaderboard).

**Key products:** Data Engine (annotation platform), Generative AI (RLHF + fine-tuning data), Government (GEOINT, autonomous vehicles for DoD), Spellbook (enterprise LLM), SEAL Leaderboards (safety + capability evals), Evaluation API.

**Who they sell to:** OpenAI (first customer), Anthropic, Meta, Microsoft, US Air Force, US Army, self-driving companies (Waymo, Cruise).

---

## What Scale AI Engineers Work On

### 1. Data Annotation Pipeline at Scale

Scale's core product is a managed workforce + platform for data annotation:

```
Scale's annotation pipeline:

1. Task creation: customer defines annotation task
   (bounding boxes, RLHF preferences, text classification, NER, etc.)

2. Quality control (key differentiator vs Amazon Mechanical Turk):
   - Worker skill assessment: pre-task qualification tests
   - Consensus: multiple workers per task, agree = label, disagree = escalate
   - Gold standard: inject known-correct tasks → measure worker accuracy
   - Audit queue: borderline cases reviewed by senior annotators

3. Task routing:
   - Match task complexity to worker skill level
   - Route specialized tasks (medical, legal, code) to domain experts
   - Maintain per-worker error rates and auto-adjust routing

4. Output: high-quality labeled dataset delivered via API

Quality metrics:
  Inter-Annotator Agreement (IAA): Cohen's kappa > 0.8 target for most tasks
  Precision/Recall vs gold standard: > 95% for production tasks
```

### 2. RLHF Preference Collection

Scale's Generative AI product collects the preference data used for RLHF fine-tuning:

```python
# Scale API for RLHF preference collection
import scaleai

client = scaleai.ScaleClient(api_key=SCALE_API_KEY)

# Create a comparison task (A vs B preference)
task = client.create_task(
    task_type="comparison",
    project="rlhf-reward-model-training",
    callback_url="https://api.mycompany.com/scale-callback",
    attachment_type="text",
    attachments=[
        {"response_a": model_response_1, "response_b": model_response_2},
    ],
    instruction="""
    You are evaluating two AI assistant responses to the same user prompt.
    Rate which response is better overall, considering:
    - Helpfulness: does it fully answer the question?
    - Accuracy: is it factually correct?
    - Safety: does it avoid harmful content?
    - Conciseness: is it appropriately brief?
    
    Select: A is better / B is better / roughly equal
    Optionally rate severity: strongly prefer / slightly prefer
    """,
    # Quality controls
    min_worker_agreement=0.7,
    enable_consensus=True,
    consensus_count=3,  # 3 annotators per task
)

# Poll for completion
completed_task = client.get_task(task.task_id)
preference = completed_task.response["choice"]  # "a" | "b" | "equal"
confidence = completed_task.response["agreement_score"]
```

**Scale's RLHF workflow for OpenAI/Anthropic:**
1. Model generates response pairs on prompt dataset
2. Scale annotators compare pairs (which is better?)
3. Outputs: preference dataset `{prompt, chosen, rejected}`
4. Customer trains reward model on preference data
5. PPO/DPO uses reward model to fine-tune policy

### 3. LLM Evaluation and SEAL

SEAL (Scale Evaluation and Leaderboard) is Scale's LLM benchmark platform:

```python
# Scale Evaluation API
from scale_evals import EvalClient

client = EvalClient(api_key=SCALE_API_KEY)

# Run automated evaluation pipeline
eval_run = client.run_evaluation(
    model_endpoint="https://api.openai.com/v1/chat/completions",
    model_config={"model": "gpt-4o", "temperature": 0},
    benchmark="scale-seal-safety-v2",
    num_samples=500,
)

# SEAL safety benchmark tasks:
# - Jailbreak resistance: can it be manipulated to produce harmful content?
# - Instruction following fidelity: does it follow constraints in the prompt?
# - Factual accuracy: on knowledge-intensive questions
# - Refusal calibration: does it refuse when it should, answer when it should?

results = client.get_eval_results(eval_run.run_id)
for metric, score in results.metrics.items():
    print(f"{metric}: {score:.3f}")
```

**Key evaluation concepts Scale works with:**
- **Human eval vs automated eval:** LLM-as-judge is fast/cheap but has biases (longer = better, sycophancy). Human eval is gold standard.
- **Adversarial red-teaming:** Hired red teamers try to break the model via jailbreaks, prompt injection, social engineering
- **MMLU/HellaSwag/TruthfulQA:** Standard benchmarks often gamed; Scale builds harder, more realistic evals
- **MT-Bench:** Conversational multi-turn evaluation with GPT-4 as judge

### 4. Self-Driving Data Pipeline (Government + Auto)

Scale's government and autonomous vehicle work involves complex sensor data annotation:

```
Autonomous vehicle annotation:
  Input: LiDAR point cloud + camera images (synchronized)
  Task: 3D bounding box annotation for each detected object
  
LiDAR point cloud annotation:
  - Each point = (x, y, z, intensity, timestamp)
  - Annotator draws 3D bounding box in point cloud viewer
  - Camera image used for confirmation (ambiguous cases)
  
Scale's quality pipeline for 3D annotation:
  1. Worker draws box in LiDAR view
  2. Automated consistency check: box must be consistent across frames
  3. Lidar-camera projection check: box must align with camera image
  4. Senior review queue for low-confidence annotations

Speed: ~2-3 minutes per 3D bounding box (manual)
Scale's AI-assisted annotation: pre-label with model → human verify
  → 10× faster annotation with same quality
```

### 5. Active Learning and Data Curation

Scale helps customers figure out what data to label next:

```python
# Active learning pipeline: label the most valuable data first
import numpy as np
from sklearn.cluster import KMeans

def select_most_valuable_samples(
    unlabeled_embeddings: np.ndarray,
    labeled_embeddings: np.ndarray,
    model_uncertainties: np.ndarray,
    n_select: int,
    diversity_weight: float = 0.5,
) -> np.ndarray:
    """
    Hybrid active learning: uncertainty + diversity sampling.
    
    Uncertainty: pick samples the model is least confident about
    Diversity: pick samples that cover underrepresented regions
    """
    # Normalize scores to [0, 1]
    uncertainty_scores = (model_uncertainties - model_uncertainties.min()) / \
                         (model_uncertainties.max() - model_uncertainties.min() + 1e-8)
    
    # Diversity: distance from nearest labeled example
    from sklearn.metrics.pairwise import euclidean_distances
    distances = euclidean_distances(unlabeled_embeddings, labeled_embeddings)
    diversity_scores = distances.min(axis=1)
    diversity_scores = (diversity_scores - diversity_scores.min()) / \
                       (diversity_scores.max() - diversity_scores.min() + 1e-8)
    
    # Combined score
    combined = (1 - diversity_weight) * uncertainty_scores + diversity_weight * diversity_scores
    
    return np.argsort(combined)[-n_select:]  # top-n most valuable
```

---

## Key Questions

**Data / Quality:**
- "Design a human-in-the-loop annotation pipeline for RLHF preference collection"
- "How would you measure inter-annotator agreement? What do you do when annotators disagree?"
- "How do you detect and handle low-quality annotators at scale?"
- "Design an active learning system to select the most valuable data to label"

**Evaluation:**
- "How do you evaluate whether an LLM is 'safe'? What are the failure modes of automated safety evals?"
- "What is LLM-as-judge? What are its biases and how do you correct for them?"
- "Design a benchmark for evaluating instruction following that can't be gamed by training on it"

**ML Systems:**
- "How would you build a data pipeline to process and store 10M preference pairs for RLHF training?"
- "Design a system to detect and deduplicate near-duplicate annotations"

---

## 7-Day Learning Path

| Day | Focus |
|---|---|
| 1 | RLHF: preference collection, reward model training, PPO + DPO |
| 2 | Human evaluation: IAA, Cohen's kappa, annotation quality metrics |
| 3 | LLM evaluation: benchmarks (MMLU, MT-Bench), LLM-as-judge, red-teaming |
| 4 | Active learning: uncertainty sampling, diversity sampling, query strategies |
| 5 | Data quality: deduplication (MinHash), calibration, bias in labeled data |
| 6 | System design: RLHF data pipeline, annotation platform |
| 7 | Autonomous vehicle perception: 3D object detection, sensor fusion basics |
