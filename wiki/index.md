# AI Engineer Wiki — Index

> Last audited: 2026-05-16 | Pages: 82 | Stubs: 53

## Transformer Architecture
- [[attention-mechanism]] — Attention lets each token in a sequence selectively gather information from every other to
- [[flash-attention]] — Flash Attention is an IO-aware implementation of standard scaled dot-product attention tha
- [[kv-cache]] — KV cache stores the Key and Value projections for previously generated tokens so they don'
- [[positional-encoding]] — Attention is permutation-invariant — shuffle the input tokens and you get the same output.
- [[transformer-architecture]] — The transformer is a sequence model built entirely on attention (no recurrence, no convolu

## Fine-tuning & Alignment
- [[dpo]] — DPO reformulates the RLHF objective as a supervised learning problem — no reward model, no
- [[grpo]] — GRPO (Group Relative Policy Optimization) is the RL algorithm used to train DeepSeek-R1 re
- [[lora-qlora]] — LoRA fine-tunes a pretrained model by adding low-rank decomposition matrices to frozen wei
- [[ppo]] — PPO (Proximal Policy Optimization) is the on-policy reinforcement learning algorithm used 
- [[rlhf]] — RLHF is the three-stage pipeline that turns a pretrained language model into a helpful, ha

## Inference & Serving
- [[continuous-batching]] — Process new requests at each decode step rather than waiting for the whole batch to finish
- [[paged-attention]] — Manages KV cache in fixed-size blocks (like OS virtual memory paging) to eliminate fragmen
- [[quantization]] — Reducing model weight/activation precision to shrink memory and increase throughput at the
- [[speculative-decoding]] — Use a cheap draft model to propose multiple tokens, then verify in parallel with the large
- [[tensor-parallelism]] — Split individual layers across GPUs (column/row partitioning) to fit models that don't fit

## RAG & Retrieval
- [[bm25-and-sparse-retrieval]] — Keyword-based retrieval using term frequency and inverse document frequency — fast, interp
- [[chunking-strategies]] — How to split documents into retrieval units — one of the highest-leverage decisions in RAG
- [[embedding-models]] — Models that encode text into dense vectors for semantic search — the retrieval backbone of
- [[hybrid-search]] — Combining dense (semantic) and sparse (keyword) retrieval to get the best of both — better
- [[query-rewriting]] — Transforming user queries before retrieval to improve recall — from HyDE to decomposition 
- [[rag-systems]] — RAG grounds LLM responses in retrieved documents, reducing hallucination and enabling know
- [[reranking]] — Reranking is a second-stage retrieval step where a more powerful but slower model re-score
- [[vector-databases]] — Vector databases store high-dimensional embeddings and enable approximate nearest-neighbor

## Agents & Orchestration
- [[agent-architectures]] — Patterns for structuring LLM reasoning + action loops — from simple ReAct to multi-step pl
- [[agent-memory]] — How agents persist and access information across steps and sessions — from in-context to e
- [[agentic-rag]] — RAG systems where an agent decides when, what, and how to retrieve — rather than a fixed p
- [[langgraph-agents]] — LLM agents combine a language model with tools, memory, and a control loop — the LLM decid
- [[mcp-protocol]] — Model Context Protocol (MCP) is an open standard (Anthropic, 2024) for connecting LLMs to 
- [[multi-agent-systems]] — Systems where multiple LLM agents collaborate — each specialized, with a coordinator manag
- [[tool-use]] — Giving LLMs access to external functions (search, code execution, APIs) to extend their ca

## Evaluation
- [[agent-evaluation]] — Measuring agent performance via task success, trajectory quality, and tool-use correctness
- [[embedding-evaluation]] — Measuring embedding model quality via retrieval benchmarks (MTEB) and ranking metrics (NDC
- [[llm-evaluation]] — Methods for measuring LLM quality at scale, from task benchmarks to LLM-as-judge.
- [[offline-vs-online-eval]] — Offline eval is fast and cheap; online eval (A/B tests) is the ground truth but slow and r
- [[rag-evaluation]] — Frameworks and metrics for evaluating retrieval-augmented generation pipelines end-to-end.

## Production AI Systems
- [[cost-optimization]] — Techniques to reduce LLM API and inference costs: caching, routing, batching, distillation
- [[model-routing]] — Dynamically selecting which model to use for each request based on cost, latency, and comp
- [[observability-llm]] — Tracking token usage, latency, cost, and quality across LLM calls — enabling debugging and
- [[prompt-injection]] — Attacks that hijack LLM behavior by embedding adversarial instructions in user input or re
- [[safety-and-guardrails]] — Input and output filtering layers that prevent harmful, off-topic, or policy-violating LLM

## Classic ML & Statistics
- [[ab-testing]] — Randomized experiments to measure the causal effect of changes (model updates, UI changes)
- [[bayesian-methods]] — Probabilistic reasoning that incorporates prior beliefs and updates them with data — princ
- [[calibration]] — Making model-predicted probabilities match empirical frequencies — essential for risk-sens
- [[causal-inference]] — Methods for estimating causal effects when randomized experiments are infeasible — DiD, IV
- [[feature-engineering]] — Transforming raw data into model-ready features — often the highest-leverage activity in a
- [[gradient-boosting]] — Sequentially fitting decision trees to residuals — the dominant algorithm for tabular data
- [[imbalanced-classification]] — Techniques for learning from datasets where one class is far rarer than another (e.g., fra
- [[interpretability]] — Methods for understanding why a model makes a specific prediction — from global feature im
- [[model-monitoring]] — Detecting when a deployed model's inputs or performance change over time, requiring retrai
- [[time-series]] — Methods for modeling and forecasting sequential temporal data — from ARIMA to gradient boo

## Prompting & Generation
- [[chain-of-thought]] — Prompting LLMs to reason step-by-step before answering — substantially improves performanc
- [[function-calling]] — Structured output format where the LLM produces a JSON function call that a host applicati
- [[prompt-engineering]] — Designing inputs to elicit desired LLM behavior — from few-shot examples to role prompting
- [[structured-output]] — Guaranteeing LLM output conforms to a schema — via constrained decoding or schema-aware pr

## Training Mechanics
- [[distributed-training]] — Splitting training computation across multiple GPUs — via data parallelism (DDP), model pa
- [[gradient-checkpointing]] — Trade compute for memory by not storing all activations — recompute them during the backwa
- [[learning-rate-schedules]] — How the learning rate changes during training — warmup + cosine decay is the de facto stan
- [[mixed-precision-training]] — Training with lower-precision floats (FP16/BF16) for speed and memory savings, with FP32 m
- [[optimizers]] — Algorithms that update model weights from gradients — Adam/AdamW dominate, with memory-eff

## Coding & Algorithms
- [[dynamic-programming]] — Breaking problems into overlapping subproblems and caching results — eliminates exponentia
- [[graph-algorithms]] — Traversal, shortest path, and connectivity algorithms on graph-structured data.
- [[heap-patterns]] — Using heaps for top-K, streaming medians, k-way merge, and scheduling problems — O(log n) 
- [[pytorch-internals]] — How PyTorch's autograd engine works and how to write custom layers, loss functions, and tr
- [[two-pointers-sliding-window]] — Linear-time techniques for array/string problems using one or two index pointers moving in

## System Design
- [[embedding-service-design]] — Design of a scalable embedding service supporting batch ingestion, real-time query encodin
- [[feature-store]]
- [[llm-serving-infra]] — LLM serving is fundamentally a memory-bandwidth-bound problem at small batch sizes and a c
- [[ml-feature-pipeline-design]] — Design of pipelines that compute, serve, and monitor ML features across batch and streamin
- [[ml-platform]]
- [[rag-pipeline-design]] — Production RAG consists of two pipelines: an offline ingestion pipeline (documents → vecto
- [[recommendation-system-design]] — End-to-end design of a scalable recommender: two-tower retrieval, candidate generation, an
- [[search-system-design]] — End-to-end design of a production search system: query understanding, multi-stage retrieva

## Q&A
- [[agents-qa]]
- [[behavioral-qa]]
- [[mlops-qa]]
- [[rag-qa]]
- [[rl-qa]]
- [[system-design-qa]]
- [[transformers-qa]]

## Cheatsheets
- [[acronyms]]
- [[complexity-guide]]
- [[math-and-notation]]

---

## Obsidian Dataview Dashboards

### Stubs to fill
```dataview
TABLE relevance, last_updated
FROM "wiki/concepts"
WHERE status = "stub"
SORT relevance DESC
```

### High-priority pages not updated recently
```dataview
TABLE last_updated
FROM "wiki"
WHERE relevance = "high" AND last_updated < date(today) - dur(30 days)
SORT last_updated ASC
```
