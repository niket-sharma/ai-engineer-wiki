---
title: "Acronyms Cheatsheet"
aliases: ["acronym list", "abbreviations", "terminology"]
tags: [cheatsheet, reference]
sources: ["training-knowledge"]
relevance: medium
last_updated: 2026-04-22
status: current
---

# Acronyms Cheatsheet

---

## Transformer Architecture

| Acronym | Full Name | One-line |
|---|---|---|
| MHA | Multi-Head Attention | Parallel attention heads over full key/value |
| MQA | Multi-Query Attention | Single shared K/V head, multiple Q heads |
| GQA | Grouped-Query Attention | g shared K/V groups (Llama 3, Mistral) |
| KV | Key-Value (cache) | Cached K and V tensors for inference |
| PE | Positional Encoding | Injects sequence order into attention |
| RoPE | Rotary Position Embedding | Rotates Q/K by position angle; relative PE |
| ALiBi | Attention with Linear Biases | Linear decay bias on attention logits |
| YaRN | Yet Another RoPE extensioN | Extends RoPE to longer contexts |
| FFN | Feed-Forward Network | Two-layer MLP inside transformer layer |
| MLP | Multi-Layer Perceptron | Same as FFN in transformer context |
| LN | Layer Normalization | Normalize over feature dim per token |
| BPE | Byte Pair Encoding | Subword tokenization (GPT-2, LLaMA) |
| SwiGLU | Swish Gated Linear Unit | Activation in modern FFN (Llama, PaLM) |

---

## Fine-Tuning & Alignment

| Acronym | Full Name | One-line |
|---|---|---|
| SFT | Supervised Fine-Tuning | Stage 1 of RLHF: train on demonstrations |
| RLHF | Reinforcement Learning from Human Feedback | Human-preference-based alignment (InstructGPT) |
| RM | Reward Model | Predicts human preference score |
| RLAIF | RL from AI Feedback | Use AI critique instead of human labels |
| PPO | Proximal Policy Optimization | RL algorithm used in RLHF stage 3 |
| DPO | Direct Preference Optimization | Offline alternative to RLHF+PPO |
| GRPO | Group Relative Policy Optimization | PPO variant without value model (DeepSeek-R1) |
| KTO | Kahneman-Tversky Optimization | Single-response preference optimization |
| ORPO | Odds Ratio Preference Optimization | No reference model needed |
| IPO | Identity Preference Optimization | DPO variant with squared loss |
| PEFT | Parameter-Efficient Fine-Tuning | Umbrella term for LoRA, prefix, etc. |
| LoRA | Low-Rank Adaptation | Add low-rank ΔW = BA to frozen weights |
| QLoRA | Quantized LoRA | LoRA + 4-bit NF4 base model |
| DoRA | Weight-Decomposed Low-Rank Adaptation | Decompose weight into magnitude + direction |
| NF4 | Normal Float 4 | 4-bit datatype optimized for Gaussian weights |
| GAE | Generalized Advantage Estimation | Advantage estimation method for PPO |
| KL | KL Divergence | Measure of distribution difference; used as penalty |

---

## RAG & Retrieval

| Acronym | Full Name | One-line |
|---|---|---|
| RAG | Retrieval-Augmented Generation | Retrieve docs at query time, insert into context |
| ANN | Approximate Nearest Neighbor | Fast but approximate vector search |
| HNSW | Hierarchical Navigable Small World | Graph-based ANN index (default in most vector DBs) |
| IVF | Inverted File Index | Cluster-based ANN (used in FAISS) |
| PQ | Product Quantization | Compress vectors for memory efficiency |
| RRF | Reciprocal Rank Fusion | Rank-based fusion for hybrid retrieval |
| BM25 | Best Match 25 | TF-IDF variant, standard sparse retrieval |
| HyDE | Hypothetical Document Embedding | Embed generated hypothetical answer for retrieval |
| ColBERT | Contextualized Late Interaction BERT | Token-level late interaction retrieval |

---

## MLOps & Serving

| Acronym | Full Name | One-line |
|---|---|---|
| TTFT | Time to First Token | Latency to first streamed token |
| TPOT | Time Per Output Token | Time between generated tokens |
| TPS | Tokens Per Second | Throughput metric for LLM serving |
| HBM | High Bandwidth Memory | GPU DRAM (A100: 80GB HBM3) |
| SRAM | Static RAM | On-chip fast memory (FlashAttention tiles here) |
| TP | Tensor Parallelism | Split model weights across GPUs |
| PP | Pipeline Parallelism | Different layers on different GPUs |
| TRT | TensorRT | NVIDIA inference optimization compiler |
| ONNX | Open Neural Network Exchange | Model format for cross-framework deployment |
| PTQ | Post-Training Quantization | Quantize after training (GPTQ, AWQ) |
| QAT | Quantization-Aware Training | Simulate quant during training |
| FP16 | 16-bit floating point (float16) | Standard inference dtype |
| BF16 | BFloat16 | Better range than FP16, used for training |
| INT8 | 8-bit integer | 2× memory savings, minimal quality loss |
| INT4 | 4-bit integer | 4× savings, used in GPTQ/AWQ |
| GPTQ | Generative Pretrained Transformer Quantization | Layer-wise weight-only quantization |
| AWQ | Activation-aware Weight Quantization | Better INT4 method (protects salient weights) |

---

## Agents & Orchestration

| Acronym | Full Name | One-line |
|---|---|---|
| MCP | Model Context Protocol | Open standard for LLM tool/resource connections |
| ReAct | Reason + Act | Agent prompting pattern (Thought/Action/Observation) |
| CoT | Chain of Thought | Multi-step reasoning in LLM prompts |
| TOT | Tree of Thoughts | Tree-search reasoning for complex problems |
| HITL | Human In The Loop | Human approval step in agentic workflows |

---

## Evaluation

| Acronym | Full Name | One-line |
|---|---|---|
| BLEU | Bilingual Evaluation Understudy | N-gram precision for translation/summarization |
| ROUGE | Recall-Oriented Understudy for Gisting Evaluation | N-gram recall for summarization |
| RAGAS | Retrieval-Augmented Generation Assessment | RAG eval framework (faithfulness, relevancy, recall) |
| PPL | Perplexity | Exp(-avg log likelihood) — lower is better |
| AUC | Area Under the Curve | ROC-AUC for classifiers |
| PSI | Population Stability Index | Feature/prediction drift metric |
| KS | Kolmogorov-Smirnov (test) | Statistical test for distribution shift |
