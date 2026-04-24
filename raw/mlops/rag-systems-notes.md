# RAG: Retrieval-Augmented Generation — Comprehensive Notes

**Primary paper:** "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
**Authors:** Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, et al. (Facebook AI Research)
**Published:** 2020-05-22
**arXiv ID:** 2005.11401
**Venue:** NeurIPS 2020

Also covers: Gao et al. 2024 "Modular RAG" survey; RAGAS framework

---

## Original RAG Paper (Lewis et al. 2020)

### Problem

Knowledge-intensive tasks (open-domain QA, fact verification, knowledge-grounded generation) require LMs to access specific facts. Options:
1. Store all knowledge in LM parameters (expensive, can't update)
2. Retrieve relevant documents at query time (flexible, updatable)

### Architecture

```
RAG-Sequence: P(y|x) = Σ_z P_η(z|x) · P_θ(y|x,z)^(1/|y|)
RAG-Token:    P(y_i|x,y_{1:i-1}) = Σ_z P_η(z|x,y_{1:i-1}) · P_θ(y_i|x,z,y_{1:i-1})
```

- `P_η(z|x)`: retriever (Dense Passage Retrieval — DPR) returns top-K documents z
- `P_θ(y|x,z)`: generator (BART) conditions on both query x and retrieved documents z
- Marginalization over K retrieved documents

**DPR (Dense Passage Retrieval):**
- Bi-encoder: separate BERT for question, separate BERT for passage
- Trained on NQ, TriviaQA with in-batch negatives
- FAISS index over Wikipedia (21M 100-word passages)

### Key Result (2020)

RAG-Sequence outperforms T5-11B on NaturalQuestions (44.5% vs 36.6% exact match) despite using a 400M BART generator — retrieval beats scale for knowledge tasks.

---

## Evolution: Naive → Advanced → Modular RAG

### Naive RAG (2020–2022)

**Pipeline:**
```
Query → Embed (e.g., sentence-transformers) → FAISS top-K → prepend to prompt → generate
```

**Components:**
- Fixed-size chunking (512 tokens, 50% overlap)
- Single dense retriever
- Top-K documents concatenated into prompt
- No reranking, no query processing

**Failure modes:**
1. Poor chunking splits semantically connected content
2. Dense retrieval fails on exact-match queries (product codes, names)
3. No quality control on retrieved context
4. "Lost in the middle": LLMs ignore context in the middle of long inputs

### Advanced RAG (2022–2023)

**Pre-retrieval improvements:**
- **Query rewriting**: rephrase vague queries before retrieval
- **HyDE**: generate hypothetical answer, embed that
- **Query expansion**: multi-query retrieval (generate N variations, merge results)
- **Step-back prompting**: abstract the specific question to a general principle, retrieve on that

**Retrieval improvements:**
- **Hybrid search**: dense + BM25 with RRF fusion
- **Reranking**: cross-encoder after initial retrieval
- **Metadata filtering**: filter by date, source, category before ANN search
- **Ensemble retrieval**: multiple embedding models, fuse results

**Post-retrieval improvements:**
- **Context compression**: summarize retrieved chunks to fit more in context
- **LLM re-ranking**: use GPT-4 to judge relevance before generating
- **Selective context**: only include chunks above a relevance threshold

### Modular RAG (2023–2025)

**Self-RAG** (Asai et al. 2023):
- Model decides WHEN to retrieve (not always)
- Generates special tokens: `[Retrieve]`, `[Relevant]`, `[Supported]`, `[No support]`
- Critique tokens allow the model to self-evaluate its own outputs
- Result: better quality + efficiency (fewer unnecessary retrievals)

**Corrective RAG (CRAG)** (Yan et al. 2024):
- After retrieval, evaluate document relevance
- If all documents are irrelevant (confidence score < threshold): web search fallback
- If ambiguous: query reformulation + re-retrieve

**Agentic RAG** (2024):
- LangGraph/ReAct agent that decides to retrieve, retrieve more, or synthesize
- Multi-step: "Retrieve about X → read results → decide I need more about Y → retrieve Y → answer"
- Better for complex multi-hop questions

---

## RAGAS: RAG Evaluation Framework

**Paper:** "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (Es et al. 2023)

### Core Metrics

**Faithfulness:**
- Is every claim in the answer supported by the retrieved context?
- Method: extract claims from answer → for each claim, ask LLM "is this supported by the context?"
- Range: 0–1 (1 = fully grounded)
- Critical: hallucination = faithfulness < 1

**Answer Relevancy:**
- Does the answer actually address the question?
- Method: generate N questions from the answer → embed → compare to original question
- High relevancy = the answer is about the same thing as the question

**Context Precision:**
- What fraction of retrieved chunks are actually relevant to the question?
- Method: for each chunk, ask LLM "is this relevant to the question?" → mean
- Low precision = noisy retrieval (irrelevant chunks confuse generation)

**Context Recall:**
- What fraction of the ground-truth answer's facts appear in the retrieved context?
- Method: decompose ground-truth answer into statements → check if each is in context
- Low recall = retrieval missed important information

### RAGAS Evaluation Matrix

| Score | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| Ideal | 1.0 | 1.0 | 1.0 | 1.0 |
| Production minimum | > 0.85 | > 0.80 | > 0.70 | > 0.70 |

### Interpretations

| Pattern | Likely Cause |
|---|---|
| Low faithfulness | LLM hallucinating despite having context |
| Low answer relevancy | Prompt is wrong; LLM is off-topic |
| Low context precision | Too many irrelevant chunks in retrieval |
| Low context recall | Right document not retrieved (chunking/embedding failure) |

---

## Production RAG Patterns

### Chunking Strategies in Detail

**Fixed-size chunking:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_text(document)
```
- Simple, fast
- Breaks at character count, not semantics
- 50-token overlap prevents losing context at boundaries

**Semantic chunking:**
```python
# Embed sentences, find semantic breakpoints (large embedding distance jumps)
# Group sentences into chunks at breakpoints
```
- Better semantic boundaries
- Slower (requires embedding every sentence)
- Better quality for RAG

**Hierarchical chunking (parent-child):**
```python
# Parent: paragraph level (512 tokens) → stored for full-context retrieval
# Child: sentence level (128 tokens) → indexed for precise retrieval
# At query time: retrieve child chunks → look up parent → return parent to LLM
```
- Best quality — precision of small chunks + completeness of large context

### Multi-Vector Retrieval

Index multiple representations of the same document:
- **Summary**: summarize each chunk → embed summary → retrieve by summary, return original chunk
- **Hypothetical questions**: generate questions this chunk could answer → retrieve by question similarity
- **Both chunk and parent**: retrieve granular, return broad

---

## Common -Relevant RAG Details

### "Lost in the Middle" (Liu et al. 2023)

LLMs perform best when relevant information is at the **beginning or end** of the context window, not the middle. With 10 retrieved chunks, the 5th chunk is most likely to be ignored.

**Mitigation:**
1. Use fewer, higher-quality chunks (reranker → top-3, not top-10)
2. Put most relevant chunk first
3. Use models specifically tested for long-context recall (Needle-in-a-Haystack)

### Embedding Model Selection

| Model | Dimensions | English MTEB | Multilingual | Speed |
|---|---|---|---|---|
| text-embedding-3-small | 512–1536 | 62.3 | No | Fast |
| text-embedding-3-large | 256–3072 | 64.6 | No | Slow |
| BGE-large-en | 1024 | 63.6 | No | Medium |
| BGE-M3 | 1024 | 62.8 | Yes | Medium |
| E5-mistral-7b | 4096 | 66.6 | Partial | Very slow |

**Selection rule:**
- Internal English-only: BGE-large-en (open source, free, strong)
- Multilingual: BGE-M3
- Managed, highest quality English: text-embedding-3-large
- Best quality (research): E5-mistral-7b (expensive to run)

---

## Common  Questions

- "Walk me through the RAG pipeline from document ingestion to answer generation."
- "What chunking strategy would you use for technical documentation?"
- "What is RAGAS and what metrics does it measure?"
- "How does HyDE improve retrieval? What are its failure modes?"
- "What is 'lost in the middle' and how do you mitigate it?"
- "Design a RAG system for a legal document corpus (high accuracy required)."
- "When would you use agentic RAG vs. naive RAG?"
