# HyDE: Hypothetical Document Embeddings

**Paper:** "Precise Zero-Shot Dense Retrieval without Relevance Labels"
**Authors:** Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan (CMU / University of Waterloo)
**Published:** 2022-12-20
**arXiv ID:** 2212.10496
**Venue:** ACL 2023

---

## The Problem: Dense Retrieval Requires Relevance Labels

Standard dense retrieval (DPR, bi-encoders) requires fine-tuning on (query, relevant_passage) pairs:
- NaturalQuestions training split: 79,168 annotated pairs
- MS MARCO: 502,939 annotated pairs

**Problem:** For a new domain (medical literature, legal docs, proprietary internal docs), you need thousands of annotated pairs to fine-tune the retriever. Annotation is expensive and slow.

**Goal:** Zero-shot dense retrieval that works on any domain without annotation.

---

## HyDE Core Idea

Instead of embedding the **query** and searching for similar documents:

1. **Generate a hypothetical document** that would answer the query (using an LLM)
2. **Embed the hypothetical document** (not the query)
3. **Search** for real documents similar to the hypothetical one

```
Query: "What causes inflation?"
   ↓  LLM generates hypothetical answer
Hypothetical doc: "Inflation is caused by excess money supply relative to goods 
                  and services. When the government prints more money or the 
                  central bank lowers interest rates, more money chases the same 
                  goods, driving prices up. Demand-pull inflation occurs when..."
   ↓  embed hypothetical doc
Query embedding: [0.23, -0.14, 0.87, ...]  # embedding of the hypothetical doc
   ↓  ANN search
Retrieved docs: real documents about inflation causes
```

---

## Why This Works

**The distributional shift problem:** A query ("What causes inflation?") is grammatically and semantically different from an answer ("Inflation is caused by..."). The bi-encoder embeds both into the same space, but queries and passages have different distributions.

**HyDE's insight:** A hypothetical *answer* is in the same distribution as real *answers*. The embedding space geometry between (hypothetical answer, real answer) is better than (query, answer).

**Analogy:** "Find documents similar to a document like this" vs "Find documents that answer this question." The former is what bi-encoders are optimized for (symmetric similarity); the latter requires asymmetric understanding.

---

## HyDE vs. Alternatives

```
Standard dense: Query → embed → search
HyDE:           Query → LLM → HypDoc → embed → search
Query expansion: Query → expand → multiple embeddings → merge → search
```

**Difference from query expansion:**
- Query expansion generates *variations of the query*
- HyDE generates a *document in the answer space*

---

## Prompt for HyDE

```python
HYDE_PROMPT = """Please write a passage to answer the question
Question: {question}
Passage:"""

def hyde_retrieve(question: str, llm, retriever, n_docs: int = 5) -> list[str]:
    # Step 1: Generate hypothetical document
    hypothetical_doc = llm.invoke(HYDE_PROMPT.format(question=question))
    
    # Step 2: Embed and search using the hypothetical doc
    results = retriever.invoke(hypothetical_doc)  # not the original question!
    
    return results[:n_docs]
```

**Optional enhancement:** Generate multiple hypothetical documents, average their embeddings:
```python
def hyde_multi(question: str, llm, embedder, index, n_hyp=5, k=10):
    hypotheticals = [llm.invoke(HYDE_PROMPT.format(question=question)) 
                     for _ in range(n_hyp)]
    embeddings = [embedder.embed(h) for h in hypotheticals]
    avg_embedding = np.mean(embeddings, axis=0)
    return index.search(avg_embedding, k=k)
```

---

## Experimental Results

### BEIR Benchmark (18 retrieval tasks, zero-shot)

| Method | Average nDCG@10 |
|---|---|
| BM25 (sparse baseline) | 43.0 |
| DPR (in-domain fine-tuned) | 37.2 |
| Contriever (unsupervised dense) | 41.9 |
| **HyDE (GPT-3 + Contriever)** | **46.2** |
| Supervised models (with labels) | ~50+ |

HyDE beats DPR (which was fine-tuned on NQ) in zero-shot settings. Nearly competitive with supervised retrieval methods.

### WebQuestions (Open-Domain QA)

| Method | Top-20 Recall |
|---|---|
| DPR | 73.8% |
| Contriever | 75.1% |
| **HyDE** | **79.0%** |

HyDE improves recall significantly over both DPR and Contriever.

---

## Failure Modes

1. **Hallucinated hypothetical docs:** If the LLM generates factually wrong hypotheticals, they may retrieve confidently wrong documents.
   - Example: "What is the GDP of Atlantis?" → LLM might generate plausible-sounding fake GDP stats → retrieves documents about fictional economies.

2. **Domain mismatch:** If the LLM doesn't know the domain well enough to generate plausible hypotheticals, HyDE degrades.
   - Example: Highly specialized medical literature where GPT-3 lacks domain knowledge.

3. **Latency:** Adds an LLM call before retrieval (~200–500ms) compared to direct embedding (~10ms).

4. **Not always better than query expansion:** For factual lookup queries ("What year was X founded?"), direct embedding works fine.

---

## When to Use HyDE

**Use HyDE when:**
- No annotated retrieval pairs available (zero-shot domain)
- Queries are short and abstract; documents are long and detailed
- The LLM has reasonable domain knowledge to generate plausible answers
- Latency budget allows for an extra LLM call

**Don't use HyDE when:**
- Latency-sensitive (adds ~200–500ms per query)
- Domain is highly technical and the LLM will hallucinate hypotheticals
- You have labeled data → fine-tune the retriever instead
- Queries are keyword-like ("invoice 2024 Q3") → BM25 or hybrid is better

---

## HyDE in Practice

**LangChain HyDE:**
```python
from langchain.chains import HypotheticalDocumentEmbedder

# Wrap your embedder with HyDE
hyde_embedder = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=embeddings,
    custom_prompt=HYDE_PROMPT  # optional
)

# Use like any embedder
vectorstore = FAISS.from_documents(docs, hyde_embedder)
retriever = vectorstore.as_retriever()
```

---

## Common  Questions

- "What is HyDE and why does it outperform standard dense retrieval in zero-shot settings?"
- "What is the distributional shift problem that HyDE solves?"
- "What are the failure modes of HyDE?"
- "When would you choose HyDE over query expansion?"
- "How does HyDE affect RAG latency?"
- "How would you evaluate whether HyDE improves your RAG pipeline?"
