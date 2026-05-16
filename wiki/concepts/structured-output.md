---
title: "Structured Output Generation"
aliases: ["JSON mode", "constrained decoding", "Instructor", "Outlines", "Guidance"]
tags: [prompting, structured-output, pydantic]
related:
- "[[function-calling]]"
- "[[prompt-engineering]]"
- "[[rag-systems]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Structured Output Generation

## TL;DR
Guaranteeing LLM output conforms to a schema — via constrained decoding or schema-aware prompting.

## Intuition
LLMs produce free text; applications need structured data. Two approaches: (1) schema-aware prompting (Instructor library — prompt with Pydantic model, retry on validation failure); (2) constrained decoding (Outlines, Guidance — modify the sampling distribution at each token step to only allow tokens valid under the current schema state). Constrained decoding is guaranteed to produce valid output; schema-aware prompting may need retries but works with any API.

## Technical Detail
<!-- to be filled -->

## Variants & Extensions
<!-- to be filled -->

## Tradeoffs
| Advantage | Disadvantage |
|---|---|
| ... | ... |

## Practical Applications
- Common use cases and when to apply
- Common follow-up questions
- Gotchas / misconceptions to avoid

## Connections
- [[function-calling]] — Function calling is the API-level form of structured output
- [[prompt-engineering]] — Output format specification is a key prompt engineering technique

## Sources
<!-- Add raw/ source paths after ingestion -->
