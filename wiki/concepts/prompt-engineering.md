---
title: "Prompt Engineering"
aliases: ["few-shot prompting", "instruction tuning", "system prompts"]
tags: [prompting, llm, few-shot]
related:
- "[[chain-of-thought]]"
- "[[structured-output]]"
- "[[function-calling]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Prompt Engineering

## TL;DR
Designing inputs to elicit desired LLM behavior — from few-shot examples to role prompting to output formatting.

## Intuition
Prompt engineering is the primary interface to LLM behavior before fine-tuning. Key techniques: few-shot examples (show don't tell); role prompting ('You are an expert...'); explicit output format specification; instruction ordering (most important instruction first and last); negative constraints ('Do not...'); and temperature/top-p for creativity vs. determinism. Prompts are fragile — small changes can have outsized effects.

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
- [[chain-of-thought]] — CoT is a specialized prompting technique for reasoning
- [[structured-output]] — Output format prompting combined with constrained decoding
- [[function-calling]] — Tool schemas are a form of structured prompting

## Sources
<!-- Add raw/ source paths after ingestion -->
