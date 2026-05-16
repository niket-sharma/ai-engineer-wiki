---
title: "Chain-of-Thought Prompting"
aliases: ["CoT", "self-consistency", "tree of thoughts", "ToT"]
tags: [prompting, reasoning, few-shot]
related:
- "[[prompt-engineering]]"
- "[[function-calling]]"
- "[[llm-evaluation]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Chain-of-Thought Prompting

## TL;DR
Prompting LLMs to reason step-by-step before answering — substantially improves performance on reasoning tasks.

## Intuition
Chain-of-thought works because reasoning tasks require intermediate computation. By prompting 'Let's think step by step' or providing few-shot examples with reasoning steps, you force the model to allocate additional forward passes to intermediate reasoning. Self-consistency samples multiple reasoning chains and takes the majority vote — averaging over diverse paths reduces errors from any single chain.

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
- [[prompt-engineering]] — CoT is a core prompting technique
- [[llm-evaluation]] — CoT improves performance on reasoning benchmarks (GSM8K, MATH)
- [[function-calling]] — CoT can be combined with tool use for agentic reasoning

## Sources
<!-- Add raw/ source paths after ingestion -->
