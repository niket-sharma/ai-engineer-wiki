---
title: "Heap & Priority Queue Patterns"
aliases: ["heap", "min-heap", "max-heap", "priority queue", "top-K", "k-way merge"]
tags: [coding, algorithms, heaps, priority-queue]
related:
- "[[graph-algorithms]]"
- "[[dynamic-programming]]"
sources: []
relevance: medium
last_updated: 2026-05-16
status: stub
---

# Heap & Priority Queue Patterns

## TL;DR
Using heaps for top-K, streaming medians, k-way merge, and scheduling problems — O(log n) push/pop.

## Intuition
A heap gives O(log n) insert and O(1) peek at min/max. Core patterns: Top-K elements — push all, maintain heap of size k (for top-K largest, use min-heap; pop when size > k, leaving k largest). K-way merge — seed with first element from each sorted list; pop min, push next from same list. Running median — two heaps: max-heap for lower half, min-heap for upper half; balance so sizes differ by at most 1. Scheduling — use min-heap keyed on deadline or end time.

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
- [[graph-algorithms]] — Dijkstra and Prim use a min-heap as their core data structure

## Sources
<!-- Add raw/ source paths after ingestion -->
