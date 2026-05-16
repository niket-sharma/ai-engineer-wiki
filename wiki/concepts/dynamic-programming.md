---
title: "Dynamic Programming"
aliases: ["DP", "memoization", "tabulation", "optimal substructure"]
tags: [coding, algorithms, dynamic-programming]
related:
- "[[graph-algorithms]]"
- "[[heap-patterns]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Dynamic Programming

## TL;DR
Breaking problems into overlapping subproblems and caching results — eliminates exponential recomputation.

## Intuition
DP applies when a problem has optimal substructure (optimal solution composed of optimal subsolutions) and overlapping subproblems (same sub-problems solved multiple times). Two implementations: top-down (memoization — recursive + cache) and bottom-up (tabulation — iterative, fill a table). Pattern recognition: if the problem asks for 'minimum/maximum/count of ways' and has a 'decision at each step', suspect DP. Common patterns: 1D DP (climbing stairs, house robber), 2D DP (longest common subsequence, edit distance), interval DP (matrix chain multiplication), knapsack.

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
- [[graph-algorithms]] — Shortest path (Bellman-Ford, Floyd-Warshall) uses DP on graphs
- [[heap-patterns]] — Dijkstra combines DP/greedy with a min-heap

## Sources
<!-- Add raw/ source paths after ingestion -->
