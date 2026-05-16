---
title: "Graph Algorithms"
aliases: ["BFS", "DFS", "Dijkstra", "topological sort", "union-find", "graphs"]
tags: [coding, algorithms, graphs]
related:
- "[[dynamic-programming]]"
- "[[heap-patterns]]"
sources: []
relevance: high
last_updated: 2026-05-16
status: stub
---

# Graph Algorithms

## TL;DR
Traversal, shortest path, and connectivity algorithms on graph-structured data.

## Intuition
Core algorithms by category: Traversal — BFS (shortest path in unweighted graph, level-order), DFS (cycle detection, topological sort, connected components). Shortest path — Dijkstra (non-negative weights, greedy + min-heap, O((V+E) log V)), Bellman-Ford (negative weights, DP, O(VE)), Floyd-Warshall (all-pairs, O(V³)). Connectivity — Union-Find (disjoint sets, near-O(1) with path compression + union by rank). Topological sort — Kahn's algorithm (BFS-based) or DFS post-order. Minimum spanning tree — Kruskal (sort edges + Union-Find) or Prim (greedy + min-heap).

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
- [[dynamic-programming]] — Many graph problems (shortest path, DAG DP) use DP
- [[heap-patterns]] — Dijkstra and Prim require a min-heap for efficient priority extraction

## Sources
<!-- Add raw/ source paths after ingestion -->
