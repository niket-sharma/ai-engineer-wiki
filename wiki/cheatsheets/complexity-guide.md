---
title: "Algorithm Complexity Guide"
aliases: ["complexity cheatsheet", "Big O cheatsheet", "time complexity"]
tags: [cheatsheet, algorithms, coding]
sources: ["training-knowledge"]
interview_relevance: medium
last_updated: 2026-04-22
status: current
---

# Algorithm Complexity Guide

---

## Big O Hierarchy (slowest to fastest)

```
O(n!) >> O(2^n) >> O(n³) >> O(n²) >> O(n log n) >> O(n) >> O(log n) >> O(1)
```

**Practical limits for interview problems (1 second budget):**
| Complexity | Max n |
|---|---|
| O(n!) | ≤ 11 |
| O(2^n) | ≤ 25 |
| O(n³) | ≤ 500 |
| O(n²) | ≤ 10,000 |
| O(n log n) | ≤ 1,000,000 |
| O(n) | ≤ 100,000,000 |
| O(log n) | ≤ 10^18 |

---

## Standard Data Structures

| Structure | Access | Search | Insert | Delete | Space |
|---|---|---|---|---|---|
| Array | O(1) | O(n) | O(n) | O(n) | O(n) |
| Dynamic Array | O(1) | O(n) | O(1) amort | O(n) | O(n) |
| Linked List | O(n) | O(n) | O(1) | O(1) | O(n) |
| Stack/Queue | O(1) top | O(n) | O(1) | O(1) | O(n) |
| Hash Map | O(1) avg | O(1) avg | O(1) avg | O(1) avg | O(n) |
| Binary Search Tree | O(log n) | O(log n) | O(log n) | O(log n) | O(n) |
| Balanced BST (AVL/RB) | O(log n) | O(log n) | O(log n) | O(log n) | O(n) |
| Heap | O(1) top | O(n) | O(log n) | O(log n) | O(n) |
| Trie | O(k) | O(k) | O(k) | O(k) | O(n·k) |

k = key/string length

---

## Sorting Algorithms

| Algorithm | Time (avg) | Time (worst) | Space | Stable |
|---|---|---|---|---|
| Merge sort | O(n log n) | O(n log n) | O(n) | Yes |
| Quick sort | O(n log n) | O(n²) | O(log n) | No |
| Heap sort | O(n log n) | O(n log n) | O(1) | No |
| Tim sort (Python) | O(n log n) | O(n log n) | O(n) | Yes |
| Counting sort | O(n + k) | O(n + k) | O(k) | Yes |
| Radix sort | O(nk) | O(nk) | O(n+k) | Yes |

Python's built-in `sort()` is Timsort. Use counting/radix when k is small (sorting integers in a range).

---

## Graph Algorithms

| Algorithm | Time | Space | Use case |
|---|---|---|---|
| BFS | O(V+E) | O(V) | Shortest path (unweighted), level order |
| DFS | O(V+E) | O(V) | Cycle detection, connected components, topological sort |
| Dijkstra | O((V+E) log V) | O(V) | Shortest path (non-negative weights) |
| Bellman-Ford | O(VE) | O(V) | Shortest path (negative weights) |
| Floyd-Warshall | O(V³) | O(V²) | All-pairs shortest path |
| Prim's MST | O(E log V) | O(V) | Minimum spanning tree |
| Kruskal's MST | O(E log E) | O(V) | MST with Union-Find |
| Topological sort | O(V+E) | O(V) | DAG scheduling |

---

## LeetCode Patterns → Algorithm

| Pattern signal | Algorithm | Complexity |
|---|---|---|
| "Shortest path in unweighted graph" | BFS | O(V+E) |
| "Find all combinations/subsets" | Backtracking | O(2^n) or O(n!) |
| "Optimal substructure" | Dynamic Programming | varies |
| "Sliding window on array/string" | Two pointers | O(n) |
| "Top K elements" | Heap | O(n log k) |
| "Search in sorted array" | Binary search | O(log n) |
| "Interval scheduling/merging" | Sort + greedy | O(n log n) |
| "String matching" | KMP or sliding window | O(n+m) |
| "Tree traversal" | DFS/BFS recursive | O(n) |
| "Number of islands / connected components" | Union-Find or DFS | O(n·α(n)) |

---

## ML Operation Complexities

| Operation | Time | Space | Notes |
|---|---|---|---|
| Attention | O(n²d) | O(n²) | n = seq len, d = dim |
| Flash Attention | O(n²d) | O(n) | Same FLOPs, less memory |
| Matrix multiply (A×B) m×n × n×k | O(mnk) | O(mk) | Core of all linear layers |
| Embedding lookup | O(1) | O(vocab×d) | Just an array index |
| Softmax (length n) | O(n) | O(n) | Numerically stable: subtract max first |
| Layer Norm | O(n) | O(1) | Over feature dimension |
| BM25 scoring | O(|q|·avg_doc_len) | O(vocab) | q = query terms |
| ANN search (HNSW) | O(log n) | O(n) | n = num vectors |
| ANN search (IVF) | O(nprobe·cluster_size) | O(n) | nprobe = clusters searched |
| Cross-encoder reranking | O(K·n²) | O(n) | K candidates, n = seq len |

---

## Dynamic Programming Patterns

| Problem type | Recurrence template | Example |
|---|---|---|
| 0/1 Knapsack | `dp[i][w] = max(dp[i-1][w], dp[i-1][w-wt[i]] + val[i])` | Subset sum |
| Unbounded Knapsack | `dp[w] = max(dp[w], dp[w-wt[i]] + val[i])` | Coin change |
| LCS | `dp[i][j] = dp[i-1][j-1]+1 if match else max(dp[i-1][j], dp[i][j-1])` | Diff, edit distance |
| LIS | `dp[i] = max(dp[j]+1 for j<i if arr[j]<arr[i])` | Longest increasing |
| Edit distance | `dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(s1[i]!=s2[j]))` | Levenshtein |
| Matrix chain | `dp[i][j] = min(dp[i][k]+dp[k+1][j]+cost)` | Optimal parenthesization |

---

## Python Complexity Gotchas

| Operation | Complexity | Note |
|---|---|---|
| `list.append()` | O(1) amortized | |
| `list.insert(0, x)` | O(n) | Use deque for front-insertion |
| `list.pop()` | O(1) | |
| `list.pop(0)` | O(n) | Use deque.popleft() |
| `x in list` | O(n) | Use set for O(1) |
| `x in set` | O(1) avg | |
| `dict[key]` | O(1) avg | |
| `sorted(list)` | O(n log n) | Returns new list |
| `list.sort()` | O(n log n) | In-place |
| `heapq.heappush` | O(log n) | |
| `heapq.heappop` | O(log n) | |
