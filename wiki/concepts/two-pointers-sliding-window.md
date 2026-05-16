---
title: "Two Pointers & Sliding Window"
aliases: ["two pointers", "sliding window", "fast and slow pointers"]
tags: [coding, algorithms, arrays, strings]
related:
- "[[dynamic-programming]]"
sources: []
relevance: medium
last_updated: 2026-05-16
status: stub
---

# Two Pointers & Sliding Window

## TL;DR
Linear-time techniques for array/string problems using one or two index pointers moving inward or forward.

## Intuition
Two pointers: use two indices moving toward each other (sorted array pair sum) or in the same direction (removing duplicates, partitioning). Sliding window: maintain a window [left, right] over an array; expand right to include new elements, shrink left to maintain invariant (e.g., 'at most k distinct characters'). Fixed window: move both pointers together. Variable window: expand until constraint broken, then shrink. Fast & slow pointers: detect cycles (Floyd's), find middle of linked list. These patterns reduce O(n²) brute force to O(n).

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
- [[dynamic-programming]] — Some sliding window problems have DP equivalents; prefer sliding window when O(n) is achievable

## Sources
<!-- Add raw/ source paths after ingestion -->
