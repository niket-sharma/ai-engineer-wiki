# LeetCode Patterns for ML/AI Engineer Interviews

**Sources:** Blind 75, NeetCode 150, Grokking the Coding Interview, personal interview reports from ML engineering interviews

---

## Pattern Recognition Guide

When you see... → think...

| Problem Signal | Pattern |
|---|---|
| Sorted array / BST | Binary search, two pointers |
| Subarray sum / sliding window | Prefix sums, sliding window |
| Parentheses, valid string | Stack |
| Top K / K smallest | Heap (min-heap for top K largest, max-heap for top K smallest) |
| Shortest path in unweighted graph | BFS |
| Shortest path in weighted graph | Dijkstra |
| All paths / combinations | DFS + backtracking |
| Optimal substructure + overlapping subproblems | Dynamic programming |
| Maximize/minimize over choices | DP or greedy |
| Tree traversal | DFS (recursive) or BFS (level-order) |
| Cycle detection in graph | DFS with colors, or Union-Find |
| Connected components | Union-Find or BFS/DFS flood fill |
| Intervals | Sort by start, sweep line |

---

## Two Pointers

**When:** Sorted array, pairs summing to target, palindrome check, remove duplicates.

```python
# Two Sum II (sorted array) — O(n)
def two_sum_sorted(numbers, target):
    left, right = 0, len(numbers) - 1
    while left < right:
        s = numbers[left] + numbers[right]
        if s == target:
            return [left + 1, right + 1]  # 1-indexed
        elif s < target:
            left += 1
        else:
            right -= 1

# Container With Most Water — O(n)
def max_area(height):
    left, right = 0, len(height) - 1
    max_water = 0
    while left < right:
        max_water = max(max_water, min(height[left], height[right]) * (right - left))
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return max_water
```

---

## Sliding Window

**When:** Maximum/minimum subarray of length k, longest substring with constraint.

```python
# Longest Substring Without Repeating Characters — O(n)
def length_of_longest_substring(s):
    char_idx = {}
    left = max_len = 0
    for right, char in enumerate(s):
        if char in char_idx and char_idx[char] >= left:
            left = char_idx[char] + 1
        char_idx[char] = right
        max_len = max(max_len, right - left + 1)
    return max_len

# Maximum Sum Subarray of Size K — O(n)
def max_sum_subarray(nums, k):
    window_sum = sum(nums[:k])
    max_sum = window_sum
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum
```

---

## Binary Search

**Template — always use this:**

```python
def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2  # avoid overflow (Python: not needed, but habit)
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Find leftmost / first occurrence — key variant
def search_leftmost(nums, target):
    left, right = 0, len(nums)
    while left < right:      # note: right = len(nums), not len-1
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid      # don't +1 here — might be the answer
    return left if left < len(nums) and nums[left] == target else -1

# Binary search on the answer — O(n log(search_space))
def min_eating_speed(piles, h):
    """Koko eating bananas. Binary search on speed k."""
    def can_finish(k):
        return sum((p + k - 1) // k for p in piles) <= h
    
    left, right = 1, max(piles)
    while left < right:
        mid = (left + right) // 2
        if can_finish(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

---

## Heap (Priority Queue)

```python
import heapq

# Python: heapq is a min-heap by default
# For max-heap: negate values

# Top K Frequent Elements — O(n log k)
def top_k_frequent(nums, k):
    from collections import Counter
    freq = Counter(nums)
    # Use min-heap of size k — always keep k largest
    heap = []
    for num, count in freq.items():
        heapq.heappush(heap, (count, num))
        if len(heap) > k:
            heapq.heappop(heap)  # removes smallest
    return [num for count, num in heap]

# K Closest Points to Origin — O(n log k)
def k_closest(points, k):
    heap = []
    for x, y in points:
        dist = -(x**2 + y**2)  # negate for max-heap behavior
        heapq.heappush(heap, (dist, x, y))
        if len(heap) > k:
            heapq.heappop(heap)
    return [[x, y] for _, x, y in heap]

# Merge K Sorted Lists — O(n log k)
def merge_k_lists(lists):
    heap = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))
    
    dummy = ListNode(0)
    curr = dummy
    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next
```

---

## Dynamic Programming

### 1D DP

```python
# Climbing Stairs / Fibonacci — O(n)
def climb_stairs(n):
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b

# House Robber — O(n)
def rob(nums):
    prev2, prev1 = 0, 0
    for n in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + n)
    return prev1
```

### 2D DP / Subsequences

```python
# Longest Common Subsequence — O(m*n)
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

# Knapsack — O(n*W)
def knapsack(weights, values, W):
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            dp[i][w] = dp[i-1][w]
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w-weights[i-1]] + values[i-1])
    return dp[n][W]

# Edit Distance — O(m*n)
def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]
```

---

## Graphs

### BFS (Shortest Path, Level Order)

```python
from collections import deque

def bfs(graph, start, target):
    visited = {start}
    queue = deque([(start, 0)])  # (node, distance)
    while queue:
        node, dist = queue.popleft()
        if node == target:
            return dist
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1

# Rotting Oranges / Walls and Gates: multi-source BFS
def oranges_rotting(grid):
    fresh = 0
    rotten = deque()
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 2:
                rotten.append((r, c, 0))
            elif grid[r][c] == 1:
                fresh += 1
    
    max_time = 0
    while rotten:
        r, c, time = rotten.popleft()
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == 1:
                grid[nr][nc] = 2
                fresh -= 1
                max_time = max(max_time, time + 1)
                rotten.append((nr, nc, time + 1))
    
    return max_time if fresh == 0 else -1
```

### DFS (Backtracking, Connected Components)

```python
# Number of Islands — O(m*n)
def num_islands(grid):
    def dfs(r, c):
        if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]):
            return
        if grid[r][c] != '1':
            return
        grid[r][c] = '0'  # mark visited
        dfs(r+1, c); dfs(r-1, c); dfs(r, c+1); dfs(r, c-1)
    
    count = 0
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == '1':
                dfs(r, c)
                count += 1
    return count

# Combination Sum (Backtracking) — O(n^(T/min_val))
def combination_sum(candidates, target):
    result = []
    def backtrack(start, current, remaining):
        if remaining == 0:
            result.append(list(current))
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break
            current.append(candidates[i])
            backtrack(i, current, remaining - candidates[i])  # i (not i+1) for reuse
            current.pop()
    
    candidates.sort()
    backtrack(0, [], target)
    return result
```

---

## Common ML-Specific Coding Questions

```python
# Implement softmax — numerically stable
def softmax(x):
    x = x - x.max()  # subtract max for stability
    e_x = np.exp(x)
    return e_x / e_x.sum()

# Implement sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# K-means clustering
def kmeans(X, k, max_iter=100):
    centroids = X[np.random.choice(len(X), k, replace=False)]
    for _ in range(max_iter):
        # Assign clusters
        dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(dists, axis=1)
        # Update centroids
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

# Moving average (streaming)
from collections import deque
class MovingAverage:
    def __init__(self, size):
        self.window = deque(maxlen=size)
        self.total = 0
    def next(self, val):
        if len(self.window) == self.window.maxlen:
            self.total -= self.window[0]
        self.window.append(val)
        self.total += val
        return self.total / len(self.window)
```

---

## Time/Space Complexity Quick Reference

| Algorithm | Time | Space |
|---|---|---|
| Binary search | O(log n) | O(1) |
| Sliding window | O(n) | O(k) |
| Two pointers | O(n) | O(1) |
| BFS/DFS | O(V + E) | O(V) |
| Heap (push/pop) | O(log n) | O(n) |
| Sort | O(n log n) | O(log n) |
| DP (2D, m×n) | O(m·n) | O(m·n) or O(n) optimized |
| Quick sort (avg) | O(n log n) | O(log n) |
| Merge sort | O(n log n) | O(n) |

---

## Interview Strategy

1. **Clarify first:** edge cases, input constraints, expected output format
2. **Brute force first:** state the O(n²) or recursive solution verbally
3. **Optimize:** identify the bottleneck, propose the pattern
4. **Code the optimized:** clean, with good variable names
5. **Test:** null input, single element, duplicates, large input

**Time allocation:** 5 min clarify → 5 min brute force → 5 min optimize → 20 min code → 5 min test
