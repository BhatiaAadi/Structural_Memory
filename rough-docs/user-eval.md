# Code Evaluation Pipeline — Full Technical Documentation

## Overview

The evaluation pipeline assesses a student's DSA understanding by analyzing their Python code
through a two-stage architecture: deterministic AST analysis followed by LLM-based rubric evaluation,
with a final deterministic correction layer on top.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EVALUATION PIPELINE                              │
│                                                                         │
│  Student Code ──► STAGE 1: AST Analysis (deterministic)                 │
│                       │                                                 │
│                       ▼                                                 │
│               Structured Signals (JSON)                                 │
│                       │                                                 │
│  Code + Signals ──► STAGE 2: LLM Rubric Evaluation (probabilistic)     │
│                       │                                                 │
│                       ▼                                                 │
│               Raw LLM Scores (JSON)                                     │
│                       │                                                 │
│  Signals + Scores ─► STAGE 3: Rule-Based Correction (deterministic)    │
│                       │                                                 │
│                       ▼                                                 │
│               Final Mastery Scores ──► User Skeleton Graph              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1 — AST Analysis (Deterministic)

### What is tree-sitter?

tree-sitter is a parser generator that builds a concrete syntax tree (CST) from source code.
Unlike regex or string matching, it understands Python's grammar — it knows that `heapq.heappush`
is a method call, that `def dijkstra` is a function definition, and that a `for` loop inside a
`while` loop is nested at depth 2. This gives us ground-truth structural facts about the code.

### What we parse

The student's Python code is fed to `tree-sitter-python`, which produces a full syntax tree.
Every token, expression, statement, and block is a node in this tree with type information.

### What we extract (ast_analyzer.py)

We walk the tree and extract 7 categories of signals:

#### 1. Recursion Detection
- Walk all `function_definition` nodes
- For each function, search its body for `call` nodes where the callee name matches the function name
- If found: `has_recursion = true`
- Then check if any `if_statement` inside the recursive function contains a `return_statement`
- If found: `has_base_case = true` (the student wrote a base case)

#### 2. Loop Analysis
- Find all `for_statement` and `while_statement` nodes
- For each loop, walk upward through parents to count how many ancestor nodes are also loops
- This gives `loop_depth_max` (e.g., a for-inside-a-while = depth 2)
- Also record `loop_types` (["for"], ["while"], or ["for", "while"])

#### 3. Import Detection
- Find `import_statement` and `import_from_statement` nodes
- Extract module and name (e.g., "heapq", "collections.deque", "functools.lru_cache")
- This tells us what tools the student chose to use

#### 4. Data Structure Detection
- **Constructor calls**: `list()`, `dict()`, `set()`, `deque()`, `defaultdict()`
- **Literals**: `[]` (list), `{}` (dict/set), `{k:v}` (dict)
- **Special patterns**: `float('inf')` detected by finding `call` to `float` with "inf" argument
- **Attribute calls**: `heapq.heappush` detected from `attribute` nodes
- **Import-implied**: if `deque` is imported, we add it even without seeing a constructor
- **Membership tests**: `x in some_var` detected from `comparison_operator` nodes

#### 5. Built-in Call Detection
- Track method calls: `.append()`, `.pop()`, `.popleft()`, `.sort()`, `.add()`, etc.
- Track function calls: `sorted()`, `len()`, `range()`, `min()`, `max()`, `enumerate()`
- Track decorators: `@lru_cache`, `@cache`

#### 6. Early Return Detection
- For each function, check the first 3 statements in the body
- If any is an `if_statement` containing a `return_statement`, it's an early return (guard clause)
- This signals edge case awareness

#### 7. Comprehension Detection
- Find `list_comprehension`, `dictionary_comprehension`, `set_comprehension`, `generator_expression` nodes

### Output format

All signals are collected into a single JSON dict:

```json
{
  "has_recursion": false,
  "has_base_case": false,
  "recursive_functions": [],
  "loop_depth_max": 2,
  "loop_types": ["for", "while"],
  "loop_count": 3,
  "data_structures_used": ["dict", "float_inf", "heapq", "list", "set"],
  "ds_details": ["membership_test_found"],
  "builtin_calls": ["add", "heappop", "heappush"],
  "imports": ["heapq"],
  "has_early_returns": true,
  "early_return_count": 1,
  "has_comprehensions": true,
  "comprehension_types": ["dict_comprehension"],
  "function_count": 1,
  "pattern_signatures": [],
  "absent_patterns": []
}
```

### Pattern Detection (pattern_detector.py)

On top of the raw signals, a second deterministic pass detects **composite algorithm skeletons**.
These are conjunctions of multiple signals that together identify a known algorithm:

| Pattern | Required signals | What it means |
|---------|-----------------|---------------|
| **bfs** | deque + popleft + while loop | Proper BFS with O(1) dequeue |
| **bfs_with_list** | list + pop + append + while + NO heap + NO recursion + NO set | BFS using list (suboptimal O(n) dequeue) |
| **dfs_recursive** | recursion + set + NO heap | DFS via recursive function with visited tracking |
| **dfs_iterative** | list + pop + set + while + NO deque + NO heap | DFS via explicit stack |
| **dijkstra** | heapq + float('inf') + while loop | Dijkstra's shortest path |
| **topdown_dp_cached** | recursion + lru_cache decorator | Memoized DP using @lru_cache |
| **topdown_dp_manual** | recursion + dict + base case + NO heap | Manual memoization with dict |
| **bottomup_dp** | list + nested for loops + NO recursion + NO heap + NO set + NO while | Table-filling DP |
| **divide_and_conquer** | recursion + base case + 2+ functions + NO heap | D&C with helper function |
| **greedy** | sorted/sort + for loop + shallow nesting | Sort-then-scan greedy |
| **backtracking** | recursion + append + pop | State-modify-recurse-undo pattern |
| **union_find** | 3+ functions + list/dict | Parent array with find/union helpers |

Each pattern also carries a **confidence level** (high/medium) and maps to specific SG nodes.

The detector also computes **absent_patterns**: expected patterns (from the question definition)
that were NOT detected. For example, if the question expects `bfs` but only `bfs_with_list` was
found, it's treated as a partial match (not absent). But if a DP question finds no DP pattern
at all, the DP patterns appear in `absent_patterns`.

---

## Stage 2 — LLM Rubric Evaluation (Probabilistic)

### Why the LLM is needed

The AST analysis gives us **structural facts** — WHAT the student wrote. But it cannot tell us:

1. **Why the student chose this approach** — Did they use a heap because they understand greedy
   selection, or did they copy a template? The LLM reads the code holistically and judges intent.

2. **Whether the approach is appropriate for the problem** — Using Dijkstra on a graph with
   negative weights is syntactically correct but logically wrong. The AST sees valid code;
   only the LLM can judge if the algorithm matches the problem constraints.

3. **Quality of implementation beyond structure** — Two Dijkstra implementations can have
   identical AST signals but differ in relaxation correctness, initialization logic, or
   edge case handling. The LLM reads the actual logic.

4. **Conceptual gaps that aren't structural** — A student who implements BFS but uses `dist[v]`
   without initializing it shows a gap that tree-sitter can't catch (it's a runtime error,
   not a structural issue).

5. **Evidence and explanation** — The LLM provides human-readable evidence strings ("The code
   uses heapq correctly for priority queue operations") and gap descriptions that feed into
   the tutor's next response.

In short: **AST tells you WHAT they wrote. The LLM tells you WHAT THEY UNDERSTAND.**

### Model selection: Qwen 2.5 Coder 7B

**Model**: `qwen2.5-coder:7b` via Ollama (local inference)

**Why this model**:
- **Code specialization**: The `-coder` variant is specifically trained on code understanding,
  code generation, and code analysis tasks. It outperforms general-purpose models of the same
  size on code comprehension benchmarks.
- **JSON output**: Qwen 2.5 Coder is strong at structured output. Ollama's `format: "json"`
  mode further constrains the output to valid JSON.



### How we prompt the LLM

The LLM receives two parts:

**System prompt** : Contains the full hierarchical rubric definition.
The rubric divides the 0.0–1.0 mastery score into 5 hierarchical tiers:

| Tier | Range | Dimension | What it measures |
|------|-------|-----------|------------------|
| 1 | 0.00–0.20 | Syntax | Is the concept's syntax correct and idiomatic? |
| 2 | 0.20–0.40 | Logical Use | Is the concept applied correctly for this problem? |
| 3 | 0.40–0.60 | Implementation Depth | How complete is the implementation? |
| 4 | 0.60–0.80 | Edge Case Awareness | Does the code handle failure modes? |
| 5 | 0.80–1.00 | Conceptual Transfer | Does the student understand abstractly, not just procedurally? |

The rubric is HIERARCHICAL: if syntax scores 0.07, the evaluation stops there — logical use,
depth, edge cases, and transfer are all 0.0. A student must demonstrate each tier before
being evaluated on the next.

The system prompt also includes:
- Calibration examples (so the LLM knows what score range to give for good/mediocre/bad code)
- Critical rules (must evaluate ALL target concepts, don't under-score correct code)
- Strict JSON output schema

**User prompt** : Contains the specific evaluation context:
- Question title and description
- Target SG concept nodes to evaluate (with current mastery scores)
- The student's raw code
- The pre-extracted AST signals (as JSON)

### LLM output

The LLM returns a JSON object with per-node assessments:

```json
{
  "node_assessments": {
    "sg_shortest_path": {
      "mastery_score": 0.87,
      "tier_reached": 5,
      "tier_scores": {
        "syntax": 0.20,
        "logical_use": 0.20,
        "implementation_depth": 0.20,
        "edge_case": 0.20,
        "transfer_signal": 0.07
      },
      "evidence": "The code uses heapq, float('inf'), and a visited set correctly...",
      "gaps": "No handling for negative edge weights",
      "misconceptions_triggered": []
    }
  },
  "strengths": ["Correct Dijkstra with priority queue"],
  "weaknesses": ["No negative weight guard"]
}
```

---

## Stage 3 — Rule-Based Score Correction (Deterministic)

### Why this layer exists

Rather than switching to a larger (and slower) model, we add a deterministic correction layer
that uses the AST signals — which are ground-truth facts — to cap or adjust the LLM's scores.

### The key insight

The LLM's QUALITATIVE output is valuable (evidence, gaps, misconceptions). We keep all of that.
We only adjust the QUANTITATIVE tier scores where the AST provides hard evidence that the
LLM's score is wrong.

Think of it as: **LLM proposes, AST constrains.**

### The correction rules

#### Rule 1 — Absent primary pattern → cap implementation tiers
If the question expected a specific algorithm pattern (e.g., any DP pattern for a DP question)
and ALL expected patterns are absent from the AST, then on primary SG nodes:
- `implementation_depth` → capped at 0.0
- `edge_case` → capped at 0.0
- `transfer_signal` → capped at 0.0

**Why**: If the AST confirms the student didn't implement DP (no memo dict, no lru_cache,
no table-filling loops), then the LLM shouldn't give implementation credit. The AST is
ground truth here — it parsed every line of code and found no DP structure.

#### Rule 2 — Suboptimal data structure choice → cap syntax + transfer
If `bfs_with_list` was detected (student used list.pop(0) instead of deque.popleft()):
- `sg_stack_queue` syntax → capped at 0.13 (not idiomatic)
- `sg_stack_queue` edge_case → capped at 0.07 (no O(1) awareness)
- `sg_stack_queue` transfer → capped at 0.0 (template usage)
- `sg_bfs_dfs` syntax → capped at 0.13 (works but wrong tool)
- `sg_bfs_dfs` transfer → capped at 0.07

**Why**: This is a structural fact — the AST knows exactly that `.pop(0)` was called on a list,
not `.popleft()` on a deque. The LLM might miss this or not penalize it appropriately.

#### Rule 3 — No early returns → cap edge_case
If the AST found zero guard-clause returns (no `if <condition>: return` at the top of any
function), then:
- `edge_case` → capped at 0.13 for all nodes

**Why**: Edge case handling almost always manifests as early returns (`if not nums: return []`).
If the AST sees none, the student probably doesn't handle edge cases, regardless of what
the LLM thinks.

#### Rule 4 — Single-function template → cap transfer_signal
If `function_count <= 1` (student wrote only one function):
- `transfer_signal` → capped at 0.07 for all nodes

**Why**: Tier 5 (conceptual transfer) requires evidence of abstract understanding — helper
functions, non-obvious concept combinations, or structural adaptations beyond the template.
A single-function solution, even if correct, is likely a template reproduction. The AST
can verify this structural fact definitively.

#### Rule 5 — Secondary concepts → cap at tier 3
If a node is only a secondary concept for the question (e.g., sg_heap is secondary for a
Dijkstra question), then:
- `edge_case` → capped at 0.13
- `transfer_signal` → capped at 0.0

**Why**: A single question shouldn't grant full mastery on a concept it only tangentially tests.
The student should solve a heap-specific question to prove heap mastery at tier 4+.

### After correction

Mastery scores and tier_reached are recomputed from the adjusted tier_scores.
The qualitative fields (evidence, gaps, misconceptions, strengths, weaknesses) are kept
unchanged — they came from the LLM and are still valid as descriptions.

---

## Stage 4 — Mastery Update (Blended Write)

The final scores are not written directly to the user's skeleton graph. Instead, they are
blended with the existing mastery using a weighted average:

```
new_mastery = 0.6 × evaluation_score + 0.4 × old_mastery
```

This prevents a single evaluation from wildly swinging the mastery score.
- A student who consistently performs well will see their mastery rise steadily
- A single bad submission won't erase progress
- The 60/40 split favors recent evidence while respecting history

---

## Summary: Why Each Stage Matters

| Stage | Type | What it does | What it can't do |
|-------|------|-------------|------------------|
| **AST Analysis** | Deterministic | Extracts ground-truth structural facts: recursion, loops, data structures, algorithm patterns | Can't judge intent, appropriateness, or code quality beyond structure |
| **LLM Evaluation** | Probabilistic | Reads code holistically, judges understanding, explains reasoning, identifies misconceptions | Inconsistent at quantitative scoring, sometimes contradicts itself, may skip nodes |
| **Rule Correction** | Deterministic | Uses AST facts to cap implausible LLM scores, enforces structural consistency | Can't generate new assessments, only adjusts existing ones |
| **Blended Update** | Deterministic | Smooths mastery trajectory over multiple evaluations | Single evaluation gives partial signal, not full mastery |

The three-stage design means:
- We get the LLM's code comprehension abilities (no pure-AST system can match this)
- We don't trust the LLM's numbers blindly (the 7B model isn't reliable enough for that)
- The AST provides an objective anchor that prevents the most egregious scoring errors
- The result is a system that's more consistent than the LLM alone, and more insightful
  than rules alone