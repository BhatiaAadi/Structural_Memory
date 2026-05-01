# DSA Code Evaluator — Structural Memory

Evaluates a student's DSA understanding by analyzing their code (tree-sitter AST + LLM rubric) and scoring them on the 22-node Skeleton Graph.

## How It Works

```
Student writes code  ──►  tree-sitter AST parser  ──►  Signal Extractor
        │                                                     │
        │                                                     ▼
        └────────────────────────────────────►  LLM Evaluator (Qwen 2.5 Coder 7B)
                                                     │
                                                     ▼
                                              Hierarchical Rubric Scoring
                                              (syntax → logic → depth → edge → transfer)
                                                     │
                                                     ▼
                                              Mastery updates on Skeleton Graph
```

### The Pipeline

1. Student submits Python code for a DSA question
2. **AST Analyzer** parses the code with tree-sitter and extracts structural signals:
   - Recursion, base cases, loop depth/types
   - Data structures used (heapq, deque, set, dict, etc.)
   - Built-in calls (append, pop, sorted, etc.)
3. **Pattern Detector** identifies composite algorithm skeletons from those signals:
   - BFS, DFS, Dijkstra, top-down DP, bottom-up DP, greedy, binary search, etc.
   - Also computes **absent patterns** (expected but not found)
4. **LLM Evaluator** receives the code + AST signals + question context and scores using a **hierarchical rubric**
5. Mastery scores are blended into the user's Skeleton Graph JSON

### The Rubric (0.0 → 1.0, hierarchical)

The score is divided into 5 tiers. A student only advances to the next tier when the previous is satisfactory:

| Range | Tier | What it measures |
|-------|------|------------------|
| 0.00 – 0.20 | **Syntax** | Does the concept appear with correct, idiomatic syntax? |
| 0.20 – 0.40 | **Logical Use** | Is the concept applied in the right place, for the right reason? |
| 0.40 – 0.60 | **Implementation Depth** | How deep is the implementation vs. template-copying? |
| 0.60 – 0.80 | **Edge Case Awareness** | Does the code handle failure modes and known misconceptions? |
| 0.80 – 1.00 | **Conceptual Transfer** | Does the student understand the concept abstractly, not just procedurally? |

**Example**: A score of 0.54 means the student has solid syntax and logical use, but their implementation depth has gaps. They haven't reached edge case handling yet.

---

## Prerequisites

```bash
# In your virtual environment:
pip install tree-sitter tree-sitter-python requests

# Pull the LLM model:
ollama pull qwen2.5-coder:7b

# Make sure Ollama is running:
ollama serve
```

---

## Usage

### Step 0 — Create a User (if not already done)

```bash
cd /path/to/AoLM/USER
python3 -c "from user_sg import create_user_sg; create_user_sg('diana')"
```

This creates `USER/users/diana.json` with all 22 nodes at mastery 0.0.

### Step 1 — Evaluate Code (Python API)

```python
import sys, os
sys.path.insert(0, "/path/to/AoLM")

from evaluator.evaluator import evaluate_code

# Student's code for a Dijkstra question
code = """
import heapq

def dijkstra(graph, source):
    dist = {node: float('inf') for node in graph}
    dist[source] = 0
    pq = [(0, source)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))

    return dist
"""

result = evaluate_code(
    username="diana",
    question_id="q_dijkstra",
    code=code,
    apply_updates=True,   # writes mastery to diana.json
    verbose=True,         # prints each pipeline step
)
```

### Step 2 — Read the Output

The result dict looks like:

```json
{
  "question": {
    "id": "q_dijkstra",
    "title": "Dijkstra's Shortest Path",
    "difficulty": 4
  },
  "ast_signals": {
    "has_recursion": false,
    "has_base_case": false,
    "loop_depth_max": 2,
    "data_structures_used": ["dict", "float_inf", "heapq", "list", "set"],
    "builtin_calls": ["add", "heappop", "heappush"],
    "pattern_signatures": ["dijkstra"],
    "absent_patterns": []
  },
  "llm_evaluation": {
    "node_assessments": {
      "sg_shortest_path": {
        "mastery_score": 0.73,
        "tier_reached": 4,
        "tier_scores": {
          "syntax": 0.20,
          "logical_use": 0.20,
          "implementation_depth": 0.20,
          "edge_case": 0.13,
          "transfer_signal": 0.0
        },
        "evidence": "Correct heapq usage, proper relaxation step...",
        "gaps": "No handling for disconnected components",
        "misconceptions_triggered": []
      }
    },
    "strengths": ["Correct Dijkstra implementation with visited set"],
    "weaknesses": ["No early termination when target is reached"]
  },
  "mastery_updates": {
    "sg_shortest_path": {
      "old": 0.0,
      "new": 0.438,
      "raw_score": 0.73,
      "delta": 0.438
    }
  }
}
```

### Step 3 — Run from CLI

```bash
cd /path/to/AoLM
python3 -m evaluator.evaluator
```

This runs the built-in sample (Dijkstra code) against a test user.

---

## Available Questions

| ID | Title | Difficulty | Primary SG Nodes |
|----|-------|-----------|------------------|
| `q_binary_search` | Binary Search | 1 | sg_arrays |
| `q_fibonacci` | Fibonacci Number | 1 | sg_recursion |
| `q_two_sum` | Two Sum | 1 | sg_hash_table |
| `q_valid_parentheses` | Valid Parentheses | 2 | sg_stack_queue |
| `q_linked_list_cycle` | Linked List Cycle Detection | 2 | sg_linked_list |
| `q_bst_validate` | Validate BST | 2 | sg_bst |
| `q_kth_largest` | Kth Largest Element | 2 | sg_heap |
| `q_merge_sort` | Merge Sort | 3 | sg_sorting, sg_divide_conquer |
| `q_bfs_shortest_path` | BFS Shortest Path | 3 | sg_bfs_dfs |
| `q_coin_change` | Coin Change (Min Coins) | 3 | sg_dp |
| `q_activity_selection` | Activity Selection | 3 | sg_greedy |
| `q_dijkstra` | Dijkstra's Shortest Path | 4 | sg_shortest_path |
| `q_topological_sort` | Topological Sort | 4 | sg_advanced_graphs |
| `q_lcs` | Longest Common Subsequence | 4 | sg_string_algo, sg_dp |

### Adding New Questions

Edit `evaluator/question_bank.py` and add a new dict:

```python
{
    "id": "q_your_question",
    "title": "Your Question Title",
    "description": "Problem statement...",
    "difficulty": 3,
    "primary_sg_nodes": ["sg_dp"],
    "secondary_sg_nodes": ["sg_recursion"],
    "expected_patterns": ["topdown_dp_manual"],
    "starter_code": "def solve(args):\n    pass",
}
```

---

## Files

| File | Purpose |
|------|---------|
| `evaluator/__init__.py` | Package init — exposes `evaluate_code()` |
| `evaluator/ast_analyzer.py` | tree-sitter parsing → structured AST signals |
| `evaluator/pattern_detector.py` | Composite algorithm skeleton detection (BFS, DP, etc.) |
| `evaluator/question_bank.py` | 15 DSA questions mapped to SG nodes |
| `evaluator/prompts.py` | LLM prompt templates with hierarchical rubric |
| `evaluator/llm_evaluator.py` | Ollama API call + JSON parsing + validation |
| `evaluator/evaluator.py` | Main pipeline orchestrator |
| `USER/rub.md` | Rubric design doc (reference, not used by code) |

---

## Mastery Update Logic

When `apply_updates=True`, the evaluator doesn't overwrite the user's mastery directly. It uses **weighted blending**:

```
new_mastery = 0.6 × llm_score + 0.4 × old_mastery
```

This means:
- A single good/bad evaluation won't wildly swing the score
- Consistent performance over multiple evaluations moves the needle
- The 60/40 split favors recent evidence while respecting history

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: tree_sitter_python` | Make sure you're running from the venv where you installed it |
| `ConnectionError` from Ollama | Run `ollama serve` in another terminal first |
| `FileNotFoundError: User not found` | Create the user first: `create_user_sg("username")` |
| LLM returns invalid JSON | The evaluator has fallback parsing. If it still fails, try again (temperature randomness) |
