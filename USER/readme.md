# Skeleton Graph — Usage Guide

## What is the Skeleton Graph?

The Skeleton Graph (SG) is a **22-node curriculum index** that sits between your users and your Knowledge Graph. It is a JSON file — not a Neo4j graph.

Neo4j is used exactly **once**, offline, to build it. After that, the JSON is the only thing the system ever reads.

```
[Neo4j KG]  ──(once, offline)──▶  skeleton_graph.json
                                        ↓  copy + mastery scores
                                  users/diana.json
                                  users/bob.json
                                  users/charlie.json
                                        ↓
                                  tutor.py  +  Qwen 2.5 7B
```

Each user has their own JSON — a copy of the skeleton with `mastery` values (0.0–1.0) on every node. The KG context (prerequisites, misconceptions, techniques) is embedded inside each node's `kg_anchor` at build time, so nothing needs to query Neo4j at runtime.

---

## Files

| File | Purpose |
|---|---|
| `build_skeleton_graph.py` | **Run once.** Queries Neo4j, embeds KG context into each node, writes `skeleton_graph.json` |
| `user_sg.py` | Create, load, and update user JSON files. No Neo4j. |
| `tutor-context.py` | Reads a user's JSON, builds prompt context, calls Qwen 2.5 7B via Ollama |
| `skeleton_graph.json` | The base SG — shared template, never modified after build |
| `users/diana.json` | Diana's SG — her own mastery scores on each of the 22 nodes |

---

## Step 1 — Build `skeleton_graph.json` (once)

This is the only step that needs Neo4j. Run it once when you first set up, and again only if you re-ingest the KG with new material.

```python
# build_skeleton_graph.py — set your credentials and run

NEO4J_URI  = "neo4j+s://YOUR_INSTANCE.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASS = "YOUR_PASSWORD"
```

```bash
python build_skeleton_graph.py
```

What it does for each of the 22 curriculum nodes:

1. Fuzzy-searches Neo4j for the best-matching KG concept (e.g. `sg_dp` → `dynamic_programming`)
2. Pulls that concept's full KG neighborhood: prerequisites, techniques, parents, misconceptions
3. Embeds everything into the node's `kg_anchor` field
4. Writes `skeleton_graph.json`

After this, Neo4j can be offline forever.

**What a node looks like in `skeleton_graph.json`:**

```json
{
  "id": "sg_dp",
  "name": "Dynamic Programming",
  "tier": 3,
  "sg_requires": ["sg_recursion", "sg_divide_conquer"],
  "kg_search_aliases": ["dynamic programming", "dp", "memoization", ...],
  "kg_anchor": {
    "kg_id": "dynamic_programming",
    "kg_name": "Dynamic Programming",
    "kg_definition": "A method for solving problems by breaking them into overlapping subproblems...",
    "kg_section": "Chapter 15",
    "prerequisites": [
      { "id": "recursion", "name": "Recursion" },
      { "id": "divide_and_conquer", "name": "Divide and Conquer" }
    ],
    "misconceptions": [
      "DP always requires recursion — tabulation is iterative.",
      "The term 'programming' refers to writing code — it does not."
    ],
    "techniques": [...],
    "parents": [...]
  }
}
```

---

## Step 2 — Create a User

When a new student starts, create their JSON from the skeleton:

```python
from user_sg import create_user_sg

create_user_sg("diana")
# writes users/diana.json
# every node starts at mastery = 0.0
```

---

## Step 3 — Set Initial Knowledge (Intake)

At the start of a course or first session, you'll know something about what the user already knows — from a placement test, self-report, or prior session data. Write those mastery values directly:

```python
from user_sg import update_mastery

# Diana already knows these topics well
update_mastery("diana", "sg_recursion",      0.9)
update_mastery("diana", "sg_sorting",        0.9)
update_mastery("diana", "sg_divide_conquer", 0.85)
update_mastery("diana", "sg_complexity",     0.8)
update_mastery("diana", "sg_arrays",         0.8)
update_mastery("diana", "sg_pointers",       0.7)
update_mastery("diana", "sg_linked_list",    0.6)
# sg_dp and all others stay at 0.0 — what we'll teach
```

`update_mastery` adds the delta to the current value, clamps to [0.0, 1.0], and saves the file immediately.

---

## Step 4 — Use the User's SG at Query Time

When a student asks a question, `tutor.py` does the following entirely from the JSON:

1. Load the user's JSON (`users/diana.json`)
2. Fuzzy-match the question to an SG node
3. Read the `kg_anchor` embedded in that node — this has the definition, prerequisites, misconceptions
4. Check which SG prerequisites the user has/hasn't mastered
5. Find analogy bridges: KG prerequisites of the target that the user already knows
6. Assemble context string and call Qwen

```python
from tutor import answer

response = answer("diana", "explain dynamic programming")
print(response)
```

No Neo4j. No external calls except Qwen via Ollama.

---

## Step 5 — Update Mastery After Each Turn

After evaluating a student's response in a dialogue turn, update their SG:

```python
from user_sg import update_mastery

update_mastery("diana", "sg_dp", +0.10)   # answered correctly with a hint
update_mastery("diana", "sg_dp", -0.05)   # showed a misconception
update_mastery("diana", "sg_dp", +0.20)   # correct, unprompted explanation
```

The file is updated immediately. The next call to `answer("diana", ...)` will reflect the new mastery.

---

## How the Same Question Produces Different Responses

The context string injected into Qwen is built entirely from the user's JSON. Two users asking the same question get different context strings, so Qwen produces structurally different responses.

**Diana** asking "explain dynamic programming":
```
User    : Diana
Level   : intermediate
Mastery : 10% on this topic

Unmet SG prerequisites — explain these FIRST:
  ✗ (none — all met)

Analogy bridges:
  → Recursion
  → Divide & Conquer
  → Merge Sort
```

**Bob** asking the same question (knows nothing):
```
User    : Bob
Level   : beginner
Mastery : 0% on this topic

Unmet SG prerequisites — explain these FIRST:
  ✗ Recursion
  ✗ Divide & Conquer

Analogy bridges: (none)
```

Qwen receives entirely different instructions for each user — same model, same question, different pedagogical response.

---

## Checking the Learning Frontier

At any point you can check which topics a user is ready to learn next (all prerequisites met, topic not yet mastered):

```python
from user_sg import load_user_sg, learning_frontier

user_sg = load_user_sg("diana")
for node in learning_frontier(user_sg):
    print(node["name"], "—", int(node["mastery"] * 100), "%")

# Dynamic Programming — 10%
# Hash Tables — 0%
```

---

## Quick Reference — SG Node IDs

| ID | Name | Tier | Requires |
|---|---|---|---|
| `sg_complexity` | Asymptotic Complexity | 1 | — |
| `sg_recursion` | Recursion & Recurrences | 1 | complexity |
| `sg_arrays` | Arrays & Dynamic Arrays | 1 | — |
| `sg_pointers` | Pointers & Memory | 1 | — |
| `sg_linked_list` | Linked Lists | 2 | pointers |
| `sg_stack_queue` | Stacks & Queues | 2 | arrays, linked_list |
| `sg_hash_table` | Hash Tables | 2 | arrays, complexity |
| `sg_bst` | Binary Search Trees | 2 | recursion, pointers |
| `sg_heap` | Heaps & Priority Queues | 2 | arrays, complexity |
| `sg_sorting` | Sorting Algorithms | 3 | arrays, recursion, complexity |
| `sg_divide_conquer` | Divide & Conquer | 3 | recursion, sorting |
| `sg_graphs` | Graph Representations | 3 | arrays, linked_list |
| `sg_bfs_dfs` | BFS & DFS | 3 | graphs, stack_queue |
| `sg_greedy` | Greedy Algorithms | 3 | sorting, complexity |
| `sg_dp` | Dynamic Programming | 3 | recursion, divide_conquer |
| `sg_balanced_trees` | Balanced Trees | 4 | bst, complexity |
| `sg_shortest_path` | Shortest Path Algorithms | 4 | graphs, heap, greedy |
| `sg_mst` | Minimum Spanning Trees | 4 | graphs, greedy, heap |
| `sg_amortized` | Amortized Analysis | 4 | complexity, arrays |
| `sg_advanced_graphs` | Advanced Graph Algorithms | 4 | bfs_dfs, dp |
| `sg_string_algo` | String Algorithms | 4 | arrays, dp |
| `sg_np` | NP-Completeness | 4 | greedy, dp, advanced_graphs |