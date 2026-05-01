# Skeleton Graph — Architecture & Design Document

## 1. What is the Skeleton Graph?

The Skeleton Graph (SG) is a **fixed 22-node directed acyclic graph** that represents the
complete DSA curriculum. Each node is a topic (e.g., *Dynamic Programming*, *BFS & DFS*,
*Heaps*), and each directed edge encodes a prerequisite dependency between topics.

It is **not** the Knowledge Graph. It is a lightweight, hand-designed curriculum index that
sits between the student and the Knowledge Graph. Its purpose is to:

1. Define **what** a student should learn (the 22 curriculum topics)
2. Define the **order** they should learn it in (prerequisite edges)
3. Track **how much** a student knows about each topic (mastery scores, 0.0–1.0)
4. Carry **embedded KG context** so the runtime never needs to query Neo4j

The SG exists as a single JSON file (`skeleton_graph.json`). Each student gets their own copy
with personalized mastery scores (`users/diana.json`, `users/bob.json`, etc.).

---

## 2. Why Do We Need It?

### The Knowledge Graph is too large and too detailed for runtime use

The Knowledge Graph (KG) lives in Neo4j. It contains hundreds of concepts, each with
definitions, relationships (`REQUIRES`, `USES`, `SUBTYPE_OF`, `HAS_MISCONCEPTION`), section
references, and fine-grained prerequisite chains. This is the source of truth for
*what DSA knowledge exists*.

But the KG is not designed for student tracking. It has no notion of mastery, no curriculum
ordering, no learning tiers. If you tried to track a student's progress directly on the KG:

- You'd need to maintain mastery scores on hundreds of nodes
- You'd need Neo4j running at all times for every query
- You'd have no pedagogical structure — the KG is a knowledge *map*, not a learning *path*
- The tutor would need to query Neo4j on every student interaction

The Skeleton Graph solves all of these problems by being a **small, static, self-contained
curriculum layer** that carries exactly the KG information it needs, embedded inside itself.

### What the SG provides that the KG alone cannot

| Need | KG | SG |
|------|----|----|
| Curriculum scope (what to teach) | Hundreds of concepts, no curation | 22 curated nodes across 4 tiers |
| Learning order | `REQUIRES` edges between concepts, but no tiers | `SG_REQUIRES` edges + explicit tier ordering (1→4) |
| Student mastery tracking | Not supported | Per-user mastery scores (0.0–1.0) on every node |
| Runtime performance | Requires Neo4j connection | Pure JSON — no database needed |
| Prerequisite gating | Possible but requires live graph queries | `check_prerequisites()` on local JSON |
| Learning frontier | Not defined | `learning_frontier()` — nodes ready to learn next |
| Personalized context for LLM | Requires Cypher queries per interaction | Pre-embedded in `kg_anchor` field |

---

## 3. The 22 Nodes — Curriculum Structure

The SG organizes DSA into 4 tiers of increasing complexity. A student must master tier N
prerequisites before advancing to tier N+1 topics.

### Tier 1 — Foundations (4 nodes)
These have no prerequisites. Every student starts here.

| Node ID | Topic | Prerequisites |
|---------|-------|---------------|
| `sg_complexity` | Asymptotic Complexity | — |
| `sg_recursion` | Recursion & Recurrences | complexity |
| `sg_arrays` | Arrays & Dynamic Arrays | — |
| `sg_pointers` | Pointers & Memory | — |

### Tier 2 — Core Data Structures (5 nodes)
Require foundational knowledge from Tier 1.

| Node ID | Topic | Prerequisites |
|---------|-------|---------------|
| `sg_linked_list` | Linked Lists | pointers |
| `sg_stack_queue` | Stacks & Queues | arrays, linked_list |
| `sg_hash_table` | Hash Tables | arrays, complexity |
| `sg_bst` | Binary Search Trees | recursion, pointers |
| `sg_heap` | Heaps & Priority Queues | arrays, complexity |

### Tier 3 — Core Algorithms (6 nodes)
Where algorithmic thinking begins. Each requires both data structures and foundational skills.

| Node ID | Topic | Prerequisites |
|---------|-------|---------------|
| `sg_sorting` | Sorting Algorithms | arrays, recursion, complexity |
| `sg_divide_conquer` | Divide & Conquer | recursion, sorting |
| `sg_graphs` | Graph Representations | arrays, linked_list |
| `sg_bfs_dfs` | BFS & DFS | graphs, stack_queue |
| `sg_greedy` | Greedy Algorithms | sorting, complexity |
| `sg_dp` | Dynamic Programming | recursion, divide_conquer |

### Tier 4 — Advanced (7 nodes)
Capstone topics that compose multiple tier 2–3 concepts.

| Node ID | Topic | Prerequisites |
|---------|-------|---------------|
| `sg_balanced_trees` | Balanced Trees (AVL, Red-Black) | bst, complexity |
| `sg_shortest_path` | Shortest Path (Dijkstra, Bellman-Ford) | graphs, heap, greedy |
| `sg_mst` | Minimum Spanning Trees | graphs, greedy, heap |
| `sg_amortized` | Amortized Analysis | complexity, arrays |
| `sg_advanced_graphs` | Advanced Graphs (SCC, Topo Sort) | bfs_dfs, dp |
| `sg_string_algo` | String Algorithms (KMP, Rabin-Karp) | arrays, dp |
| `sg_np` | NP-Completeness | greedy, dp, advanced_graphs |

### The prerequisite DAG

```
Tier 1:   complexity ─────────────────────┐
               │                          │
          recursion ──────┐               │
               │          │               │
          arrays ────┬────┼───────┬───────┤
               │     │    │       │       │
          pointers ──┼────┤       │       │
                     │    │       │       │
Tier 2:    linked_list    bst   hash_table│
               │                  │       │
          stack_queue ─────┐    heap ─────┤
               │           │      │       │
Tier 3:     graphs ───── bfs_dfs  │    greedy
               │           │      │       │
               │           │   sorting    │
               │           │      │       │
               │     divide_conquer       │
               │           │              │
               │          dp ─────────────┤
               │           │              │
Tier 4:   shortest_path  advanced_graphs  │
               │                          │
              mst      string_algo    balanced_trees
                                          │
                          np ─────────────┘
```

---

## 4. How the Skeleton Graph Connects to the Knowledge Graph

This is the critical architectural relationship. The SG is **not** a subset of the KG.
It is a separate, hand-designed graph that **anchors** into the KG at specific points.

### The Anchoring Process

When `build_skeleton_graph.py` runs (once, offline), it connects to Neo4j and performs
the following for each of the 22 SG nodes:

#### Step 1 — Fuzzy-match the SG node to a KG concept

Each SG node has a list of `kg_search_aliases` — human-written search terms that should
match KG concept names. For example:

```python
{
    "id": "sg_dp",
    "name": "Dynamic Programming",
    "aliases": ["dynamic programming", "dp", "memoization", "tabulation", "optimal substructure"]
}
```

The build script runs a Cypher query that searches for KG concepts whose `id` or `name`
contains any of these aliases:

```cypher
MATCH (c:Concept)
WHERE toLower(c.id) CONTAINS 'dynamic programming'
   OR toLower(c.name) CONTAINS 'dp'
   OR toLower(c.name) CONTAINS 'memoization'
   ...
RETURN c.id, c.name, c.definition, c.section
```

If multiple candidates match, the one with the most alias hits is selected as the **anchor**.

#### Step 2 — Pull the anchor's full KG neighborhood

Once the best-matching KG concept is found, the script pulls everything connected to it:

```cypher
MATCH (c:Concept {id: $kg_id})
OPTIONAL MATCH (c)-[:REQUIRES]->(prereq:Concept)
OPTIONAL MATCH (c)-[:USES]->(technique:Concept)
OPTIONAL MATCH (c)-[:SUBTYPE_OF]->(parent:Concept)
OPTIONAL MATCH (c)-[:HAS_MISCONCEPTION]->(m:Misconception)
RETURN
    collect(DISTINCT prereq)       AS prerequisites,
    collect(DISTINCT technique)    AS techniques,
    collect(DISTINCT parent)       AS parents,
    collect(DISTINCT m.description) AS misconceptions
```

This pulls:
- **KG prerequisites** — what the KG says you need to know before this concept
- **Techniques** — specific methods or algorithms associated with this concept
- **Parents** — broader category this concept belongs to
- **Misconceptions** — known student misunderstandings about this concept

#### Step 3 — Store the mapping in the base SG

The matched KG concept ID is stored in the SG node as `kg_concept_id`. This is the
only KG data stored in the SG — it acts as a pointer, not a cache:

```json
{
  "id": "sg_dp",
  "name": "Dynamic Programming",
  "tier": 3,
  "sg_requires": ["sg_recursion", "sg_divide_conquer"],
  "kg_search_aliases": ["dynamic programming", "dp", "memoization", ...],
  "kg_concept_id": "dynamic_programming"
}
```

At runtime, when the tutor or evaluator needs context for `sg_dp`, the system uses
`kg_concept_id` to query Neo4j for the full neighborhood (prerequisites, misconceptions,
techniques, parents). This means the KG can be updated without rebuilding the SG or
any user files.

### Two different prerequisite systems, one node

Each SG node has **two** kinds of prerequisites, and they serve different purposes:

| Field | Source | Purpose | Example (for sg_dp) |
|-------|--------|---------|---------------------|
| `sg_requires` | Hand-designed | **Curriculum ordering** — what the student must learn first in our curriculum | `["sg_recursion", "sg_divide_conquer"]` |
| KG `REQUIRES` edges | From live KG query | **Conceptual dependencies** — what the KG says this concept requires intellectually | `recursion`, `divide_and_conquer` (returned by Neo4j) |

These are **not** the same lists, even though they often overlap:

- `sg_requires` uses SG node IDs and defines the learning path we've designed
- KG prerequisites use KG concept IDs and reflect the KG's own dependency structure
- The KG may have prerequisites that aren't in our 22-node curriculum (e.g., a KG prerequisite
  like `mathematical_induction` is intellectually required but isn't an SG node)
- The SG may require something the KG doesn't explicitly list (e.g., `sg_arrays` is required
  for `sg_stack_queue` in our curriculum design, even if the KG doesn't have a direct
  `REQUIRES` edge between them)

### The dual-prerequisite system at runtime

When the tutor processes a student query, it checks **both** prerequisite systems:

1. **SG prerequisites** (`sg_requires`) — determines if the student should be learning this
   topic at all. If `sg_recursion` mastery is below 0.65, the tutor won't teach `sg_dp`.

2. **KG prerequisites** (queried live via `REQUIRES` edges) — finds **analogy bridges**. If
   the student already knows a KG prerequisite well (mastery ≥ 0.6), the tutor uses it as an
   anchor: *"You already understand recursion — DP extends that idea by storing results..."*

The SG handles gating (should we teach this?), the KG handles enrichment (how should we teach it?).

---

## 5. How the SG is Used at Runtime

### For the Tutor (teaching)

```
Student asks "explain dynamic programming"
    │
    ▼
find_sg_node_for_query() — fuzzy-match to sg_dp
    │
    ▼
check_prerequisites() — are sg_recursion and sg_divide_conquer mastered?
    │
    ├── YES → proceed to teach sg_dp
    │           │
    │           ▼
    │         query_kg(kg_concept_id) — live Neo4j query for:
    │           - KG definition
    │           - KG misconceptions (inject into prompt)
    │           - KG prerequisites (find analogy bridges)
    │           │
    │           ▼
    │         ask_model() — call LLM with personalized context
    │
    └── NO → teach unmet prerequisites first
```

### For the Evaluator (code assessment)

```
Student submits code for q_dijkstra
    │
    ▼
Map question to SG nodes:
    primary:   [sg_shortest_path]
    secondary: [sg_heap, sg_greedy, sg_graphs]
    │
    ▼
AST analysis → extract structural signals
    │
    ▼
LLM evaluation → score each SG node on 5-tier rubric
    │
    ▼
Rule-based correction → cap scores using AST evidence
    │
    ▼
Blended mastery update → write to user's JSON
    new_mastery = 0.6 × score + 0.4 × old_mastery
```

### For the Learning Frontier (what to teach next)

```python
learning_frontier(user_sg)
# Returns nodes where:
#   - mastery < 0.65 (not yet known)
#   - ALL sg_requires have mastery >= 0.65 (ready)
#
# For a student who knows Tier 1 + some Tier 2:
# → ["Hash Tables", "Heaps", "BST"]
```

---

## 6. SG vs KG — The Full Relationship

```
┌───────────────────────────────────────────────────────────────────────┐
│                        KNOWLEDGE GRAPH (Neo4j)                        │
│                                                                       │
│   Hundreds of Concept nodes, each with:                               │
│     - id, name, definition, section                                   │
│     - REQUIRES edges to other concepts                                │
│     - USES edges to techniques                                        │
│     - SUBTYPE_OF edges to parent categories                           │
│     - HAS_MISCONCEPTION edges to misconception nodes                  │
│                                                                       │
│   This is the SOURCE OF TRUTH for DSA knowledge.                      │
│   Built from CLRS, lecture materials, and domain expertise.           │
│   Shared across ALL users. Queried at RUNTIME.                        │
│                                                                       │
└───────────────────────────┬───────────────────────────────────────────┘
                            │
              ┌─────────────┼──────────────┐
              │             │              │
     (build time)    (runtime: tutor)  (runtime: evaluator)
              │             │              │
              ▼             ▼              ▼
┌───────────────────────────────────────────────────────────────────────┐
│                     SKELETON GRAPH (base JSON)                        │
│                                                                       │
│   22 curated curriculum nodes, each with:                             │
│     - id, name, tier (1-4)                                            │
│     - sg_requires (curriculum prerequisites)                          │
│     - kg_concept_id (maps this SG node to a KG concept)               │
│     - kg_search_aliases (for fuzzy-matching student queries)          │
│                                                                       │
│   This is the CURRICULUM STRUCTURE. Shared, read-only.                │
│   Defines what to teach and in what order.                            │
│   Does NOT store KG content — just the mapping to KG.                 │
│                                                                       │
└───────────────────────────┬───────────────────────────────────────────┘
                            │
                   create_user_sg(username)
                   (create JSON with mastery = 0.0 per node)
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────────┐
│                     USER JSON (per student)                            │
│                                                                       │
│   Only stores: { node_id → mastery_score } for all 22 SG nodes.      │
│                                                                       │
│   users/diana.json — Diana's mastery scores                           │
│   users/bob.json   — Bob's mastery scores                             │
│                                                                       │
│   No KG content stored here. Lightweight.                             │
│   Updated after every evaluation or tutoring session.                 │
│   Drives personalized responses via mastery-based gating.             │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### How they interact at runtime

When a student asks a question:

1. The **SG** identifies which curriculum node is relevant and what its prerequisites are
2. The **user JSON** provides the mastery scores to determine which prerequisites are met/unmet
3. The **KG** is queried live for that concept's full context: definition, misconceptions,
   techniques, and KG-level prerequisites (for analogy bridges)

This means:
- The KG can be updated (new misconceptions, refined definitions) without rebuilding user files
- User files stay small — just 22 mastery floats per student
- The SG-to-KG mapping is maintained in the base SG, not duplicated per user

### In summary

The **Knowledge Graph** is the library. It knows everything about every DSA concept.
It is queried at runtime whenever the tutor or evaluator needs concept context.

The **Skeleton Graph** is the syllabus. It selects 22 topics, orders them into a learning
path, and maps each topic to a KG concept for live context retrieval.

The **User JSON** is the report card. It stores only mastery scores — how far each student
has progressed through the syllabus. It is the only per-user state.

The bridge between KG and SG is the `kg_concept_id` mapping — the SG knows which KG concept
each curriculum node corresponds to, and queries the KG at runtime for the full context.
