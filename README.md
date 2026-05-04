# Structural Memory

Structural Memory is a research prototype for building an **LLM-powered Socratic tutor for Data Structures & Algorithms (DSA)** that maintains **persistent pedagogical state**.

Out-of-the-box LLM tutors often behave like an *oracle*: they reveal solutions too early, skip prerequisite checks, and forget what the learner previously struggled with. Structural Memory bridges this pedagogical gap by combining:

- a **DSA Knowledge Graph (KG)** for grounded concepts + misconceptions,
- a compact **22-node Skeleton Graph (SG)** for curriculum structure and prerequisite gating,
- a per-student **User Graph** (JSON) that persists mastery scores,
- an **AST → rubric → mastery update** evaluator that turns student code into structured learning signals,
- a **Socratic tutor model** aligned via **SFT + DPO (RLAIF)** to guide through inquiry rather than answers.

---

## End-to-end system overview

Structural Memory runs as a closed learning loop:

1. **Student query / code submission** arrives for a DSA topic.
2. The request is mapped to a **Skeleton Graph node** (the curriculum concept).
3. A **prerequisite gate** checks the student’s mastery on required SG nodes (mastery threshold used in our experiments: **0.65**).
4. The system retrieves grounded context from the **KG** (definitions, prerequisite bridges, misconceptions).
5. A **Socratic tutor** generates a response that:
   - stays within the student’s learning frontier,
   - uses analogies anchored to already-mastered prerequisites,
   - avoids “answer leaks” (guided discovery over revelation).
6. When code is available, the **code evaluator** extracts structural signals, assesses rubric mastery, and **updates the student’s persistent mastery state**.

The key design principle is **decoupling state from generation**: the student’s evolving competence lives in structured graphs, and the LLM consumes that state to choose the right pedagogical move.

---

## Knowledge Graph (KG)

We construct a DSA KG (from CLRS) as a shared source of truth for:

- concept nodes and semantic chunks,
- prerequisite dependencies (`REQUIRES`),
- taxonomies (`SUBTYPE_OF`),
- technique relations (`USES`),
- misconception patterns (`HAS_MISCONCEPTION`).

In the overall tutoring loop:

- **SG prerequisites** decide *whether we should teach a concept now* (curriculum gating).
- **KG prerequisites** and misconceptions influence *how we teach it* (enrichment + analogical bridging + misconception preemption).

---

## Skeleton Graph (SG) + User Structural Memory

The Skeleton Graph is the curriculum layer:

- **22 curated nodes** across 4 tiers (foundations → advanced topics),
- explicit SG prerequisite edges (`sg_requires`),
- mapping to KG concept IDs for retrieval and bridging.

Each student has a lightweight **User JSON graph** that stores:

- `mastery_score` per SG node (0.0–1.0)

This is the project’s “structural memory”: it persists across sessions and drives prerequisite gating, personalization, and analogy selection.

---

## Code evaluator: AST → rubric → SG mastery update

Student code evaluation is hybrid (symbolic + LLM):

### 1) AST analysis (deterministic)

Using tree-sitter, we extract structural signals such as:

- recursion + base case detection,
- loop depth/types (complexity proxies),
- imports and tool usage (`heapq`, `deque`, `lru_cache`…),
- data structure usage (`dict`, `set`, `heap`, `deque`…),
- early returns / guard clauses (edge-case awareness),
- composite pattern signatures (BFS/DFS, Dijkstra, top-down DP, bottom-up DP, etc.).

### 2) LLM rubric scoring (hierarchical)

An LLM judge scores mastery using a **5-tier hierarchical rubric**:

1. Syntax
2. Logical use
3. Implementation depth
4. Edge cases
5. Conceptual transfer

### 3) Rule correction + blended updates

AST-driven rules cap implausible scores (“LLM proposes, AST constrains”), and mastery is updated smoothly over time (e.g., blending new signal with prior mastery).

### How this connects to tutoring

The evaluator produces **interpretable evidence about which SG nodes the student demonstrated** and writes updated mastery into the student’s persistent graph.

The tutor then uses that stored state to:

- gate advanced topics until prerequisites are mastered,
- select analogies from concepts the student has already shown they understand,
- focus on misconceptions the KG predicts for the next concept.

---

## Socratic tutor: alignment via SFT + DPO (RLAIF)

We ground tutoring behavior in ConvoLearn-style human tutoring moves (even though the source domain is Earth Science) and transfer those pedagogical tactics into DSA tutoring.

The alignment stack:

- **SFT (QLoRA)** teaches the model *what Socratic responses look like*.
- **DPO** trains preferences (chosen = Socratic scaffolding, rejected = direct answer), penalizing answer leaks and rewarding inquiry.

This produces a tutor that is more consistent at:

- resisting “just give me the solution” pressure,
- asking diagnostic questions,
- providing graded hints,
- sequencing help in alignment with learning-sciences principles.

---

## Evaluation framework

We evaluate Structural Memory with three independent modules:

- **E1 — Prerequisite gate enforcement:** precision/recall/specificity of gating decisions across controlled user archetypes.
- **E2 — Socratic answer leak detection:** leak rate and first-leak-turn (T*) under pressure across multi-turn simulations.
- **E3 — Learning sciences principle evaluation:** a 20-principle rubric measuring coverage, depth, and sequencing quality.

---

## Repository map

- `evaluator/` — **DSA Code Evaluator** (tree-sitter AST + LLM rubric scoring)
  - Extracts structural signals from student code
  - Detects algorithmic patterns (BFS/DFS, Dijkstra, DP, greedy…)
  - Scores mastery on SG skill nodes and can write updates into a user graph
  - See `evaluator/README.md` for the detailed pipeline

- `rag/` — DSA **RAG engine** + evaluation artifacts
  - Core RAG logic and evaluation scripts
  - Reports (`report.md`, `report.pdf`, `report.tex`) and visuals

- `USER/` — User state / skeleton graph
  - `skeleton_graph.json`: the 22-node curriculum graph + prerequisites
  - User JSONs that store per-node `mastery_score` and persist across sessions
  - Utilities to create/update a user graph

- `Data_pipeline/` — Dataset generation and preprocessing pipeline
  - Contains `data-pipeline.ipynb` for processing and preparing learning data

- `rough-docs/` — Working notes and evaluation writeups

- `docs/` — PDFs and report artifacts

- `llama_finetuning.ipynb` — Fine-tuning and experimentation notebook
