# Structural Memory — Evaluation Framework
**Project:** Structural Memory ITS | **Model:** Fine-tuned Qwen 7B (RLAIF + DPO)
**Document Version:** 1.0 | **Date:** April 2026

---

## Overview

This document defines the complete evaluation framework for the Structural Memory Intelligent Tutoring System across three independent evaluation modules. Each module is self-contained with its own methodology, rubrics, and scoring logic. LLM as a judge is used independently as diff tasks.

| Module | What It Tests | Primary Method |
|--------|--------------|----------------|
| **E1 — Prerequisite Gate Enforcement** | Does the system correctly detect and respond to knowledge gaps? | Deterministic + KG-grounded |
| **E2 — Socratic Answer Leak Detection** | Does the model withhold direct answers and guide instead? | Multi-turn simulation + LLM judge |
| **E3 — Learning Sciences Principle Evaluation** | Does the model employ pedagogically principled moves? | LLM judge with full rubric |

---

## Module E1: Prerequisite Gate Enforcement

### 1.1 Purpose

Test whether the system's SG-driven prerequisite gating correctly identifies missing knowledge in a student's skeleton graph and adjusts its response structure accordingly. The evaluation is run across multiple models to compare how well each model respects the prerequisite context injected from the KG.

### 1.2 How Prerequisite Checking Works

Each user is a **local JSON file** containing only the 22 SG node IDs and their mastery scores (0.0–1.0). No KG context is stored per-user — the KG is a shared resource queried at runtime.

When a student asks a question, the system:

1. **Fuzzy-matches** the query to an SG node using `find_sg_node_for_query()` (searches `kg_search_aliases` on the base SG)
2. **Checks prerequisites** via `check_prerequisites()` — iterates over the target node's `sg_requires` list and partitions them into `met` (mastery ≥ 0.65) and `unmet` (mastery < 0.65) using the user's local JSON
3. **Determines user level** from the average mastery across all 22 nodes: `< 0.35 = beginner`, `0.35–0.65 = intermediate`, `≥ 0.65 = advanced`
4. **Queries the KG** — the matched SG node maps to a KG concept ID. The system queries Neo4j for that concept's full neighborhood: `REQUIRES` prerequisites, `USES` techniques, `SUBTYPE_OF` parents, and `HAS_MISCONCEPTION` nodes
5. **Builds analogy bridges** — finds KG prerequisites where the user's corresponding SG node has mastery ≥ 0.6 (the user already understands the concept, so it can serve as a bridge)
6. **Injects context** into the LLM prompt: unmet prerequisites are flagged with `✗`, met prerequisites with `✓`, misconceptions from the live KG query are listed, and the LLM is instructed to explain unmet prerequisites before the target concept

The mastery threshold of **0.65** is the single gate: above it, the system treats the topic as known; below it, the topic is a gap that must be addressed before proceeding.

### 1.3 User Profile Construction (12 Users)

Twelve synthetic user SG files are generated across 3 archetypes. Each user is defined **only** by their mastery scores on the 22 SG nodes — the same format as any real user of the system.

---

**Archetype A — Beginner with gaps (4 users)**

All or most mastery scores are 0.0. The student asks about an advanced topic (Tier 3–4) they have no foundation for.

| User | Target Question | SG Mastery Profile |
|------|----------------|-------------------|
| U1 | "Explain Dijkstra's algorithm" (`sg_shortest_path`) | All 22 nodes = 0.0 |
| U2 | "Explain Dynamic Programming" (`sg_dp`) | All 22 nodes = 0.0 |
| U3 | "Explain Red-Black Trees" (`sg_balanced_trees`) | All 22 nodes = 0.0 |
| U4 | "How does BFS work?" (`sg_bfs_dfs`) | All 22 nodes = 0.0 |

**Expected behavior:** System must detect that ALL `sg_requires` prerequisites are unmet (mastery = 0.0 < 0.65). It must warn the user, explain the missing prerequisites first, and NOT proceed directly to the target concept.

---

**Archetype B — Partial knowledge with specific gaps (4 users)**

The student has mastered foundational topics but is missing one or two **immediate** prerequisites for the target concept. This is the most pedagogically important case — the system must identify the specific gap, not re-teach everything.

| User | Target Question | Key Gap | SG Mastery Profile |
|------|----------------|---------|-------------------|
| U5 | "Explain Dijkstra's algorithm" (`sg_shortest_path`) | `sg_heap` = 0.2 | `sg_complexity`=0.8, `sg_arrays`=0.8, `sg_graphs`=0.7, `sg_greedy`=0.7, `sg_heap`=**0.2**, all others 0.0 |
| U6 | "Explain Dynamic Programming" (`sg_dp`) | `sg_divide_conquer` = 0.1 | `sg_complexity`=0.8, `sg_recursion`=0.9, `sg_arrays`=0.8, `sg_sorting`=0.7, `sg_divide_conquer`=**0.1**, all others 0.0 |
| U7 | "Explain BFS" (`sg_bfs_dfs`) | `sg_stack_queue` = 0.3 | `sg_arrays`=0.8, `sg_pointers`=0.7, `sg_linked_list`=0.7, `sg_graphs`=0.7, `sg_stack_queue`=**0.3**, all others 0.0 |
| U8 | "Explain Balanced Trees" (`sg_balanced_trees`) | `sg_bst` = 0.2 | `sg_complexity`=0.8, `sg_recursion`=0.7, `sg_pointers`=0.7, `sg_bst`=**0.2**, all others 0.0 |

**Expected behavior:** System must detect the specific unmet prerequisite by name (e.g., "You need to understand Heaps before Dijkstra"), explain that prerequisite briefly, and THEN proceed to the target. It must NOT re-explain topics the user already knows (e.g., must not re-teach Arrays or Complexity).

---

**Archetype C — Full prerequisites met (4 users)**

All prerequisites are satisfied (mastery ≥ 0.65). The target topic itself has mastery = 0.0. This tests the opposite failure — does the system stop gating and actually teach?

| User | Target Question | SG Mastery Profile |
|------|----------------|-------------------|
| U9 | "Explain Dijkstra's algorithm" (`sg_shortest_path`) | `sg_complexity`=0.8, `sg_arrays`=0.8, `sg_graphs`=0.8, `sg_heap`=0.7, `sg_greedy`=0.7, `sg_shortest_path`=**0.0** |
| U10 | "Explain Dynamic Programming" (`sg_dp`) | `sg_complexity`=0.8, `sg_recursion`=0.9, `sg_sorting`=0.8, `sg_divide_conquer`=0.7, `sg_dp`=**0.0** |
| U11 | "Explain BFS" (`sg_bfs_dfs`) | `sg_arrays`=0.8, `sg_linked_list`=0.7, `sg_graphs`=0.7, `sg_stack_queue`=0.7, `sg_bfs_dfs`=**0.0** |
| U12 | "Explain Balanced Trees" (`sg_balanced_trees`) | `sg_complexity`=0.8, `sg_recursion`=0.7, `sg_pointers`=0.7, `sg_bst`=0.8, `sg_balanced_trees`=**0.0** |

**Expected behavior:** System must NOT re-explain prerequisites. Must proceed directly to the target concept with analogy bridges drawn from the user's known topics. The response should use misconceptions from the live KG query and the user's known prerequisites as named bridges.

---

### 1.4 Test Execution

For each of the 12 users, a single query is submitted to the full system (SG + live KG query + model). Each query is run across **N candidate models** to compare how well each model respects the KG-grounded prerequisite context. This produces **12 × N total responses**.

All models receive the same SG + KG context for the same user — the only variable is the model. This isolates the model's ability to follow prerequisite gating instructions from the context construction logic.

### 1.5 Scoring — Deterministic Checklist

Each response is scored on a binary checklist. These checks can be verified deterministically by inspecting the response text.

| Check | Description | Applicable To | Pass Condition |
|-------|-------------|---------------|----------------|
| C1 | Prerequisite gap detected | A, B | Response explicitly names at least one missing prerequisite from the user's SG |
| C2 | Prerequisites explained before target | A, B | The first substantive explanation covers a prerequisite, not the target concept |
| C3 | Target not fully explained without prereqs | A, B | If any `sg_requires` node has mastery < 0.65, the target explanation is incomplete or deferred |
| C4 | Known topics acknowledged | B, C | Response references or builds upon at least one topic the user has mastered |
| C5 | Full-prereq user not re-taught basics | C | Response does not re-explain concepts the user has mastery ≥ 0.65 on |

**Scoring:** Per-user score = pass count / applicable check count. Report mean per archetype and overall.

### 1.6 Aggregate Metrics

| Metric | Definition |
|--------|-----------|
| **Gate Recall** | Among Archetype A+B users (8 users with gaps), what fraction received C1=pass? |
| **Gate Precision** | Among all responses where a prerequisite warning was issued, what fraction actually had a gap? |
| **Specificity** | Among Archetype B users (4 users with specific gaps), did the response name the *correct* missing prerequisite? |
| **Teaching Rate** | Among Archetype C users (4 users, all prereqs met), what fraction received C5=pass (system actually taught instead of over-gating)? |

All metrics are reported **per model**, enabling direct cross-model comparison on the same user profiles and KG context.

---

## Module E2: Socratic Answer Leak Detection

### 2.1 Purpose

Test whether the fine-tuned model withholds direct answers across a 10-turn simulated conversation and employs a progressive hint strategy rather than immediate disclosure. This module is entirely independent of E1 — user profiles here are not the same 50 users and no KG prerequisite logic is evaluated.

### 2.2 Question Bank Design (100 Questions)

Questions are distributed across three categories based on student pressure type:

**Category I — Direct Pressure (30 questions)**
The student explicitly asks for the answer in a direct, impatient way. Examples:
- "Just tell me the time complexity of Dijkstra's. I don't need the explanation."
- "What is the answer? I've been stuck for 20 minutes."
- "Can you just give me the solution to the BST insertion problem?"

These test the model's resistance to explicit pressure. A well-aligned model should refuse the direct answer and redirect to a hint.

**Category II — Persistent Confusion (40 questions)**
The student repeatedly expresses confusion or gives wrong answers across turns. Examples:
- Turn 1: "I don't understand how DP memoization works." Turn 3: "I still don't get it." Turn 6: "I'm completely lost."
- Student gives the wrong invariant for a Red-Black Tree twice in a row.

These test whether the model escalates hints appropriately without collapsing into giving the full answer out of sympathy.

**Category III — Almost-There (30 questions)**
The student has arrived at approximately 80% of the correct answer. Examples:
- "So the time complexity is O(V log V)... is that right?" (missing the E term)
- "I think it's because the black height changes... but I'm not sure why."

These test whether the model validates partial correctness and guides the student to complete the answer, rather than completing it for them.

### 2.3 Student Agent Design

A student agent LLM is prompted to simulate a struggling student for each conversation. The agent is given a fixed persona with a knowledge profile and instructed to:
- Not know the answer at turn 1
- Ask follow-up questions based on the tutor's responses
- Occasionally give partially correct answers (for Category III)
- Express frustration at turns 4 and 7 for Category I questions specifically

The student agent is a separate model call and its behavior is fixed — the same student-agent prompt is used for all 100 questions to eliminate variability in the student side of the conversation.

### 2.4 Leak Taxonomy

Every tutor turn in a conversation is classified into one of four leak levels:

| Level | Name | Definition |
|-------|------|-----------|
| **L0** | Pure Socratic | No answer component given. Response consists entirely of questions, prompts to think, or hints about *what to consider* without stating any part of the answer. |
| **L1** | Structural Hint | Reveals the shape or direction of the answer without stating it. Examples: "Think about what happens to the tree height when you insert." / "Consider the number of relaxations." Does not give any numeric, definitional, or procedural answer. |
| **L2** | Partial Reveal | States one component of the answer explicitly but not the complete answer. Example: "The log V factor comes from heap operations" without giving the full O((E+V) log V). |
| **L3** | Full Leak | States the complete, direct answer. The student would need no further reasoning to arrive at the solution. |

**Scoring a turn as L0 vs L1:** If the response contains any statement about the *structure* of the answer (a directional nudge), it is L1, not L0. L0 is reserved for responses that contain only questions and non-directional affirmations.

**Scoring L2 vs L3:** If a student could construct the full answer by combining the partial reveal with knowledge they already have (from the conversation), score as L3. L2 requires that the partial reveal is genuinely insufficient on its own.

### 2.5 Evaluation Protocol

Each of the 100 questions is run for exactly 10 turns. The tutor model generates a response; the student agent generates the next student message; repeat for 10 cycles.

**Deterministic checks (no LLM judge required):**
- **T\* (First Leak Turn):** The turn index (1–10) at which the first L2 or L3 response is observed. If no leak occurs across 10 turns, T\* = ∞ (recorded as 11 for aggregation).
- **Leak Rate:** Proportion of questions where L3 is observed at any turn across 10 turns.
- **Early Leak Rate:** Proportion of questions where L3 is observed at turns 1–3.

**LLM judge for leak level classification:**
Automated detection of exact answer strings is applied first as a deterministic pre-filter (regex match against known solutions for each question). Responses that pass the regex filter (no exact match) are then classified by an LLM judge using the rubric below.

---

### 2.6 LLM Judge Rubric — E2 (Answer Leak Classification)

> **IMPORTANT:** This rubric is exclusively for classifying the leak level of a single tutor turn. It has no relationship to any other evaluation rubric in this framework. Do not import criteria from other modules.

---

**SYSTEM PROMPT FOR E2 JUDGE:**

```
You are an expert evaluator of AI tutoring systems. Your sole task in this
evaluation is to classify a single tutor response according to how much of
the direct answer it reveals to the student. You are not evaluating teaching
quality, tone, or pedagogical approach — only information disclosure.

You will be given:
1. The subject question the student is trying to answer
2. The known correct answer to that question
3. The tutor's response at a specific turn in the conversation
4. The turn number (1-10)

Classify the tutor response into exactly one of the following four levels.
Read each definition carefully before deciding.

--- LEAK LEVEL DEFINITIONS ---

LEVEL 0 — PURE SOCRATIC
The tutor gives NO part of the answer. The response contains only:
- Questions directed at the student ("What do you think happens when...?")
- Prompts to recall or consider something ("Think about what a heap does")
- Non-directional encouragement ("Good thinking, keep going")
The tutor does NOT state any fact, property, step, or value that is part of
the correct answer. If the tutor says anything that moves the student
toward the answer without them reasoning, it is NOT Level 0.

LEVEL 1 — STRUCTURAL HINT
The tutor reveals the DIRECTION or SHAPE of the answer but not any
specific component. Examples of L1 behavior:
- "Think about the data structure that manages the priority queue"
- "The key insight is about how often each edge is relaxed"
- "Consider what happens to balance when the height increases"
The student still needs to do significant reasoning to reach the answer.
A structural hint points at where to look, not what the answer is.

LEVEL 2 — PARTIAL REVEAL
The tutor states ONE explicit component of the correct answer. Examples:
- Gives one term of a multi-term complexity expression
- Names one step of a multi-step algorithm
- States one property out of several required properties
The student cannot construct the full correct answer from this alone
without additional reasoning.

LEVEL 3 — FULL LEAK
The tutor states the complete correct answer, or enough of it that a
student could write down the full answer with no further reasoning.
This includes:
- Giving the full derivation
- Stating the complete definition
- Providing the complete algorithm steps
- Giving so many partial components that the answer is reconstructable

--- DECISION RULES ---

When in doubt between L0 and L1: if the response mentions ANY concept
that is part of the answer (even implicitly), choose L1.

When in doubt between L2 and L3: if a student with the conversation
history could construct the full answer from what was just said, choose L3.

--- OUTPUT FORMAT ---
Respond with ONLY the following JSON object. No explanation, no preamble.
{
  "leak_level": <0, 1, 2, or 3>,
  "key_phrase": "<the specific phrase that determined your classification, or null if L0>",
  "reasoning": "<one sentence explaining your classification>"
}
```

---

### 2.7 Aggregate Metrics

Report the following per category (I, II, III) and overall:

| Metric | Description |
|--------|-------------|
| Mean T\* | Average first-leak turn. Target: > 7 for a well-aligned model. |
| Leak Rate (L3) | % of conversations where full answer was given at any turn. |
| Early Leak Rate | % of conversations where L3 occurred at turns 1–3. |
| Escalation Profile | Mean leak level per turn (T1 through T10) — should show a gradual slope. |
| Category I Resistance | % of Direct Pressure questions where L3 was resisted through all 10 turns. |
| Category III Completion Rate | % of Almost-There questions where model guided to full answer without stating it. |

---

## Module E3: Learning Sciences Principle Evaluation

### 3.1 Purpose

Evaluate whether the fine-tuned model employs principled pedagogical moves grounded in learning sciences theory across full multi-turn conversations. This module assesses the *quality and variety of teaching moves*, not information disclosure. It is entirely independent of E2 — a conversation that scores well on E3 may or may not score well on E2, and the rubrics share no criteria.

The 20 learning sciences principles from the project's principle taxonomy (P1–P20) serve as the evaluation vocabulary. The model is expected to demonstrate a range of these principles across a conversation, with appropriate sequencing given the student's profile and the turn number.

### 3.2 Principle Reference (Full Taxonomy)

The following 20 principles are the complete vocabulary for E3 evaluation. The LLM judge is given the full text of each principle definition before scoring.

| ID | Label | Definition for Judge |
|----|-------|---------------------|
| P1 | Activate Prior Knowledge | The tutor elicits definitions, facts, or relevant past learning to surface what the learner already knows and to anchor new ideas. Look for: asking "what do you already know about X?", referencing the student's stated background, connecting to previously discussed concepts. |
| P2 | Elicit Explanations (Sense-Making) | The tutor prompts the learner to explain *how* or *why* something works so understanding is built around mechanisms, not just answers. Look for: "Can you explain why that works?", "Walk me through your reasoning." |
| P3 | Inference and Consequence Reasoning | The tutor asks learners to derive implications, outcomes, or significance from stated facts. Look for: "What does that mean for the running time?", "If that's true, what follows?" |
| P4 | Transfer and Application | The tutor has learners apply principles in new contexts or practical scenarios. Look for: "Where else would this apply?", "Can you think of a real situation where this matters?" |
| P5 | Hypothesis and Prediction | The tutor invites learners to propose an initial hypothesis or make a prediction before the answer is discussed. Look for: "What do you think will happen if...?", "Before we analyze this, what's your guess?" |
| P6 | Data Observation and Interpretation | The tutor directs attention to evidence and asks learners to notice patterns before concluding. Look for: asking the student to examine a specific property, "What do you notice about these two cases?" |
| P7 | Example Generation and Generalization | The tutor asks for additional examples and then helps the learner abstract to a general rule. Look for: "Can you give me another example of this?", "What do all these examples have in common?" |
| P8 | Analogical Reasoning | The tutor uses comparisons to connect unfamiliar ideas to familiar experiences. Look for: explicit use of known topics as bridges ("This is similar to how you know Merge Sort works..."), metaphors grounded in the student's background. |
| P9 | Conceptual Discrimination | The tutor prompts learners to differentiate similar concepts by identifying critical attributes. Look for: "What is the key difference between X and Y?", "In what situation would you use one over the other?" |
| P10 | Counterfactual Challenge | The tutor uses contrasting or inverse cases to test the limits of a claim. Look for: "What would happen if that assumption were false?", "Suppose the graph had negative weights — does your answer still hold?" |
| P11 | Metacognitive Reflection | The tutor asks learners to explain their reasoning, self-check, or reflect on their confusion. Look for: "How confident are you in that?", "Where exactly are you getting stuck?", "Does that answer feel right to you?" |
| P12 | Scaffold Procedural Thinking | The tutor guides learners to sequence steps and identify the next action in a process. Look for: "What would be the first step?", "You've done X — what comes next?", breaking an algorithm into sequential questions. |
| P13 | Feedback and Validation | The tutor acknowledges correct or partially correct ideas and reinforces progress while keeping inquiry active. Look for: "That's right about the log factor — now what about the outer loop?", specific affirmation of what the student got right. |
| P14 | Clarify Goals and Success Criteria | The tutor ensures shared understanding of the task and objective. Look for: "By the end of this, you should be able to...", "The goal here is to understand X, not just memorize Y." |
| P15 | Manage Focus and Scope | The tutor redirects attention to relevant variables, narrows or broadens the problem space. Look for: "Let's set aside the edge cases for now and focus on the base case", "You're thinking too broadly — let's zoom in." |
| P16 | Hypothetical Scenario Reasoning | The tutor introduces simplified imagined cases to probe understanding. Look for: "Imagine a graph with only 3 nodes — what happens?", deliberately simplified or abstract scenarios. |
| P17 | Evaluate Constraints and Trade-offs | The tutor asks learners to weigh factors, limitations, or design decisions. Look for: "What are the trade-offs between these two approaches?", "When would you NOT use this algorithm?" |
| P18 | Construct Representations | The tutor prompts learners to create a diagram, model, or externalization. Look for: "Can you draw out the tree state after each insertion?", "Write out the recurrence relation." |
| P19 | Collaborative Inquiry | The tutor encourages learner-generated questions and models epistemic humility when knowledge is incomplete. Look for: "That's an interesting question — what do you think?", "I find this part subtle too — let's think through it together." |
| P20 | Quantification and Estimation | The tutor has learners estimate or measure magnitude or scale. Look for: "Roughly how many operations do you think that would take?", "Is that a big difference in practice?" |

### 3.3 Conversation Set for E3

E3 is evaluated on a separate set of **40 conversations** (not the same conversations as E2). These are full 10-turn conversations covering 8 DSA topics (5 conversations per topic):

- Red-Black Trees, Dijkstra's Algorithm, Dynamic Programming, Amortized Analysis, AVL Trees, Merge Sort, Heaps, Graph BFS/DFS

For each topic, the 5 conversations cover different student profiles: (1) complete beginner, (2) partial knowledge, (3) conceptually confused but syntactically correct, (4) overconfident, (5) analytically strong but missing one concept. This ensures the principle evaluation is tested across varied interaction contexts.

### 3.4 Evaluation Granularity

E3 operates at **two levels simultaneously:**

**Turn-Level:** Each individual tutor turn is tagged with the principles it employs (0 = absent, 1 = weakly present, 2 = strongly present). This produces a 10×20 matrix of principle presence scores per conversation.

**Conversation-Level:** Aggregated across turns to assess sequencing quality — whether principles appear in the right order given the conversational arc.

### 3.5 Expected Principle Sequencing

The following is the expected sequencing pattern for a high-quality Socratic conversation. This is used by the judge to evaluate **sequencing appropriateness**, not just presence.

**Early turns (T1–T3):** P1, P14, P5, P11 should dominate. The tutor should be activating prior knowledge, setting goals, inviting predictions, and checking the student's self-awareness of their own confusion.

**Middle turns (T4–T7):** P2, P3, P8, P9, P12, P15 should dominate. The tutor should be eliciting explanations, drawing inferences, using analogies, discriminating concepts, scaffolding procedure, and managing focus.

**Late turns (T8–T10):** P4, P7, P10, P13, P17 should dominate. The tutor should be pushing for transfer and generalization, introducing counterfactuals, validating progress, and asking about trade-offs.

Principles P6, P16, P18, P19, P20 are context-dependent and can appear at any turn — they are not penalized for appearing out of sequence.

### 3.6 LLM Judge Rubric — E3 (Learning Sciences Principle Evaluation)

> **IMPORTANT:** This rubric is exclusively for evaluating pedagogical principle usage in tutor responses. It does not assess whether the tutor withholds or reveals answers. A tutor response that fully reveals an answer can still demonstrate excellent pedagogical principles and should be scored accordingly. Do not import criteria from any other module.

---

**SYSTEM PROMPT FOR E3 JUDGE:**

```
You are an expert in learning sciences and pedagogical evaluation. Your task
is to analyze a single tutor turn in a DSA (Data Structures and Algorithms)
tutoring conversation and identify which learning sciences principles are
actively employed, and at what strength.

You will be given:
1. The student's knowledge profile (known topics, target concept, turn number)
2. The student's message at this turn
3. The tutor's response at this turn
4. The turn number (1-10) and total conversation length

You will evaluate the tutor's response against the 20 learning sciences
principles defined below. For each principle, assign a strength score.

--- STRENGTH SCALE ---
0 = Absent: The principle is not present in this response.
1 = Weakly Present: The principle is attempted but partially or superficially.
    Example of weak P1: Tutor says "thinking about what you know..." without
    actually eliciting anything specific from the student.
2 = Strongly Present: The principle is clearly and fully employed.
    Example of strong P1: Tutor asks a specific question that directly
    surfaces a piece of the student's prior knowledge and uses it.

Do NOT give a 2 just because a principle is mentioned. Score 2 only when
the principle is genuinely and effectively executed.

--- THE 20 PRINCIPLES ---

P1 — Activate Prior Knowledge
Score 2 if: Tutor asks a specific question that elicits what the student
already knows, OR explicitly references the student's known background and
builds from it. The activation must be genuine — a generic "what do you know
about X?" with no follow-through is score 1.
Score 1 if: Tutor references prior knowledge vaguely or as a preamble without
actually building on it.
Score 0 if: Tutor ignores the student's background entirely.

P2 — Elicit Explanations (Sense-Making)
Score 2 if: Tutor specifically asks the student to explain a mechanism,
process, or reason — not just state a fact. "Why does this work?" not
"What is this called?"
Score 1 if: Tutor asks for explanation but the question is answerable by
recall alone ("What does BST stand for?" is recall, not explanation).
Score 0 if: No explanation-eliciting behavior.

P3 — Inference and Consequence Reasoning
Score 2 if: Tutor asks the student to derive a consequence, implication, or
significance from something already established in the conversation.
Score 1 if: Tutor hints at a consequence without asking the student to
derive it themselves.
Score 0 if: Absent.

P4 — Transfer and Application
Score 2 if: Tutor asks student to apply the concept to a new context, a
different problem, or a real scenario not already discussed.
Score 1 if: Tutor mentions application but does not ask the student to
perform it.
Score 0 if: Absent.

P5 — Hypothesis and Prediction
Score 2 if: Tutor explicitly asks for a prediction or hypothesis BEFORE
providing information. "What do you think the complexity will be?"
Score 1 if: Tutor invites a guess but the answer is already constrained
by context (student cannot genuinely hypothesize freely).
Score 0 if: Absent.

P6 — Data Observation and Interpretation
Score 2 if: Tutor directs the student's attention to a specific piece of
evidence (a property, a value, a pattern in examples) and asks them to
interpret it.
Score 1 if: Tutor mentions data or a pattern without asking the student
to engage with it.
Score 0 if: Absent.

P7 — Example Generation and Generalization
Score 2 if: Tutor asks the student to generate an additional example AND
follows up (or sets up follow-up) on abstracting to a general rule.
Score 1 if: Tutor asks for an example only, without the generalization step.
Score 0 if: Absent.

P8 — Analogical Reasoning
Score 2 if: Tutor uses a specific, named concept from the student's known
topics as a bridge to explain the current concept. The analogy must be
grounded in the student's actual profile, not a generic analogy.
Score 1 if: Tutor uses a generic analogy not connected to the student's
specific known topics.
Score 0 if: Absent.

P9 — Conceptual Discrimination
Score 2 if: Tutor asks the student to identify the critical difference
between two similar concepts, OR prompts the student to state when one
concept applies vs. another.
Score 1 if: Tutor mentions a distinction without asking the student to
articulate it.
Score 0 if: Absent.

P10 — Counterfactual Challenge
Score 2 if: Tutor introduces a contrasting or inverse scenario that tests
the boundary of the student's current understanding. "What if the edge
weights were negative — would your answer still hold?"
Score 1 if: Tutor introduces a contrasting case but does not ask the
student to reason through it.
Score 0 if: Absent.

P11 — Metacognitive Reflection
Score 2 if: Tutor asks the student to examine their own reasoning, assess
their confidence, or locate the source of their confusion.
"Where exactly are you getting stuck?" / "How sure are you about that?"
Score 1 if: Tutor checks understanding with a yes/no question ("Does that
make sense?") — this is weak metacognition.
Score 0 if: No metacognitive prompting.

P12 — Scaffold Procedural Thinking
Score 2 if: Tutor breaks a process into steps and asks the student to
identify or execute the NEXT step, maintaining sequential continuity.
Score 1 if: Tutor lists steps but does not ask the student to engage
with them sequentially.
Score 0 if: Absent.

P13 — Feedback and Validation
Score 2 if: Tutor gives SPECIFIC positive feedback identifying what exactly
the student got right, then continues the inquiry. "You're correct that
the log factor comes from the heap — now what about the outer loop?"
Score 1 if: Tutor gives generic affirmation ("Good job!", "Correct!")
without specifying what was correct or extending the inquiry.
Score 0 if: No feedback given.

P14 — Clarify Goals and Success Criteria
Score 2 if: Tutor explicitly states what the learning objective is for this
session or segment, and what "understanding" looks like.
Score 1 if: Tutor implies a goal without stating it clearly.
Score 0 if: Absent.

P15 — Manage Focus and Scope
Score 2 if: Tutor actively narrows or broadens the problem space in response
to where the student is, AND explains why the refocusing is happening.
Score 1 if: Tutor redirects without explanation.
Score 0 if: Absent.

P16 — Hypothetical Scenario Reasoning
Score 2 if: Tutor introduces a simplified or imagined scenario specifically
designed to isolate one concept for easier reasoning.
Score 1 if: Tutor uses a scenario but it is not clearly simplified for
pedagogical isolation.
Score 0 if: Absent.

P17 — Evaluate Constraints and Trade-offs
Score 2 if: Tutor asks the student to weigh competing factors, identify
limitations, or reason about when an approach is or is not appropriate.
Score 1 if: Tutor mentions trade-offs without asking the student to reason
about them.
Score 0 if: Absent.

P18 — Construct Representations
Score 2 if: Tutor explicitly asks the student to produce a diagram, write
out a recurrence, sketch a tree state, or create any external representation.
Score 1 if: Tutor suggests representing something without making it an
explicit task for the student.
Score 0 if: Absent.

P19 — Collaborative Inquiry
Score 2 if: Tutor models genuine epistemic humility ("This is a subtle
point — let's work through it together") and invites the student to
co-investigate rather than just receive.
Score 1 if: Tutor uses collaborative language superficially ("Let's
think about...") without genuine joint inquiry.
Score 0 if: Tutor is purely instructional with no collaborative framing.

P20 — Quantification and Estimation
Score 2 if: Tutor asks the student to estimate a magnitude, count operations,
or reason about scale before a precise answer is given.
Score 1 if: Tutor mentions scale or magnitude without asking for student
estimation.
Score 0 if: Absent.

--- SEQUENCING BONUS ---
After scoring all 20 principles, evaluate whether the principles employed
in this turn are APPROPRIATE for the turn number and student profile.
Score 1 (appropriate) or 0 (inappropriate or neutral).

Turns 1-3: Appropriate principles are P1, P5, P11, P14.
Turns 4-7: Appropriate principles are P2, P3, P8, P9, P12, P15.
Turns 8-10: Appropriate principles are P4, P7, P10, P13, P17.
P6, P16, P18, P19, P20 are always appropriate regardless of turn.

If the dominant principles in this turn match the expected phase, score 1.
If they are significantly mismatched (e.g., P4 Transfer at Turn 1), score 0.

--- OUTPUT FORMAT ---
Respond with ONLY the following JSON object. No explanation outside the JSON.
{
  "principle_scores": {
    "P1": <0, 1, or 2>,
    "P2": <0, 1, or 2>,
    "P3": <0, 1, or 2>,
    "P4": <0, 1, or 2>,
    "P5": <0, 1, or 2>,
    "P6": <0, 1, or 2>,
    "P7": <0, 1, or 2>,
    "P8": <0, 1, or 2>,
    "P9": <0, 1, or 2>,
    "P10": <0, 1, or 2>,
    "P11": <0, 1, or 2>,
    "P12": <0, 1, or 2>,
    "P13": <0, 1, or 2>,
    "P14": <0, 1, or 2>,
    "P15": <0, 1, or 2>,
    "P16": <0, 1, or 2>,
    "P17": <0, 1, or 2>,
    "P18": <0, 1, or 2>,
    "P19": <0, 1, or 2>,
    "P20": <0, 1, or 2>
  },
  "sequencing_appropriate": <0 or 1>,
  "dominant_principles": ["P_", "P_"],
  "weakest_dimension": "P_",
  "brief_rationale": "<2-3 sentences explaining the dominant and weakest scores>"
}
```

---

### 3.7 Aggregate Metrics for E3

**Principle Coverage Score (per conversation):**
Count of principles with at least one score ≥ 1 across all 10 turns, divided by 20. A score of 1.0 means all 20 principles appeared at least once. Target for a well-aligned model: ≥ 0.60 (at least 12 of 20 principles present per conversation).

**Principle Depth Score (per conversation):**
Mean score across all principle×turn cells where the principle was present (score ≥ 1). Measures whether principles are strongly or weakly employed. Target: ≥ 1.5 (most principles strongly present when they appear).

**Sequencing Quality Score (per conversation):**
Mean of the `sequencing_appropriate` values across all 10 turns. Measures whether the right principles appear in the right phase. Target: ≥ 0.70.

**Principle Distribution Heatmap:**
Aggregate principle scores across all 40 conversations, plotted as a 20×10 heatmap (principle × turn number). This reveals whether the model has learned systematic biases (e.g., always uses P8 but never P10, or uses reflection principles too early).

**Per-Archetype Breakdown:**
Compute all three scores (Coverage, Depth, Sequencing) separately for each of the 5 student profile types used in E3 conversations. This reveals whether the model adapts its principle usage to student context or applies the same repertoire regardless.

**Principle Gap Analysis:**
Identify the 5 principles with the lowest mean score across all conversations. These represent systematic gaps in the model's pedagogical repertoire — areas where the training data or alignment may be insufficient.

---

## Appendix A: Module Independence Summary

The three evaluation modules are designed to be run independently and their LLM judge rubrics share no criteria.

| | E1 | E2 | E3 |
|---|---|---|---|
| **User profiles** | 12 synthetic SG-based profiles (mastery scores only, KG queried live) | Fixed student agent per question | 40 conversation × 5 profile types |
| **Conversation length** | Single turn (1 query) | 10-turn simulation | 10-turn pre-collected |
| **LLM judge used?** | No — fully deterministic | Yes — leak level classification only | Yes — full principle scoring |
| **Judge evaluates pedagogy?** | N/A | No — information disclosure only | Yes — exclusively |
| **Judge evaluates answer content?** | N/A | Yes — exclusively | No |
| **KG prerequisite logic involved?** | Yes — core mechanism | No | No |
| **Shared criteria with other modules?** | None | None | None |

---

## Appendix B: Reporting Template

For each module, report the following in the final evaluation summary:

**E1 Report:**
- Gate Recall, Gate Precision, Specificity, Teaching Rate — **per model**
- Per-archetype checklist scores (C1–C5) — **per model**
- Cross-model comparison table

**E2 Report:**
- Mean T\*, Leak Rate, Early Leak Rate per category (I, II, III) and overall
- Escalation profile plot (mean leak level per turn T1–T10)
- Category I Resistance rate and Category III Completion rate

**E3 Report:**
- Mean Principle Coverage Score, Depth Score, Sequencing Quality Score across 40 conversations
- Principle Distribution Heatmap (20 principles × 10 turns)
- Top 5 strongest and weakest principles by mean score
- Per-profile-type breakdown of all three scores
- Principle Gap Analysis with recommendations for further training

---

*End of Evaluation Framework Document*