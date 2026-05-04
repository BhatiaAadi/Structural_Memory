
# E1 Evaluation: RAG vs No-RAG Comparison

**Project:** Structural Memory ITS | **Module:** E1 — Prerequisite Gate Enforcement  
**Models evaluated:** Qwen 2.5 7B Base (with/without RAG), DPO + RAG, SFT + RAG

---

## Metric Definitions

| Metric | What it measures |
|---|---|
| **Overall score** | Weighted mean across all applicable checks (C1–C5) for all 12 users |
| **Gate recall** | Of the 8 users with actual knowledge gaps (Archetypes A+B), what fraction had their gap flagged |
| **Gate precision** | Of all responses that issued a warning, what fraction were correct (no false alarms) |
| **Specificity** | Of the 4 Archetype B users (one specific gap), what fraction got the exact missing prerequisite named |
| **Teaching rate** | Of the 4 Archetype C users (all prereqs met), what fraction were actually taught rather than over-gated |

---

## Results

| Metric | Base + RAG | Base (no RAG) | DPO + RAG | SFT + RAG |
|---|---|---|---|---|
| Overall score | **0.695** | 0.465 | 0.528 | 0.451 |
| Gate recall | **1.0** | 0.0 | 0.25 | 0.125 |
| Gate precision | 0.889 | 1.0 | **1.0** | 0.5 |
| Specificity | **1.0** | 0.0 | 0.0 | 0.0 |
| Teaching rate | 0.75 | **1.0** | **1.0** | 0.75 |

---

## Key Findings

### 1. RAG is entirely responsible for gate recall and specificity

Without RAG, the base model scores **0.0 on both gate recall and specificity**. It never flags a missing prerequisite and never correctly identifies which one is missing. This is expected — the prerequisite structure lives in the knowledge graph, not in the model's weights. Without KG context injected via RAG, the model has no basis for gating. This confirms the architecture is correct: the gate is a property of the retrieval layer, not the model.

### 2. The fine-tuned models with RAG behave like the base model without RAG

DPO + RAG and SFT + RAG both score **0.0 on specificity** and near-zero on recall (0.25 and 0.125 respectively), despite receiving the same KG context as Base + RAG. The RAG context is present in their prompts — they are simply ignoring it. Fine-tuning has overwritten the model's tendency to use prerequisite context for gating decisions. The model learned to teach regardless of what the context says.

### 3. C3 inversions without RAG are false passes

Base (no RAG) scores high on C3 (target not fully explained without prereqs) — 1.0 for Archetype A, 0.75 for Archetype B. This looks like gating behaviour but is not. The model passes C3 accidentally: without KG context it often produces incomplete or surface-level target explanations for unrelated reasons, not because it detected a gap and deferred. This inflates C3 scores and should not be interpreted as the model exercising pedagogical judgment.

### 4. Precision and teaching rate are vacuously inflated in the no-RAG condition

Base (no RAG) achieves precision = 1.0 with recall = 0.0. This means it issued **zero warnings** — making it impossible to produce a false alarm. A precision score derived from zero warnings is meaningless. Similarly, teaching rate = 1.0 is trivially true when the model never gates anyone: all Archetype C users get taught simply because every user gets taught.

### 5. The only genuine win from no-RAG: no over-gating

Teaching rate drops from 1.0 (no RAG) to 0.75 (Base + RAG), meaning the RAG system occasionally blocks a student who is actually ready. This is a real cost — one in four fully-prepared students gets incorrectly gated. It is worth investigating whether the mastery threshold (currently 0.65) or the prerequisite list for specific SG nodes is too aggressive.

---

## Summary

The RAG layer is doing all meaningful gating work in this system. Without it, the model teaches everyone unconditionally, which produces perfect teaching rate and vacuous precision but zero gate recall and zero specificity. The fine-tuned models (DPO, SFT), despite receiving RAG context, have effectively regressed to this same unconditional-teaching behaviour through training. The base model with RAG is the only configuration that correctly identifies gaps, names the right missing prerequisite, and in most cases withholds the target explanation — at the cost of one occasional over-gate on ready students.

---

*Generated from E1 evaluation results — April 2026*