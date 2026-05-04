# Structural Memory

Structural Memory is a research prototype for building an **LLM-powered Socratic tutor for Data Structures & Algorithms (DSA)** that maintains **persistent pedagogical state**.

Out-of-the-box LLM tutors often behave like an *oracle*: they reveal solutions too early, skip prerequisite checks, and forget what the learner previously struggled with. Structural Memory bridges this pedagogical gap by combining:

- a **DSA Knowledge Graph (KG)** for grounded concepts + misconceptions,
- a compact **22-node Skeleton Graph (SG)** for curriculum structure and prerequisite gating,
- a per-student **User Graph** (JSON) that persists mastery scores,
- an **AST → rubric → mastery update** evaluator that turns student code into structured learning signals,
- a **Socratic tutor model** aligned via **SFT + DPO (RLAIF)** to guide through inquiry rather than answers.

---
## Compute and API Key Requirements

We used freely available p100 and T4*2 GPUs available at [kaggle](https://www.kaggle.com/) and Google gemini 2.5 pro API key to run the experiments.

## Prerequisites & Environment Setup

### Python Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

pip install google-genai neo4j PyMuPDF requests tree-sitter matplotlib numpy
```

### API Keys Required

| Service | Where to Set | Used By |
|---------|-------------|---------|
| **Google Gemini API** | Hardcoded in scripts (replace `GOOGLE_API_KEY` / `GEMINI_API_KEY`) | KG builder, data pipeline, all evaluation judges |
| **Neo4j Aura** | Hardcoded in `rag/build_dsa_graph.py`, `rag/rag_engine.py`, `USER/build_skeleton_graph.py` | KG storage & retrieval |
| **HuggingFace Token** | Hardcoded in simulation scripts (replace `HF_TOKEN`) | Model loading for evaluation simulations |
| **Ollama (local)** | Runs on `localhost:11434` | RAG engine, tutor runtime |

### Hardware Requirements

| Phase | Hardware | Notes |
|-------|----------|-------|
| KG build, data pipeline, judges | CPU + Gemini API | Runs locally |
| RAG evaluation | CPU + Ollama (Qwen 2.5 7B) | Needs ~8 GB RAM for Qwen |
| E1/E2/E3 simulations | 2× T4 GPU (Kaggle) | 4-bit quantization, `device_map="auto"` |
| Fine-tuning (SFT/DPO) | 1× A100 GPU (Kaggle) | QLoRA, ~40 GB VRAM |

---

## End-to-End Pipeline — Execution Order

The project runs as a 7-phase pipeline. Each phase depends on the outputs of the previous one.

```
Phase 1: KG Construction
    ↓
Phase 2: Skeleton Graph Build
    ↓
Phase 3: Synthetic Data Pipeline
    ↓
Phase 4: Fine-Tuning (SFT + DPO)
    ↓
Phase 5: RAG Engine + Evaluation
    ↓
Phase 6: Evaluation Simulations (E1/E2/E3)
    ↓
Phase 7: Evaluation Judging (E1/E2/E3)
```

---

## Phase 1 — Knowledge Graph Construction

**Script:** `rag/build_dsa_graph.py`
**Runs on:** Local machine (CPU)
**Dependencies:** `google-genai`, `neo4j`, `PyMuPDF`
**Input:** `rag/Cormen Introduction to Algorithms.pdf`
**Output:** Neo4j graph database (cloud-hosted on Aura)

This script extracts DSA concepts from the CLRS textbook, chunking it into semantic sections, calling Gemini to extract concepts/relationships/misconceptions, and upserting them into Neo4j.

```bash
cd rag/
python build_dsa_graph.py
```

**Key configuration (top of file):**
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` — your Neo4j Aura credentials
- `GOOGLE_API_KEY` — your Gemini API key
- `MODEL_NAME` — `gemini-2.5-pro` (default) or `gemini-2.0-flash` (cheaper)
- `BATCH_SIZE` — chunks per API call (default: 5)
- `DELAY_SECONDS` — rate limiting delay (default: 1s)

**Resume support:** Progress is saved to `rag/progress.json`. Re-running the script resumes from the last processed chunk.

**Estimated cost:** ~$2–3 with `gemini-2.5-pro`, ~$0.20 with `gemini-2.0-flash`.

---

## Phase 2 — Skeleton Graph Build

**Script:** `USER/build_skeleton_graph.py`
**Runs on:** Local machine (CPU)
**Dependencies:** `neo4j`
**Input:** Neo4j KG (from Phase 1)
**Output:** `USER/skeleton_graph.json`

Queries Neo4j to resolve KG anchors for each of the 22 curriculum nodes and writes a self-contained JSON. After this runs, Neo4j is never needed at runtime.

```bash
cd USER/
python build_skeleton_graph.py
```

**Key configuration (bottom of file, `__main__` block):**
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` — replace the placeholders with your credentials

**Output:** `skeleton_graph.json` — contains 22 nodes across 4 tiers with embedded KG anchors (prerequisites, misconceptions, techniques).

---

## Phase 3 — Synthetic Data Pipeline

**Notebook:** `Data_Pipeline/data-pipeline.ipynb`
**Runs on:** Jupyter / Google Colab (CPU)
**Dependencies:** `google-genai`, `pandas`
**Input:** ConvoLearn CSV (embedded/downloaded in notebook)
**Output:** `sft_dialogues.json`, `dpo_pairs.json`

Three-phase pipeline that extracts pedagogical tactics from ConvoLearn (earth-science domain) and generates synthetic DSA-grounded training data.

```bash
# Open in Jupyter
jupyter notebook Data_Pipeline/data-pipeline.ipynb
```

Run cells sequentially:

| Phase | What It Does | Output |
|-------|-------------|--------|
| **Phase 1** — Tactic Extraction | Filters 2,134 ConvoLearn dialogues → 1,345 teacher turns, calls Gemini to extract Socratic tactics | 180 raw tactics, 177 unique labels |
| **Phase 1b** — Tactic Consolidation | Single Gemini clustering call: 177 labels → 20 canonical tactics | Canonical tactic bank |
| **Phase 2** — Dual-Agent Generation | Simulates Student + Tutor across 30 DSA concepts × 10 dialogues | `sft_dialogues.json` (300 samples), `dpo_pairs.json` (300 pairs) |

**API key:** Set `GEMINI_API_KEY` in the first code cell.

---

## Phase 4 — Fine-Tuning (SFT + DPO)

**Notebook:** `llama_finetuning.ipynb`
**Runs on:** Kaggle (1× A100 GPU recommended)
**Dependencies:** `transformers`, `peft`, `trl`, `bitsandbytes`, `datasets`
**Input:** `sft_dialogues.json`, `dpo_pairs.json` (from Phase 3)
**Output:** Fine-tuned model weights pushed to HuggingFace Hub

```bash
# Upload to Kaggle and run as notebook, or:
jupyter notebook llama_finetuning.ipynb
```

**Two-stage alignment:**

1. **SFT (QLoRA):** Fine-tunes Qwen 2.5 7B on the 300 SFT dialogues
   - Output: `Aryan3it/socratic-tutor-qwen2.5-7b` (merged weights on HF)

2. **DPO:** Trains preference model on 300 chosen/rejected pairs
   - Output: `Aryan3it/socratic-tutor-qwen2.5-7b_dpo_lora` (LoRA adapter on HF)

**Key configuration (in notebook cells):**
- `HF_TOKEN` — your HuggingFace write token
- Adjust `per_device_train_batch_size`, `gradient_accumulation_steps` for your GPU

---

## Phase 5 — RAG Engine & Evaluation

### 5a. RAG Engine (Library)

**Script:** `rag/rag_engine.py`
**Runs on:** Local machine
**Dependencies:** `neo4j`, `requests`, Ollama running locally
**Prerequisite:** Start Ollama with Qwen 2.5 7B

```bash
# Start Ollama (separate terminal)
ollama serve

# Pull the model (first time only)
ollama pull qwen2.5:7b
```

The RAG engine is a library, not a standalone script. It's imported by `rag/evaluate.py` and `USER/tutor-context.py`.

### 5b. RAG Evaluation

**Script:** `rag/evaluate.py`
**Runs on:** Local machine
**Dependencies:** Ollama running + Neo4j accessible
**Output:** `rag/evaluation_results.txt`

Runs 3 evaluation scenarios comparing RAG-augmented vs plain LLM answers:
- **S1:** Same question, two users with different prerequisite knowledge
- **S2:** Direct factual questions
- **S3:** Expert in X, beginner in Y (analogical bridging)

```bash
cd rag/
python evaluate.py
```

Results are printed to stdout and saved to `evaluation_results.txt`.

### 5c. Socratic Tutor Runtime

**Script:** `USER/tutor-context.py`
**Runs on:** Local machine
**Dependencies:** Ollama running, `USER/skeleton_graph.json`

Interactive tutor that builds KG-grounded context from the user's JSON and calls Qwen.

```bash
cd USER/
python tutor-context.py
```

This runs the example flow:
1. Creates a user `diana` with preset mastery scores
2. Asks "explain dynamic programming"
3. Prints the context-aware Socratic response
4. Updates mastery by +0.10 on the `sg_dp` node

### 5d. User Graph Management

**Script:** `USER/user_sg.py`
**Runs on:** Local machine
**Dependencies:** `USER/skeleton_graph.json`

Library for creating and managing per-user mastery state:

```python
from USER.user_sg import create_user_sg, load_user_sg, update_mastery, learning_frontier

# Create a new user (all mastery = 0.0)
create_user_sg("alice")

# Update mastery after a session
update_mastery("alice", "sg_arrays", +0.15)

# Get topics the user is ready to learn next
user = load_user_sg("alice")
frontier = learning_frontier(user)
```

---

## Phase 6 — Code Evaluator

**Package:** `evaluator/`
**Runs on:** Local machine
**Dependencies:** `tree-sitter`, `google-genai`

Hybrid AST + LLM evaluator that scores student code against the 22-node SG.

```python
from evaluator import evaluate_code

result = evaluate_code(
    username="diana",
    question_id="q_dijkstra",
    code=student_code_string,
    apply_updates=True,   # writes mastery updates to user JSON
    verbose=True,
)
```

Or run the built-in sample:

```bash
python -m evaluator.evaluator
```

**Pipeline steps:**
1. Parse code with tree-sitter → extract AST signals (recursion, loop depth, imports, patterns)
2. Detect algorithm patterns (BFS, DFS, Dijkstra, DP, greedy, etc.)
3. Build LLM rubric prompt with 5-tier hierarchical scoring
4. Call Gemini evaluator → per-node mastery scores
5. Apply rule correction ("LLM proposes, AST constrains")
6. Blend new scores with existing mastery (60/40 recency weight)

See `evaluator/README.md` for detailed pipeline documentation.

---

## Phase 7 — Evaluation Framework (E1 / E2 / E3)

All evaluation scripts live in `evaluation/`. Each evaluation has 3 stages:

```
Stage 1: Generate profiles/questions/scenarios  (local, Gemini API)
Stage 2: Run simulations with tutor models       (Kaggle GPU)
Stage 3: Judge responses with LLM judge           (local, Gemini API)
```

### E1 — Prerequisite Gate Enforcement

Tests whether the tutor correctly detects knowledge gaps and gates content.

#### Stage 1: Generate Profiles

```bash
python evaluation/e1_profiles_generate.py
```

- **Input:** `USER/skeleton_graph.json`
- **Output:** `evaluation/e1_profiles.json`
- **Produces:** 12 synthetic user profiles across 3 archetypes (A=beginner, B=partial knowledge, C=full prerequisites)

#### Stage 2: Run Simulation (Kaggle)

```bash
# Run on Kaggle with 2× T4 GPUs
python evaluation/e1_simulation.py
```

- **Input:** `e1_profiles.json`
- **Output:** `e1_responses.json`
- **Models tested:** Qwen Base (no RAG), Qwen Base (RAG), Qwen SFT, Qwen DPO
- **Config:** Set `DRY_RUN = False` for full run, `HF_TOKEN` for model access
- **Install on Kaggle:**
  ```
  !pip install -U bitsandbytes>=0.46.1 --no-deps --quiet
  !pip install transformers>=4.45.0 peft>=0.13.0 accelerate einops huggingface_hub --upgrade --quiet
  ```

#### Stage 3: Judge Responses

```bash
python evaluation/e1_judge.py
```

- **Input:** `e1_responses-qwenbase.json` (or `-rag-sft-dpo.json`)
- **Output:** `e1_results-*.json`, `e1_metrics-*.json`, heatmap PNGs
- **Evaluates:** C1–C5 checklist (gap detection, prereq-first, gating, acknowledgment, no re-teaching)
- **Config:** Update `RESPONSES_PATH`, `RESULTS_PATH` etc. at top of file to match your input/output filenames

---

### E2 — Socratic Answer-Leak Detection

Tests whether the tutor resists "just tell me the answer" pressure across 10-turn conversations.

#### Stage 1: Generate Question Bank

```bash
python evaluation/generate_questions.py
```

- **Input:** None (generates from Gemini)
- **Output:** `evaluation/e2_question_bank.json`
- **Produces:** 100 DSA questions across 3 categories (I=direct pressure, II=persistent confusion, III=almost-there) and 8 topics

#### Stage 2: Run Multi-Turn Simulation (Kaggle)

```bash
# Run on Kaggle with 2× T4 GPUs
python evaluation/e2_simulation.py
```

- **Input:** `e2_question_bank.json`
- **Output:** `e2_conversations_<model_name>.json` (one per model)
- **Student agent:** Gemini 2.5 Flash simulates 3 student archetypes
- **Tutor models:** Qwen Base, Qwen SFT, Qwen DPO
- **Config:** Set `DRY_RUN = False`, `GEMINI_API_KEY`, `HF_TOKEN`
- **Resume support:** Saves `e2_conversations_<model>_partial.json` after each conversation

#### Stage 3: Judge Leak Levels

```bash
python evaluation/e2_judge.py
```

- **Input:** `e2_conversations_<model_name>.json`
- **Output:** `e2_results_<model>.json`, `e2_metrics_<model>.json`, escalation profile PNGs
- **Classifies:** Each tutor turn as L0 (pure Socratic) → L3 (full leak)
- **Config:** Update `MODEL_NAMES` list to match your model files. Uncomment models as needed.

---

### E3 — Learning Sciences Principle Evaluation

Tests tutor responses against 20 learning sciences principles (P1–P20).

#### Stage 1: Generate Scenarios

```bash
python evaluation/e3_generate_scenarios.py
```

- **Input:** None (generates from Gemini)
- **Output:** `evaluation/e3_scenarios.json`
- **Produces:** 40 scenarios (8 topics × 5 student profiles: beginner, partial, confused, overconfident, strong-with-gap)

#### Stage 2: Run Simulation (Kaggle)

```bash
# Run on Kaggle with 2× T4 GPUs
python evaluation/e3_simulation.py
```

- **Input:** `e3_scenarios.json`
- **Output:** `e3_conversations_<model_name>.json`
- **Models:** Qwen SFT, Qwen DPO
- **Config:** Set `DRY_RUN = False`, `GEMINI_API_KEY`, `HF_TOKEN`

#### Stage 3: Judge Principles

```bash
python evaluation/e3_judge.py
```

- **Input:** `e3_conversations_<model_name>.json`
- **Output:** `e3_results_<model>.json`, `e3_metrics_<model>.json`, heatmap + archetype + gap analysis PNGs
- **Scores:** Each tutor turn on 20 principles (0=absent, 1=weak, 2=strong) + sequencing appropriateness
- **Config:** Update `MODEL_NAMES` list to match your model files

---

## Repository Map

```
Structural_Memory/
├── rag/                          # Knowledge Graph + RAG Engine
│   ├── build_dsa_graph.py        # Phase 1: KG construction from CLRS PDF
│   ├── rag_engine.py             # RAG engine (Neo4j → context → Ollama)
│   ├── evaluate.py               # RAG evaluation (3 scenarios)
│   ├── Cormen Introduction to Algorithms.pdf
│   ├── progress.json             # KG build resume state
│   └── evaluation_results.txt    # RAG evaluation output
│
├── USER/                         # Skeleton Graph + User State
│   ├── build_skeleton_graph.py   # Phase 2: SG construction from KG
│   ├── skeleton_graph.json       # 22-node curriculum graph (self-contained)
│   ├── user_sg.py                # User graph CRUD operations
│   ├── tutor-context.py          # Runtime Socratic tutor
│   └── readme.md                 # SG documentation
│
├── Data_Pipeline/                # Synthetic Data Pipeline
│   └── data-pipeline.ipynb       # Phase 3: Tactic extraction + dialogue generation
│
├── evaluator/                    # Code Evaluator (AST + LLM)
│   ├── __init__.py               # Public API: evaluate_code()
│   ├── evaluator.py              # Pipeline orchestrator
│   ├── ast_analyzer.py           # tree-sitter AST analysis
│   ├── pattern_detector.py       # Algorithm pattern detection
│   ├── llm_evaluator.py          # Gemini rubric scoring
│   ├── prompts.py                # LLM prompt templates
│   ├── question_bank.py          # Question definitions
│   └── README.md                 # Evaluator documentation
│
├── evaluation/                   # Evaluation Framework (E1/E2/E3)
│   ├── e1_profiles_generate.py   # E1 Stage 1: profile generation
│   ├── e1_simulation.py          # E1 Stage 2: simulation (Kaggle)
│   ├── e1_judge.py               # E1 Stage 3: judge
│   ├── generate_questions.py     # E2 Stage 1: question bank
│   ├── e2_simulation.py          # E2 Stage 2: multi-turn sim (Kaggle)
│   ├── e2_judge.py               # E2 Stage 3: leak judge
│   ├── e3_generate_scenarios.py  # E3 Stage 1: scenario generation
│   ├── e3_simulation.py          # E3 Stage 2: simulation (Kaggle)
│   ├── e3_judge.py               # E3 Stage 3: principle judge
│   └── *.json                    # Generated profiles, results, metrics
│
├── llama_finetuning.ipynb        # Phase 4: SFT + DPO fine-tuning
├── structural_memory_acl.tex     # ACL-style research paper
├── references.bib                # BibTeX references
├── docs/                         # Report PDFs
└── rough-docs/                   # Working notes
```

---

## Quick Start (Minimal Demo)

If you just want to see the tutor in action without running the full pipeline:

```bash
# 1. Install dependencies
pip install requests

# 2. Start Ollama with Qwen
ollama serve &
ollama pull qwen2.5:7b

# 3. Run the tutor demo (uses pre-built skeleton_graph.json)
cd USER/
python tutor-context.py
```

This creates user `diana`, sets up her mastery profile, and generates a personalized Socratic response to "explain dynamic programming".

---


