"""
evaluate.py
───────────
Runs 3 evaluation scenarios from res.txt and prints a side-by-side
comparison report of answers WITH vs WITHOUT RAG.
Results are saved to evaluation_results.txt automatically.

Scenarios
─────────
S1 – Two users, same question on a topic that has prerequisites.
     User A knows the prereqs.  User B does NOT.
     Expected: RAG answer warns User B, plain LLM doesn't.

S2 – Direct factual questions answered with and without RAG.
     Expected: RAG answer is more precise, CLRS-grounded, misconception-aware.

S3 – User with high proficiency in topic X asks about topic Y (the next topic).
     Expected: RAG explains Y using X as an anchor / bridge.
"""

import sys
import textwrap
from datetime import datetime
from rag_engine import (
    KnowledgeGraph,
    answer_with_rag,
    OLLAMA_MODEL,
    LEVEL_BEGINNER,
    LEVEL_INTERMEDIATE,
    LEVEL_ADVANCED,
)

OUTPUT_FILE = "evaluation_results.txt"

# ── Tee writer — prints to screen AND collects for file ───────────────────────
class Tee:
    def __init__(self):
        self._lines: list[str] = []

    def write(self, text: str) -> None:
        sys.__stdout__.write(text)
        self._lines.append(text)

    def flush(self) -> None:
        sys.__stdout__.flush()

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(self._lines)
        sys.__stdout__.write(f"\n📄 Results saved to: {path}\n")

# ── Pretty printer ─────────────────────────────────────────────────────────────
SEP  = "─" * 72
SEP2 = "═" * 72

def wrap(text: str, width: int = 68, indent: str = "  ") -> str:
    lines = text.splitlines()
    wrapped = []
    for line in lines:
        if len(line) > width:
            wrapped.extend(textwrap.wrap(line, width, initial_indent=indent,
                                         subsequent_indent=indent))
        else:
            wrapped.append(indent + line)
    return "\n".join(wrapped)

def print_comparison(result: dict, scenario_label: str) -> None:
    print(f"\n{SEP2}")
    print(f"  {scenario_label}")
    print(f"  User     : {result['user']}  |  Level: {result['level'].upper()}")
    print(f"  Question : {result['question']}")
    print(SEP2)

    print(f"\n  ── WITHOUT RAG (Plain {OLLAMA_MODEL}) ──────────────────────────")
    print(wrap(result["without_rag"] or "[No response]"))

    print(f"\n{SEP}")
    print(f"\n  ── WITH RAG (KG + {OLLAMA_MODEL}) ────────────────────────────")
    print(wrap(result["with_rag"] or "[No response]"))
    print(f"\n{SEP}\n")

# ── User profiles ──────────────────────────────────────────────────────────────
# Score 0-10 maps to: 0-3 = beginner, 4-6 = intermediate, 7-10 = advanced
def score_to_level(score: int) -> str:
    if score <= 3:   return LEVEL_BEGINNER
    elif score <= 6: return LEVEL_INTERMEDIATE
    else:            return LEVEL_ADVANCED

# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 1
# Two users ask about "Red-Black Tree".
# User A (score 8) knows BSTs, rotations, sorting.
# User B (score 2) knows nothing about trees.
# ─────────────────────────────────────────────────────────────────────────────
def scenario_1(kg: KnowledgeGraph) -> None:
    print(f"\n{'#'*72}")
    print("  SCENARIO 1 — Same question, different prerequisite knowledge")
    print(f"{'#'*72}")

    question = "How does a Red-Black Tree maintain balance after insertion?"
    topic    = "red-black tree"

    user_a = {
        "name":         "Alice (knows prerequisites)",
        "score":        8,
        "level":        score_to_level(8),
        "known_topics": ["Binary Search Tree", "Tree Rotation",
                         "Sorting", "Balanced Tree", "Binary Tree"],
    }

    user_b = {
        "name":         "Bob (does NOT know prerequisites)",
        "score":        2,
        "level":        score_to_level(2),
        "known_topics": [],   # knows nothing
    }

    for user in [user_a, user_b]:
        result = answer_with_rag(question, topic, user, kg)
        print_comparison(result, f"SCENARIO 1 — {user['name']}")


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 2
# Direct factual questions — compare plain LLM vs RAG-grounded answer
# ─────────────────────────────────────────────────────────────────────────────
def scenario_2(kg: KnowledgeGraph) -> None:
    print(f"\n{'#'*72}")
    print("  SCENARIO 2 — Direct questions: Plain LLM vs KG-grounded RAG")
    print(f"{'#'*72}")

    questions = [
        ("What is the time complexity of Dijkstra's algorithm?",   "dijkstra",          "intermediate"),
        ("Explain dynamic programming with an example.",            "dynamic programming", "beginner"),
        ("What is amortized analysis and when do you use it?",     "amortized analysis",  "advanced"),
    ]

    user = {
        "name":         "Charlie (intermediate student)",
        "score":        5,
        "level":        LEVEL_INTERMEDIATE,
        "known_topics": ["Arrays", "Recursion", "Sorting", "Graph"],
    }

    for question, topic, level_override in questions:
        user["level"] = level_override
        result = answer_with_rag(question, topic, user, kg)
        print_comparison(result, f"SCENARIO 2 — Q: {question[:50]}…")


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 3
# User is expert in Merge Sort (score 9), but new to Dynamic Programming.
# RAG should explain DP using Merge Sort / divide-and-conquer as a bridge.
# ─────────────────────────────────────────────────────────────────────────────
def scenario_3(kg: KnowledgeGraph) -> None:
    print(f"\n{'#'*72}")
    print("  SCENARIO 3 — Expert in topic X, new to topic Y")
    print("               RAG bridges explanation using known knowledge")
    print(f"{'#'*72}")

    question = (
        "I understand Merge Sort very well. "
        "Can you explain Dynamic Programming to me in a way that connects "
        "to what I already know?"
    )
    topic = "dynamic programming"

    user = {
        "name":         "Diana (Merge Sort expert, DP beginner)",
        "score":        2,           # low score on DP specifically
        "level":        LEVEL_BEGINNER,
        "known_topics": [
            "Merge Sort", "Divide and Conquer", "Recursion",
            "Sorting", "Time Complexity", "Recurrence",
        ],
    }

    result = answer_with_rag(question, topic, user, kg)
    print_comparison(result, "SCENARIO 3 — Expert in X, new to Y")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tee = Tee()
    sys.stdout = tee   # redirect all print() output through Tee

    print(SEP2)
    print("  RAG EVALUATION — KG-Augmented Qwen vs Plain Qwen")
    print(f"  Model  : {OLLAMA_MODEL}")
    print(f"  Run at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Knowledge Graph source: CLRS Introduction to Algorithms")
    print(SEP2)

    kg = KnowledgeGraph()
    try:
        scenario_1(kg)
        scenario_2(kg)
        scenario_3(kg)
    finally:
        kg.close()

    print(f"\n{SEP2}")
    print("  Evaluation complete.")
    print(SEP2)

    sys.stdout = sys.__stdout__   # restore stdout before saving
    tee.save(OUTPUT_FILE)
