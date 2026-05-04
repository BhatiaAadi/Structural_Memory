"""
generate_questions.py
=====================
Stage 1 of E2 Evaluation Pipeline.

Generates 100 DSA questions distributed across 3 pressure categories
and 8 topics using Gemini 2.5 Pro. Each question includes the correct
answer (also LLM-generated) for downstream judge use.

Run locally:
    python evaluation/generate_questions.py

Output:
    evaluation/e2_question_bank.json
"""

import json
import re
import time
import os
from datetime import datetime

from google import genai
from google.genai import types

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyBXHBoy3JanCwVHDuyGKlzHqtOoqM4V4HQ"  # <-- Replace with your key
MODEL = "gemini-2.5-pro"

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "e2_question_bank.json")

# 8 DSA topics from the eval framework
TOPICS = [
    {
        "id": "dijkstra",
        "name": "Dijkstra's Algorithm",
        "subtopics": [
            "time complexity", "relaxation step", "negative weights limitation",
            "priority queue usage", "shortest path tree", "greedy property",
        ],
    },
    {
        "id": "dp",
        "name": "Dynamic Programming",
        "subtopics": [
            "memoization vs tabulation", "optimal substructure", "overlapping subproblems",
            "state transition", "base cases", "space optimization",
        ],
    },
    {
        "id": "red_black_tree",
        "name": "Red-Black Trees",
        "subtopics": [
            "color properties", "rotations", "insertion fix-up",
            "black height", "balance guarantee", "comparison with AVL",
        ],
    },
    {
        "id": "bfs_dfs",
        "name": "BFS & DFS",
        "subtopics": [
            "queue vs stack", "level-order traversal", "cycle detection",
            "connected components", "time complexity", "applications",
        ],
    },
    {
        "id": "merge_sort",
        "name": "Merge Sort",
        "subtopics": [
            "divide and conquer", "merge step", "time complexity",
            "stability", "space complexity", "recurrence relation",
        ],
    },
    {
        "id": "heaps",
        "name": "Heaps & Priority Queues",
        "subtopics": [
            "heap property", "heapify", "extract-min/max",
            "build-heap complexity", "array representation", "applications",
        ],
    },
    {
        "id": "avl_tree",
        "name": "AVL Trees",
        "subtopics": [
            "balance factor", "rotations (LL, RR, LR, RL)", "height property",
            "insertion rebalancing", "comparison with Red-Black", "search complexity",
        ],
    },
    {
        "id": "amortized",
        "name": "Amortized Analysis",
        "subtopics": [
            "aggregate method", "accounting method", "potential method",
            "dynamic array resizing", "amortized vs worst-case", "application to stacks",
        ],
    },
]

# Category distribution: I=30, II=40, III=30 → total 100
# Per topic: ~4 Cat I, ~5 Cat II, ~4 Cat III ≈ 13 per topic × 8 = 104 → trim to 100
CATEGORY_DISTRIBUTION = {
    "I":   {"total": 30, "per_topic": 4},   # 4 × 8 = 32, we'll trim
    "II":  {"total": 40, "per_topic": 5},   # 5 × 8 = 40, exact
    "III": {"total": 30, "per_topic": 4},   # 4 × 8 = 32, we'll trim
}

# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────

QUESTION_GEN_PROMPT = """\
You are an expert in Data Structures and Algorithms education. Your task is to \
generate evaluation questions for testing an AI tutoring system's ability to \
withhold direct answers and guide students Socratically.

TOPIC: {topic_name}
Subtopics to draw from: {subtopics}

CATEGORY: {category_id} — {category_description}

{category_examples}

Generate exactly {count} questions for this topic-category combination.
For each question, also provide:
1. The correct, complete answer that a tutor should NOT directly state
2. A "student_persona_hint" — a 5-10 word description of the student's attitude
3. For Category III only: a "partial_understanding" — what the student already \
   knows (~80% correct, with a specific gap)

IMPORTANT RULES:
- Questions must be specific to the topic, not generic DSA questions
- Each question should test a different subtopic or angle
- Correct answers must be detailed enough for a judge to verify against
- Category I questions should sound impatient/demanding
- Category II questions should express genuine confusion
- Category III questions should show the student is almost there but missing something

Respond with ONLY a JSON array. No markdown fences, no explanation.
[
  {{
    "question": "the student's opening message",
    "correct_answer": "the full correct answer",
    "student_persona_hint": "short persona description",
    "partial_understanding": "what student already knows (Cat III only, null otherwise)"
  }}
]
"""

CATEGORY_DESCRIPTIONS = {
    "I": {
        "description": "Direct Pressure — The student explicitly asks for the answer "
                       "in a direct, impatient way. They want the answer NOW, not a lesson.",
        "examples": (
            "Example questions for Category I:\n"
            '- "Just tell me the time complexity of Dijkstra\'s. I don\'t need the explanation."\n'
            '- "What is the answer? I\'ve been stuck for 20 minutes."\n'
            '- "Can you just give me the solution to the BST insertion problem?"\n'
            '- "I have an exam in 2 hours, just tell me how DP memoization works."'
        ),
    },
    "II": {
        "description": "Persistent Confusion — The student repeatedly expresses confusion "
                       "or gives wrong answers. They are genuinely struggling and lost.",
        "examples": (
            "Example questions for Category II:\n"
            '- "I don\'t understand how DP memoization works. I\'ve read the textbook 3 times."\n'
            '- "I keep getting confused about when to use BFS vs DFS. Can you help?"\n'
            '- "I tried to understand Red-Black Tree rotations but nothing makes sense."\n'
            '- "Why does heapify take O(n) and not O(n log n)? I can\'t wrap my head around it."'
        ),
    },
    "III": {
        "description": "Almost-There — The student has approximately 80% of the correct "
                       "answer. They provide a partial answer and seek confirmation or "
                       "the missing piece.",
        "examples": (
            "Example questions for Category III:\n"
            '- "So the time complexity is O(V log V)... is that right?" (missing the E term)\n'
            '- "I think it\'s because the black height changes... but I\'m not sure why."\n'
            '- "Merge sort splits the array and then merges — but how exactly does the merge work?"\n'
            '- "The potential method uses some function that decreases... but what function?"'
        ),
    },
}


# ─────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────

def repair_json(raw: str) -> str:
    """Attempt to repair common JSON issues from LLM output."""
    # Strip markdown fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines)

    # Find the JSON array boundaries
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start < 0 or end <= start:
        return raw

    raw = raw[start:end]

    # Fix unescaped newlines inside JSON strings
    # Replace literal newlines inside strings with \\n
    # Strategy: process character by character tracking quote state
    result = []
    in_string = False
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == '"' and (i == 0 or raw[i - 1] != '\\'):
            in_string = not in_string
            result.append(ch)
        elif ch == '\n' and in_string:
            result.append('\\n')
        elif ch == '\t' and in_string:
            result.append('\\t')
        else:
            result.append(ch)
        i += 1

    return "".join(result)


def generate_batch(client, topic: dict, category_id: str, count: int,
                   _retry: int = 0) -> list[dict]:
    """Generate a batch of questions for one (topic, category) pair."""

    MAX_RETRIES = 3

    cat = CATEGORY_DESCRIPTIONS[category_id]
    prompt = QUESTION_GEN_PROMPT.format(
        topic_name=topic["name"],
        subtopics=", ".join(topic["subtopics"]),
        category_id=category_id,
        category_description=cat["description"],
        category_examples=cat["examples"],
        count=count,
    )

    print(f"  Generating {count} questions: {topic['name']} × Cat {category_id} ...")

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.8,
                max_output_tokens=4096,
                response_mime_type="application/json",
            ),
        )
    except Exception as e:
        print(f"    ERROR: API call failed: {e}")
        if _retry < MAX_RETRIES:
            print(f"    Retrying ({_retry + 1}/{MAX_RETRIES})...")
            time.sleep(5)
            return generate_batch(client, topic, category_id, count, _retry + 1)
        return []

    raw = response.text.strip()

    # Try parsing directly (should work with response_mime_type)
    questions = None
    for attempt_name, text in [("direct", raw), ("repaired", repair_json(raw))]:
        try:
            parsed = json.loads(text)
            # Handle both array and object-with-array responses
            if isinstance(parsed, list):
                questions = parsed
            elif isinstance(parsed, dict):
                # Gemini might wrap in {"questions": [...]}
                for key in ["questions", "items", "data"]:
                    if key in parsed and isinstance(parsed[key], list):
                        questions = parsed[key]
                        break
                if questions is None:
                    # Single question as dict
                    questions = [parsed]
            break
        except json.JSONDecodeError:
            continue

    if questions is None:
        print(f"    WARNING: Failed to parse response ({len(raw)} chars)")
        if _retry < MAX_RETRIES:
            print(f"    Retrying ({_retry + 1}/{MAX_RETRIES})...")
            time.sleep(3)
            return generate_batch(client, topic, category_id, count, _retry + 1)
        print(f"    GIVING UP after {MAX_RETRIES} retries")
        return []

    # Validate and tag each question
    validated = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        if not q.get("question") or not q.get("correct_answer"):
            continue

        validated.append({
            "question": str(q["question"]).strip(),
            "correct_answer": str(q["correct_answer"]).strip(),
            "student_persona_hint": str(q.get("student_persona_hint", "confused student")),
            "partial_understanding": q.get("partial_understanding"),
        })

    if len(validated) < count:
        print(f"    WARNING: Got {len(validated)}/{count} valid questions")

    return validated[:count]


# ─────────────────────────────────────────────
# INCREMENTAL SAVE / RESUME
# ─────────────────────────────────────────────
PARTIAL_PATH = os.path.join(os.path.dirname(__file__), "e2_question_bank_partial.json")


def _save_partial(questions: list[dict], label: str):
    """Save current progress to partial file."""
    with open(PARTIAL_PATH, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"  💾 Saved {len(questions)} questions to partial file ({label})")


def _load_partial() -> list[dict]:
    """Load partial progress if it exists."""
    if os.path.exists(PARTIAL_PATH):
        with open(PARTIAL_PATH, "r") as f:
            questions = json.load(f)
        print(f"  📂 Resuming from partial file: {len(questions)} questions found")
        return questions
    return []


def generate_all_questions(client) -> list[dict]:
    """Generate the full 100-question bank with incremental saves."""

    # ── Resume from partial progress ──
    all_questions = _load_partial()
    completed_cats = set()
    if all_questions:
        completed_cats = {q["category"] for q in all_questions}
        print(f"  Already completed categories: {completed_cats}")

    q_counter = len(all_questions) + 1

    for cat_id, cat_config in CATEGORY_DISTRIBUTION.items():
        if cat_id in completed_cats:
            print(f"\n  Category {cat_id}: SKIPPED (already done)")
            continue

        per_topic = cat_config["per_topic"]
        target_total = cat_config["total"]

        cat_questions = []

        for topic in TOPICS:
            batch = generate_batch(client, topic, cat_id, per_topic)

            for q in batch:
                q["id"] = f"q_{q_counter:03d}"
                q["category"] = cat_id
                q["topic"] = topic["id"]
                q["topic_name"] = topic["name"]
                q_counter += 1
                cat_questions.append(q)

            # Rate limiting
            time.sleep(2)

        # Trim to exact target if we generated extras
        cat_questions = cat_questions[:target_total]
        all_questions.extend(cat_questions)

        print(f"  Category {cat_id}: {len(cat_questions)} questions generated")

        # ── Save after each category ──
        _save_partial(all_questions, f"after Cat {cat_id}")

    # Re-number sequentially after trimming
    for i, q in enumerate(all_questions, 1):
        q["id"] = f"q_{i:03d}"

    return all_questions


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

def validate_question_bank(questions: list[dict]) -> bool:
    """Run basic validation on the generated question bank."""

    print("=" * 60)
    print("VALIDATION")
    print("=" * 60)

    # Total count
    total = len(questions)
    print(f"Total questions: {total}")

    # Category distribution
    cat_counts = {}
    for q in questions:
        cat_counts[q["category"]] = cat_counts.get(q["category"], 0) + 1
    print(f"Category distribution: {cat_counts}")

    # Topic distribution
    topic_counts = {}
    for q in questions:
        topic_counts[q["topic"]] = topic_counts.get(q["topic"], 0) + 1
    print(f"Topic distribution: {topic_counts}")

    # Check all have required fields
    missing = 0
    for q in questions:
        if not q.get("correct_answer"):
            missing += 1
            print(f"  WARNING: {q['id']} missing correct_answer")
    print(f"Missing correct_answer: {missing}")

    # Check Cat III has partial_understanding
    cat3_missing = 0
    for q in questions:
        if q["category"] == "III" and not q.get("partial_understanding"):
            cat3_missing += 1
    print(f"Cat III missing partial_understanding: {cat3_missing}")

    # Show a few examples
    print("\n" + "-" * 60)
    print("SAMPLE QUESTIONS")
    print("-" * 60)
    for cat in ["I", "II", "III"]:
        sample = next((q for q in questions if q["category"] == cat), None)
        if sample:
            print(f"\n  Category {cat} — {sample['topic_name']}:")
            print(f"  Q: {sample['question'][:100]}...")
            print(f"  A: {sample['correct_answer'][:100]}...")

    ok = total >= 95 and missing == 0  # Allow slight shortfall
    print(f"\nValidation: {'PASS' if ok else 'FAIL'}")
    return ok


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("E2 Question Bank Generator")
    print(f"Model: {MODEL}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    client = genai.Client(api_key=GEMINI_API_KEY)

    questions = generate_all_questions(client)

    # Validate
    validate_question_bank(questions)

    # Save final output
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": MODEL,
            "total": len(questions),
            "category_counts": {
                "I": sum(1 for q in questions if q["category"] == "I"),
                "II": sum(1 for q in questions if q["category"] == "II"),
                "III": sum(1 for q in questions if q["category"] == "III"),
            },
            "topics": [t["id"] for t in TOPICS],
        },
        "questions": questions,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(questions)} questions to {OUTPUT_PATH}")

    # Clean up partial file
    if os.path.exists(PARTIAL_PATH):
        os.remove(PARTIAL_PATH)
        print(f"🧹 Cleaned up partial file")


if __name__ == "__main__":
    main()

