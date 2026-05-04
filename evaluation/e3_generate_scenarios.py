"""
generate_e3_scenarios.py
========================
Stage 1 of E3 Evaluation Pipeline.

Generates 40 conversation scenarios: 8 DSA topics × 5 student profiles.
Uses Gemini to generate varied opening questions and persona details.
The structure is largely deterministic — the LLM adds flavor, not structure.

Run locally:
    python evaluation/generate_e3_scenarios.py

Output:
    evaluation/e3_scenarios.json
"""

import json
import time
import os
from datetime import datetime

from google import genai
from google.genai import types

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyBXHBoy3JanCwVHDuyGKlzHqtOoqM4V4HQ"  # <-- Replace with your key
MODEL = "gemini-2.5-flash"  # Flash is sufficient for scenario generation

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "e3_scenarios.json")

# ─────────────────────────────────────────────
# 8 DSA TOPICS
# ─────────────────────────────────────────────
TOPICS = [
    {
        "id": "red_black_tree",
        "name": "Red-Black Trees",
        "core_concept": "Self-balancing BST with color properties and rotations",
        "prerequisites": ["binary search trees", "tree rotations", "tree traversals"],
        "key_subtopics": ["color properties", "rotations", "insertion fix-up", "black height"],
        "common_confusion": "Confusing Red-Black tree rotations with AVL rotations",
        "missing_concept_option": "why double-red violations require uncle-checking",
    },
    {
        "id": "dijkstra",
        "name": "Dijkstra's Algorithm",
        "core_concept": "Greedy shortest-path algorithm using a priority queue",
        "prerequisites": ["graphs", "adjacency lists", "priority queues", "BFS"],
        "key_subtopics": ["relaxation", "priority queue", "negative weights", "greedy property"],
        "common_confusion": "Thinking Dijkstra works with negative edge weights",
        "missing_concept_option": "the relaxation step and why it guarantees optimality",
    },
    {
        "id": "dp",
        "name": "Dynamic Programming",
        "core_concept": "Solving problems by combining solutions to overlapping subproblems",
        "prerequisites": ["recursion", "memoization basics", "divide and conquer"],
        "key_subtopics": ["optimal substructure", "overlapping subproblems", "tabulation", "state transition"],
        "common_confusion": "Confusing DP with divide-and-conquer (missing overlap requirement)",
        "missing_concept_option": "how to identify the state transition function",
    },
    {
        "id": "amortized",
        "name": "Amortized Analysis",
        "core_concept": "Averaging cost over a sequence of operations rather than per-operation worst case",
        "prerequisites": ["asymptotic complexity", "arrays", "stacks"],
        "key_subtopics": ["aggregate method", "accounting method", "potential method", "dynamic arrays"],
        "common_confusion": "Thinking amortized = average-case analysis",
        "missing_concept_option": "the potential method and how to choose a potential function",
    },
    {
        "id": "avl_tree",
        "name": "AVL Trees",
        "core_concept": "Height-balanced BST with balance factor constraint and rotations",
        "prerequisites": ["binary search trees", "tree height", "tree rotations"],
        "key_subtopics": ["balance factor", "LL/RR/LR/RL rotations", "height maintenance", "rebalancing"],
        "common_confusion": "Not understanding when double rotation (LR/RL) is needed vs single",
        "missing_concept_option": "how LR and RL double rotations work mechanically",
    },
    {
        "id": "merge_sort",
        "name": "Merge Sort",
        "core_concept": "Divide-and-conquer sorting by splitting and merging sorted subarrays",
        "prerequisites": ["arrays", "recursion", "divide and conquer concept"],
        "key_subtopics": ["divide step", "merge step", "stability", "space complexity", "recurrence"],
        "common_confusion": "Not understanding why the merge step is O(n) or why it's stable",
        "missing_concept_option": "how the merge step preserves order and achieves O(n)",
    },
    {
        "id": "heaps",
        "name": "Heaps & Priority Queues",
        "core_concept": "Complete binary tree with heap property, used for priority-based access",
        "prerequisites": ["arrays", "binary trees", "asymptotic complexity"],
        "key_subtopics": ["heap property", "heapify", "build-heap", "extract-min/max", "array representation"],
        "common_confusion": "Thinking build-heap is O(n log n) instead of O(n)",
        "missing_concept_option": "why build-heap is O(n) despite calling heapify n/2 times",
    },
    {
        "id": "bfs_dfs",
        "name": "Graph BFS/DFS",
        "core_concept": "Systematic graph traversal using queue (BFS) or stack/recursion (DFS)",
        "prerequisites": ["graphs", "adjacency lists", "stacks", "queues"],
        "key_subtopics": ["queue vs stack", "level-order", "cycle detection", "connected components"],
        "common_confusion": "Not knowing when to use BFS vs DFS, or confusing their traversal order",
        "missing_concept_option": "how BFS guarantees shortest path in unweighted graphs",
    },
]

# ─────────────────────────────────────────────
# 5 STUDENT PROFILES
# ─────────────────────────────────────────────
PROFILES = [
    {
        "id": 1,
        "name": "complete_beginner",
        "label": "Complete Beginner",
        "description_template": (
            "You know absolutely nothing about {topic_name}. You have never "
            "encountered this concept before. You need everything explained from scratch."
        ),
        "known_topics_rule": "empty",  # knows nothing about the topic
        "persona_hint": "Lost, overwhelmed, asks very basic questions",
        "opening_style": "What is {topic_name}? I've never heard of it before.",
    },
    {
        "id": 2,
        "name": "partial_knowledge",
        "label": "Partial Knowledge",
        "description_template": (
            "You understand the prerequisites ({prereqs}) but have not learned "
            "{topic_name} itself. You can connect new information to what you know."
        ),
        "known_topics_rule": "prerequisites",  # knows prerequisites
        "persona_hint": "Has foundation, asks focused follow-up questions",
        "opening_style": "I understand {prereq_example} but I'm not sure how {topic_name} builds on that. Can you help?",
    },
    {
        "id": 3,
        "name": "conceptually_confused",
        "label": "Conceptually Confused",
        "description_template": (
            "You can write code that looks correct but your mental model of {topic_name} "
            "is wrong. Specifically, you confuse: {confusion}. You give plausible-sounding "
            "but incorrect explanations."
        ),
        "known_topics_rule": "prerequisites_plus_partial",
        "persona_hint": "Sounds confident but explanations are subtly wrong",
        "opening_style": "I think I understand {topic_name} — {wrong_explanation}. Is that right?",
    },
    {
        "id": 4,
        "name": "overconfident",
        "label": "Overconfident",
        "description_template": (
            "You believe you fully understand {topic_name} but you have significant gaps. "
            "You assert wrong answers confidently and push back when corrected. "
            "Your specific misconception: {confusion}."
        ),
        "known_topics_rule": "prerequisites_plus_partial",
        "persona_hint": "Confident, slightly dismissive, pushes back on corrections",
        "opening_style": "I already know {topic_name} pretty well. {wrong_assertion}. Right?",
    },
    {
        "id": 5,
        "name": "strong_with_gap",
        "label": "Analytically Strong, Missing One Concept",
        "description_template": (
            "You have a strong understanding of {topic_name} overall, but you are "
            "missing one specific piece: {missing_concept}. You get everything else right "
            "and can reason well, but stumble on this one aspect."
        ),
        "known_topics_rule": "full_minus_one",
        "persona_hint": "Articulate, mostly correct, stumbles on one specific thing",
        "opening_style": "I've been studying {topic_name} and I think I get most of it, but I'm stuck on one part.",
    },
]


# ─────────────────────────────────────────────
# SCENARIO GENERATION
# ─────────────────────────────────────────────

OPENING_QUESTION_PROMPT = """\
You are generating opening messages for a student in a tutoring conversation \
about {topic_name}.

STUDENT PROFILE: {profile_label}
{profile_description}

TOPIC: {topic_name} — {core_concept}

Generate ONE opening message (1-3 sentences) that this student would say to \
start a tutoring conversation. The message should:
- Match the student's knowledge level and personality
- Be natural-sounding, not robotic
- Reference the topic specifically

{style_hint}

Respond with ONLY the student's message. No quotes, no explanation.
"""


def generate_opening(client, topic: dict, profile: dict) -> str:
    """Generate an opening question for a specific topic-profile pair."""

    # Build profile description
    prereqs = ", ".join(topic["prerequisites"])
    description = profile["description_template"].format(
        topic_name=topic["name"],
        prereqs=prereqs,
        prereq_example=topic["prerequisites"][0] if topic["prerequisites"] else "basics",
        confusion=topic["common_confusion"],
        wrong_explanation=topic["common_confusion"],
        wrong_assertion=topic["common_confusion"],
        missing_concept=topic["missing_concept_option"],
    )

    style_hint = f"Example style: \"{profile['opening_style'].format(topic_name=topic['name'], prereq_example=topic['prerequisites'][0] if topic['prerequisites'] else 'basics', wrong_explanation=topic['common_confusion'], wrong_assertion=topic['common_confusion'])}\""

    prompt = OPENING_QUESTION_PROMPT.format(
        topic_name=topic["name"],
        profile_label=profile["label"],
        profile_description=description,
        core_concept=topic["core_concept"],
        style_hint=style_hint,
    )

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.8,
                max_output_tokens=256,
            ),
        )
        return response.text.strip().strip('"').strip("'")
    except Exception as e:
        print(f"    WARNING: Failed to generate opening: {e}")
        # Fallback to template
        return profile["opening_style"].format(
            topic_name=topic["name"],
            prereq_example=topic["prerequisites"][0] if topic["prerequisites"] else "basics",
            wrong_explanation=topic["common_confusion"],
            wrong_assertion=topic["common_confusion"],
        )


def get_known_topics(topic: dict, profile: dict) -> list[str]:
    """Determine the student's known topics based on profile rule."""
    rule = profile["known_topics_rule"]

    if rule == "empty":
        return []
    elif rule == "prerequisites":
        return list(topic["prerequisites"])
    elif rule == "prerequisites_plus_partial":
        return list(topic["prerequisites"]) + [f"{topic['name']} (superficial)"]
    elif rule == "full_minus_one":
        # Knows everything except the missing concept
        known = list(topic["prerequisites"]) + list(topic["key_subtopics"])
        # Remove the missing concept area
        missing = topic["missing_concept_option"].lower()
        known = [k for k in known if missing.split()[0] not in k.lower()]
        return known
    return []


def generate_all_scenarios(client) -> list[dict]:
    """Generate all 40 scenarios (8 topics × 5 profiles)."""
    scenarios = []
    counter = 1

    for topic in TOPICS:
        print(f"\n  Topic: {topic['name']}")

        for profile in PROFILES:
            print(f"    Profile {profile['id']}: {profile['label']} ...", end=" ")

            opening = generate_opening(client, topic, profile)
            known = get_known_topics(topic, profile)

            # Build profile description for downstream use
            prereqs = ", ".join(topic["prerequisites"])
            description = profile["description_template"].format(
                topic_name=topic["name"],
                prereqs=prereqs,
                prereq_example=topic["prerequisites"][0] if topic["prerequisites"] else "basics",
                confusion=topic["common_confusion"],
                wrong_explanation=topic["common_confusion"],
                wrong_assertion=topic["common_confusion"],
                missing_concept=topic["missing_concept_option"],
            )

            scenario = {
                "id": f"e3_{counter:03d}",
                "topic": topic["id"],
                "topic_name": topic["name"],
                "core_concept": topic["core_concept"],
                "profile_id": profile["id"],
                "profile_name": profile["name"],
                "profile_label": profile["label"],
                "profile_description": description,
                "opening_question": opening,
                "known_topics": known,
                "target_concept": topic["core_concept"],
                "student_persona": profile["persona_hint"],
                "common_confusion": topic["common_confusion"],
                "missing_concept": topic["missing_concept_option"],
            }

            scenarios.append(scenario)
            print(f"✓ ({opening[:50]}...)")
            counter += 1

            time.sleep(1)  # Rate limiting

    return scenarios


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

def validate_scenarios(scenarios: list[dict]) -> bool:
    """Validate the generated scenarios."""
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    total = len(scenarios)
    print(f"Total scenarios: {total}")

    # Topic distribution
    topic_counts = {}
    for s in scenarios:
        topic_counts[s["topic"]] = topic_counts.get(s["topic"], 0) + 1
    print(f"Topic distribution: {topic_counts}")
    all_five = all(v == 5 for v in topic_counts.values())
    print(f"All topics have 5 profiles: {'✓' if all_five else '✗'}")

    # Profile distribution
    profile_counts = {}
    for s in scenarios:
        profile_counts[s["profile_name"]] = profile_counts.get(s["profile_name"], 0) + 1
    print(f"Profile distribution: {profile_counts}")
    all_eight = all(v == 8 for v in profile_counts.values())
    print(f"All profiles cover 8 topics: {'✓' if all_eight else '✗'}")

    # Check required fields
    missing = 0
    for s in scenarios:
        for field in ["opening_question", "known_topics", "target_concept", "profile_description"]:
            if not s.get(field) and field != "known_topics":
                missing += 1
                print(f"  WARNING: {s['id']} missing {field}")
    print(f"Missing required fields: {missing}")

    # Verify beginner profiles have empty known_topics
    beginner_with_knowledge = 0
    for s in scenarios:
        if s["profile_name"] == "complete_beginner" and s["known_topics"]:
            beginner_with_knowledge += 1
    print(f"Beginner profiles with known_topics (should be 0): {beginner_with_knowledge}")

    # Sample
    print("\n" + "-" * 60)
    print("SAMPLES")
    print("-" * 60)
    for profile_id in [1, 3, 5]:
        sample = next((s for s in scenarios if s["profile_id"] == profile_id), None)
        if sample:
            print(f"\n  {sample['id']} | {sample['topic_name']} | {sample['profile_label']}")
            print(f"  Opening: {sample['opening_question'][:80]}...")
            print(f"  Known: {sample['known_topics'][:3]}...")

    ok = total == 40 and missing == 0 and all_five and all_eight
    print(f"\nValidation: {'PASS' if ok else 'FAIL'}")
    return ok


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("E3 Scenario Generator")
    print(f"Model: {MODEL}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Scenarios: 8 topics × 5 profiles = 40")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    client = genai.Client(api_key=GEMINI_API_KEY)
    scenarios = generate_all_scenarios(client)

    validate_scenarios(scenarios)

    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": MODEL,
            "total": len(scenarios),
            "topics": [t["id"] for t in TOPICS],
            "profiles": [p["name"] for p in PROFILES],
        },
        "scenarios": scenarios,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(scenarios)} scenarios to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
