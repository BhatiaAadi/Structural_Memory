"""
generate_e1_profiles.py
=======================
Stage 1 of E1 Evaluation Pipeline.

Runs locally. Generates 12 synthetic user profiles from eval.md §1.3
using skeleton_graph.json as the template.

Input:  USER/skeleton_graph.json
Output: evaluation/e1_profiles.json
"""

import json
import copy
import os
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SKELETON_PATH = os.path.join(PROJECT_ROOT, "USER", "skeleton_graph.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "e1_profiles.json")

MASTERY_THRESHOLD = 0.65  # Above this = "knows" the topic


# ─────────────────────────────────────────────
# 12 USER PROFILES (from eval.md §1.3)
# ─────────────────────────────────────────────

USER_PROFILES = [
    # ── Archetype A — Beginner with gaps (4 users) ──
    # All mastery = 0.0. Asking about advanced topics.
    {
        "user_id": "U1",
        "archetype": "A",
        "archetype_label": "Beginner with gaps",
        "target_sg_node": "sg_shortest_path",
        "target_question": "Explain Dijkstra's algorithm",
        "mastery_overrides": {},  # All 0.0 (default)
        "applicable_checks": ["C1", "C2", "C3"],
    },
    {
        "user_id": "U2",
        "archetype": "A",
        "archetype_label": "Beginner with gaps",
        "target_sg_node": "sg_dp",
        "target_question": "Explain Dynamic Programming",
        "mastery_overrides": {},
        "applicable_checks": ["C1", "C2", "C3"],
    },
    {
        "user_id": "U3",
        "archetype": "A",
        "archetype_label": "Beginner with gaps",
        "target_sg_node": "sg_balanced_trees",
        "target_question": "Explain Red-Black Trees",
        "mastery_overrides": {},
        "applicable_checks": ["C1", "C2", "C3"],
    },
    {
        "user_id": "U4",
        "archetype": "A",
        "archetype_label": "Beginner with gaps",
        "target_sg_node": "sg_bfs_dfs",
        "target_question": "How does BFS work?",
        "mastery_overrides": {},
        "applicable_checks": ["C1", "C2", "C3"],
    },

    # ── Archetype B — Partial knowledge with specific gaps (4 users) ──
    # Most prereqs met, but one immediate prerequisite is missing.
    {
        "user_id": "U5",
        "archetype": "B",
        "archetype_label": "Partial knowledge with specific gaps",
        "target_sg_node": "sg_shortest_path",
        "target_question": "Explain Dijkstra's algorithm",
        "mastery_overrides": {
            "sg_complexity": 0.8,
            "sg_arrays": 0.8,
            "sg_graphs": 0.7,
            "sg_greedy": 0.7,
            "sg_heap": 0.2,  # ← THE GAP
        },
        "key_gap": "sg_heap",
        "applicable_checks": ["C1", "C2", "C3", "C4"],
    },
    {
        "user_id": "U6",
        "archetype": "B",
        "archetype_label": "Partial knowledge with specific gaps",
        "target_sg_node": "sg_dp",
        "target_question": "Explain Dynamic Programming",
        "mastery_overrides": {
            "sg_complexity": 0.8,
            "sg_recursion": 0.9,
            "sg_arrays": 0.8,
            "sg_sorting": 0.7,
            "sg_divide_conquer": 0.1,  # ← THE GAP
        },
        "key_gap": "sg_divide_conquer",
        "applicable_checks": ["C1", "C2", "C3", "C4"],
    },
    {
        "user_id": "U7",
        "archetype": "B",
        "archetype_label": "Partial knowledge with specific gaps",
        "target_sg_node": "sg_bfs_dfs",
        "target_question": "Explain BFS",
        "mastery_overrides": {
            "sg_arrays": 0.8,
            "sg_pointers": 0.7,
            "sg_linked_list": 0.7,
            "sg_graphs": 0.7,
            "sg_stack_queue": 0.3,  # ← THE GAP
        },
        "key_gap": "sg_stack_queue",
        "applicable_checks": ["C1", "C2", "C3", "C4"],
    },
    {
        "user_id": "U8",
        "archetype": "B",
        "archetype_label": "Partial knowledge with specific gaps",
        "target_sg_node": "sg_balanced_trees",
        "target_question": "Explain Balanced Trees",
        "mastery_overrides": {
            "sg_complexity": 0.8,
            "sg_recursion": 0.7,
            "sg_pointers": 0.7,
            "sg_bst": 0.2,  # ← THE GAP
        },
        "key_gap": "sg_bst",
        "applicable_checks": ["C1", "C2", "C3", "C4"],
    },

    # ── Archetype C — Full prerequisites met (4 users) ──
    # All prereqs ≥ 0.65. Target topic = 0.0.
    {
        "user_id": "U9",
        "archetype": "C",
        "archetype_label": "Full prerequisites met",
        "target_sg_node": "sg_shortest_path",
        "target_question": "Explain Dijkstra's algorithm",
        "mastery_overrides": {
            "sg_complexity": 0.8,
            "sg_arrays": 0.8,
            "sg_graphs": 0.8,
            "sg_heap": 0.7,
            "sg_greedy": 0.7,
            "sg_shortest_path": 0.0,
        },
        "applicable_checks": ["C4", "C5"],
    },
    {
        "user_id": "U10",
        "archetype": "C",
        "archetype_label": "Full prerequisites met",
        "target_sg_node": "sg_dp",
        "target_question": "Explain Dynamic Programming",
        "mastery_overrides": {
            "sg_complexity": 0.8,
            "sg_recursion": 0.9,
            "sg_sorting": 0.8,
            "sg_divide_conquer": 0.7,
            "sg_dp": 0.0,
        },
        "applicable_checks": ["C4", "C5"],
    },
    {
        "user_id": "U11",
        "archetype": "C",
        "archetype_label": "Full prerequisites met",
        "target_sg_node": "sg_bfs_dfs",
        "target_question": "Explain BFS",
        "mastery_overrides": {
            "sg_arrays": 0.8,
            "sg_linked_list": 0.7,
            "sg_graphs": 0.7,
            "sg_stack_queue": 0.7,
            "sg_bfs_dfs": 0.0,
        },
        "applicable_checks": ["C4", "C5"],
    },
    {
        "user_id": "U12",
        "archetype": "C",
        "archetype_label": "Full prerequisites met",
        "target_sg_node": "sg_balanced_trees",
        "target_question": "Explain Balanced Trees",
        "mastery_overrides": {
            "sg_complexity": 0.8,
            "sg_recursion": 0.7,
            "sg_pointers": 0.7,
            "sg_bst": 0.8,
            "sg_balanced_trees": 0.0,
        },
        "applicable_checks": ["C4", "C5"],
    },
]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def build_user_sg(skeleton: dict, mastery_overrides: dict, user_id: str) -> dict:
    """Create a user SG dict with specific mastery overrides."""
    user_sg = copy.deepcopy(skeleton)
    user_sg["user"] = user_id

    for node in user_sg["nodes"]:
        node["mastery"] = mastery_overrides.get(node["id"], 0.0)

    return user_sg


def get_node(user_sg: dict, sg_id: str) -> dict | None:
    for node in user_sg["nodes"]:
        if node["id"] == sg_id:
            return node
    return None


def check_prerequisites(user_sg: dict, sg_id: str) -> dict:
    """Check which SG prerequisites are met/unmet for a target node."""
    node = get_node(user_sg, sg_id)
    if not node:
        return {"met": [], "unmet": [], "met_ids": [], "unmet_ids": []}

    met, unmet = [], []
    met_ids, unmet_ids = [], []

    for req_id in node["sg_requires"]:
        req_node = get_node(user_sg, req_id)
        if req_node and req_node["mastery"] >= MASTERY_THRESHOLD:
            met.append(req_node["name"])
            met_ids.append(req_id)
        else:
            unmet.append(req_node["name"] if req_node else req_id)
            unmet_ids.append(req_id)

    return {"met": met, "unmet": unmet, "met_ids": met_ids, "unmet_ids": unmet_ids}


def user_level(user_sg: dict) -> str:
    """Determine user level from average mastery."""
    scores = [n["mastery"] for n in user_sg["nodes"]]
    avg = sum(scores) / len(scores) if scores else 0
    if avg < 0.35:
        return "beginner"
    if avg < 0.65:
        return "intermediate"
    return "advanced"


def get_known_topics(user_sg: dict) -> list[str]:
    """Get names of all topics the user has mastery ≥ 0.65 on."""
    return [n["name"] for n in user_sg["nodes"] if n["mastery"] >= MASTERY_THRESHOLD]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  E1 Profile Generator — Prerequisite Gate Enforcement")
    print(f"  Skeleton: {SKELETON_PATH}")
    print(f"  Output:   {OUTPUT_PATH}")
    print("=" * 60)

    # Load skeleton graph
    if not os.path.exists(SKELETON_PATH):
        print(f"\nERROR: {SKELETON_PATH} not found.")
        print("Run USER/build_skeleton_graph.py first.")
        return

    with open(SKELETON_PATH, "r") as f:
        skeleton = json.load(f)

    print(f"\nLoaded skeleton graph: {len(skeleton['nodes'])} nodes")

    # Check kg_anchor status
    anchored = sum(1 for n in skeleton["nodes"] if n.get("kg_anchor"))
    print(f"KG anchors populated: {anchored}/22")
    if anchored == 0:
        print("WARNING: All kg_anchor are null. Run build_skeleton_graph.py first!")
        print("Continuing anyway — profiles will work but context will be incomplete.\n")

    # Generate profiles
    profiles = []
    for profile_def in USER_PROFILES:
        user_id = profile_def["user_id"]
        print(f"\n[{user_id}] Archetype {profile_def['archetype']} | "
              f"Target: {profile_def['target_sg_node']} | "
              f"Question: {profile_def['target_question']}")

        # Build user SG
        user_sg = build_user_sg(skeleton, profile_def["mastery_overrides"], user_id)

        # Compute prerequisite state
        prereqs = check_prerequisites(user_sg, profile_def["target_sg_node"])
        level = user_level(user_sg)
        known = get_known_topics(user_sg)

        # Get target node info
        target_node = get_node(user_sg, profile_def["target_sg_node"])
        target_name = target_node["name"] if target_node else profile_def["target_sg_node"]

        print(f"  Level: {level}")
        print(f"  Met prereqs:   {prereqs['met'] if prereqs['met'] else '(none)'}")
        print(f"  Unmet prereqs: {prereqs['unmet'] if prereqs['unmet'] else '(none)'}")
        print(f"  Known topics:  {len(known)}")
        print(f"  Checks: {profile_def['applicable_checks']}")

        profile = {
            "user_id": user_id,
            "archetype": profile_def["archetype"],
            "archetype_label": profile_def["archetype_label"],
            "target_sg_node": profile_def["target_sg_node"],
            "target_name": target_name,
            "target_question": profile_def["target_question"],
            "user_level": level,
            "expected_met_prereqs": prereqs["met"],
            "expected_met_prereq_ids": prereqs["met_ids"],
            "expected_unmet_prereqs": prereqs["unmet"],
            "expected_unmet_prereq_ids": prereqs["unmet_ids"],
            "known_topics": known,
            "key_gap": profile_def.get("key_gap"),
            "applicable_checks": profile_def["applicable_checks"],
            "user_sg": user_sg,
        }

        profiles.append(profile)

    # Save
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_users": len(profiles),
            "skeleton_path": SKELETON_PATH,
            "kg_anchors_populated": anchored,
            "archetypes": {
                "A": sum(1 for p in profiles if p["archetype"] == "A"),
                "B": sum(1 for p in profiles if p["archetype"] == "B"),
                "C": sum(1 for p in profiles if p["archetype"] == "C"),
            },
        },
        "profiles": profiles,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  DONE — {len(profiles)} profiles saved to {OUTPUT_PATH}")
    print(f"  File size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")
    print(f"  Archetypes: A={output['metadata']['archetypes']['A']}, "
          f"B={output['metadata']['archetypes']['B']}, "
          f"C={output['metadata']['archetypes']['C']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
