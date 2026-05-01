"""
evaluator.py
============
Main pipeline orchestrator — the public API.

Usage:
    from evaluator import evaluate_code
    result = evaluate_code("diana", "q_dijkstra", code_string)
"""

import json
import os
import sys

# Add parent directory to path so we can import USER modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "USER"))

from .ast_analyzer import extract_signals
from .pattern_detector import enrich_signals
from .question_bank import get_question
from .prompts import build_evaluator_prompt
from .llm_evaluator import evaluate as llm_evaluate


# ── SKELETON GRAPH HELPERS ────────────────────────────────────────────────────

USER_DIR = os.path.join(os.path.dirname(__file__), "..", "USER")
SKELETON_PATH = os.path.join(USER_DIR, "skeleton_graph.json")
USERS_DIR = os.path.join(USER_DIR, "users")


def _load_skeleton_graph():
    with open(SKELETON_PATH) as f:
        return json.load(f)


def _load_user_sg(username: str):
    path = os.path.join(USERS_DIR, f"{username.lower()}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"User '{username}' not found. Create with user_sg.create_user_sg('{username}') first."
        )
    with open(path) as f:
        return json.load(f)


def _save_user_sg(username: str, user_sg: dict):
    path = os.path.join(USERS_DIR, f"{username.lower()}.json")
    with open(path, "w") as f:
        json.dump(user_sg, f, indent=2)


def _get_node(sg: dict, node_id: str):
    for node in sg["nodes"]:
        if node["id"] == node_id:
            return node
    return None


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def evaluate_code(username: str, question_id: str, code: str,
                  apply_updates: bool = True, verbose: bool = False) -> dict:
    """
    Full evaluation pipeline:
    1. Load question metadata
    2. Parse code with tree-sitter
    3. Extract AST signals + detect algorithm patterns
    4. Load user's current skeleton graph
    5. Build LLM prompt with rubric
    6. Call LLM evaluator
    7. Optionally apply mastery updates to user SG
    8. Return full evaluation result

    Args:
        username:       User whose SG to evaluate against
        question_id:    Question ID from question_bank
        code:           Student's Python source code
        apply_updates:  If True, write mastery deltas to user JSON
        verbose:        If True, print intermediate steps

    Returns:
        dict with: question, ast_signals, llm_evaluation, mastery_updates
    """
    # 1. Load question
    question = get_question(question_id)
    if not question:
        raise ValueError(f"Question '{question_id}' not found in question bank")

    if verbose:
        print(f"  [1/7] Question: {question['title']}")

    # 2-3. Parse code + extract signals + detect patterns
    signals = extract_signals(code)
    signals = enrich_signals(signals, question.get("expected_patterns", []))

    if verbose:
        print(f"  [2/7] AST signals extracted")
        print(f"         Patterns detected: {signals['pattern_signatures']}")
        print(f"         Absent patterns:   {signals['absent_patterns']}")

    # 4. Load user SG and gather target concept info
    user_sg = _load_user_sg(username)
    sg = _load_skeleton_graph()

    all_target_ids = list(set(
        question["primary_sg_nodes"] + question["secondary_sg_nodes"]
    ))

    target_concepts = []
    current_mastery = {}
    for nid in all_target_ids:
        sg_node = _get_node(sg, nid)
        user_node = _get_node(user_sg, nid)
        if sg_node:
            target_concepts.append({
                "id": nid,
                "name": sg_node["name"],
            })
            current_mastery[nid] = user_node["mastery"] if user_node and "mastery" in user_node else 0.0

    if verbose:
        print(f"  [3/7] Target concepts: {[c['name'] for c in target_concepts]}")

    # 5. Build LLM prompt
    system_prompt, user_prompt = build_evaluator_prompt(
        question=question,
        code=code,
        ast_signals=signals,
        target_concepts=target_concepts,
        current_mastery=current_mastery,
    )

    if verbose:
        print(f"  [4/7] LLM prompt built ({len(system_prompt) + len(user_prompt)} chars)")

    # 6. Call LLM evaluator
    if verbose:
        print(f"  [5/7] Calling LLM evaluator...")

    llm_result = llm_evaluate(system_prompt, user_prompt,
                               ast_signals=signals, question=question)

    if verbose:
        print(f"  [6/7] LLM evaluation received")
        for nid, assessment in llm_result["node_assessments"].items():
            print(f"         {nid}: {assessment['mastery_score']:.3f} (tier {assessment['tier_reached']})")

    # 7. Apply mastery updates
    mastery_updates = {}
    if apply_updates and llm_result["node_assessments"]:
        for nid, assessment in llm_result["node_assessments"].items():
            user_node = _get_node(user_sg, nid)
            if user_node is None:
                continue

            old_mastery = user_node.get("mastery", 0.0)
            new_mastery = assessment["mastery_score"]

            # Blend: weighted average of old and new, favoring recent evidence
            # This prevents a single evaluation from overwriting history
            RECENCY_WEIGHT = 0.6
            blended = round(
                RECENCY_WEIGHT * new_mastery + (1 - RECENCY_WEIGHT) * old_mastery,
                3
            )
            blended = max(0.0, min(1.0, blended))

            user_node["mastery"] = blended
            mastery_updates[nid] = {
                "old": old_mastery,
                "new": blended,
                "raw_score": new_mastery,
                "delta": round(blended - old_mastery, 3),
            }

        _save_user_sg(username, user_sg)

        if verbose:
            print(f"  [7/7] Mastery updates applied:")
            for nid, upd in mastery_updates.items():
                print(f"         {nid}: {upd['old']:.3f} → {upd['new']:.3f} (Δ{upd['delta']:+.3f})")

    # 8. Return full result
    return {
        "question": {
            "id": question["id"],
            "title": question["title"],
            "difficulty": question["difficulty"],
        },
        "ast_signals": {
            k: v for k, v in signals.items()
            if k not in ("pattern_details",)  # exclude internal details
        },
        "llm_evaluation": llm_result,
        "mastery_updates": mastery_updates,
    }


# ── CLI ENTRY POINT ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example usage — run a sample evaluation
    sample_code = """\
import heapq

def dijkstra(graph, source):
    dist = {node: float('inf') for node in graph}
    dist[source] = 0
    pq = [(0, source)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))

    return dist
"""

    print("=" * 60)
    print("DSA Code Evaluator — Sample Run")
    print("=" * 60)

    # Check if a user exists, create if not
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "USER"))

    try:
        result = evaluate_code(
            username="test_user",
            question_id="q_dijkstra",
            code=sample_code,
            apply_updates=False,  # don't write to file in sample run
            verbose=True,
        )
        print("\n" + "=" * 60)
        print("Full Result:")
        print(json.dumps(result, indent=2))
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Create a user first with: python -c \"from USER.user_sg import create_user_sg; create_user_sg('test_user')\"")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
