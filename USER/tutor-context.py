"""
tutor.py
========
Runtime tutor — reads user SG JSON, builds context, calls Qwen 2.5 7B.
No Neo4j. Everything comes from the user's JSON.
"""

import requests
from user_sg import (
    load_user_sg,
    find_sg_node_for_query,
    check_prerequisites,
    user_level,
    learning_frontier,
    update_mastery,
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "qwen2.5:7b"


# ── CONTEXT BUILDER ───────────────────────────────────────────────────────────

def build_context(user_sg: dict, sg_node: dict) -> str:
    """
    Assemble the full context string from the user's JSON alone.
    The kg_anchor embedded in each node has everything we need:
    definition, prerequisites, misconceptions, techniques — all from KG,
    all already stored in the JSON at build time.
    """
    anchor  = sg_node.get("kg_anchor") or {}
    prereqs = check_prerequisites(user_sg, sg_node["id"])
    level   = user_level(user_sg)

    # Analogy bridges: SG nodes the user knows that are KG prerequisites
    # of the target concept (pulled directly from the anchor's prerequisites list)
    known_kg_ids = {
        n["kg_anchor"]["kg_id"]
        for n in user_sg["nodes"]
        if n["mastery"] >= 0.6 and n.get("kg_anchor")
    }
    kg_prereq_ids = {p["id"] for p in anchor.get("prerequisites", []) if p.get("id")}
    bridges = [
        p["name"] for p in anchor.get("prerequisites", [])
        if p.get("id") in known_kg_ids
    ]

    lines = [
        "=== KNOWLEDGE GRAPH CONTEXT ===",
        f"Topic      : {anchor.get('kg_name', sg_node['name'])}",
        f"Definition : {anchor.get('kg_definition', 'N/A')}",
        f"CLRS       : {anchor.get('kg_section', 'N/A')}",
        "",
        "KG Prerequisites:",
    ]
    for p in anchor.get("prerequisites", []):
        status = "✓" if p.get("id") in known_kg_ids else "✗"
        lines.append(f"  {status} {p['name']}")

    lines += ["", "Misconceptions to address:"]
    for m in anchor.get("misconceptions", [])[:4]:
        lines.append(f"  ⚠ {m}")

    lines += [
        "",
        "=== USER STATE ===",
        f"User    : {user_sg['user']}",
        f"Level   : {level}",
        f"Mastery : {int(sg_node['mastery'] * 100)}% on this topic",
        "",
    ]

    if prereqs["unmet"]:
        lines.append("Unmet SG prerequisites — explain these FIRST:")
        for p in prereqs["unmet"]:
            lines.append(f"  ✗ {p}")
    else:
        lines.append("All SG prerequisites met ✓")

    if bridges:
        lines += ["", "Analogy bridges (user knows these — anchor your explanation here):"]
        for b in bridges:
            lines.append(f"  → {b}")

    lines += [
        "",
        "=== INSTRUCTIONS ===",
        f"Speak at {level} level.",
        "If unmet prerequisites exist, explain them before the main topic.",
        "Use analogy bridges to connect the new concept to what the user knows.",
        "Proactively address the misconceptions listed above.",
        "Do not give direct solutions — guide with questions and scaffolded hints.",
    ]

    return "\n".join(lines)


# ── QWEN CALL ─────────────────────────────────────────────────────────────────

def ask_qwen(context: str, question: str) -> str:
    response = requests.post(OLLAMA_URL, json={
        "model":  MODEL,
        "system": "You are an adaptive DSA tutor. Use the knowledge graph context "
                  "and user state provided to personalise every response.",
        "prompt": f"{context}\n\nStudent: {question}",
        "stream": False,
    })
    return response.json()["response"]


# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────

def answer(username: str, question: str) -> str:
    """
    Full pipeline for one user + one question.
    Loads the user's JSON, builds context from it, calls Qwen.
    """
    user_sg = load_user_sg(username)
    sg_node = find_sg_node_for_query(user_sg, question)

    if not sg_node:
        # No SG match — call Qwen without context
        return ask_qwen("", question)

    context = build_context(user_sg, sg_node)
    return ask_qwen(context, question)


# ── EXAMPLE USAGE ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from user_sg import create_user_sg, update_mastery

    # First time: create user
    create_user_sg("diana")

    # Manually set what Diana already knows (her first session intake)
    for sg_id, mastery in [
        ("sg_complexity",     0.8),
        ("sg_recursion",      0.9),
        ("sg_arrays",         0.8),
        ("sg_pointers",       0.7),
        ("sg_linked_list",    0.6),
        ("sg_sorting",        0.9),
        ("sg_divide_conquer", 0.85),
    ]:
        # set directly — positive delta from 0.0
        update_mastery("diana", sg_id, mastery)

    # Ask a question
    response = answer("diana", "explain dynamic programming")
    print(response)

    # After evaluating her answer in the dialogue, update her DP mastery
    update_mastery("diana", "sg_dp", +0.10)