"""
user_sg.py
==========
Everything to do with a real user's SG JSON.
No Neo4j. No simulation. Just user files.

Functions
---------
create_user_sg(username)        — create a new user JSON from skeleton_graph.json
load_user_sg(username)          — load an existing user JSON
update_mastery(username, sg_id, delta)  — write a mastery delta after a session
get_node(user_sg, sg_id)        — get one node dict by id
get_anchor(user_sg, sg_id)      — get the kg_anchor dict for a node
learning_frontier(user_sg)      — nodes the user is ready for next
"""

import json
import copy
import os

SKELETON_PATH = "skeleton_graph.json"
USERS_DIR     = "users"          # all user JSONs live here
MASTERY_THRESHOLD = 0.65          # above this = "knows" the topic


# ── CREATE ────────────────────────────────────────────────────────────────────

def create_user_sg(username: str) -> dict:
    """
    Create a new user JSON from skeleton_graph.json.
    Every node starts with mastery = 0.0.
    Writes users/{username}.json and returns the dict.
    """
    with open(SKELETON_PATH) as f:
        sg = json.load(f)

    user_sg = copy.deepcopy(sg)
    user_sg["user"] = username

    for node in user_sg["nodes"]:
        node["mastery"] = 0.0

    os.makedirs(USERS_DIR, exist_ok=True)
    path = _path(username)
    with open(path, "w") as f:
        json.dump(user_sg, f, indent=2)

    print(f"Created user SG → {path}")
    return user_sg


# ── LOAD ──────────────────────────────────────────────────────────────────────

def load_user_sg(username: str) -> dict:
    with open(_path(username)) as f:
        return json.load(f)


# ── UPDATE MASTERY ────────────────────────────────────────────────────────────

def update_mastery(username: str, sg_id: str, delta: float) -> float:
    """
    Apply delta to a node's mastery score and save.
    Returns the new mastery value.

    Typical deltas:
      +0.15  correct unprompted answer
      +0.10  correct with a hint
      +0.05  partially correct
      -0.05  misconception shown
      -0.10  completely wrong
    """
    user_sg = load_user_sg(username)

    for node in user_sg["nodes"]:
        if node["id"] == sg_id:
            node["mastery"] = round(max(0.0, min(1.0, node["mastery"] + delta)), 3)
            new_val = node["mastery"]
            break
    else:
        raise ValueError(f"sg_id '{sg_id}' not found in user SG")

    with open(_path(username), "w") as f:
        json.dump(user_sg, f, indent=2)

    return new_val


# ── QUERY HELPERS ─────────────────────────────────────────────────────────────

def get_node(user_sg: dict, sg_id: str) -> dict | None:
    for node in user_sg["nodes"]:
        if node["id"] == sg_id:
            return node
    return None


def get_anchor(user_sg: dict, sg_id: str) -> dict | None:
    """Return the kg_anchor dict for a node, or None if unresolved."""
    node = get_node(user_sg, sg_id)
    if node and node.get("kg_anchor"):
        return node["kg_anchor"]
    return None


def knows(user_sg: dict, sg_id: str) -> bool:
    node = get_node(user_sg, sg_id)
    return node is not None and node["mastery"] >= MASTERY_THRESHOLD


def learning_frontier(user_sg: dict) -> list[dict]:
    """
    Nodes the user is ready to learn next:
      - mastery below threshold (not yet known)
      - all sg_requires already known
    """
    return [
        node for node in user_sg["nodes"]
        if not knows(user_sg, node["id"])
        and all(knows(user_sg, req) for req in node["sg_requires"])
    ]


def check_prerequisites(user_sg: dict, sg_id: str) -> dict:
    """
    For a target node, return which SG prerequisites are met and unmet.
    Used to decide whether to teach prerequisites first.
    """
    node = get_node(user_sg, sg_id)
    met, unmet = [], []
    for req_id in node["sg_requires"]:
        req_node = get_node(user_sg, req_id)
        if knows(user_sg, req_id):
            met.append(req_node["name"])
        else:
            unmet.append(req_node["name"])
    return {"met": met, "unmet": unmet}


def find_sg_node_for_query(user_sg: dict, query: str) -> dict | None:
    """Fuzzy match a user's natural language query to an SG node."""
    q = query.lower()
    for node in user_sg["nodes"]:
        if any(alias in q or q in alias for alias in node["kg_search_aliases"]):
            return node
    return None


def user_level(user_sg: dict) -> str:
    scores = [n["mastery"] for n in user_sg["nodes"]]
    avg = sum(scores) / len(scores)
    if avg < 0.35:  return "beginner"
    if avg < 0.65:  return "intermediate"
    return "advanced"


# ── INTERNAL ──────────────────────────────────────────────────────────────────

def _path(username: str) -> str:
    return os.path.join(USERS_DIR, f"{username.lower()}.json")