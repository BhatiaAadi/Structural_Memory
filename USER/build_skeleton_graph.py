"""
build_skeleton_graph.py
=======================
ONE-TIME offline script.

Queries Neo4j, resolves KG anchors for each curriculum seed,
and writes skeleton_graph.json.

After this runs, Neo4j is never needed again.
The JSON is the only thing the runtime uses.
"""

import json
from neo4j import GraphDatabase  # pip install neo4j


# ── CURRICULUM SEEDS ──────────────────────────────────────────────────────────
# Fixed 22-node DSA curriculum. Never changes.
# aliases = search terms used to fuzzy-match against KG concept names in Neo4j.

CURRICULUM_SEEDS = [
    # Tier 1 — Foundations
    {
        "id": "sg_complexity",
        "name": "Asymptotic Complexity",
        "aliases": ["asymptotic", "big-o", "big o", "theta", "omega", "time complexity"],
        "tier": 1,
        "sg_requires": [],
    },
    {
        "id": "sg_recursion",
        "name": "Recursion & Recurrences",
        "aliases": ["recursion", "recurrence", "master theorem", "recursive"],
        "tier": 1,
        "sg_requires": ["sg_complexity"],
    },
    {
        "id": "sg_arrays",
        "name": "Arrays & Dynamic Arrays",
        "aliases": ["array", "dynamic array", "amortized", "contiguous"],
        "tier": 1,
        "sg_requires": [],
    },
    {
        "id": "sg_pointers",
        "name": "Pointers & Memory",
        "aliases": ["pointer", "memory", "allocation", "reference"],
        "tier": 1,
        "sg_requires": [],
    },
    # Tier 2 — Core Data Structures
    {
        "id": "sg_linked_list",
        "name": "Linked Lists",
        "aliases": ["linked list", "singly linked", "doubly linked", "node pointer"],
        "tier": 2,
        "sg_requires": ["sg_pointers"],
    },
    {
        "id": "sg_stack_queue",
        "name": "Stacks & Queues",
        "aliases": ["stack", "queue", "deque", "lifo", "fifo"],
        "tier": 2,
        "sg_requires": ["sg_arrays", "sg_linked_list"],
    },
    {
        "id": "sg_hash_table",
        "name": "Hash Tables",
        "aliases": ["hash table", "hash map", "hashing", "hash function", "collision"],
        "tier": 2,
        "sg_requires": ["sg_arrays", "sg_complexity"],
    },
    {
        "id": "sg_bst",
        "name": "Binary Search Trees",
        "aliases": ["binary search tree", "bst", "search tree", "tree traversal"],
        "tier": 2,
        "sg_requires": ["sg_recursion", "sg_pointers"],
    },
    {
        "id": "sg_heap",
        "name": "Heaps & Priority Queues",
        "aliases": ["heap", "priority queue", "max-heap", "min-heap", "heapify"],
        "tier": 2,
        "sg_requires": ["sg_arrays", "sg_complexity"],
    },
    # Tier 3 — Core Algorithms
    {
        "id": "sg_sorting",
        "name": "Sorting Algorithms",
        "aliases": ["sorting", "mergesort", "merge sort", "quicksort", "heapsort", "comparison sort"],
        "tier": 3,
        "sg_requires": ["sg_arrays", "sg_recursion", "sg_complexity"],
    },
    {
        "id": "sg_divide_conquer",
        "name": "Divide & Conquer",
        "aliases": ["divide and conquer", "divide-and-conquer", "d&c", "subproblem"],
        "tier": 3,
        "sg_requires": ["sg_recursion", "sg_sorting"],
    },
    {
        "id": "sg_graphs",
        "name": "Graph Representations",
        "aliases": ["graph", "adjacency list", "adjacency matrix", "directed graph", "weighted graph"],
        "tier": 3,
        "sg_requires": ["sg_arrays", "sg_linked_list"],
    },
    {
        "id": "sg_bfs_dfs",
        "name": "BFS & DFS",
        "aliases": ["breadth first search", "depth first search", "bfs", "dfs", "graph traversal"],
        "tier": 3,
        "sg_requires": ["sg_graphs", "sg_stack_queue"],
    },
    {
        "id": "sg_greedy",
        "name": "Greedy Algorithms",
        "aliases": ["greedy", "greedy algorithm", "activity selection", "greedy choice"],
        "tier": 3,
        "sg_requires": ["sg_sorting", "sg_complexity"],
    },
    {
        "id": "sg_dp",
        "name": "Dynamic Programming",
        "aliases": ["dynamic programming", "dp", "memoization", "tabulation", "optimal substructure"],
        "tier": 3,
        "sg_requires": ["sg_recursion", "sg_divide_conquer"],
    },
    # Tier 4 — Advanced
    {
        "id": "sg_balanced_trees",
        "name": "Balanced Trees",
        "aliases": ["red-black tree", "avl tree", "balanced tree", "tree rotation", "red black"],
        "tier": 4,
        "sg_requires": ["sg_bst", "sg_complexity"],
    },
    {
        "id": "sg_shortest_path",
        "name": "Shortest Path Algorithms",
        "aliases": ["dijkstra", "bellman-ford", "bellman ford", "shortest path", "single source"],
        "tier": 4,
        "sg_requires": ["sg_graphs", "sg_heap", "sg_greedy"],
    },
    {
        "id": "sg_mst",
        "name": "Minimum Spanning Trees",
        "aliases": ["minimum spanning tree", "mst", "kruskal", "prim", "spanning tree"],
        "tier": 4,
        "sg_requires": ["sg_graphs", "sg_greedy", "sg_heap"],
    },
    {
        "id": "sg_amortized",
        "name": "Amortized Analysis",
        "aliases": ["amortized", "amortized analysis", "accounting method", "potential function"],
        "tier": 4,
        "sg_requires": ["sg_complexity", "sg_arrays"],
    },
    {
        "id": "sg_advanced_graphs",
        "name": "Advanced Graph Algorithms",
        "aliases": ["strongly connected", "scc", "topological sort", "topological", "max flow"],
        "tier": 4,
        "sg_requires": ["sg_bfs_dfs", "sg_dp"],
    },
    {
        "id": "sg_string_algo",
        "name": "String Algorithms",
        "aliases": ["string matching", "kmp", "knuth morris pratt", "rabin-karp", "pattern matching"],
        "tier": 4,
        "sg_requires": ["sg_arrays", "sg_dp"],
    },
    {
        "id": "sg_np",
        "name": "NP-Completeness",
        "aliases": ["np-complete", "np completeness", "np complete", "reduction", "p vs np"],
        "tier": 4,
        "sg_requires": ["sg_greedy", "sg_dp", "sg_advanced_graphs"],
    },
]


# ── KG ANCHOR RESOLVER ────────────────────────────────────────────────────────

def resolve_anchor(session, seed: dict) -> dict | None:
    """
    Find best-matching KG concept for a seed and pull its full neighborhood.
    Everything is embedded into the JSON — no Neo4j needed after this.
    """
    alias_conditions = " OR ".join(
        f"toLower(c.id) CONTAINS '{a}' OR toLower(c.name) CONTAINS '{a}'"
        for a in seed["aliases"]
    )

    candidates_result = session.run(f"""
        MATCH (c:Concept)
        WHERE {alias_conditions}
        RETURN c.id AS id, c.name AS name,
               c.definition AS definition, c.section AS section
    """)
    candidates = [dict(r) for r in candidates_result]

    if not candidates:
        return None

    def score(c):
        text = ((c.get("name") or "") + " " + (c.get("id") or "")).lower()
        return sum(1 for alias in seed["aliases"] if alias in text)

    best = max(candidates, key=score)

    # Pull full neighborhood — this is what makes the JSON self-contained
    nbr_result = session.run("""
        MATCH (c:Concept {id: $kg_id})
        OPTIONAL MATCH (c)-[:REQUIRES]->(prereq:Concept)
        OPTIONAL MATCH (c)-[:USES]->(technique:Concept)
        OPTIONAL MATCH (c)-[:SUBTYPE_OF]->(parent:Concept)
        OPTIONAL MATCH (c)-[:HAS_MISCONCEPTION]->(m:Misconception)
        RETURN
            collect(DISTINCT {id: prereq.id, name: prereq.name})       AS prerequisites,
            collect(DISTINCT {id: technique.id, name: technique.name}) AS techniques,
            collect(DISTINCT {id: parent.id, name: parent.name})       AS parents,
            collect(DISTINCT m.description)                             AS misconceptions
    """, kg_id=best["id"])

    nbr = dict(nbr_result.single())

    def clean_nodes(lst):
        return [x for x in lst if x and x.get("id")]

    def clean_strings(lst):
        return [x for x in lst if x]

    return {
        "kg_id":          best["id"],
        "kg_name":        best["name"],
        "kg_definition":  best["definition"],
        "kg_section":     best["section"],
        "prerequisites":  clean_nodes(nbr.get("prerequisites", [])),
        "techniques":     clean_nodes(nbr.get("techniques", [])),
        "parents":        clean_nodes(nbr.get("parents", [])),
        "misconceptions": clean_strings(nbr.get("misconceptions", [])),
    }


# ── BUILD ─────────────────────────────────────────────────────────────────────

def build_skeleton_graph(uri: str, user: str, password: str) -> dict:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    nodes = []
    unresolved = []

    with driver.session() as session:
        for seed in CURRICULUM_SEEDS:
            print(f"  resolving {seed['id']} ...", end=" ")
            anchor = resolve_anchor(session, seed)

            if anchor:
                print(f"✓  → {anchor['kg_name']}")
            else:
                print("✗  NOT FOUND")
                unresolved.append(seed["id"])

            nodes.append({
                "id":                seed["id"],
                "name":              seed["name"],
                "tier":              seed["tier"],
                "sg_requires":       seed["sg_requires"],
                "kg_search_aliases": seed["aliases"],
                "kg_anchor":         anchor,
            })

    driver.close()

    edges = [
        {"from": req, "to": seed["id"], "type": "SG_REQUIRES"}
        for seed in CURRICULUM_SEEDS
        for req in seed["sg_requires"]
    ]

    return {
        "meta": {
            "total_nodes":      len(nodes),
            "total_edges":      len(edges),
            "resolved_anchors": len(nodes) - len(unresolved),
            "unresolved":       unresolved,
        },
        "nodes": nodes,
        "edges": edges,
    }


def save_skeleton_graph(sg: dict, path: str = "skeleton_graph.json"):
    with open(path, "w") as f:
        json.dump(sg, f, indent=2)
    print(f"\nSaved → {path}")
    print(f"  Nodes    : {sg['meta']['total_nodes']}")
    print(f"  Edges    : {sg['meta']['total_edges']}")
    print(f"  Anchored : {sg['meta']['resolved_anchors']} / {sg['meta']['total_nodes']}")
    if sg["meta"]["unresolved"]:
        print(f"  Unresolved: {sg['meta']['unresolved']}")


if __name__ == "__main__":
    NEO4J_URI="<url>"
    NEO4J_USERNAME="<name>"
    NEO4J_PASSWORD="<pwd>"

    print("Building Skeleton Graph from KG...\n")
    sg = build_skeleton_graph(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    save_skeleton_graph(sg)