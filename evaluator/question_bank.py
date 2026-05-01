"""
question_bank.py
================
Static question definitions. Each question maps to SG nodes and defines
what patterns a correct solution should exhibit.

Easy to extend — just add more dicts to QUESTIONS.
"""

QUESTIONS = [
    # ── TIER 1 ────────────────────────────────────────────────────────────
    {
        "id": "q_binary_search",
        "title": "Binary Search",
        "description": "Given a sorted array and a target value, return the index of the target. Return -1 if not found.",
        "difficulty": 1,
        "primary_sg_nodes": ["sg_arrays"],
        "secondary_sg_nodes": ["sg_complexity"],
        "expected_patterns": ["binary_search"],
        "starter_code": "def binary_search(nums: list[int], target: int) -> int:\n    pass",
    },
    {
        "id": "q_fibonacci",
        "title": "Fibonacci Number",
        "description": "Compute the n-th Fibonacci number. You may use any approach (recursive, iterative, memoized).",
        "difficulty": 1,
        "primary_sg_nodes": ["sg_recursion"],
        "secondary_sg_nodes": ["sg_complexity", "sg_dp"],
        "expected_patterns": ["topdown_dp_manual", "topdown_dp_cached", "bottomup_dp"],
        "starter_code": "def fibonacci(n: int) -> int:\n    pass",
    },
    {
        "id": "q_two_sum",
        "title": "Two Sum",
        "description": "Given an array of integers and a target sum, return indices of two numbers that add up to the target.",
        "difficulty": 1,
        "primary_sg_nodes": ["sg_hash_table"],
        "secondary_sg_nodes": ["sg_arrays", "sg_complexity"],
        "expected_patterns": [],
        "starter_code": "def two_sum(nums: list[int], target: int) -> list[int]:\n    pass",
    },

    # ── TIER 2 ────────────────────────────────────────────────────────────
    {
        "id": "q_valid_parentheses",
        "title": "Valid Parentheses",
        "description": "Given a string containing just '(', ')', '{', '}', '[', ']', determine if the input string is valid.",
        "difficulty": 2,
        "primary_sg_nodes": ["sg_stack_queue"],
        "secondary_sg_nodes": ["sg_arrays"],
        "expected_patterns": [],
        "starter_code": "def is_valid(s: str) -> bool:\n    pass",
    },
    {
        "id": "q_linked_list_cycle",
        "title": "Linked List Cycle Detection",
        "description": "Given head of a linked list, determine if it has a cycle. Use Floyd's cycle detection.",
        "difficulty": 2,
        "primary_sg_nodes": ["sg_linked_list"],
        "secondary_sg_nodes": ["sg_pointers"],
        "expected_patterns": ["two_pointer"],
        "starter_code": "def has_cycle(head) -> bool:\n    pass",
    },
    {
        "id": "q_bst_validate",
        "title": "Validate Binary Search Tree",
        "description": "Given root of a binary tree, determine if it is a valid BST.",
        "difficulty": 2,
        "primary_sg_nodes": ["sg_bst"],
        "secondary_sg_nodes": ["sg_recursion"],
        "expected_patterns": ["divide_and_conquer"],
        "starter_code": "def is_valid_bst(root) -> bool:\n    pass",
    },
    {
        "id": "q_kth_largest",
        "title": "Kth Largest Element",
        "description": "Find the kth largest element in an unsorted array. Use a heap-based approach.",
        "difficulty": 2,
        "primary_sg_nodes": ["sg_heap"],
        "secondary_sg_nodes": ["sg_arrays", "sg_complexity"],
        "expected_patterns": [],
        "starter_code": "def find_kth_largest(nums: list[int], k: int) -> int:\n    pass",
    },

    # ── TIER 3 ────────────────────────────────────────────────────────────
    {
        "id": "q_merge_sort",
        "title": "Merge Sort",
        "description": "Implement merge sort to sort an array in ascending order.",
        "difficulty": 3,
        "primary_sg_nodes": ["sg_sorting", "sg_divide_conquer"],
        "secondary_sg_nodes": ["sg_recursion", "sg_arrays"],
        "expected_patterns": ["divide_and_conquer"],
        "starter_code": "def merge_sort(nums: list[int]) -> list[int]:\n    pass",
    },
    {
        "id": "q_bfs_shortest_path",
        "title": "BFS Shortest Path in Unweighted Graph",
        "description": "Given an unweighted graph as adjacency list and a source node, find shortest distance to all nodes using BFS.",
        "difficulty": 3,
        "primary_sg_nodes": ["sg_bfs_dfs"],
        "secondary_sg_nodes": ["sg_graphs", "sg_stack_queue"],
        "expected_patterns": ["bfs"],
        "starter_code": "def bfs_shortest(graph: dict, source: int) -> dict:\n    pass",
    },
    {
        "id": "q_coin_change",
        "title": "Coin Change (Minimum Coins)",
        "description": "Given coin denominations and a target amount, find the minimum number of coins needed. Return -1 if impossible.",
        "difficulty": 3,
        "primary_sg_nodes": ["sg_dp"],
        "secondary_sg_nodes": ["sg_recursion", "sg_arrays"],
        "expected_patterns": ["bottomup_dp", "topdown_dp_manual", "topdown_dp_cached"],
        "starter_code": "def coin_change(coins: list[int], amount: int) -> int:\n    pass",
    },
    {
        "id": "q_activity_selection",
        "title": "Activity Selection (Greedy)",
        "description": "Given start and end times of activities, find the maximum number of non-overlapping activities.",
        "difficulty": 3,
        "primary_sg_nodes": ["sg_greedy"],
        "secondary_sg_nodes": ["sg_sorting", "sg_complexity"],
        "expected_patterns": ["greedy"],
        "starter_code": "def max_activities(starts: list[int], ends: list[int]) -> int:\n    pass",
    },

    # ── TIER 4 ────────────────────────────────────────────────────────────
    {
        "id": "q_dijkstra",
        "title": "Dijkstra's Shortest Path",
        "description": "Given a weighted graph (adjacency list with weights) and source, find shortest distances to all nodes using Dijkstra's algorithm.",
        "difficulty": 4,
        "primary_sg_nodes": ["sg_shortest_path"],
        "secondary_sg_nodes": ["sg_heap", "sg_greedy", "sg_graphs"],
        "expected_patterns": ["dijkstra"],
        "starter_code": "def dijkstra(graph: dict, source: int) -> dict:\n    pass",
    },
    {
        "id": "q_topological_sort",
        "title": "Topological Sort",
        "description": "Given a DAG as adjacency list, return a valid topological ordering of the nodes.",
        "difficulty": 4,
        "primary_sg_nodes": ["sg_advanced_graphs"],
        "secondary_sg_nodes": ["sg_bfs_dfs", "sg_graphs"],
        "expected_patterns": ["bfs", "dfs_recursive"],
        "starter_code": "def topological_sort(graph: dict) -> list:\n    pass",
    },
    {
        "id": "q_lcs",
        "title": "Longest Common Subsequence",
        "description": "Given two strings, find the length of their longest common subsequence.",
        "difficulty": 4,
        "primary_sg_nodes": ["sg_string_algo", "sg_dp"],
        "secondary_sg_nodes": ["sg_arrays"],
        "expected_patterns": ["bottomup_dp", "topdown_dp_manual", "topdown_dp_cached"],
        "starter_code": "def lcs(text1: str, text2: str) -> int:\n    pass",
    },
]


def get_question(question_id: str) -> dict | None:
    """Look up a question by ID."""
    for q in QUESTIONS:
        if q["id"] == question_id:
            return q
    return None


def get_questions_for_node(sg_node_id: str) -> list[dict]:
    """Get all questions that test a given SG node (primary or secondary)."""
    return [
        q for q in QUESTIONS
        if sg_node_id in q["primary_sg_nodes"] or sg_node_id in q["secondary_sg_nodes"]
    ]


def list_questions() -> list[dict]:
    """Return all questions (without starter_code for brevity)."""
    return [
        {k: v for k, v in q.items() if k != "starter_code"}
        for q in QUESTIONS
    ]
