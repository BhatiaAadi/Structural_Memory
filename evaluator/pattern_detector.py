"""
pattern_detector.py
===================
Detects composite algorithm skeletons from raw AST signals.
Takes signals from ast_analyzer.extract_signals() and identifies
higher-level patterns like BFS, Dijkstra, top-down DP, etc.

Also computes absent_patterns by comparing detected patterns
against expected patterns for a given question.

v2 — tightened rules to reduce false positives:
  - bottomup_dp requires NO heapq/set (exclude Dijkstra-like code)
  - binary_search removed (too noisy, let LLM judge)
  - two_pointer removed (too noisy, let LLM judge)
  - bfs_with_list added (list.pop(0) BFS — detected as partial BFS)
  - dfs_iterative requires set (visited) to avoid false match on BFS-with-list)
  - Patterns only reported at medium+ confidence
"""


def _has(signals, key, values=None):
    """Check if a signal key exists and optionally contains specific values."""
    val = signals.get(key)
    if val is None:
        return False
    if values is None:
        return bool(val)
    if isinstance(val, list):
        return any(v in val for v in values)
    return val in values if isinstance(values, (list, set, tuple)) else val == values


def _has_all(signals, key, values):
    """Check if signal list contains ALL specified values."""
    val = signals.get(key)
    if not isinstance(val, list):
        return False
    return all(v in val for v in values)


# ── INDIVIDUAL PATTERN DETECTORS ─────────────────────────────────────────────

def _detect_bfs(signals):
    """
    BFS (proper): deque + popleft + while loop.
    BFS (with list): list + pop + append + while loop + NO recursion + NO heapq.
    """
    has_deque = _has(signals, "data_structures_used", ["deque"])
    has_popleft = _has(signals, "builtin_calls", ["popleft"])
    has_while = "while" in signals.get("loop_types", [])

    # Proper BFS with deque
    if has_deque and has_popleft and has_while:
        has_set = _has(signals, "data_structures_used", ["set"])
        return {"pattern": "bfs", "confidence": "high" if has_set else "medium",
                "sg_nodes": ["sg_bfs_dfs", "sg_stack_queue", "sg_graphs"]}

    # BFS with list (suboptimal but functionally correct)
    has_list = _has(signals, "data_structures_used", ["list"])
    has_pop = _has(signals, "builtin_calls", ["pop"])
    has_append = _has(signals, "builtin_calls", ["append"])
    no_heap = not _has(signals, "data_structures_used", ["heapq"])
    no_recursion = not signals.get("has_recursion", False)
    no_set = not _has(signals, "data_structures_used", ["set"])

    if has_list and has_pop and has_append and has_while and no_heap and no_recursion and no_set:
        return {"pattern": "bfs_with_list", "confidence": "medium",
                "sg_nodes": ["sg_bfs_dfs", "sg_graphs"]}

    return None


def _detect_dfs_recursive(signals):
    """DFS recursive: recursive function + visited set"""
    has_rec = signals.get("has_recursion", False)
    has_set = _has(signals, "data_structures_used", ["set"])
    # Must not have heapq (that would be Dijkstra-like)
    no_heap = not _has(signals, "data_structures_used", ["heapq"])

    if has_rec and has_set and no_heap:
        return {"pattern": "dfs_recursive", "confidence": "high",
                "sg_nodes": ["sg_bfs_dfs", "sg_recursion", "sg_graphs"]}
    return None


def _detect_dfs_iterative(signals):
    """DFS iterative: stack (list + pop) + visited set + while loop + NO deque + NO heapq"""
    has_list = _has(signals, "data_structures_used", ["list"])
    has_pop = _has(signals, "builtin_calls", ["pop"])
    has_set = _has(signals, "data_structures_used", ["set"])  # REQUIRED — distinguishes from BFS-with-list
    has_while = "while" in signals.get("loop_types", [])
    no_deque = not _has(signals, "data_structures_used", ["deque"])
    no_heap = not _has(signals, "data_structures_used", ["heapq"])

    if has_list and has_pop and has_set and has_while and no_deque and no_heap:
        return {"pattern": "dfs_iterative", "confidence": "medium",
                "sg_nodes": ["sg_bfs_dfs", "sg_stack_queue"]}
    return None


def _detect_dijkstra(signals):
    """Dijkstra: heapq + float('inf') init + while loop"""
    has_heap = _has(signals, "data_structures_used", ["heapq"])
    has_inf = _has(signals, "data_structures_used", ["float_inf"])
    has_while = "while" in signals.get("loop_types", [])

    if has_heap and has_inf and has_while:
        return {"pattern": "dijkstra", "confidence": "high",
                "sg_nodes": ["sg_shortest_path", "sg_heap", "sg_greedy"]}
    return None


def _detect_topdown_dp(signals):
    """Top-down DP: memo dict + recursive function + base case (or @lru_cache)"""
    has_lru = _has(signals, "builtin_calls", ["lru_cache_decorator"])
    has_lru_ds = _has(signals, "data_structures_used", ["lru_cache"])
    has_rec = signals.get("has_recursion", False)
    has_dict = _has(signals, "data_structures_used", ["dict"])
    has_base = signals.get("has_base_case", False)

    if has_rec and (has_lru or has_lru_ds):
        return {"pattern": "topdown_dp_cached", "confidence": "high",
                "sg_nodes": ["sg_dp", "sg_recursion"]}
    if has_rec and has_dict and has_base:
        # Only if no heapq (otherwise it's Dijkstra, not DP)
        no_heap = not _has(signals, "data_structures_used", ["heapq"])
        if no_heap:
            return {"pattern": "topdown_dp_manual", "confidence": "high",
                    "sg_nodes": ["sg_dp", "sg_recursion"]}
    return None


def _detect_bottomup_dp(signals):
    """
    Bottom-up DP: table (list) init + nested for-loops + NO recursion.
    EXCLUDE: code with heapq or set (those are graph algorithms, not DP tables).
    """
    has_list = _has(signals, "data_structures_used", ["list"])
    deep_loops = signals.get("loop_depth_max", 0) >= 2
    has_for = "for" in signals.get("loop_types", [])
    no_recursion = not signals.get("has_recursion", False)
    no_heap = not _has(signals, "data_structures_used", ["heapq"])
    no_set = not _has(signals, "data_structures_used", ["set"])
    # Must have ONLY for loops (while+for suggests graph traversal, not table fill)
    no_while = "while" not in signals.get("loop_types", [])

    if has_list and deep_loops and has_for and no_recursion and no_heap and no_set and no_while:
        return {"pattern": "bottomup_dp", "confidence": "medium",
                "sg_nodes": ["sg_dp", "sg_arrays"]}
    return None


def _detect_divide_and_conquer(signals):
    """D&C: recursive function + base case + multiple functions (helper)"""
    has_rec = signals.get("has_recursion", False)
    has_base = signals.get("has_base_case", False)
    multi_func = signals.get("function_count", 0) >= 2  # need helper function (e.g. merge)
    no_heap = not _has(signals, "data_structures_used", ["heapq"])

    if has_rec and has_base and multi_func and no_heap:
        return {"pattern": "divide_and_conquer", "confidence": "medium",
                "sg_nodes": ["sg_divide_conquer", "sg_recursion"]}
    return None


def _detect_greedy(signals):
    """Greedy: sorted() call + single pass (for loop, low depth)"""
    has_sort = _has(signals, "builtin_calls", ["sorted"]) or _has(signals, "builtin_calls", ["sort"])
    has_for = "for" in signals.get("loop_types", [])
    shallow = signals.get("loop_depth_max", 0) <= 1

    if has_sort and has_for and shallow:
        return {"pattern": "greedy", "confidence": "medium",
                "sg_nodes": ["sg_greedy", "sg_sorting"]}
    return None


def _detect_backtracking(signals):
    """Backtracking: recursive function + list modifications (append + pop)"""
    has_rec = signals.get("has_recursion", False)
    has_append = _has(signals, "builtin_calls", ["append"])
    has_pop = _has(signals, "builtin_calls", ["pop"])

    if has_rec and has_append and has_pop:
        return {"pattern": "backtracking", "confidence": "medium",
                "sg_nodes": ["sg_recursion"]}
    return None


def _detect_union_find(signals):
    """Union-Find: 3+ functions (find/union/main) + list/dict for parent"""
    multi_func = signals.get("function_count", 0) >= 3
    has_container = (_has(signals, "data_structures_used", ["list"]) or
                     _has(signals, "data_structures_used", ["dict"]))

    if multi_func and has_container:
        return {"pattern": "union_find", "confidence": "medium",
                "sg_nodes": ["sg_graphs", "sg_advanced_graphs"]}
    return None


# ── MAIN DETECTION ────────────────────────────────────────────────────────────

ALL_DETECTORS = [
    _detect_bfs,           # must come before dfs_iterative (shared signals)
    _detect_dfs_recursive,
    _detect_dfs_iterative,
    _detect_dijkstra,
    _detect_topdown_dp,
    _detect_bottomup_dp,
    _detect_divide_and_conquer,
    _detect_greedy,
    _detect_backtracking,
    _detect_union_find,
    # NOTE: binary_search and two_pointer removed — too noisy structurally,
    # let the LLM judge these from code context instead
]


def detect_patterns(signals: dict) -> list[dict]:
    """
    Run all pattern detectors on AST signals.
    Returns list of detected patterns with confidence and SG node mappings.
    """
    detected = []
    for detector in ALL_DETECTORS:
        result = detector(signals)
        if result:
            detected.append(result)
    return detected


def compute_absent_patterns(detected: list[dict], expected: list[str]) -> list[str]:
    """
    Compare detected patterns against expected patterns for a question.
    Returns list of expected patterns that were NOT detected.
    
    Note: bfs_with_list counts as a partial match for 'bfs' — it is NOT
    reported as absent (the LLM should see it and evaluate accordingly).
    """
    detected_names = {p["pattern"] for p in detected}

    # bfs_with_list is a partial match for bfs
    if "bfs_with_list" in detected_names:
        detected_names.add("bfs")

    return [e for e in expected if e not in detected_names]


def enrich_signals(signals: dict, expected_patterns: list[str]) -> dict:
    """
    Full pipeline: detect patterns, compute absent patterns, and
    enrich the signals dict with pattern_signatures and absent_patterns.
    Returns the enriched signals dict (modified in place).
    """
    detected = detect_patterns(signals)

    signals["pattern_signatures"] = [p["pattern"] for p in detected]
    signals["pattern_details"] = detected
    signals["absent_patterns"] = compute_absent_patterns(detected, expected_patterns)

    return signals
