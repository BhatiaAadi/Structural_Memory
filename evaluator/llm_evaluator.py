"""
llm_evaluator.py
================
Calls Ollama (qwen2.5-coder:7b) with the rubric prompt,
parses the JSON response, validates, and applies rule-based
score adjustments using AST signals.

The LLM provides qualitative judgment (evidence, gaps, misconceptions).
The rule-based post-processor enforces quantitative consistency
using hard AST evidence to cap/adjust tier scores.
"""

import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:7b"


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    """
    Call Ollama's generate API with system + user prompts.
    Returns the raw response text.
    """
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 2048,
        },
        "format": "json",  # request JSON mode from Ollama
    }, timeout=120)

    response.raise_for_status()
    return response.json()["response"]


def parse_llm_response(raw_response: str) -> dict:
    """
    Parse the LLM's JSON response. Handles common failure modes:
    - Extra text before/after JSON
    - Markdown code fences wrapping JSON
    """
    text = raw_response.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse LLM response as JSON:\n{raw_response[:500]}")


def validate_response(parsed: dict) -> dict:
    """
    Validate and normalize the LLM response against expected schema.
    Returns cleaned response with guaranteed structure.
    """
    result = {
        "node_assessments": {},
        "strengths": parsed.get("strengths", []),
        "weaknesses": parsed.get("weaknesses", []),
    }

    assessments = parsed.get("node_assessments", {})
    for node_id, assessment in assessments.items():
        if not node_id.startswith("sg_"):
            continue

        tier_scores = assessment.get("tier_scores", {})
        clamped = {}
        for tier in ["syntax", "logical_use", "implementation_depth", "edge_case", "transfer_signal"]:
            raw = tier_scores.get(tier, 0.0)
            clamped[tier] = round(max(0.0, min(0.20, float(raw))), 3)

        mastery = round(sum(clamped.values()), 3)
        mastery = max(0.0, min(1.0, mastery))

        tier_order = ["syntax", "logical_use", "implementation_depth", "edge_case", "transfer_signal"]
        tier_reached = 0
        for i, tier in enumerate(tier_order, 1):
            if clamped[tier] > 0.02:
                tier_reached = i

        result["node_assessments"][node_id] = {
            "mastery_score": mastery,
            "tier_reached": tier_reached,
            "tier_scores": clamped,
            "evidence": assessment.get("evidence", ""),
            "gaps": assessment.get("gaps", ""),
            "misconceptions_triggered": assessment.get("misconceptions_triggered", []),
        }

    return result


# ── RULE-BASED SCORE ADJUSTMENT ───────────────────────────────────────────────
#
# The 7B LLM tends to either under-score or over-score. These rules use
# hard AST evidence to enforce consistency. The LLM's qualitative output
# (evidence, gaps, misconceptions) is kept — only the tier scores are adjusted.


def _cap_tier(scores: dict, tier: str, max_val: float):
    """Cap a tier score to max_val."""
    scores[tier] = min(scores[tier], max_val)


def _zero_above(scores: dict, from_tier: str):
    """Zero all tiers at and above from_tier."""
    tiers = ["syntax", "logical_use", "implementation_depth", "edge_case", "transfer_signal"]
    idx = tiers.index(from_tier)
    for t in tiers[idx:]:
        scores[t] = 0.0


def adjust_scores(llm_result: dict, ast_signals: dict, question: dict) -> dict:
    """
    Post-process LLM scores using AST signal evidence.
    
    Rules:
    1. If a node's expected pattern is in absent_patterns, cap at tier 2 (0.40)
    2. If bfs_with_list detected (not proper bfs), cap sg_stack_queue syntax at 0.13
       and cap overall at tier 3 (0.60)
    3. If no early returns and LLM scored edge_case > 0.07, cap edge_case at 0.07
    4. Transfer signal (tier 5) requires function_count >= 2 OR unusual pattern combination
       — single-function template solutions cap at 0.07
    5. If a node is only tangentially related (secondary), cap at tier 3 (0.60)
    """
    patterns = ast_signals.get("pattern_signatures", [])
    absent = ast_signals.get("absent_patterns", [])
    has_early_returns = ast_signals.get("has_early_returns", False)
    func_count = ast_signals.get("function_count", 1)
    primary_nodes = set(question.get("primary_sg_nodes", []))
    secondary_nodes = set(question.get("secondary_sg_nodes", []))

    for node_id, assessment in llm_result["node_assessments"].items():
        ts = assessment["tier_scores"]

        # ── Rule 1: Node's concept not implemented (absent pattern for this node) ──
        # If the question expected a pattern and it's absent, student didn't demonstrate it
        # But only apply to primary nodes — secondary nodes shouldn't need the pattern
        if node_id in primary_nodes and absent:
            # Check if ANY expected pattern maps to this node
            from .pattern_detector import ALL_DETECTORS
            # If all expected patterns are absent, cap at tier 2
            if len(absent) == len(question.get("expected_patterns", [])) and absent:
                _cap_tier(ts, "implementation_depth", 0.0)
                _cap_tier(ts, "edge_case", 0.0)
                _cap_tier(ts, "transfer_signal", 0.0)

        # ── Rule 2: BFS with list — suboptimal data structure ──
        if "bfs_with_list" in patterns:
            if node_id == "sg_stack_queue":
                _cap_tier(ts, "syntax", 0.13)         # not idiomatic
                _cap_tier(ts, "edge_case", 0.07)       # no O(1) awareness
                _cap_tier(ts, "transfer_signal", 0.0)  # template use
            if node_id == "sg_bfs_dfs":
                _cap_tier(ts, "syntax", 0.13)          # works but list.pop(0)
                _cap_tier(ts, "transfer_signal", 0.07)  # not novel

        # ── Rule 3: No early returns → cap edge_case ──
        # If student has no guard clauses at all, they likely don't handle edge cases
        if not has_early_returns:
            _cap_tier(ts, "edge_case", 0.13)  # allow partial if LLM found other evidence

        # ── Rule 4: Single-function template → cap transfer_signal ──
        # Tier 5 requires evidence of abstract understanding: helper functions,
        # non-obvious combinations, or adaptations beyond the template
        if func_count <= 1:
            _cap_tier(ts, "transfer_signal", 0.07)

        # ── Rule 5: Secondary nodes → cap at tier 3 ──
        # Secondary nodes are supporting concepts. A single question shouldn't
        # give full mastery on a secondary concept.
        if node_id in secondary_nodes and node_id not in primary_nodes:
            _cap_tier(ts, "edge_case", 0.13)
            _cap_tier(ts, "transfer_signal", 0.0)

        # ── Recompute mastery_score and tier_reached ──
        mastery = round(sum(ts.values()), 3)
        mastery = max(0.0, min(1.0, mastery))
        assessment["mastery_score"] = mastery

        tier_order = ["syntax", "logical_use", "implementation_depth", "edge_case", "transfer_signal"]
        tier_reached = 0
        for i, tier in enumerate(tier_order, 1):
            if ts[tier] > 0.02:
                tier_reached = i
        assessment["tier_reached"] = tier_reached

    return llm_result


def evaluate(system_prompt: str, user_prompt: str,
             ast_signals: dict = None, question: dict = None) -> dict:
    """
    Full LLM evaluation pipeline: call → parse → validate → adjust.
    Returns validated and adjusted evaluation result.
    """
    raw = call_llm(system_prompt, user_prompt)
    parsed = parse_llm_response(raw)
    result = validate_response(parsed)

    # Apply rule-based score adjustment if AST signals are provided
    if ast_signals and question:
        result = adjust_scores(result, ast_signals, question)

    return result
