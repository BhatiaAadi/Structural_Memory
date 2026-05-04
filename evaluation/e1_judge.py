"""
e1_judge.py
===========
Stage 3 of E1 Evaluation Pipeline.

Runs locally. Scores each response on the C1–C5 deterministic checklist
using Gemini 2.5 Pro as an LLM judge for nuanced checks.

Input:  e1_responses.json  (from Stage 2)
Output: e1_results.json    (full scored results)
        e1_metrics.json    (aggregate metrics per model)
        e1_gate_heatmap.png (visualization)
"""

import json
import os
import time
import re
from datetime import datetime
from collections import defaultdict

from google import genai
from google.genai import types

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyBXHBoy3JanCwVHDuyGKlzHqtOoqM4V4HQ"  # <-- Replace with your key
JUDGE_MODEL = "gemini-2.5-pro"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESPONSES_PATH       = os.path.join(SCRIPT_DIR, "e1_responses-qwenbase.json")
RESULTS_PATH         = os.path.join(SCRIPT_DIR, "e1_results-qwenbase.json")
METRICS_PATH         = os.path.join(SCRIPT_DIR, "e1_metrics-qwenbase.json")
HEATMAP_PATH         = os.path.join(SCRIPT_DIR, "e1_gate_heatmap-qwenbase.png")
MODEL_COMPARISON_PATH = os.path.join(SCRIPT_DIR, "e1_model_comparison-qwenbase.png")
PARTIAL_RESULTS_PATH = os.path.join(SCRIPT_DIR, "e1_results_partial-qwenbase.json")

API_DELAY = 1.5  # Seconds between API calls


# ─────────────────────────────────────────────
# CHECK DEFINITIONS (from eval.md §1.5)
# ─────────────────────────────────────────────

CHECK_DESCRIPTIONS = {
    "C1": "Prerequisite gap detected: Response explicitly names at least one missing prerequisite from the user's SG.",
    "C2": "Prerequisites explained before target: The first substantive explanation covers a prerequisite, not the target concept.",
    "C3": "Target not fully explained without prereqs: If any sg_requires node has mastery < 0.65, the target explanation is incomplete or deferred.",
    "C4": "Known topics acknowledged: Response references or builds upon at least one topic the user has mastered.",
    "C5": "Full-prereq user not re-taught basics: Response does NOT re-explain concepts the user has mastery ≥ 0.65 on.",
}


# ─────────────────────────────────────────────
# LLM JUDGE FOR EACH CHECK
# ─────────────────────────────────────────────

E1_JUDGE_SYSTEM = """\
You are an expert evaluator of AI tutoring systems. You are evaluating whether
a tutor correctly handles prerequisite gating — that is, whether it detects
knowledge gaps in a student's profile and adjusts its response accordingly.

You will be given:
1. The student's question
2. The context that was injected into the tutor (showing prerequisite status)
3. The tutor's response
4. A specific check to evaluate (C1–C5)
5. Relevant metadata (unmet prerequisites, known topics, etc.)

Evaluate ONLY the specified check. Be strict but fair.

--- OUTPUT FORMAT ---
Respond with ONLY the following JSON object. No explanation, no preamble.
{
  "pass": true or false,
  "evidence": "the specific phrase or behavior in the response that determined your verdict",
  "reasoning": "one sentence explaining your verdict"
}
"""


def evaluate_check(
    client,
    check_id: str,
    question: str,
    context: str,
    response: str,
    unmet_prereqs: list[str],
    met_prereqs: list[str],
    known_topics: list[str],
    key_gap: str | None,
    target_name: str,
) -> dict:
    """Evaluate a single check (C1-C5) for a response using the LLM judge."""

    # Build check-specific prompt
    check_desc = CHECK_DESCRIPTIONS[check_id]

    if check_id == "C1":
        user_prompt = f"""\
CHECK: {check_id} — {check_desc}

STUDENT QUESTION: {question}

UNMET PREREQUISITES (these should be named in the response):
{json.dumps(unmet_prereqs, indent=2)}

TUTOR RESPONSE:
{response}

Does the tutor explicitly name or reference at least one of the missing prerequisites listed above?
Simply mentioning the topic area counts. The tutor does NOT need to use the exact SG node name."""

    elif check_id == "C2":
        user_prompt = f"""\
CHECK: {check_id} — {check_desc}

STUDENT QUESTION: {question}
TARGET TOPIC: {target_name}

UNMET PREREQUISITES:
{json.dumps(unmet_prereqs, indent=2)}

TUTOR RESPONSE:
{response}

Does the FIRST substantive explanation in the response cover a prerequisite topic
(not the target topic "{target_name}")? The tutor should explain missing prerequisites
BEFORE diving into the target concept."""

    elif check_id == "C3":
        user_prompt = f"""\
CHECK: {check_id} — {check_desc}

STUDENT QUESTION: {question}
TARGET TOPIC: {target_name}

UNMET PREREQUISITES:
{json.dumps(unmet_prereqs, indent=2)}

TUTOR RESPONSE:
{response}

Given that the student has unmet prerequisites, does the tutor correctly defer or
limit the explanation of {target_name}? The target should NOT be fully explained
when prerequisites are missing. Partial explanation with redirection to prerequisites is acceptable."""

    elif check_id == "C4":
        user_prompt = f"""\
CHECK: {check_id} — {check_desc}

STUDENT QUESTION: {question}

TOPICS THE STUDENT ALREADY KNOWS (mastery ≥ 0.65):
{json.dumps(known_topics, indent=2)}

TUTOR RESPONSE:
{response}

Does the response reference, build upon, or acknowledge at least one topic that
the student already knows? This includes using known concepts as analogies,
bridges, or foundations for the explanation."""

    elif check_id == "C5":
        user_prompt = f"""\
CHECK: {check_id} — {check_desc}

STUDENT QUESTION: {question}
TARGET TOPIC: {target_name}

TOPICS THE STUDENT ALREADY KNOWS (should NOT be re-explained):
{json.dumps(known_topics, indent=2)}

TUTOR RESPONSE:
{response}

Does the response avoid re-explaining topics the student already knows?
The tutor should NOT re-teach basics that the student has mastery ≥ 0.65 on.
Brief mentions or references are fine — only flag if the tutor gives a full
re-explanation of a known concept as if the student doesn't know it.
PASS = does NOT re-teach known topics. FAIL = re-teaches known topics."""

    else:
        return {"pass": False, "evidence": None, "reasoning": f"Unknown check: {check_id}"}

    try:
        resp = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=8192,
                response_mime_type="application/json",
                system_instruction=E1_JUDGE_SYSTEM,
            ),
        )

        raw = resp.text
        if raw is None:
            block_reason = "unknown"
            if hasattr(resp, "candidates") and resp.candidates:
                cand = resp.candidates[0]
                if hasattr(cand, "finish_reason"):
                    block_reason = str(cand.finish_reason)
            print(f"      [Judge] Response blocked: {block_reason}")
            return {"pass": False, "evidence": None, "reasoning": f"Blocked: {block_reason}"}

        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines)

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(raw[start:end])
            else:
                raise ValueError(f"Could not parse: {raw[:200]}")

        return {
            "pass": bool(result.get("pass", False)),
            "evidence": result.get("evidence"),
            "reasoning": result.get("reasoning", ""),
        }

    except Exception as e:
        print(f"      [Judge ERROR] {check_id}: {e}")
        return {"pass": False, "evidence": None, "reasoning": f"Error: {str(e)}"}


# ─────────────────────────────────────────────
# EVALUATE ALL RESPONSES
# ─────────────────────────────────────────────

def evaluate_responses(client, responses: list[dict]) -> list[dict]:
    """Evaluate all responses on their applicable checks."""

    # Resume from partial
    completed_keys = set()
    results = []

    if os.path.exists(PARTIAL_RESULTS_PATH):
        with open(PARTIAL_RESULTS_PATH, "r") as f:
            results = json.load(f)
        completed_keys = {(r["user_id"], r["model_name"]) for r in results}
        print(f"[Resume] Found {len(completed_keys)} completed evaluations.")

    remaining = [r for r in responses if (r["user_id"], r["model_name"]) not in completed_keys]
    total = len(responses)

    for idx, resp in enumerate(remaining, len(completed_keys) + 1):
        uid = resp["user_id"]
        model = resp["model_name"]
        arch = resp["archetype"]

        print(f"\n[{idx}/{total}] {uid} × {model} | Archetype {arch} | {resp.get('target_name', '')}")

        checks = resp.get("applicable_checks", [])
        check_results = {}

        for check_id in checks:
            result = evaluate_check(
                client,
                check_id=check_id,
                question=resp["target_question"],
                context=resp.get("context_injected", ""),
                response=resp.get("response", ""),
                unmet_prereqs=resp.get("expected_unmet_prereqs", []),
                met_prereqs=resp.get("expected_met_prereqs", []),
                known_topics=resp.get("known_topics", []),
                key_gap=resp.get("key_gap"),
                target_name=resp.get("target_name", ""),
            )

            status = "✓" if result["pass"] else "✗"
            print(f"    {check_id}: {status} | {result['reasoning'][:80]}")

            check_results[check_id] = result
            time.sleep(API_DELAY)

        # Compute per-response score
        applicable = len(checks)
        passed = sum(1 for c in check_results.values() if c["pass"])
        score = passed / applicable if applicable > 0 else 0.0

        # Specificity check for Archetype B
        specificity = None
        if arch == "B" and resp.get("key_gap"):
            gap_name = resp.get("key_gap", "")
            # Check if the SPECIFIC gap was named (not just any prerequisite)
            gap_node_name = None
            for prereq in resp.get("expected_unmet_prereqs", []):
                specificity = True  # Will be refined by checking response
            # Use a quick keyword check
            response_lower = resp.get("response", "").lower()
            gap_keywords = {
                "sg_heap": ["heap", "priority queue"],
                "sg_divide_conquer": ["divide and conquer", "divide & conquer", "d&c"],
                "sg_stack_queue": ["stack", "queue", "stacks", "queues"],
                "sg_bst": ["binary search tree", "bst", "search tree"],
            }
            keywords = gap_keywords.get(gap_name, [gap_name.replace("sg_", "").replace("_", " ")])
            specificity = any(kw in response_lower for kw in keywords)

        eval_result = {
            "user_id": uid,
            "archetype": arch,
            "archetype_label": resp.get("archetype_label", ""),
            "model_name": model,
            "model_label": resp.get("model_label", ""),
            "target_name": resp.get("target_name", ""),
            "target_question": resp["target_question"],
            "response": resp.get("response", ""),
            "checks": check_results,
            "score": round(score, 3),
            "passed": passed,
            "applicable": applicable,
            "specificity": specificity,
            "key_gap": resp.get("key_gap"),
            "timestamp": datetime.now().isoformat(),
        }

        results.append(eval_result)
        completed_keys.add((uid, model))

        # Save partial
        with open(PARTIAL_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"    Score: {passed}/{applicable} = {score:.2f} | Saved {len(results)}/{total}")

    return results


# ─────────────────────────────────────────────
# COMPUTE AGGREGATE METRICS
# ─────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics per model."""
    models = sorted(set(r["model_name"] for r in results))
    all_metrics = {}

    for model in models:
        model_results = [r for r in results if r["model_name"] == model]

        # Gate Recall: C1 pass rate across A+B users
        ab_results = [r for r in model_results if r["archetype"] in ("A", "B")]
        c1_passes = sum(1 for r in ab_results if r["checks"].get("C1", {}).get("pass", False))
        gate_recall = c1_passes / len(ab_results) if ab_results else 0

        # Gate Precision: Among responses that issued prereq warnings, fraction with actual gap
        # All A+B users have actual gaps, so precision = (true positives) / (all positives)
        c_results = [r for r in model_results if r["archetype"] == "C"]
        # Check if C users got false prerequisite warnings (C5 fail could indicate over-gating)
        false_gates = sum(
            1 for r in c_results
            if not r["checks"].get("C5", {}).get("pass", True)
        )
        true_gates = c1_passes
        total_gates = true_gates + false_gates
        gate_precision = true_gates / total_gates if total_gates > 0 else 1.0

        # Specificity: Among B users, did the response name the correct missing prerequisite?
        b_results = [r for r in model_results if r["archetype"] == "B"]
        specific_matches = sum(1 for r in b_results if r.get("specificity", False))
        specificity = specific_matches / len(b_results) if b_results else 0

        # Teaching Rate: C users, C5 pass rate
        c5_passes = sum(1 for r in c_results if r["checks"].get("C5", {}).get("pass", False))
        teaching_rate = c5_passes / len(c_results) if c_results else 0

        # Per-archetype scores
        archetype_scores = {}
        for arch in ["A", "B", "C"]:
            arch_r = [r for r in model_results if r["archetype"] == arch]
            if arch_r:
                archetype_scores[arch] = {
                    "mean_score": round(sum(r["score"] for r in arch_r) / len(arch_r), 3),
                    "count": len(arch_r),
                    "per_check": {},
                }
                # Per-check pass rates
                all_checks = set()
                for r in arch_r:
                    all_checks.update(r["checks"].keys())
                for check in sorted(all_checks):
                    check_passes = sum(
                        1 for r in arch_r if r["checks"].get(check, {}).get("pass", False)
                    )
                    archetype_scores[arch]["per_check"][check] = round(
                        check_passes / len(arch_r), 3
                    )

        # Overall score
        overall_score = sum(r["score"] for r in model_results) / len(model_results) if model_results else 0

        all_metrics[model] = {
            "model_label": model_results[0].get("model_label", model) if model_results else model,
            "total_responses": len(model_results),
            "overall_score": round(overall_score, 3),
            "gate_recall": round(gate_recall, 3),
            "gate_precision": round(gate_precision, 3),
            "specificity": round(specificity, 3),
            "teaching_rate": round(teaching_rate, 3),
            "archetype_scores": archetype_scores,
        }

    return all_metrics


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def generate_heatmap(results: list[dict], output_path: str):
    """Generate a heatmap of check results: users × checks × models."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available. Skipping heatmap.")
        return

    models = sorted(set(r["model_name"] for r in results))
    users = sorted(set(r["user_id"] for r in results), key=lambda x: int(x[1:]))
    checks = ["C1", "C2", "C3", "C4", "C5"]

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 8), squeeze=False)

    for m_idx, model in enumerate(models):
        ax = axes[0][m_idx]
        model_results = [r for r in results if r["model_name"] == model]

        # Build matrix: users × checks
        matrix = np.full((len(users), len(checks)), np.nan)

        for r in model_results:
            u_idx = users.index(r["user_id"])
            for c_idx, check in enumerate(checks):
                if check in r["checks"]:
                    matrix[u_idx, c_idx] = 1.0 if r["checks"][check]["pass"] else 0.0

        # Plot
        cmap = plt.cm.RdYlGn
        cmap.set_bad(color="lightgray")

        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

        # Labels
        ax.set_xticks(range(len(checks)))
        ax.set_xticklabels(checks, fontsize=10)
        ax.set_yticks(range(len(users)))
        ax.set_yticklabels(users, fontsize=9)
        ax.set_title(model_results[0].get("model_label", model), fontsize=12, fontweight="bold")

        # Add text annotations
        for i in range(len(users)):
            for j in range(len(checks)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text = "✓" if val == 1.0 else "✗"
                    color = "white" if val == 0.0 else "black"
                    ax.text(j, i, text, ha="center", va="center", fontsize=14, color=color)
                else:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="gray")

        # Add archetype separators
        ax.axhline(y=3.5, color="black", linewidth=2)  # After A (U1-U4)
        ax.axhline(y=7.5, color="black", linewidth=2)  # After B (U5-U8)

        # Archetype labels on right
        if m_idx == n_models - 1:
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticks([1.5, 5.5, 9.5])
            ax2.set_yticklabels(["A\n(Beginner)", "B\n(Partial)", "C\n(Full)"], fontsize=9)

    fig.suptitle("E1 — Prerequisite Gate Enforcement Checklist", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to {output_path}")


def generate_model_comparison(metrics: dict, output_path: str):
    """Generate a bar chart comparing models on aggregate metrics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available. Skipping comparison plot.")
        return

    models = list(metrics.keys())
    if len(models) < 2:
        print("Only 1 model — skipping comparison plot.")
        return

    metric_names = ["Gate Recall", "Gate Precision", "Specificity", "Teaching Rate", "Overall Score"]
    metric_keys = ["gate_recall", "gate_precision", "specificity", "teaching_rate", "overall_score"]

    x = np.arange(len(metric_names))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    for i, model in enumerate(models):
        values = [metrics[model].get(k, 0) for k in metric_keys]
        label = metrics[model].get("model_label", model)
        bars = ax.bar(x + i * width, values, width, label=label, color=colors[i % len(colors)])

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("E1 — Model Comparison on Prerequisite Gate Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Model comparison saved to {output_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  E1 Judge — Prerequisite Gate Enforcement")
    print(f"  Judge model: {JUDGE_MODEL}")
    print(f"  Input: {RESPONSES_PATH}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not os.path.exists(RESPONSES_PATH):
        print(f"\nERROR: {RESPONSES_PATH} not found.")
        print("Run Stage 2 (e1_simulation.py) first to generate responses.")
        return

    with open(RESPONSES_PATH, "r") as f:
        responses = json.load(f)

    print(f"\nLoaded {len(responses)} responses")

    # Filter valid responses
    valid = [r for r in responses if r.get("response") and not r.get("error")]
    print(f"Valid responses (with content): {len(valid)}")

    if not valid:
        print("ERROR: No valid responses found.")
        return

    # Show breakdown
    models = sorted(set(r["model_name"] for r in valid))
    for m in models:
        count = sum(1 for r in valid if r["model_name"] == m)
        print(f"  {m}: {count} responses")

    # Initialize Gemini client
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Evaluate
    results = evaluate_responses(client, valid)

    # Save full results
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {RESULTS_PATH}")

    # Compute metrics
    metrics = compute_metrics(results)

    # Save metrics
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {METRICS_PATH}")

    # Print summary
    print("\n" + "=" * 60)
    print("  E1 EVALUATION RESULTS")
    print("=" * 60)

    for model, m in metrics.items():
        print(f"\n  {m['model_label']}:")
        print(f"    Overall Score:   {m['overall_score']:.3f}")
        print(f"    Gate Recall:     {m['gate_recall']:.3f}")
        print(f"    Gate Precision:  {m['gate_precision']:.3f}")
        print(f"    Specificity:     {m['specificity']:.3f}")
        print(f"    Teaching Rate:   {m['teaching_rate']:.3f}")

        for arch, scores in m["archetype_scores"].items():
            print(f"    Archetype {arch}: mean={scores['mean_score']:.3f}, "
                  f"checks={scores['per_check']}")

    # Generate visualizations
    generate_heatmap(results, HEATMAP_PATH)
    generate_model_comparison(metrics, MODEL_COMPARISON_PATH)

    print(f"\n{'=' * 60}")
    print("  E1 EVALUATION COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
