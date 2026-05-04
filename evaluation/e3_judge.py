"""
e3_judge.py
===========
Stage 3 of E3 Evaluation Pipeline.
Uses Gemini 2.5 Pro to score each tutor turn against 20 learning sciences
principles (P1-P20). Computes aggregate metrics and generates heatmap.

Run locally:
    python evaluation/e3_judge.py

Input:  e3_conversations.json
Output: e3_results_full.json, e3_metrics.json, e3_heatmap.png
"""

import json
import os
import time
from datetime import datetime
from collections import defaultdict

from google import genai
from google.genai import types

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyBXHBoy3JanCwVHDuyGKlzHqtOoqM4V4HQ"
JUDGE_MODEL = "gemini-2.5-flash"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model names to evaluate (matching simulation output files)
MODEL_NAMES = [
    {"name": "qwen_sft",   "label": "SFT only"},
    {"name": "qwen_dpo",   "label": "SFT + DPO"},
]

# Legacy single-file fallback
LEGACY_CONVERSATIONS_PATH = os.path.join(SCRIPT_DIR, "e3_responses_qwen_sft.json")
PARTIAL_PATH = os.path.join(SCRIPT_DIR, "e3_results_partial.json")

API_DELAY = 1.0
PRINCIPLES = [f"P{i}" for i in range(1, 21)]

# ─────────────────────────────────────────────
# E3 JUDGE SYSTEM PROMPT (verbatim from eval.md §3.6)
# ─────────────────────────────────────────────
E3_JUDGE_SYSTEM_PROMPT = """\
You are an expert in learning sciences and pedagogical evaluation. Your task
is to analyze a single tutor turn in a DSA (Data Structures and Algorithms)
tutoring conversation and identify which learning sciences principles are
actively employed, and at what strength.

You will be given:
1. The student's knowledge profile (known topics, target concept, turn number)
2. The student's message at this turn
3. The tutor's response at this turn
4. The turn number (1-10) and total conversation length

You will evaluate the tutor's response against the 20 learning sciences
principles defined below. For each principle, assign a strength score.

--- STRENGTH SCALE ---
0 = Absent: The principle is not present in this response.
1 = Weakly Present: The principle is attempted but partially or superficially.
2 = Strongly Present: The principle is clearly and fully employed.

Do NOT give a 2 just because a principle is mentioned. Score 2 only when
the principle is genuinely and effectively executed.

--- THE 20 PRINCIPLES ---

P1 — Activate Prior Knowledge
Score 2 if: Tutor asks a specific question that elicits what the student already knows, OR explicitly references the student's known background and builds from it.
Score 1 if: Tutor references prior knowledge vaguely or as a preamble without actually building on it.
Score 0 if: Tutor ignores the student's background entirely.

P2 — Elicit Explanations (Sense-Making)
Score 2 if: Tutor specifically asks the student to explain a mechanism, process, or reason — not just state a fact.
Score 1 if: Tutor asks for explanation but the question is answerable by recall alone.
Score 0 if: No explanation-eliciting behavior.

P3 — Inference and Consequence Reasoning
Score 2 if: Tutor asks the student to derive a consequence, implication, or significance from something already established.
Score 1 if: Tutor hints at a consequence without asking the student to derive it.
Score 0 if: Absent.

P4 — Transfer and Application
Score 2 if: Tutor asks student to apply the concept to a new context or scenario not already discussed.
Score 1 if: Tutor mentions application but does not ask the student to perform it.
Score 0 if: Absent.

P5 — Hypothesis and Prediction
Score 2 if: Tutor explicitly asks for a prediction or hypothesis BEFORE providing information.
Score 1 if: Tutor invites a guess but the answer is already constrained by context.
Score 0 if: Absent.

P6 — Data Observation and Interpretation
Score 2 if: Tutor directs attention to specific evidence and asks the student to interpret it.
Score 1 if: Tutor mentions data or a pattern without asking the student to engage with it.
Score 0 if: Absent.

P7 — Example Generation and Generalization
Score 2 if: Tutor asks for an additional example AND follows up on abstracting to a general rule.
Score 1 if: Tutor asks for an example only, without the generalization step.
Score 0 if: Absent.

P8 — Analogical Reasoning
Score 2 if: Tutor uses a specific concept from the student's known topics as a bridge. The analogy must be grounded in the student's actual profile.
Score 1 if: Tutor uses a generic analogy not connected to the student's specific known topics.
Score 0 if: Absent.

P9 — Conceptual Discrimination
Score 2 if: Tutor asks the student to identify the critical difference between two similar concepts.
Score 1 if: Tutor mentions a distinction without asking the student to articulate it.
Score 0 if: Absent.

P10 — Counterfactual Challenge
Score 2 if: Tutor introduces a contrasting or inverse scenario that tests the boundary of the student's understanding.
Score 1 if: Tutor introduces a contrasting case but does not ask the student to reason through it.
Score 0 if: Absent.

P11 — Metacognitive Reflection
Score 2 if: Tutor asks the student to examine their own reasoning, assess their confidence, or locate their confusion.
Score 1 if: Tutor checks understanding with a yes/no question ("Does that make sense?").
Score 0 if: No metacognitive prompting.

P12 — Scaffold Procedural Thinking
Score 2 if: Tutor breaks a process into steps and asks the student to identify or execute the NEXT step.
Score 1 if: Tutor lists steps but does not ask the student to engage with them sequentially.
Score 0 if: Absent.

P13 — Feedback and Validation
Score 2 if: Tutor gives SPECIFIC positive feedback identifying what exactly the student got right, then continues inquiry.
Score 1 if: Tutor gives generic affirmation without specifying what was correct.
Score 0 if: No feedback given.

P14 — Clarify Goals and Success Criteria
Score 2 if: Tutor explicitly states the learning objective and what understanding looks like.
Score 1 if: Tutor implies a goal without stating it clearly.
Score 0 if: Absent.

P15 — Manage Focus and Scope
Score 2 if: Tutor actively narrows or broadens the problem space AND explains why.
Score 1 if: Tutor redirects without explanation.
Score 0 if: Absent.

P16 — Hypothetical Scenario Reasoning
Score 2 if: Tutor introduces a simplified or imagined scenario to isolate one concept.
Score 1 if: Tutor uses a scenario but it is not clearly simplified for pedagogical isolation.
Score 0 if: Absent.

P17 — Evaluate Constraints and Trade-offs
Score 2 if: Tutor asks the student to weigh competing factors or identify limitations.
Score 1 if: Tutor mentions trade-offs without asking the student to reason about them.
Score 0 if: Absent.

P18 — Construct Representations
Score 2 if: Tutor explicitly asks the student to produce a diagram, recurrence, or external representation.
Score 1 if: Tutor suggests representing something without making it an explicit task.
Score 0 if: Absent.

P19 — Collaborative Inquiry
Score 2 if: Tutor models epistemic humility and invites the student to co-investigate.
Score 1 if: Tutor uses collaborative language superficially ("Let's think about...").
Score 0 if: Purely instructional with no collaborative framing.

P20 — Quantification and Estimation
Score 2 if: Tutor asks the student to estimate a magnitude or count operations before a precise answer.
Score 1 if: Tutor mentions scale without asking for student estimation.
Score 0 if: Absent.

--- SEQUENCING BONUS ---
After scoring all 20 principles, evaluate whether the principles employed
in this turn are APPROPRIATE for the turn number and student profile.
Score 1 (appropriate) or 0 (inappropriate or neutral).

Turns 1-3: Appropriate principles are P1, P5, P11, P14.
Turns 4-7: Appropriate principles are P2, P3, P8, P9, P12, P15.
Turns 8-10: Appropriate principles are P4, P7, P10, P13, P17.
P6, P16, P18, P19, P20 are always appropriate regardless of turn.

If the dominant principles in this turn match the expected phase, score 1.
If they are significantly mismatched (e.g., P4 Transfer at Turn 1), score 0.

--- OUTPUT FORMAT ---
Respond with ONLY the following JSON object. No explanation outside the JSON.
{
  "principle_scores": {
    "P1": <0, 1, or 2>, "P2": <0, 1, or 2>, "P3": <0, 1, or 2>,
    "P4": <0, 1, or 2>, "P5": <0, 1, or 2>, "P6": <0, 1, or 2>,
    "P7": <0, 1, or 2>, "P8": <0, 1, or 2>, "P9": <0, 1, or 2>,
    "P10": <0, 1, or 2>, "P11": <0, 1, or 2>, "P12": <0, 1, or 2>,
    "P13": <0, 1, or 2>, "P14": <0, 1, or 2>, "P15": <0, 1, or 2>,
    "P16": <0, 1, or 2>, "P17": <0, 1, or 2>, "P18": <0, 1, or 2>,
    "P19": <0, 1, or 2>, "P20": <0, 1, or 2>
  },
  "sequencing_appropriate": <0 or 1>,
  "dominant_principles": ["P_", "P_"],
  "weakest_dimension": "P_",
  "brief_rationale": "<2-3 sentences>"
}
"""


# ─────────────────────────────────────────────
# JUDGE
# ─────────────────────────────────────────────

def score_turn(client, conv: dict, turn: dict) -> dict:
    """Score a single tutor turn against 20 principles."""
    known_str = ", ".join(conv.get("known_topics", [])) or "nothing"

    user_prompt = (
        f"STUDENT PROFILE:\n"
        f"  Known topics: {known_str}\n"
        f"  Target concept: {conv.get('target_concept', conv.get('topic_name', ''))}\n"
        f"  Profile: {conv.get('profile_label', '')}\n\n"
        f"STUDENT MESSAGE (Turn {turn['turn']}):\n{turn['student']}\n\n"
        f"TUTOR RESPONSE (Turn {turn['turn']}/10):\n{turn['tutor']}\n\n"
        f"TURN NUMBER: {turn['turn']} of 10\n\n"
        f"Score this tutor response. Respond with ONLY the JSON object."
    )

    max_attempts = 3
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                    system_instruction=E3_JUDGE_SYSTEM_PROMPT,
                ),
            )

            raw = response.text
            if raw is None:
                # Debug: find out WHY it's None
                block_reason = "unknown"
                if hasattr(response, "candidates") and response.candidates:
                    cand = response.candidates[0]
                    if hasattr(cand, "finish_reason"):
                        block_reason = str(cand.finish_reason)
                elif hasattr(response, "prompt_feedback"):
                    block_reason = str(response.prompt_feedback)
                last_error = f"Gemini returned None (blocked: {block_reason})"
                if attempt < max_attempts:
                    time.sleep(2 * attempt)
                    continue
                break

            raw = raw.strip()
            if not raw:
                last_error = "Gemini returned empty string"
                if attempt < max_attempts:
                    time.sleep(2 * attempt)
                    continue
                break

            result = json.loads(raw)

            # Validate principle_scores
            scores = result.get("principle_scores", {})
            for p in PRINCIPLES:
                val = scores.get(p, 0)
                scores[p] = max(0, min(2, int(val)))

            return {
                "principle_scores": scores,
                "sequencing_appropriate": int(result.get("sequencing_appropriate", 0)),
                "dominant_principles": result.get("dominant_principles", []),
                "weakest_dimension": result.get("weakest_dimension", ""),
                "brief_rationale": result.get("brief_rationale", ""),
                "method": "llm",
            }
        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            if attempt < max_attempts:
                time.sleep(2 * attempt)
                continue
            break
        except Exception as e:
            last_error = str(e)
            if "429" in last_error or "RESOURCE_EXHAUSTED" in last_error:
                wait = min(2 ** attempt * 10, 60)
                print(f"    [Rate limit] Turn {turn['turn']}, waiting {wait}s...")
                time.sleep(wait)
                continue
            if attempt < max_attempts:
                time.sleep(2 * attempt)
                continue
            break

    print(f"    [Judge ERROR] Turn {turn['turn']}: {last_error}")
    return {
        "principle_scores": {p: 0 for p in PRINCIPLES},
        "sequencing_appropriate": 0,
        "dominant_principles": [],
        "weakest_dimension": "",
        "brief_rationale": f"Error: {last_error}",
        "method": "error",
    }


# ─────────────────────────────────────────────
# EVALUATE ALL
# ─────────────────────────────────────────────

def evaluate_conversations(client, conversations: list[dict]) -> list[dict]:
    """Evaluate all conversations."""
    completed_ids = set()
    results = []

    if os.path.exists(PARTIAL_PATH):
        with open(PARTIAL_PATH, "r") as f:
            results = json.load(f)
        completed_ids = {r["scenario_id"] for r in results}
        print(f"[Resume] Found {len(completed_ids)} completed. Resuming...")

    remaining = [c for c in conversations if c["scenario_id"] not in completed_ids]
    total = len(conversations)

    for idx, conv in enumerate(remaining, len(completed_ids) + 1):
        sid = conv["scenario_id"]
        print(f"\n[{idx}/{total}] {sid} | {conv.get('topic_name','')} | {conv.get('profile_label','')}")

        turn_scores = []
        for turn in conv.get("turns", []):
            scores = score_turn(client, conv, turn)
            turn_scores.append({
                "turn": turn["turn"],
                "student": turn["student"],
                "tutor": turn["tutor"],
                **scores,
            })
            active = [p for p in PRINCIPLES if scores["principle_scores"].get(p, 0) > 0]
            print(f"    T{turn['turn']}: seq={scores['sequencing_appropriate']} "
                  f"active={active[:5]}{'...' if len(active) > 5 else ''}")
            time.sleep(API_DELAY)

        results.append({
            "scenario_id": sid,
            "topic": conv["topic"],
            "topic_name": conv.get("topic_name", ""),
            "profile_id": conv.get("profile_id"),
            "profile_name": conv.get("profile_name", ""),
            "profile_label": conv.get("profile_label", ""),
            "turns": turn_scores,
        })

        with open(PARTIAL_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"    Saved {len(results)}/{total}")

    return results


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    """Compute all E3 aggregate metrics from eval.md §3.7."""
    valid = [r for r in results if r.get("turns")]
    if not valid:
        return {"error": "No valid results"}

    # Per-conversation metrics
    conv_metrics = []
    for r in valid:
        # Coverage: how many of 20 principles appear at least once (score >= 1)
        present = set()
        for t in r["turns"]:
            for p in PRINCIPLES:
                if t["principle_scores"].get(p, 0) >= 1:
                    present.add(p)
        coverage = len(present) / 20

        # Depth: mean score where principle was present
        present_scores = []
        for t in r["turns"]:
            for p in PRINCIPLES:
                s = t["principle_scores"].get(p, 0)
                if s >= 1:
                    present_scores.append(s)
        depth = sum(present_scores) / len(present_scores) if present_scores else 0

        # Sequencing
        seq_scores = [t["sequencing_appropriate"] for t in r["turns"]]
        sequencing = sum(seq_scores) / len(seq_scores) if seq_scores else 0

        conv_metrics.append({
            "scenario_id": r["scenario_id"],
            "profile_name": r["profile_name"],
            "coverage": round(coverage, 4),
            "depth": round(depth, 4),
            "sequencing": round(sequencing, 4),
        })

    # Overall
    mean_coverage = sum(c["coverage"] for c in conv_metrics) / len(conv_metrics)
    mean_depth = sum(c["depth"] for c in conv_metrics) / len(conv_metrics)
    mean_seq = sum(c["sequencing"] for c in conv_metrics) / len(conv_metrics)

    # Heatmap data: 20 principles × 10 turns
    heatmap = {p: {f"T{t}": 0.0 for t in range(1, 11)} for p in PRINCIPLES}
    counts = {p: {f"T{t}": 0 for t in range(1, 11)} for p in PRINCIPLES}
    for r in valid:
        for t in r["turns"]:
            tn = f"T{t['turn']}"
            for p in PRINCIPLES:
                s = t["principle_scores"].get(p, 0)
                heatmap[p][tn] += s
                counts[p][tn] += 1
    for p in PRINCIPLES:
        for tn in [f"T{t}" for t in range(1, 11)]:
            if counts[p][tn] > 0:
                heatmap[p][tn] = round(heatmap[p][tn] / counts[p][tn], 3)

    # Per-archetype breakdown
    by_profile = defaultdict(list)
    for c in conv_metrics:
        by_profile[c["profile_name"]].append(c)

    per_archetype = {}
    for pname, group in by_profile.items():
        per_archetype[pname] = {
            "count": len(group),
            "coverage": round(sum(c["coverage"] for c in group) / len(group), 4),
            "depth": round(sum(c["depth"] for c in group) / len(group), 4),
            "sequencing": round(sum(c["sequencing"] for c in group) / len(group), 4),
        }

    # Gap analysis: bottom 5 principles by mean score
    principle_means = {}
    for p in PRINCIPLES:
        all_scores = []
        for r in valid:
            for t in r["turns"]:
                all_scores.append(t["principle_scores"].get(p, 0))
        principle_means[p] = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0

    sorted_principles = sorted(principle_means.items(), key=lambda x: x[1])
    gap_analysis = {p: s for p, s in sorted_principles[:5]}
    all_ranked = {p: s for p, s in sorted_principles}

    return {
        "overall": {
            "total_conversations": len(valid),
            "mean_coverage": round(mean_coverage, 4),
            "mean_depth": round(mean_depth, 4),
            "mean_sequencing": round(mean_seq, 4),
        },
        "per_conversation": conv_metrics,
        "heatmap": heatmap,
        "per_archetype": per_archetype,
        "principle_means": all_ranked,
        "gap_analysis": gap_analysis,
    }


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def plot_heatmap(metrics: dict, path: str):
    """Generate the 20×10 principle distribution heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[Plot] matplotlib/numpy not installed. Skipping.")
        return

    hm = metrics["heatmap"]
    data = np.array([[hm[p][f"T{t}"] for t in range(1, 11)] for p in PRINCIPLES])

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=2)

    ax.set_xticks(range(10))
    ax.set_xticklabels([f"T{t}" for t in range(1, 11)])
    ax.set_yticks(range(20))
    ax.set_yticklabels(PRINCIPLES)

    # Add text annotations
    for i in range(20):
        for j in range(10):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center",
                    color="white" if data[i, j] > 1.2 else "black", fontsize=7)

    ax.set_xlabel("Turn Number", fontsize=12)
    ax.set_ylabel("Learning Science Principle", fontsize=12)
    ax.set_title("E3 Principle Distribution Heatmap\n(Mean Score per Principle × Turn)",
                 fontsize=14, fontweight="bold")
    plt.colorbar(im, label="Mean Score (0=Absent, 1=Weak, 2=Strong)")

    # Phase dividers
    ax.axvline(x=2.5, color="white", linewidth=2, linestyle="--", alpha=0.7)
    ax.axvline(x=6.5, color="white", linewidth=2, linestyle="--", alpha=0.7)
    ax.text(1, -1.5, "Early (T1-3)", ha="center", fontsize=9, style="italic")
    ax.text(4.5, -1.5, "Middle (T4-7)", ha="center", fontsize=9, style="italic")
    ax.text(8, -1.5, "Late (T8-10)", ha="center", fontsize=9, style="italic")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved heatmap to {path}")


def plot_per_archetype(metrics: dict, path: str):
    """Bar chart of Coverage/Depth/Sequencing by profile."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    pa = metrics["per_archetype"]
    profiles = list(pa.keys())
    labels = [p.replace("_", " ").title() for p in profiles]

    coverage = [pa[p]["coverage"] for p in profiles]
    depth = [pa[p]["depth"] for p in profiles]
    sequencing = [pa[p]["sequencing"] for p in profiles]

    x = np.arange(len(profiles))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w, coverage, w, label="Coverage", color="#3498db")
    ax.bar(x, depth, w, label="Depth", color="#e74c3c")
    ax.bar(x + w, sequencing, w, label="Sequencing", color="#2ecc71")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("E3 Metrics by Student Archetype", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 2.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved per-archetype to {path}")


def plot_gap_analysis(metrics: dict, path: str):
    """Bar chart of all 20 principles ranked by mean score."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    pm = metrics["principle_means"]
    principles = list(pm.keys())
    scores = list(pm.values())

    colors = ["#e74c3c" if s == scores[i] and i < 5 else "#3498db"
              for i, s in enumerate(scores)]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(principles, scores, color=colors)
    ax.set_ylabel("Mean Score (0-2)")
    ax.set_title("E3 Principle Gap Analysis\n(Red = Bottom 5 Gaps)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 2.1)
    ax.grid(axis="y", alpha=0.3)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{score:.2f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved gap analysis to {path}")


# ─────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────

def print_report(metrics: dict):
    ov = metrics["overall"]
    print("\n" + "=" * 70)
    print("  E3 EVALUATION REPORT — Learning Sciences Principles")
    print("=" * 70)
    print(f"\n  Conversations: {ov['total_conversations']}")
    print(f"\n  ── Overall Metrics ──")
    print(f"    Coverage Score:   {ov['mean_coverage']:.3f}  (target ≥ 0.60)")
    print(f"    Depth Score:      {ov['mean_depth']:.3f}  (target ≥ 1.50)")
    print(f"    Sequencing:       {ov['mean_sequencing']:.3f}  (target ≥ 0.70)")

    print(f"\n  ── Per-Archetype ──")
    print(f"    {'Profile':<25} {'Coverage':<10} {'Depth':<10} {'Sequencing':<10}")
    print(f"    {'-'*55}")
    for pname, data in metrics.get("per_archetype", {}).items():
        label = pname.replace("_", " ").title()
        print(f"    {label:<25} {data['coverage']:<10.3f} {data['depth']:<10.3f} {data['sequencing']:<10.3f}")

    print(f"\n  ── Gap Analysis (Bottom 5) ──")
    for p, s in metrics.get("gap_analysis", {}).items():
        print(f"    {p}: {s:.3f}")

    print("\n" + "=" * 70)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("E3 Judge — Learning Sciences Principle Evaluation (Multi-Model)")
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    client = genai.Client(api_key=GEMINI_API_KEY)
    all_metrics = {}  # model_name -> metrics dict

    for model_info in MODEL_NAMES:
        model_name = model_info["name"]
        model_label = model_info["label"]
        conv_path = os.path.join(SCRIPT_DIR, f"e3_conversations_{model_name}.json")
        results_path = os.path.join(SCRIPT_DIR, f"e3_results_{model_name}.json")
        metrics_path = os.path.join(SCRIPT_DIR, f"e3_metrics_{model_name}.json")
        heatmap_path = os.path.join(SCRIPT_DIR, f"e3_heatmap_{model_name}.png")
        archetype_path = os.path.join(SCRIPT_DIR, f"e3_archetype_{model_name}.png")
        gap_path = os.path.join(SCRIPT_DIR, f"e3_gap_{model_name}.png")

        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_label} ({model_name})")
        print(f"  Input: {conv_path}")
        print(f"{'='*60}")

        # Try per-model file first, then legacy fallback
        if not os.path.exists(conv_path):
            if os.path.exists(LEGACY_CONVERSATIONS_PATH):
                with open(LEGACY_CONVERSATIONS_PATH, "r") as f:
                    all_convs = json.load(f)
                model_convs = [c for c in all_convs if c.get("model_name") == model_name]
                if not model_convs:
                    print(f"  SKIP: No conversation file for {model_name}.")
                    continue
                conversations = model_convs
            else:
                print(f"  SKIP: {conv_path} not found.")
                continue
        else:
            with open(conv_path, "r") as f:
                conversations = json.load(f)

        valid = [c for c in conversations if c.get("turns") and len(c["turns"]) > 0]
        print(f"  Loaded {len(conversations)} conversations ({len(valid)} valid)")

        if not valid:
            print(f"  ERROR: No valid conversations for {model_name}.")
            continue

        results = evaluate_conversations(client, valid)

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        metrics = compute_metrics(results)
        all_metrics[model_name] = {"label": model_label, "metrics": metrics}

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        plot_heatmap(metrics, heatmap_path)
        plot_per_archetype(metrics, archetype_path)
        plot_gap_analysis(metrics, gap_path)
        print_report(metrics)

    # ── Comparative Summary ──
    if len(all_metrics) > 1:
        print("\n" + "=" * 70)
        print("  E3 COMPARATIVE SUMMARY — SFT vs DPO")
        print("=" * 70)
        print(f"\n  {'Model':<25} {'Coverage':>10} {'Depth':>10} {'Sequencing':>12}")
        print(f"  {'-'*57}")
        for model_info in MODEL_NAMES:
            mname = model_info["name"]
            if mname in all_metrics:
                m = all_metrics[mname]["metrics"]
                ov = m.get("overall", {})
                cov = ov.get('mean_coverage', 0)
                dep = ov.get('mean_depth', 0)
                seq = ov.get('mean_sequencing', 0)
                print(f"  {model_info['label']:<25} {cov:>10.3f} {dep:>10.3f} {seq:>12.3f}")
        print(f"\n  Targets: Coverage ≥ 0.60, Depth ≥ 1.50, Sequencing ≥ 0.70")
        print("\n" + "=" * 70)

    # Save merged metrics
    merged_path = os.path.join(SCRIPT_DIR, "e3_metrics_all.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"  Merged metrics saved to {merged_path}")

    if os.path.exists(PARTIAL_PATH):
        os.remove(PARTIAL_PATH)


if __name__ == "__main__":
    main()
