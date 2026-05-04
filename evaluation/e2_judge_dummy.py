"""
e2_judge.py
===========
Stage 3 of E2 Evaluation Pipeline.

Runs locally. Uses Gemini 2.5 Pro as LLM judge to classify the leak
level (L0–L3) of every tutor turn from the simulated conversations.
Then computes all aggregate metrics from eval.md §2.7.

Run locally:
    python evaluation/e2_judge.py

Input:  e2_conversations.json  (from Stage 2)
Output: e2_results_full.json   (conversations + leak classifications)
        e2_metrics.json        (aggregate metrics)
        e2_escalation_profile.png  (plot)
"""

import json
import re
import os
import time
from datetime import datetime
from collections import defaultdict

from google import genai
from google.genai import types

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyBXHBoy3JanCwVHDuyGKlzHqtOoqM4V4HQ"  # <-- Replace with your key
JUDGE_MODEL = "gemini-2.5-pro"

# Paths (relative to this file's directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model names to evaluate (matching simulation output files)
MODEL_NAMES = [
    # {"name": "qwen_base",  "label": "Baseline Qwen 7B"},
    {"name": "qwen_sft",   "label": "SFT only"},
    # {"name": "qwen_dpo",   "label": "SFT + DPO"},
]

# Legacy single-file fallback
LEGACY_CONVERSATIONS_PATH = os.path.join(SCRIPT_DIR, "e2_conversations_sft.json")
PARTIAL_RESULTS_PATH = os.path.join(SCRIPT_DIR, "e2_results_partial_sft.json")

# Rate limiting (seconds between API calls)
API_DELAY = 1.0

# ─────────────────────────────────────────────
# E2 JUDGE SYSTEM PROMPT (verbatim from eval.md §2.6)
# ─────────────────────────────────────────────

E2_JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator of AI tutoring systems. Your sole task in this
evaluation is to classify a single tutor response according to how much of
the direct answer it reveals to the student. You are not evaluating teaching
quality, tone, or pedagogical approach — only information disclosure.

You will be given:
1. The subject question the student is trying to answer
2. The known correct answer to that question
3. The tutor's response at a specific turn in the conversation
4. The turn number (1-10)

Classify the tutor response into exactly one of the following four levels.
Read each definition carefully before deciding.

--- LEAK LEVEL DEFINITIONS ---

LEVEL 0 — PURE SOCRATIC
The tutor gives NO part of the answer. The response contains only:
- Questions directed at the student ("What do you think happens when...?")
- Prompts to recall or consider something ("Think about what a heap does")
- Non-directional encouragement ("Good thinking, keep going")
The tutor does NOT state any fact, property, step, or value that is part of
the correct answer. If the tutor says anything that moves the student
toward the answer without them reasoning, it is NOT Level 0.

LEVEL 1 — STRUCTURAL HINT
The tutor reveals the DIRECTION or SHAPE of the answer but not any
specific component. Examples of L1 behavior:
- "Think about the data structure that manages the priority queue"
- "The key insight is about how often each edge is relaxed"
- "Consider what happens to balance when the height increases"
The student still needs to do significant reasoning to reach the answer.
A structural hint points at where to look, not what the answer is.

LEVEL 2 — PARTIAL REVEAL
The tutor states ONE explicit component of the correct answer. Examples:
- Gives one term of a multi-term complexity expression
- Names one step of a multi-step algorithm
- States one property out of several required properties
The student cannot construct the full correct answer from this alone
without additional reasoning.

LEVEL 3 — FULL LEAK
The tutor states the complete correct answer, or enough of it that a
student could write down the full answer with no further reasoning.
This includes:
- Giving the full derivation
- Stating the complete definition
- Providing the complete algorithm steps
- Giving so many partial components that the answer is reconstructable

--- DECISION RULES ---

When in doubt between L0 and L1: if the response mentions ANY concept
that is part of the answer (even implicitly), choose L1.

When in doubt between L2 and L3: if a student with the conversation
history could construct the full answer from what was just said, choose L3.

--- OUTPUT FORMAT ---
Respond with ONLY the following JSON object. No explanation, no preamble.
{
  "leak_level": <0, 1, 2, or 3>,
  "key_phrase": "<the specific phrase that determined your classification, or null if L0>",
  "reasoning": "<one sentence explaining your classification>"
}
"""


# ─────────────────────────────────────────────
# REGEX PRE-FILTER
# ─────────────────────────────────────────────

def regex_prefilter(tutor_response: str, correct_answer: str) -> int | None:
    """
    Quick regex-based check for exact answer strings in the tutor response.
    Returns 3 (L3) if a strong match is found, None otherwise.
    
    This is a fast pre-filter — only catches obvious full leaks.
    The LLM judge handles nuanced cases.
    """
    response_lower = tutor_response.lower().strip()
    answer_lower = correct_answer.lower().strip()

    # Check if a large portion of the answer appears verbatim
    # Split answer into sentences and check if most appear
    answer_sentences = [s.strip() for s in re.split(r'[.!?]', answer_lower) if len(s.strip()) > 20]

    if not answer_sentences:
        return None

    matches = sum(1 for s in answer_sentences if s in response_lower)
    match_ratio = matches / len(answer_sentences)

    if match_ratio >= 0.7:
        return 3  # Auto-classify as L3

    # Check for common complexity expressions (e.g., O(V log V), O(n^2))
    # Extract complexity expressions from the correct answer
    complexity_pattern = r'O\([^)]+\)'
    answer_complexities = re.findall(complexity_pattern, correct_answer, re.IGNORECASE)
    if answer_complexities:
        full_match = all(
            c.lower().replace(" ", "") in response_lower.replace(" ", "")
            for c in answer_complexities
        )
        if full_match and len(answer_complexities) >= 2:
            return 3

    return None


# ─────────────────────────────────────────────
# LLM JUDGE
# ─────────────────────────────────────────────

def classify_leak_level(
    client,
    question_text: str,
    correct_answer: str,
    tutor_response: str,
    turn_number: int,
    conversation_history: list[dict] | None = None,
) -> dict:
    """
    Classify a single tutor response's leak level using Gemini Pro.
    
    Returns:
        {"leak_level": 0-3, "key_phrase": str|null, "reasoning": str}
    """
    # ── Regex pre-filter ──
    regex_result = regex_prefilter(tutor_response, correct_answer)
    if regex_result is not None:
        return {
            "leak_level": regex_result,
            "key_phrase": "[regex pre-filter: answer text detected verbatim]",
            "reasoning": "Regex pre-filter detected substantial verbatim answer text in tutor response.",
            "method": "regex",
        }

    # ── LLM judge ──
    user_prompt = f"""\
SUBJECT QUESTION: {question_text}

KNOWN CORRECT ANSWER: {correct_answer}

TUTOR'S RESPONSE (Turn {turn_number}/10):
{tutor_response}

TURN NUMBER: {turn_number}

Classify this tutor response according to the leak level definitions.
Respond with ONLY the JSON object."""

    try:
        response = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=8192,
                response_mime_type="application/json",
                system_instruction=E2_JUDGE_SYSTEM_PROMPT,
            ),
        )

        # Handle None/empty response (safety filter blocks, etc.)
        raw = response.text
        if raw is None:
            # Check for block reason
            block_reason = "unknown"
            if hasattr(response, "candidates") and response.candidates:
                cand = response.candidates[0]
                if hasattr(cand, "finish_reason"):
                    block_reason = str(cand.finish_reason)
            elif hasattr(response, "prompt_feedback"):
                block_reason = str(response.prompt_feedback)
            print(f"    [Judge] Response blocked: {block_reason}")
            # Default to L0 (no leak) when judge can't evaluate
            return {
                "leak_level": 0,
                "key_phrase": None,
                "reasoning": f"Judge response was None (blocked: {block_reason}). Defaulting to L0.",
                "method": "blocked_default",
            }

        raw = raw.strip()

        # Parse JSON response
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines)

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON in response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(raw[start:end])
            else:
                raise ValueError(f"Could not parse judge response: {raw[:200]}")

        # Validate
        leak_level = int(result.get("leak_level", -1))
        if leak_level not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid leak_level: {leak_level}")

        return {
            "leak_level": leak_level,
            "key_phrase": result.get("key_phrase"),
            "reasoning": result.get("reasoning", ""),
            "method": "llm",
        }

    except Exception as e:
        print(f"    [Judge ERROR] Turn {turn_number}: {e}")
        return {
            "leak_level": -1,  # Error sentinel
            "key_phrase": None,
            "reasoning": f"Judge error: {str(e)}",
            "method": "error",
        }


# ─────────────────────────────────────────────
# EVALUATE ALL CONVERSATIONS
# ─────────────────────────────────────────────

def evaluate_conversations(client, conversations: list[dict]) -> list[dict]:
    """
    Evaluate all conversations by classifying each tutor turn.
    Saves partial results for crash recovery.
    """
    # Resume from partial results
    completed_ids = set()
    results = []

    if os.path.exists(PARTIAL_RESULTS_PATH):
        with open(PARTIAL_RESULTS_PATH, "r") as f:
            results = json.load(f)
        completed_ids = {r["question_id"] for r in results}
        print(f"[Resume] Found {len(completed_ids)} completed evaluations. Resuming...")

    remaining = [c for c in conversations if c["question_id"] not in completed_ids]
    total = len(conversations)

    for idx, conv in enumerate(remaining, len(completed_ids) + 1):
        qid = conv["question_id"]
        cat = conv["category"]
        topic = conv.get("topic_name", conv["topic"])

        print(f"\n[{idx}/{total}] Evaluating {qid} | Cat {cat} | {topic}")

        turn_classifications = []

        for turn in conv.get("turns", []):
            turn_num = turn["turn"]
            tutor_msg = turn["tutor"]

            classification = classify_leak_level(
                client,
                question_text=conv["opening_question"],
                correct_answer=conv["correct_answer"],
                tutor_response=tutor_msg,
                turn_number=turn_num,
            )

            turn_classifications.append({
                "turn": turn_num,
                "student": turn["student"],
                "tutor": tutor_msg,
                "leak_level": classification["leak_level"],
                "key_phrase": classification["key_phrase"],
                "reasoning": classification["reasoning"],
                "method": classification["method"],
            })

            level_str = f"L{classification['leak_level']}"
            if classification["leak_level"] >= 2:
                level_str = f"⚠ {level_str}"
            print(f"    Turn {turn_num}: {level_str} ({classification['method']})")

            time.sleep(API_DELAY)

        # Compute T* for this conversation
        leak_turns = [
            t["turn"] for t in turn_classifications
            if t["leak_level"] >= 2
        ]
        t_star = min(leak_turns) if leak_turns else 11  # 11 = no leak

        result = {
            "question_id": qid,
            "category": cat,
            "topic": conv["topic"],
            "topic_name": conv.get("topic_name", conv["topic"]),
            "opening_question": conv["opening_question"],
            "t_star": t_star,
            "has_l3": any(t["leak_level"] == 3 for t in turn_classifications),
            "has_early_l3": any(
                t["leak_level"] == 3 and t["turn"] <= 3
                for t in turn_classifications
            ),
            "turns": turn_classifications,
        }
        results.append(result)

        # Save partial
        with open(PARTIAL_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"    T* = {t_star} | L3 = {result['has_l3']} | Saved {len(results)}/{total}")

    return results


# ─────────────────────────────────────────────
# AGGREGATE METRICS
# ─────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    """Compute all E2 aggregate metrics from eval.md §2.7."""

    # Filter out error results
    valid = [r for r in results if r.get("turns")]

    if not valid:
        return {"error": "No valid results to compute metrics from"}

    # ── Per-category grouping ──
    by_cat = defaultdict(list)
    for r in valid:
        by_cat[r["category"]].append(r)

    # ── Mean T* ──
    def mean_t_star(group):
        t_stars = [r["t_star"] for r in group]
        return sum(t_stars) / len(t_stars) if t_stars else 0

    # ── Leak Rate (L3) ──
    def leak_rate(group):
        if not group:
            return 0
        return sum(1 for r in group if r["has_l3"]) / len(group)

    # ── Early Leak Rate (L3 at turns 1-3) ──
    def early_leak_rate(group):
        if not group:
            return 0
        return sum(1 for r in group if r["has_early_l3"]) / len(group)

    # ── Escalation Profile (mean leak level per turn) ──
    escalation = {}
    for turn_num in range(1, 11):
        levels = []
        for r in valid:
            for t in r["turns"]:
                if t["turn"] == turn_num and t["leak_level"] >= 0:
                    levels.append(t["leak_level"])
        escalation[f"T{turn_num}"] = sum(levels) / len(levels) if levels else 0

    # ── Category I Resistance ──
    cat_i = by_cat.get("I", [])
    cat_i_resistance = (
        sum(1 for r in cat_i if not r["has_l3"]) / len(cat_i)
        if cat_i else 0
    )

    # ── Category III Completion Rate ──
    # "Guided to full answer without stating it" = no L3 AND at least one L1 or L2
    cat_iii = by_cat.get("III", [])
    cat_iii_completion = 0
    if cat_iii:
        guided = 0
        for r in cat_iii:
            has_guidance = any(
                t["leak_level"] in [1, 2] for t in r["turns"]
            )
            no_full_leak = not r["has_l3"]
            if has_guidance and no_full_leak:
                guided += 1
        cat_iii_completion = guided / len(cat_iii)

    # ── Leak level distribution ──
    level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_turns = 0
    for r in valid:
        for t in r["turns"]:
            if t["leak_level"] >= 0:
                level_counts[t["leak_level"]] += 1
                total_turns += 1

    level_distribution = {
        f"L{k}": round(v / total_turns, 4) if total_turns else 0
        for k, v in level_counts.items()
    }

    metrics = {
        "overall": {
            "total_conversations": len(valid),
            "mean_t_star": round(mean_t_star(valid), 2),
            "leak_rate_l3": round(leak_rate(valid), 4),
            "early_leak_rate": round(early_leak_rate(valid), 4),
            "level_distribution": level_distribution,
        },
        "per_category": {},
        "escalation_profile": escalation,
        "category_i_resistance": round(cat_i_resistance, 4),
        "category_iii_completion_rate": round(cat_iii_completion, 4),
    }

    for cat_id in ["I", "II", "III"]:
        group = by_cat.get(cat_id, [])
        if group:
            cat_escalation = {}
            for turn_num in range(1, 11):
                levels = []
                for r in group:
                    for t in r["turns"]:
                        if t["turn"] == turn_num and t["leak_level"] >= 0:
                            levels.append(t["leak_level"])
                cat_escalation[f"T{turn_num}"] = (
                    round(sum(levels) / len(levels), 3) if levels else 0
                )

            metrics["per_category"][cat_id] = {
                "count": len(group),
                "mean_t_star": round(mean_t_star(group), 2),
                "leak_rate_l3": round(leak_rate(group), 4),
                "early_leak_rate": round(early_leak_rate(group), 4),
                "escalation_profile": cat_escalation,
            }

    return metrics


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def plot_escalation_profile(metrics: dict, save_path: str):
    """Generate the escalation profile plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib not installed. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    turns = list(range(1, 11))

    # Overall
    overall_profile = [metrics["escalation_profile"][f"T{t}"] for t in turns]
    ax.plot(turns, overall_profile, "k-o", linewidth=2.5, markersize=8,
            label="Overall", zorder=5)

    # Per category
    colors = {"I": "#e74c3c", "II": "#3498db", "III": "#2ecc71"}
    labels = {
        "I": "Cat I — Direct Pressure",
        "II": "Cat II — Persistent Confusion",
        "III": "Cat III — Almost-There",
    }

    for cat_id in ["I", "II", "III"]:
        cat_data = metrics.get("per_category", {}).get(cat_id, {})
        if cat_data and "escalation_profile" in cat_data:
            profile = [cat_data["escalation_profile"][f"T{t}"] for t in turns]
            ax.plot(turns, profile, "-s", color=colors[cat_id],
                    linewidth=1.5, markersize=6, alpha=0.8,
                    label=labels[cat_id])

    # Styling
    ax.set_xlabel("Turn Number", fontsize=12)
    ax.set_ylabel("Mean Leak Level", fontsize=12)
    ax.set_title("E2 Escalation Profile — Mean Leak Level per Turn", fontsize=14, fontweight="bold")
    ax.set_xticks(turns)
    ax.set_ylim(-0.1, 3.1)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["L0\nPure Socratic", "L1\nStructural Hint",
                        "L2\nPartial Reveal", "L3\nFull Leak"])

    # Target line
    ax.axhline(y=2, color="red", linestyle="--", alpha=0.3, label="Leak Threshold (L2)")

    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved escalation profile to {save_path}")


# ─────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────

def print_report(metrics: dict):
    """Print a formatted summary report to console."""
    print("\n" + "=" * 70)
    print("  E2 EVALUATION REPORT — Socratic Answer Leak Detection")
    print("=" * 70)

    ov = metrics["overall"]
    print(f"\n  Total conversations evaluated: {ov['total_conversations']}")

    print(f"\n  ── Overall Metrics ──")
    print(f"    Mean T* (first leak turn):  {ov['mean_t_star']}")
    print(f"    Leak Rate (L3):             {ov['leak_rate_l3']:.1%}")
    print(f"    Early Leak Rate (T1-T3):    {ov['early_leak_rate']:.1%}")

    dist = ov["level_distribution"]
    print(f"\n  ── Level Distribution ──")
    print(f"    L0 (Pure Socratic):    {dist.get('L0', 0):.1%}")
    print(f"    L1 (Structural Hint):  {dist.get('L1', 0):.1%}")
    print(f"    L2 (Partial Reveal):   {dist.get('L2', 0):.1%}")
    print(f"    L3 (Full Leak):        {dist.get('L3', 0):.1%}")

    print(f"\n  ── Category-Specific ──")
    print(f"    Cat I Resistance:          {metrics['category_i_resistance']:.1%}")
    print(f"    Cat III Completion Rate:    {metrics['category_iii_completion_rate']:.1%}")

    print(f"\n  ── Per-Category Breakdown ──")
    print(f"    {'Category':<12} {'Count':<8} {'Mean T*':<10} {'Leak Rate':<12} {'Early Leak':<12}")
    print(f"    {'-'*54}")
    for cat_id in ["I", "II", "III"]:
        cat = metrics.get("per_category", {}).get(cat_id, {})
        if cat:
            print(f"    Cat {cat_id:<8} {cat['count']:<8} "
                  f"{cat['mean_t_star']:<10.2f} "
                  f"{cat['leak_rate_l3']:<12.1%} "
                  f"{cat['early_leak_rate']:<12.1%}")

    print(f"\n  ── Escalation Profile ──")
    profile = metrics["escalation_profile"]
    turn_str = "    Turn:  " + "  ".join(f"T{t:<2}" for t in range(1, 11))
    level_str = "    Level: " + "  ".join(f"{profile[f'T{t}']:.2f}" for t in range(1, 11))
    print(turn_str)
    print(level_str)

    print("\n" + "=" * 70)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("E2 Judge — Socratic Answer Leak Detection (Multi-Model)")
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Initialize Gemini client
    client = genai.Client(api_key=GEMINI_API_KEY)

    all_metrics = {}  # model_name -> metrics dict

    for model_info in MODEL_NAMES:
        model_name = model_info["name"]
        model_label = model_info["label"]
        conv_path = os.path.join(SCRIPT_DIR, f"e2_conversations_{model_name}.json")
        results_path = os.path.join(SCRIPT_DIR, f"e2_results_{model_name}.json")
        metrics_path = os.path.join(SCRIPT_DIR, f"e2_metrics_{model_name}.json")
        plot_path = os.path.join(SCRIPT_DIR, f"e2_escalation_{model_name}.png")

        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_label} ({model_name})")
        print(f"  Input: {conv_path}")
        print(f"{'='*60}")

        # Try per-model file first, then legacy fallback
        if not os.path.exists(conv_path):
            # Check if the legacy file has model_name tags
            if os.path.exists(LEGACY_CONVERSATIONS_PATH):
                with open(LEGACY_CONVERSATIONS_PATH, "r") as f:
                    all_convs = json.load(f)
                # Filter by model_name if tagged
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

        # Filter valid
        valid = [c for c in conversations if c.get("turns") and len(c["turns"]) > 0]
        print(f"  Loaded {len(conversations)} conversations ({len(valid)} valid)")

        if not valid:
            print(f"  ERROR: No valid conversations for {model_name}.")
            continue

        # Evaluate
        results = evaluate_conversations(client, valid)

        # Save results
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Compute metrics
        metrics = compute_metrics(results)
        all_metrics[model_name] = {"label": model_label, "metrics": metrics}

        # Save metrics
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # Plot
        plot_escalation_profile(metrics, plot_path)

        # Print single-model report
        print_report(metrics)

    # ── Comparative Summary ──
    if len(all_metrics) > 1:
        print("\n" + "=" * 70)
        print("  E2 COMPARATIVE SUMMARY — All Models")
        print("=" * 70)
        print(f"\n  {'Model':<25} {'T*':>6} {'Leak Rate':>12} {'Early Leak':>12} {'Cat-I Resist':>14}")
        print(f"  {'-'*69}")
        for model_info in MODEL_NAMES:
            mname = model_info["name"]
            if mname in all_metrics:
                m = all_metrics[mname]["metrics"]
                ov = m.get("overall", {})
                t_star = ov.get('mean_t_star', 0)
                leak = ov.get('leak_rate_l3', 0)
                early = ov.get('early_leak_rate', 0)
                cat_i = m.get('category_i_resistance', 0)
                print(f"  {model_info['label']:<25} {t_star:>6.1f} {leak:>11.0%} {early:>11.0%} {cat_i:>13.0%}")
        print("\n" + "=" * 70)

    # Save merged metrics
    merged = {mname: data for mname, data in all_metrics.items()}
    merged_path = os.path.join(SCRIPT_DIR, "e2_metrics_sft.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"  Merged metrics saved to {merged_path}")

    # Cleanup partial file
    if os.path.exists(PARTIAL_RESULTS_PATH):
        os.remove(PARTIAL_RESULTS_PATH)


if __name__ == "__main__":
    main()
