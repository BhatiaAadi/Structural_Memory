"""
e1_simulation.py
================
Stage 2 of E1 Evaluation Pipeline.

Runs on Kaggle (GPU required). Loads 3 candidate models and runs the full
SG+KG context pipeline for each of the 12 user profiles. Single-turn only.

Input:  e1_profiles.json  (from Stage 1)
Output: e1_responses.json (consumed by Stage 3)
"""


# === Cell 1 (code): Install dependencies ===
# NOTE: On Kaggle, bitsandbytes needs special handling.
# Run these in separate cells:
#
# Cell 1a:
# !pip install -U bitsandbytes>=0.46.1 --no-deps --quiet
#
# Cell 1b:
# !pip install transformers>=4.45.0 peft>=0.13.0 accelerate einops \
#     huggingface_hub --upgrade --quiet
#
# Cell 1c (if bitsandbytes still fails):
# import bitsandbytes; print(bitsandbytes.__version__)  # should be >= 0.46.1


# === Cell 2 (code): HuggingFace login ===
from huggingface_hub import login

HF_TOKEN = "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"  # <-- Replace with your HF token
login(token=HF_TOKEN)
print("Logged in to HuggingFace.")


# === Cell 3 (code): Configuration ===
import os
import json
import time
import torch
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ─────────────────────────────────────────────
# MODEL CONFIG — 4 models to compare
# ─────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SFT_MODEL_REPO = "Aryan3it/socratic-tutor-qwen2.5-7b"  # Merged SFT
DPO_ADAPTER_REPO = "Aryan3it/socratic-tutor-qwen2.5-7b_dpo_lora"  # DPO LoRA on base

MODELS = [
    {
        "name": "qwen_base_no_rag",
        "label": "Qwen 2.5 7B (Base, No RAG)",
        "type": "base",
        "repo": BASE_MODEL,
        "no_rag": True,
    },
    {
        "name": "qwen_base",
        "label": "Qwen 2.5 7B (Base)",
        "type": "base",
        "repo": BASE_MODEL,
    },
    {
        "name": "qwen_sft",
        "label": "Qwen 2.5 7B (SFT)",
        "type": "merged",
        "repo": SFT_MODEL_REPO,
    },
    {
        "name": "qwen_dpo",
        "label": "Qwen 2.5 7B (SFT + DPO)",
        "type": "lora",
        "base_repo": BASE_MODEL,
        "adapter_repo": DPO_ADAPTER_REPO,
    },
]

MAX_MEMORY = {0: "13GiB", 1: "13GiB", "cpu": "30GiB"}

# ─────────────────────────────────────────────
# EXPERIMENT CONFIG
# ─────────────────────────────────────────────
DRY_RUN = True           # Set to False for full run
DRY_RUN_COUNT = 3        # Number of profiles per model in dry run
MAX_NEW_TOKENS = 1024    # Longer for single-turn explanations
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# ─────────────────────────────────────────────
# TUTOR SYSTEM PROMPT (same as training)
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a Socratic tutor specializing in Data Structures and Algorithms. "
    "Guide students to discover answers themselves through targeted questions and "
    "scaffolding. Never state the answer directly. Always end your turn with a "
    "question that advances the student's thinking."
)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
PROFILES_PATH     = "e1_profiles.json"
RESPONSES_PATH    = "e1_responses.json"
PARTIAL_SAVE_PATH = "e1_responses_partial.json"

print(f"[Config] DRY_RUN = {DRY_RUN}")
print(f"[Config] Models: {[m['name'] for m in MODELS]}")
print(f"[Config] Max new tokens: {MAX_NEW_TOKENS}")


# === Cell 4 (code): Load profiles ===
print("\n[Profiles] Loading profiles ...")

with open(PROFILES_PATH, "r") as f:
    profile_bank = json.load(f)

profiles = profile_bank["profiles"]
print(f"[Profiles] Loaded {len(profiles)} profiles")

if DRY_RUN:
    # Pick one from each archetype for dry run
    dry_profiles = []
    seen_archetypes = set()
    for p in profiles:
        if p["archetype"] not in seen_archetypes and len(dry_profiles) < DRY_RUN_COUNT:
            dry_profiles.append(p)
            seen_archetypes.add(p["archetype"])
    profiles = dry_profiles[:DRY_RUN_COUNT]
    print(f"[DRY RUN] Using {len(profiles)} profiles:")
    for p in profiles:
        print(f"  {p['user_id']} | Archetype {p['archetype']} | {p['target_question'][:50]}...")


# === Cell 5 (code): Context builder (from tutor-context.py) ===

MASTERY_THRESHOLD = 0.65

def get_node(user_sg: dict, sg_id: str) -> dict | None:
    for node in user_sg["nodes"]:
        if node["id"] == sg_id:
            return node
    return None

def check_prerequisites(user_sg: dict, sg_id: str) -> dict:
    node = get_node(user_sg, sg_id)
    met, unmet = [], []
    for req_id in node["sg_requires"]:
        req_node = get_node(user_sg, req_id)
        if req_node and req_node["mastery"] >= MASTERY_THRESHOLD:
            met.append(req_node["name"])
        else:
            unmet.append(req_node["name"] if req_node else req_id)
    return {"met": met, "unmet": unmet}

def user_level(user_sg: dict) -> str:
    scores = [n["mastery"] for n in user_sg["nodes"]]
    avg = sum(scores) / len(scores) if scores else 0
    if avg < 0.35: return "beginner"
    if avg < 0.65: return "intermediate"
    return "advanced"

def find_sg_node_for_query(user_sg: dict, query: str) -> dict | None:
    q = query.lower()
    for node in user_sg["nodes"]:
        if any(alias in q or q in alias for alias in node.get("kg_search_aliases", [])):
            return node
    return None

def build_context(user_sg: dict, sg_node: dict) -> str:
    """
    Build the full context string from the user's SG.
    Identical logic to USER/tutor-context.py:build_context().
    """
    anchor = sg_node.get("kg_anchor") or {}
    prereqs = check_prerequisites(user_sg, sg_node["id"])
    level = user_level(user_sg)

    # Analogy bridges: SG nodes the user knows that are KG prerequisites
    known_kg_ids = {
        n["kg_anchor"]["kg_id"]
        for n in user_sg["nodes"]
        if n["mastery"] >= 0.6 and n.get("kg_anchor") and n["kg_anchor"]
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
        f"User    : {user_sg.get('user', 'unknown')}",
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


print("[Context Builder] Ready.")


# === Cell 6 (code): Model loading utilities ===
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(model_config: dict):
    """Load a model based on its config. Returns (model, tokenizer)."""
    print(f"\n[Model] Loading: {model_config['label']}")

    # ── Try bitsandbytes 4-bit quantization, fall back to bfloat16 ──
    USE_4BIT = False
    bnb_config = None
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes as bnb
        from packaging import version
        bnb_ver = bnb.__version__
        if version.parse(bnb_ver) >= version.parse("0.46.1"):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            USE_4BIT = True
            print(f"  bitsandbytes {bnb_ver} — using 4-bit")
        else:
            print(f"  bitsandbytes {bnb_ver} too old — using bfloat16")
    except (ImportError, Exception):
        print("  bitsandbytes not available — using bfloat16")

    load_kwargs = {
        "device_map": "auto",
        "max_memory": MAX_MEMORY,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    if USE_4BIT and bnb_config:
        load_kwargs["quantization_config"] = bnb_config

    model_type = model_config["type"]

    if model_type == "base":
        repo = model_config["repo"]
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(repo, **load_kwargs)

    elif model_type == "merged":
        repo = model_config["repo"]
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(repo, **load_kwargs)

    elif model_type == "lora":
        base_repo = model_config["base_repo"]
        adapter_repo = model_config["adapter_repo"]
        tokenizer = AutoTokenizer.from_pretrained(base_repo, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(base_repo, **load_kwargs)
        base_model.config.use_cache = True
        model = PeftModel.from_pretrained(base_model, adapter_repo, is_trainable=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.eval()
    print(f"  Ready. 4-bit={USE_4BIT}. Device: {next(model.parameters()).device}")

    return model, tokenizer


def unload_model(model, tokenizer):
    """Free GPU memory after using a model."""
    del model
    del tokenizer
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    print("  Model unloaded. GPU memory freed.")


def generate_response(model, tokenizer, context: str, question: str) -> str:
    """Generate a single tutor response given context and question.
    
    For no-RAG baseline: pass context="" and question=<bare question>.
    The system prompt adapts based on whether context is provided.
    """
    if context and context.strip():
        system_msg = (
            "You are an adaptive DSA tutor. Use the knowledge graph context "
            "and user state provided to personalise every response."
        )
        user_content = f"{context}\n\nStudent: {question}"
    else:
        # No RAG context — plain Socratic tutor
        system_msg = SYSTEM_PROMPT
        user_content = question

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    # Clean Qwen markers
    for marker in ["<|im_start|>", "<|im_end|>", "<|im_start|>assistant\n",
                    "<|im_start|>user\n", "<|im_start|>system\n"]:
        response = response.replace(marker, "")

    del inputs, outputs
    torch.cuda.empty_cache()

    return response.strip()


print("[Model Utils] Ready.")


# === Cell 7 (code): Run E1 evaluation ===

print("\n" + "=" * 60)
print(f"  STARTING E1 EVALUATION — {'DRY RUN' if DRY_RUN else 'FULL RUN'}")
print(f"  Profiles: {len(profiles)}")
print(f"  Models: {len(MODELS)}")
print(f"  Total responses: {len(profiles) * len(MODELS)}")
print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ── Resume from partial results ──
completed_keys = set()
all_results = []

if os.path.exists(PARTIAL_SAVE_PATH):
    with open(PARTIAL_SAVE_PATH, "r") as f:
        all_results = json.load(f)
    completed_keys = {(r["user_id"], r["model_name"]) for r in all_results}
    print(f"[Resume] Found {len(completed_keys)} completed responses. Resuming...")

# ── Run each model ──
for model_idx, model_config in enumerate(MODELS, 1):
    model_name = model_config["name"]
    print(f"\n{'='*60}")
    print(f"  MODEL {model_idx}/{len(MODELS)}: {model_config['label']}")
    print(f"{'='*60}")

    # Check if all profiles for this model are already done
    remaining = [p for p in profiles if (p["user_id"], model_name) not in completed_keys]
    if not remaining:
        print(f"  All profiles already completed for {model_name}. Skipping.")
        continue

    # Load model
    model, tokenizer = load_model(model_config)

    for p_idx, profile in enumerate(remaining, 1):
        user_id = profile["user_id"]
        key = (user_id, model_name)

        if key in completed_keys:
            continue

        print(f"\n  [{p_idx}/{len(remaining)}] {user_id} | Archetype {profile['archetype']} | "
              f"{profile['target_question'][:40]}...")

        try:
            is_no_rag = model_config.get("no_rag", False)

            # Build context from user's SG (skip for no_rag)
            if is_no_rag:
                context = ""  # Empty context triggers plain Socratic mode
            else:
                user_sg = profile["user_sg"]
                target_node = get_node(user_sg, profile["target_sg_node"])

                if target_node:
                    context = build_context(user_sg, target_node)
                else:
                    context = ""
                    print(f"    WARNING: Could not find target node {profile['target_sg_node']}")

            # Generate response
            t0 = time.time()
            response = generate_response(model, tokenizer, context, profile["target_question"])
            gen_time = time.time() - t0

            print(f"    Response ({gen_time:.1f}s): {response[:100]}...")

            result = {
                "user_id": user_id,
                "archetype": profile["archetype"],
                "archetype_label": profile["archetype_label"],
                "target_sg_node": profile["target_sg_node"],
                "target_name": profile["target_name"],
                "target_question": profile["target_question"],
                "user_level": profile["user_level"],
                "model_name": model_name,
                "model_label": model_config["label"],
                "context_injected": context,
                "response": response,
                "expected_met_prereqs": profile["expected_met_prereqs"],
                "expected_unmet_prereqs": profile["expected_unmet_prereqs"],
                "expected_unmet_prereq_ids": profile["expected_unmet_prereq_ids"],
                "known_topics": profile["known_topics"],
                "key_gap": profile.get("key_gap"),
                "applicable_checks": profile["applicable_checks"],
                "gen_time_s": round(gen_time, 2),
                "timestamp": datetime.now().isoformat(),
            }

            all_results.append(result)
            completed_keys.add(key)

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "user_id": user_id,
                "archetype": profile["archetype"],
                "model_name": model_name,
                "error": str(e),
                "response": "",
                "timestamp": datetime.now().isoformat(),
            })

        # Save after each response
        with open(PARTIAL_SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"    [Saved] {len(all_results)} total responses")

    # Unload model before loading next
    unload_model(model, tokenizer)

# ── Final save ──
with open(RESPONSES_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"  DONE — {len(all_results)} responses saved to {RESPONSES_PATH}")
print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)


# === Cell 8 (code): Quick summary ===

print("\n" + "=" * 60)
print("  E1 RESPONSE SUMMARY")
print("=" * 60)

for model_config in MODELS:
    mname = model_config["name"]
    model_results = [r for r in all_results if r.get("model_name") == mname]
    successful = [r for r in model_results if r.get("response") and not r.get("error")]
    failed = [r for r in model_results if r.get("error")]

    print(f"\n  {model_config['label']}:")
    print(f"    Successful: {len(successful)}/{len(model_results)}")
    print(f"    Failed:     {len(failed)}")

    if successful:
        avg_len = sum(len(r["response"]) for r in successful) / len(successful)
        avg_time = sum(r.get("gen_time_s", 0) for r in successful) / len(successful)
        print(f"    Avg response length: {avg_len:.0f} chars")
        print(f"    Avg generation time: {avg_time:.1f}s")

        # Per-archetype breakdown
        for arch in ["A", "B", "C"]:
            arch_results = [r for r in successful if r.get("archetype") == arch]
            if arch_results:
                a_len = sum(len(r["response"]) for r in arch_results) / len(arch_results)
                print(f"    Archetype {arch}: {len(arch_results)} responses, avg {a_len:.0f} chars")

# Sample response
if all_results and all_results[0].get("response"):
    sample = all_results[0]
    print(f"\n  --- Sample: {sample['user_id']} × {sample['model_name']} ---")
    print(f"  Question: {sample['target_question']}")
    print(f"  Response: {sample['response'][:200]}...")

print(f"\n  Output file: {RESPONSES_PATH}")
print(f"  File size: {os.path.getsize(RESPONSES_PATH) / 1024:.1f} KB")
