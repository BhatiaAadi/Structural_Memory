"""
e3_simulation.py
================
Stage 2 of E3 Evaluation Pipeline.

Runs on Kaggle (GPU required). Loads the fine-tuned Qwen 7B model from
HuggingFace, simulates 40 × 10-turn conversations using Gemini 2.5 Flash
as the student agent with 5 different student profiles per topic.

Input:  e3_scenarios.json    (from Stage 1)
Output: e3_conversations.json (consumed by Stage 3)
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
#     google-genai huggingface_hub --upgrade --quiet
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
# MODEL CONFIG — SFT and DPO (base not meaningful for E3)
# ─────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SFT_MODEL_REPO = "Aryan3it/socratic-tutor-qwen2.5-7b"  # Merged SFT
DPO_ADAPTER_REPO = "Aryan3it/socratic-tutor-qwen2.5-7b_dpo_lora"  # DPO LoRA on base

MODELS = [
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
# GEMINI CONFIG (Student Agent)
# ─────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyBXHBoy3JanCwVHDuyGKlzHqtOoqM4V4HQ"  # <-- Replace with your Gemini key
STUDENT_MODEL = "gemini-2.5-flash"

# ─────────────────────────────────────────────
# EXPERIMENT CONFIG
# ─────────────────────────────────────────────
DRY_RUN = True           # Set to False for full 40-conversation run
DRY_RUN_COUNT = 5        # Number of conversations in dry run
NUM_TURNS = 10
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# ─────────────────────────────────────────────
# TUTOR SYSTEM PROMPT (same as training / E2)
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
SCENARIOS_PATH     = "e3_scenarios.json"
# Per-model output files generated dynamically:
#   e3_conversations_<model_name>.json  (per model)

print(f"[Config] DRY_RUN = {DRY_RUN}")
print(f"[Config] Models: {[m['name'] for m in MODELS]}")
print(f"[Config] Student agent: {STUDENT_MODEL}")
print(f"[Config] Turns per conversation: {NUM_TURNS}")


# === Cell 4 (code): Load scenarios ===
print("\n[Scenarios] Loading scenarios ...")

with open(SCENARIOS_PATH, "r") as f:
    scenario_bank = json.load(f)

scenarios = scenario_bank["scenarios"]
print(f"[Scenarios] Loaded {len(scenarios)} scenarios")

if DRY_RUN:
    # Pick one from each profile for dry run
    dry_scenarios = []
    seen_profiles = set()
    for s in scenarios:
        if s["profile_name"] not in seen_profiles and len(dry_scenarios) < DRY_RUN_COUNT:
            dry_scenarios.append(s)
            seen_profiles.add(s["profile_name"])
    scenarios = dry_scenarios[:DRY_RUN_COUNT]
    print(f"[DRY RUN] Using {len(scenarios)} scenarios:")
    for s in scenarios:
        print(f"  {s['id']} | {s['topic_name']} | {s['profile_label']} | {s['opening_question'][:50]}...")


# === Cell 5 (code): Model loading utilities ===
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(model_config: dict):
    """Load a model based on its config. Returns (model, tokenizer)."""
    print(f"\n[Model] Loading: {model_config['label']}")

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


print("[Model Utils] Ready.")


# === Cell 6 (code): Initialize student agent (Gemini Flash) ===
from google import genai

print("\n[Student Agent] Initializing Gemini Flash ...")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ── Student system prompts per profile ──

STUDENT_BASE_PROMPT = """\
You are a college student in a Data Structures and Algorithms course.
You are chatting with an AI tutor for help on {topic_name}.

YOUR PROFILE: {profile_label}
{profile_description}

YOUR KNOWLEDGE: You already know: {known_topics_str}
YOU DO NOT KNOW: {target_concept}

RULES:
- Stay in character as this specific type of student
- Keep responses to 1-3 sentences maximum
- Never break character or acknowledge you are an AI
- Ask follow-up questions based on the tutor's responses
"""

PROFILE_INSTRUCTIONS = {
    "complete_beginner": (
        "\nBEHAVIOR:\n"
        "- Ask very basic 'what is X?' questions\n"
        "- Admit when you don't understand at all\n"
        "- Need things explained in simple terms\n"
        "- Don't use technical vocabulary you wouldn't know yet\n"
        "- Express feeling overwhelmed"
    ),
    "partial_knowledge": (
        "\nBEHAVIOR:\n"
        "- Reference your prerequisite knowledge naturally\n"
        "- Ask how {topic_name} builds on what you already know\n"
        "- Make connections between new and known concepts\n"
        "- Ask focused, specific questions\n"
        "- Show curiosity about how things fit together"
    ),
    "conceptually_confused": (
        "\nBEHAVIOR:\n"
        "- Give explanations that SOUND correct but are actually WRONG\n"
        "- Confuse related concepts: {common_confusion}\n"
        "- When corrected, initially resist — 'but isn't that the same thing?'\n"
        "- Eventually acknowledge confusion when the tutor provides evidence\n"
        "- Your wrong mental model should be consistent across turns"
    ),
    "overconfident": (
        "\nBEHAVIOR:\n"
        "- Assert your understanding confidently even when wrong\n"
        "- Push back when the tutor hints you might be incorrect\n"
        "- Say things like 'I'm pretty sure that's right' or 'I've seen this before'\n"
        "- Your misconception: {common_confusion}\n"
        "- Only gradually accept correction after strong evidence\n"
        "- Be slightly dismissive of basic questions"
    ),
    "strong_with_gap": (
        "\nBEHAVIOR:\n"
        "- Demonstrate strong understanding on most subtopics\n"
        "- Give correct, detailed answers when asked about things you know\n"
        "- BUT stumble specifically on: {missing_concept}\n"
        "- When the topic turns to your gap, say something like 'this is the part I'm stuck on'\n"
        "- Show frustration at not being able to figure out this one piece\n"
        "- Be articulate and precise in your language"
    ),
}


def build_student_system_prompt(scenario: dict) -> str:
    """Build the student agent system prompt for a given scenario."""
    known_str = ", ".join(scenario["known_topics"]) if scenario["known_topics"] else "nothing about this topic"

    base = STUDENT_BASE_PROMPT.format(
        topic_name=scenario["topic_name"],
        profile_label=scenario["profile_label"],
        profile_description=scenario["profile_description"],
        known_topics_str=known_str,
        target_concept=scenario["target_concept"],
    )

    profile_instr = PROFILE_INSTRUCTIONS[scenario["profile_name"]]
    profile_instr = profile_instr.format(
        topic_name=scenario["topic_name"],
        common_confusion=scenario.get("common_confusion", ""),
        missing_concept=scenario.get("missing_concept", ""),
    )

    return base + profile_instr


def generate_student_response(
    scenario: dict,
    conversation_history: list[dict],
    turn_number: int,
) -> str:
    """Generate the next student message using Gemini Flash."""

    system_prompt = build_student_system_prompt(scenario)

    # Build conversation from student's perspective
    contents = []
    for msg in conversation_history:
        if msg["role"] == "user":
            contents.append({"role": "model", "parts": [{"text": msg["content"]}]})
        elif msg["role"] == "assistant":
            contents.append({"role": "user", "parts": [{"text": msg["content"]}]})

    if not contents or contents[-1]["role"] != "user":
        contents.append({
            "role": "user",
            "parts": [{"text": "(The tutor is waiting for your response. Stay in character.)"}],
        })

    try:
        response = gemini_client.models.generate_content(
            model=STUDENT_MODEL,
            contents=contents,
            config={
                "temperature": 0.7,
                "max_output_tokens": 256,
                "system_instruction": system_prompt,
            },
        )
        return response.text.strip()
    except Exception as e:
        print(f"    [Student Agent ERROR] Turn {turn_number}: {e}")
        fallbacks = {
            "complete_beginner": "I'm really confused. Can you explain that more simply?",
            "partial_knowledge": "How does that connect to what I already know?",
            "conceptually_confused": "Wait, isn't that the same thing? I'm confused about the difference.",
            "overconfident": "I think I already understand this. Can we move on to the harder stuff?",
            "strong_with_gap": "I get that part, but I'm still stuck on the specific thing I mentioned.",
        }
        return fallbacks.get(scenario["profile_name"], "Can you explain that again?")


print("[Student Agent] Ready.")


# === Cell 7 (code): Tutor generation function ===

def generate_tutor_response(conversation_history: list[dict]) -> str:
    """Generate a tutor response using the fine-tuned Qwen model."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

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

    for marker in ["<|im_start|>", "<|im_end|>", "<|im_start|>assistant\n",
                    "<|im_start|>user\n", "<|im_start|>system\n"]:
        response = response.replace(marker, "")

    del inputs, outputs
    torch.cuda.empty_cache()

    return response.strip()


print("[Tutor] Generation function ready.")


# === Cell 8 (code): Conversation simulation engine ===

def run_conversation(scenario: dict, conv_index: int, total: int) -> dict:
    """Run a single 10-turn student-tutor conversation."""
    print(f"\n{'='*60}")
    print(f"[{conv_index}/{total}] {scenario['id']} | {scenario['topic_name']} | "
          f"{scenario['profile_label']}")
    print(f"{'='*60}")

    conversation_history = []
    turns = []

    # First student message is the opening question
    student_msg = scenario["opening_question"]

    for turn_num in range(1, NUM_TURNS + 1):
        print(f"\n  --- Turn {turn_num}/{NUM_TURNS} ---")

        # Student speaks
        print(f"  Student: {student_msg[:80]}...")
        conversation_history.append({"role": "user", "content": student_msg})

        # Tutor responds
        t0 = time.time()
        tutor_msg = generate_tutor_response(conversation_history)
        gen_time = time.time() - t0
        print(f"  Tutor ({gen_time:.1f}s): {tutor_msg[:80]}...")
        conversation_history.append({"role": "assistant", "content": tutor_msg})

        turns.append({
            "turn": turn_num,
            "student": student_msg,
            "tutor": tutor_msg,
            "tutor_gen_time_s": round(gen_time, 2),
        })

        # Generate next student message (unless last turn)
        if turn_num < NUM_TURNS:
            student_msg = generate_student_response(
                scenario, conversation_history, turn_num + 1,
            )
            time.sleep(0.5)

    result = {
        "scenario_id": scenario["id"],
        "topic": scenario["topic"],
        "topic_name": scenario["topic_name"],
        "profile_id": scenario["profile_id"],
        "profile_name": scenario["profile_name"],
        "profile_label": scenario["profile_label"],
        "profile_description": scenario["profile_description"],
        "known_topics": scenario["known_topics"],
        "target_concept": scenario["target_concept"],
        "opening_question": scenario["opening_question"],
        "turns": turns,
        "timestamp": datetime.now().isoformat(),
    }

    return result


print("[Engine] Conversation simulation engine ready.")


# === Cell 9 (code): Run all conversations — Multi-model loop ===

print("\n" + "=" * 60)
print(f"  STARTING E3 SIMULATION — {'DRY RUN' if DRY_RUN else 'FULL RUN'}")
print(f"  Scenarios: {len(scenarios)}")
print(f"  Models: {[m['name'] for m in MODELS]}")
print(f"  Turns per conversation: {NUM_TURNS}")
print(f"  Total responses: {len(scenarios) * NUM_TURNS * len(MODELS)}")
print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

all_model_results = {}  # model_name -> list of conversation dicts

for model_idx, model_config in enumerate(MODELS, 1):
    model_name = model_config["name"]
    conv_path = f"e3_conversations_{model_name}.json"
    partial_path = f"e3_conversations_{model_name}_partial.json"

    print(f"\n{'='*60}")
    print(f"  MODEL {model_idx}/{len(MODELS)}: {model_config['label']}")
    print(f"  Output: {conv_path}")
    print(f"{'='*60}")

    # ── Resume from partial results ──
    completed_ids = set()
    model_results = []

    if os.path.exists(conv_path):
        with open(conv_path, "r") as f:
            model_results = json.load(f)
        completed_ids = {r["scenario_id"] for r in model_results}
        if len(completed_ids) >= len(scenarios):
            print(f"  All {len(completed_ids)} conversations already done. Skipping.")
            all_model_results[model_name] = model_results
            continue
        print(f"  [Resume] {len(completed_ids)} done, {len(scenarios) - len(completed_ids)} remaining.")
    elif os.path.exists(partial_path):
        with open(partial_path, "r") as f:
            model_results = json.load(f)
        completed_ids = {r["scenario_id"] for r in model_results}
        print(f"  [Resume] {len(completed_ids)} from partial.")

    remaining = [s for s in scenarios if s["id"] not in completed_ids]
    if not remaining:
        print(f"  All conversations done for {model_name}. Skipping.")
        all_model_results[model_name] = model_results
        continue

    # ── Load model ──
    model, tokenizer = load_model(model_config)

    for i, scenario in enumerate(remaining, len(completed_ids) + 1):
        try:
            result = run_conversation(scenario, i, len(scenarios))
            result["model_name"] = model_name
            result["model_label"] = model_config["label"]
            model_results.append(result)
        except Exception as e:
            print(f"\n  ERROR on {scenario['id']}: {e}")
            import traceback
            traceback.print_exc()
            model_results.append({
                "scenario_id": scenario["id"],
                "topic": scenario["topic"],
                "profile_name": scenario["profile_name"],
                "model_name": model_name,
                "model_label": model_config["label"],
                "error": str(e),
                "turns": [],
                "timestamp": datetime.now().isoformat(),
            })

        # Save after each conversation
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)
        print(f"  [Saved] {len(model_results)}/{len(scenarios)}")

    # ── Unload model ──
    unload_model(model, tokenizer)

    # ── Save final per-model file ──
    with open(conv_path, "w", encoding="utf-8") as f:
        json.dump(model_results, f, indent=2, ensure_ascii=False)
    all_model_results[model_name] = model_results
    print(f"  Saved {len(model_results)} conversations to {conv_path}")

print(f"\n{'='*60}")
print(f"  DONE — All models complete")
print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)


# === Cell 10 (code): Quick summary ===

print("\n" + "=" * 60)
print("  E3 CONVERSATION SUMMARY")
print("=" * 60)

for model_config in MODELS:
    mname = model_config["name"]
    results = all_model_results.get(mname, [])
    successful = [r for r in results if len(r.get("turns", [])) == NUM_TURNS]
    failed = [r for r in results if r.get("error")]

    print(f"\n  {model_config['label']}:")
    print(f"    Successful: {len(successful)}/{len(results)}")
    print(f"    Failed:     {len(failed)}")

    if successful:
        avg_len = sum(
            len(t["tutor"]) for r in successful for t in r["turns"]
        ) / sum(len(r["turns"]) for r in successful)
        print(f"    Avg tutor response length: {avg_len:.0f} chars")

        for profile_name in ["complete_beginner", "partial_knowledge", "conceptually_confused",
                              "overconfident", "strong_with_gap"]:
            profile_results = [r for r in successful if r.get("profile_name") == profile_name]
            if profile_results:
                print(f"    {profile_name}: {len(profile_results)} conversations")
