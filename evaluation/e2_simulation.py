"""
e2_simulation.py
================
Stage 2 of E2 Evaluation Pipeline.

Runs on Kaggle (GPU required). Loads the fine-tuned Qwen 7B model from
HuggingFace, simulates 10-turn student-tutor conversations using Gemini
2.5 Flash as the student agent, and saves all conversations to JSON.

Cell structure mirrors a Jupyter notebook for Kaggle execution.

Input:  e2_question_bank.json  (from Stage 1)
Output: e2_conversations.json  (consumed by Stage 3)
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

HF_TOKEN = "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"  # <-- Replace with your HuggingFace token
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
# MODEL CONFIG — 3 models to compare
# ─────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SFT_MODEL_REPO = "Aryan3it/socratic-tutor-qwen2.5-7b"  # Merged SFT
DPO_ADAPTER_REPO = "Aryan3it/socratic-tutor-qwen2.5-7b_dpo_lora"  # DPO LoRA on base

MODELS = [
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
# GEMINI CONFIG (Student Agent)
# ─────────────────────────────────────────────
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"  # <-- Replace with your Gemini key
STUDENT_MODEL = "gemini-2.5-flash"

# ─────────────────────────────────────────────
# EXPERIMENT CONFIG
# ─────────────────────────────────────────────
DRY_RUN = False           # Set to False for full 100-question run
DRY_RUN_COUNT = 5        # Number of conversations in dry run
NUM_TURNS = 10           # Turns per conversation (from eval.md)
MAX_NEW_TOKENS = 512     # Tutor response length
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
QUESTION_BANK_PATH = "e2_question_bank.json"      # Input
# Per-model output files are generated dynamically:
#   e2_conversations_<model_name>.json  (per model)
#   e2_conversations_all.json           (merged final)

print(f"[Config] DRY_RUN = {DRY_RUN}")
print(f"[Config] Models: {[m['name'] for m in MODELS]}")
print(f"[Config] Student agent: {STUDENT_MODEL}")
print(f"[Config] Turns per conversation: {NUM_TURNS}")


# === Cell 4 (code): Load question bank ===
print("\n[Questions] Loading question bank ...")

with open(QUESTION_BANK_PATH, "r") as f:
    question_bank = json.load(f)

questions = question_bank["questions"]
print(f"[Questions] Loaded {len(questions)} questions")

if DRY_RUN:
    # Pick one from each category for dry run
    dry_questions = []
    for cat in ["I", "II", "III"]:
        cat_qs = [q for q in questions if q["category"] == cat]
        dry_questions.extend(cat_qs[:max(1, DRY_RUN_COUNT // 3)])
    # Fill remaining
    remaining = DRY_RUN_COUNT - len(dry_questions)
    if remaining > 0:
        used_ids = {q["id"] for q in dry_questions}
        for q in questions:
            if q["id"] not in used_ids and remaining > 0:
                dry_questions.append(q)
                remaining -= 1
    questions = dry_questions[:DRY_RUN_COUNT]
    print(f"[DRY RUN] Using {len(questions)} questions:")
    for q in questions:
        print(f"  {q['id']} | Cat {q['category']} | {q['topic']} | {q['question'][:60]}...")


# === Cell 5 (code): Model loading utilities ===
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


print("[Model Utils] Ready.")


# === Cell 6 (code): Initialize student agent (Gemini Flash) ===
from google import genai

print("\n[Student Agent] Initializing Gemini Flash ...")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ── Student system prompts per category ──

STUDENT_BASE_PROMPT = """\
You are a college student struggling with Data Structures and Algorithms.
You are chatting with an AI tutor for help.

RULES:
- You do NOT know the complete answer
- Ask follow-up questions based on the tutor's responses
- Show genuine confusion — you are learning, not testing
- Keep responses to 1-3 sentences maximum
- Stay in character as a student. Never break character.
- Do NOT acknowledge you are an AI or a simulation

TOPIC: {topic_name}
YOUR KNOWLEDGE LEVEL: {persona_hint}
"""

CATEGORY_INSTRUCTIONS = {
    "I": (
        "\nSPECIAL BEHAVIOR — Direct Pressure Student:\n"
        "- You are impatient and want the answer quickly\n"
        "- If the tutor keeps asking questions instead of answering, express frustration\n"
        "- At your 4th and 7th message, be especially insistent: say something like "
        "'Can you just tell me the answer?' or 'I don't have time for this, just tell me'\n"
        "- You want the DIRECT answer, not a guided exploration"
    ),
    "II": (
        "\nSPECIAL BEHAVIOR — Persistently Confused Student:\n"
        "- You are genuinely trying but keep getting confused\n"
        "- Occasionally give WRONG answers when the tutor asks you questions\n"
        "- At your 3rd and 6th message, say something like "
        "'I still don't get it' or 'I'm completely lost'\n"
        "- Show that you're making effort but struggling to understand"
    ),
    "III": (
        "\nSPECIAL BEHAVIOR — Almost-There Student:\n"
        "- You already have a partial understanding of the answer\n"
        "- Your current (partially correct) understanding: {partial_understanding}\n"
        "- In your first message, share what you know and ask if it's correct\n"
        "- You are close but missing a specific piece — seek that missing piece\n"
        "- Show confidence in what you know and curiosity about what you're missing"
    ),
}


def build_student_system_prompt(question: dict) -> str:
    """Build the student agent system prompt for a given question."""
    cat = question["category"]

    base = STUDENT_BASE_PROMPT.format(
        topic_name=question.get("topic_name", question["topic"]),
        persona_hint=question.get("student_persona_hint", "confused student"),
    )

    cat_instr = CATEGORY_INSTRUCTIONS[cat]
    if cat == "III":
        cat_instr = cat_instr.format(
            partial_understanding=question.get("partial_understanding", "some basic understanding"),
        )

    return base + cat_instr


def generate_student_response(
    question: dict,
    conversation_history: list[dict],
    turn_number: int,
) -> str:
    """Generate the next student message using Gemini Flash."""

    system_prompt = build_student_system_prompt(question)

    # Build the conversation as user/model turns from the student's perspective
    # The student's messages are "model" turns (what Gemini should generate)
    # The tutor's messages are "user" turns (what the student sees)
    contents = []

    # Add conversation so far
    for msg in conversation_history:
        if msg["role"] == "user":
            # This was a student message → "model" in Gemini's view
            contents.append({"role": "model", "parts": [{"text": msg["content"]}]})
        elif msg["role"] == "assistant":
            # This was a tutor message → "user" in Gemini's view
            contents.append({"role": "user", "parts": [{"text": msg["content"]}]})

    # If conversation_history ends with an assistant (tutor) message,
    # Gemini will naturally generate the next student response.
    # If it's empty or ends with a user message, we need to handle that.
    if not contents or contents[-1]["role"] != "user":
        # Force by adding a prompt
        contents.append({
            "role": "user",
            "parts": [{"text": "(The tutor is waiting for your response. Stay in character as a student.)"}],
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
        # Fallback generic responses per category
        fallbacks = {
            "I": "Can you just tell me the answer? I really need it.",
            "II": "I still don't understand. Can you explain it differently?",
            "III": "Am I on the right track? What am I missing?",
        }
        return fallbacks.get(question["category"], "I'm confused. Can you help?")


print("[Student Agent] Ready.")


# === Cell 7 (code): Tutor generation function ===

def generate_tutor_response(conversation_history: list[dict]) -> str:
    """
    Generate a tutor response using the fine-tuned Qwen model.
    
    Args:
        conversation_history: List of {"role": "system"/"user"/"assistant", "content": str}
    
    Returns:
        The tutor's response text.
    """
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

    # Clean up any residual chat markers
    for marker in ["<|im_start|>", "<|im_end|>", "<|im_start|>assistant\n",
                    "<|im_start|>user\n", "<|im_start|>system\n"]:
        response = response.replace(marker, "")

    # Free GPU memory
    del inputs, outputs
    torch.cuda.empty_cache()

    return response.strip()


print("[Tutor] Generation function ready.")


# === Cell 8 (code): Conversation simulation engine ===

def run_conversation(question: dict, conv_index: int, total: int) -> dict:
    """
    Run a single 10-turn student-tutor conversation.
    
    Returns:
        dict with question metadata and list of turns
    """
    print(f"\n{'='*60}")
    print(f"[{conv_index}/{total}] {question['id']} | Cat {question['category']} | "
          f"{question['topic']} | {question['question'][:50]}...")
    print(f"{'='*60}")

    conversation_history = []  # For the tutor model (role: user/assistant)
    turns = []

    # The first student message is the question itself
    student_msg = question["question"]

    for turn_num in range(1, NUM_TURNS + 1):
        print(f"\n  --- Turn {turn_num}/{NUM_TURNS} ---")

        # ── Student speaks ──
        print(f"  Student: {student_msg[:80]}...")
        conversation_history.append({"role": "user", "content": student_msg})

        # ── Tutor responds ──
        t0 = time.time()
        tutor_msg = generate_tutor_response(conversation_history)
        gen_time = time.time() - t0
        print(f"  Tutor ({gen_time:.1f}s): {tutor_msg[:80]}...")
        conversation_history.append({"role": "assistant", "content": tutor_msg})

        # ── Record turn ──
        turns.append({
            "turn": turn_num,
            "student": student_msg,
            "tutor": tutor_msg,
            "tutor_gen_time_s": round(gen_time, 2),
        })

        # ── Generate next student message (unless last turn) ──
        if turn_num < NUM_TURNS:
            student_msg = generate_student_response(
                question, conversation_history, turn_num + 1,
            )
            # Small delay for rate limiting
            time.sleep(0.5)

    result = {
        "question_id": question["id"],
        "category": question["category"],
        "topic": question["topic"],
        "topic_name": question.get("topic_name", question["topic"]),
        "opening_question": question["question"],
        "correct_answer": question["correct_answer"],
        "turns": turns,
        "timestamp": datetime.now().isoformat(),
    }

    return result


print("[Engine] Conversation simulation engine ready.")


# === Cell 9 (code): Run all conversations — Multi-model loop ===

print("\n" + "=" * 60)
print(f"  STARTING E2 EVALUATION — {'DRY RUN' if DRY_RUN else 'FULL RUN'}")
print(f"  Questions: {len(questions)}")
print(f"  Models: {[m['name'] for m in MODELS]}")
print(f"  Turns per conversation: {NUM_TURNS}")
print(f"  Total responses: {len(questions) * NUM_TURNS * len(MODELS)}")
print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

all_model_results = {}  # model_name -> list of conversation dicts

for model_idx, model_config in enumerate(MODELS, 1):
    model_name = model_config["name"]
    conv_path = f"e2_conversations_{model_name}.json"
    partial_path = f"e2_conversations_{model_name}_partial.json"

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
        completed_ids = {r["question_id"] for r in model_results}
        if len(completed_ids) >= len(questions):
            print(f"  All {len(completed_ids)} conversations already done. Skipping.")
            all_model_results[model_name] = model_results
            continue
        print(f"  [Resume] {len(completed_ids)} done, {len(questions) - len(completed_ids)} remaining.")
    elif os.path.exists(partial_path):
        with open(partial_path, "r") as f:
            model_results = json.load(f)
        completed_ids = {r["question_id"] for r in model_results}
        print(f"  [Resume] {len(completed_ids)} from partial.")

    remaining = [q for q in questions if q["id"] not in completed_ids]
    if not remaining:
        print(f"  All conversations done for {model_name}. Skipping.")
        all_model_results[model_name] = model_results
        continue

    # ── Load model ──
    model, tokenizer = load_model(model_config)

    for i, question in enumerate(remaining, len(completed_ids) + 1):
        try:
            result = run_conversation(question, i, len(questions))
            result["model_name"] = model_name
            result["model_label"] = model_config["label"]
            model_results.append(result)
        except Exception as e:
            print(f"\n  ERROR on {question['id']}: {e}")
            import traceback
            traceback.print_exc()
            model_results.append({
                "question_id": question["id"],
                "category": question["category"],
                "topic": question["topic"],
                "model_name": model_name,
                "model_label": model_config["label"],
                "error": str(e),
                "turns": [],
                "timestamp": datetime.now().isoformat(),
            })

        # Save after each conversation
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)
        print(f"  [Saved] {len(model_results)}/{len(questions)}")

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
print("  E2 CONVERSATION SUMMARY")
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
        avg_tutor_len = sum(
            len(t["tutor"]) for r in successful for t in r["turns"]
        ) / sum(len(r["turns"]) for r in successful)
        print(f"    Avg tutor response length: {avg_tutor_len:.0f} chars")

        for cat in ["I", "II", "III"]:
            cat_results = [r for r in successful if r["category"] == cat]
            if cat_results:
                print(f"    Category {cat}: {len(cat_results)} conversations")
