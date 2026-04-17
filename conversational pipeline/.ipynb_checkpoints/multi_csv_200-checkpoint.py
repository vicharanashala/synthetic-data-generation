import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────────
VLLM_URL        = "http://100.100.108.100:8080/v1"
MODEL           = "Qwen/Qwen3-30B-A3B"

# Update this to point to your CSV file
INPUT_FILE      = "/home/kritika/self_instruct/data/random_200_qa_1.csv"
OUT_JSONL       = "/home/kritika/self_instruct/conversational pipeline/multiturn_from_csv.jsonl"
OUT_LOG         = "/home/kritika/self_instruct/conversational pipeline/multiturn_from_csv_log.txt"

MAX_RECORDS         = None    
N_FOLLOWUP_TURNS    = 3       
MAX_RETRIES         = 5       
RETRY_DELAY         = 5       
TEMPERATURE         = 0.75
MAX_TOKENS          = 3000    # Increased to prevent null-content cutoffs

JUDGE_MAX_TOKENS    = 400
JUDGE_TEMPERATURE   = 0.0
MAX_JUDGE_RETRIES   = 3

MIN_FOLLOWUP_WORDS  = 5
MAX_FOLLOWUP_WORDS  = 250

# ─── Filters & Guards ─────────────────────────────────────────────────────────
KVK_PATTERNS = [
    r'farmers?\s+are\s+advised\s+to\s+contact.*?(?:\.|$)',
    r'contact\s+the\s+nearest\s+krishi\s+vigyan\s+kendra.*?(?:\.|$)',
    r'consult.*?kvk.*?(?:\.|$)',
    r'local\s+agricultural\s+extension\s+officer.*?(?:\.|$)',
]

def strip_kvk_lines(text: str) -> str:
    for pattern in KVK_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return re.sub(r'\n{3,}', '\n\n', text).strip()

SELF_REF_PATTERNS = [
    r'\bthe seed\b', r'\bseed answer\b', r'\baccording to the seed\b',
    r'\bas mentioned in the seed\b', r'\bthe source says\b',
    r'\bseed says\b', r'\bthe prompt\b', r'\bthe input\b',
    r'\bas mentioned\b', r'\bmentioned earlier\b', r'\bmentioned above\b',
    r'\bmentioned previously\b', r'\blevels mentioned\b', r'\bas stated\b',
    r'\bas I said\b', r'\balready mentioned\b'
]

def has_self_reference(text: str) -> bool:
    for pattern in SELF_REF_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def normalize_q(q: str) -> str:
    return re.sub(r'[^a-z0-9\s]', '', q.lower()).strip()

def is_duplicate(new_q: str, seen_questions: set) -> bool:
    norm = normalize_q(new_q)
    if norm in seen_questions: return True
    new_words = set(norm.split())
    for seen in seen_questions:
        seen_words = set(seen.split())
        if not new_words or not seen_words: continue
        if len(new_words & seen_words) / max(len(new_words), len(seen_words)) >= 0.75:
            return True
    return False

def is_answer_length_ok(text: str) -> bool:
    return MIN_FOLLOWUP_WORDS <= len(text.split()) <= MAX_FOLLOWUP_WORDS

# ─── Prompts ──────────────────────────────────────────────────────────────────
GENERATION_SYSTEM_PROMPT = """You are an expert in Indian agricultural conversational dataset creation.

You receive:
  - SEED ANSWER: the original expert-verified answer (the ONLY allowed source of facts)
  - CONVERSATION SO FAR: the dialogue turns already generated
  - TURN NUMBER: which follow-up turn you are generating
  - EXCLUDED ASPECTS: aspects already covered — do NOT repeat them

YOUR TASK:
Generate exactly ONE follow-up (user_question, assistant_answer) pair.

ABSOLUTE RULES:
1. The follow-up user question MUST feel like a farmer continuing the chat.
2. The follow-up user question MUST focus on a DIFFERENT aspect of the seed topic.
3. The assistant answer MUST be grounded EXCLUSIVELY in the SEED ANSWER.
4. Do NOT add "consult your nearest KVK" unless it's in the seed answer.
5. Keep ALL chemical names in English.
6. Do NOT use lazy language. Never say "as mentioned earlier" or "the levels mentioned". State the actual facts directly.
7. SCOPE RULE: ONLY address what the follow-up question asks.
8. If the seed answer has no remaining unexplored aspects, return JSON with user_question=null and assistant_answer=null.

Output ONLY a valid JSON object:
{
  "user_question": "<follow-up farmer question, or null>",
  "assistant_answer": "<grounded answer, or null>"
}"""

JUDGE_SYSTEM_PROMPT = """You are a strict quality verifier.
CHECK 1 — GROUNDING: Does the FOLLOW-UP ANSWER contain any claim/fact NOT explicitly present in the SEED ANSWER?
CHECK 2 — SCOPE ALIGNMENT: Does the FOLLOW-UP ANSWER stay within the scope of the FOLLOW-UP QUESTION?

Set grounded=false if CHECK 1 fails OR if scope_overflow=true.

Output ONLY a valid JSON object:
{
  "grounded": true or false,
  "scope_overflow": true or false,
  "confidence": "high" or "medium" or "low",
  "issues": ["<specific problem>"]
}"""

FOLLOWUP_ASPECTS = [
    "dosage or quantity of application", "timing or crop growth stage for application",
    "method of application", "symptoms or how to identify the problem",
    "biological or cultural control alternatives", "safety or precautions",
    "what happens if left untreated", "preventive measures",
    "specific chemical ingredients", "irrigation or water management",
    "soil or field preparation", "frequency or number of applications"
]

def build_generation_prompt(seed_answer: str, conversation_so_far: list, turn_number: int, excluded_aspects: list) -> str:
    convo_text = "\n\n".join([f"{'Farmer' if m['role']=='user' else 'Expert'}: {m['content']}" for m in conversation_so_far if m['role'] != 'system'])
    excluded_str = (", ".join(excluded_aspects) if excluded_aspects else "none yet")
    return (
        f"SEED ANSWER:\n{seed_answer}\n\nCONVERSATION SO FAR:\n{convo_text}\n\n"
        f"TURN NUMBER: {turn_number}\n\nEXCLUDED ASPECTS:\n{excluded_str}\n\n"
        f"Available aspects to explore:\n" + "\n".join(f"  - {a}" for a in FOLLOWUP_ASPECTS) +
        "\n\nNow produce the JSON output."
    )

# ─── API Client & Parsers ─────────────────────────────────────────────────────
client = OpenAI(base_url=VLLM_URL, api_key="dummy")

def parse_json_response(text: str, is_judge=False):
    match = re.search(r"\{.*\}", re.sub(r"```json|```", "", text).strip(), re.DOTALL)
    if not match: return None if not is_judge else {"grounded": True, "scope_overflow": False, "issues": []}
    try:
        obj = json.loads(match.group())
        if is_judge:
            scope_overflow = bool(obj.get("scope_overflow", False))
            return {"grounded": False if scope_overflow else bool(obj.get("grounded", True)),
                    "scope_overflow": scope_overflow, "issues": obj.get("issues", [])}
        return obj if "user_question" in obj and "assistant_answer" in obj else None
    except json.JSONDecodeError:
        return None if not is_judge else {"grounded": True, "scope_overflow": False, "issues": []}

def call_model(seed_answer: str, conversation: list, turn_number: int, excluded: list):
    user_prompt = build_generation_prompt(seed_answer, conversation, turn_number, excluded)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": GENERATION_SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
                max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            raw = resp.choices[0].message.content
            if not raw or raw.strip() == "":
                print("    [empty] retrying...", end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

            result = parse_json_response(raw)
            if result is None: continue
            if result.get("user_question") is None or result.get("assistant_answer") is None:
                return {"exhausted": True}

            ans = result["assistant_answer"]
            if has_self_reference(ans) or has_self_reference(result["user_question"]):
                print("    [lazy-ref] retrying...", end=" ", flush=True)
                continue
            if not is_answer_length_ok(ans):
                print(f"    [len={len(ans.split())}w] retrying...", end=" ", flush=True)
                continue
            return result
        except Exception as e:
            print(f"    [error] {e}")
        time.sleep(RETRY_DELAY)
    return None

def call_judge(seed: str, q: str, a: str) -> dict:
    prompt = f"SEED ANSWER:\n{strip_kvk_lines(seed)}\n\nFOLLOW-UP QUESTION:\n{q}\n\nFOLLOW-UP ANSWER:\n{a}\n\nOutput JSON verdict."
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=JUDGE_MAX_TOKENS, temperature=JUDGE_TEMPERATURE,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return parse_json_response(resp.choices[0].message.content or "", is_judge=True)
    except:
        return {"grounded": True, "scope_overflow": False, "issues": []}

# ─── Checkpointing ────────────────────────────────────────────────────────────
def append_checkpoint(record: dict, record_log: list):
    with open(OUT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if record_log:
        with open(OUT_LOG, "a", encoding="utf-8") as f:
            f.write("\n".join(record_log) + "\n")

def load_checkpoint() -> set:
    if not os.path.exists(OUT_JSONL): return set()
    completed = set()
    with open(OUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): completed.add(json.loads(line)["source_id"])
    return completed

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\nLoading input from CSV: {INPUT_FILE}")
    records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    completed_keys = load_checkpoint()
    print(f"Resuming: {len(completed_keys)} records already completed.")

    for rec_idx, row in enumerate(records):
        source_id = row.get("Answer ID", str(rec_idx))
        if source_id in completed_keys: continue

        crop = row.get("Crop", "Unknown")
        q_text = row.get("Question Text", "")
        a_text = strip_kvk_lines(row.get("Answer Text", ""))

        print(f"\n── Record {rec_idx+1}/{len(records)}: {crop} | {source_id} ──")
        
        # Initialize conversation with System Prompt and Turn 1
        conversation = [
            {"role": "system", "content": "You are a helpful and knowledgeable agricultural advisory assistant."},
            {"role": "user", "content": q_text},
            {"role": "assistant", "content": a_text}
        ]

        seen_questions = {normalize_q(q_text)}
        excluded_aspects = []
        turns_added = 0
        record_log = []

        for turn_num in range(1, N_FOLLOWUP_TURNS + 1):
            print(f"  ┌ Turn {turn_num} ... ", end="", flush=True)
            result = call_model(a_text, conversation, turn_num, excluded_aspects)

            if result and result.get("exhausted"):
                print("[exhausted] ending record.", flush=True)
                break
            if not result:
                print("[failed] skipping.", flush=True)
                continue

            followup_q, followup_a = result["user_question"], result["assistant_answer"]
            if is_duplicate(followup_q, seen_questions):
                print("[dupe] skipping.", flush=True)
                continue

            print(f"✓ gen → judging...", end=" ", flush=True)
            judge_retries, verdict = 0, {"grounded": False}

            while judge_retries <= MAX_JUDGE_RETRIES:
                verdict = call_judge(a_text, followup_q, followup_a)
                if verdict["grounded"]: break
                judge_retries += 1
                if judge_retries > MAX_JUDGE_RETRIES: break
                print(f"⚠ FLAGGED regen...", end=" ", flush=True)
                result = call_model(a_text, conversation, turn_num, excluded_aspects)
                if not result or result.get("exhausted"): break
                followup_q, followup_a = result["user_question"], result["assistant_answer"]

            if not verdict["grounded"]:
                print("⚠ abandoned.", flush=True)
                continue

            print(f"✓ Q: {followup_q[:50]}...")
            seen_questions.add(normalize_q(followup_q))
            conversation.append({"role": "user", "content": followup_q})
            conversation.append({"role": "assistant", "content": followup_a})
            excluded_aspects.append(followup_q)
            turns_added += 1
            record_log.append(f"OK | rec={source_id} | turn={turn_num}")

        # Exact JSON format requested by user
        output_record = {
            "source_id": source_id,
            "crop": crop,
            "messages": conversation,
            "turns_added": turns_added,
            "total_turns": (len(conversation) - 1) // 2
        }

        append_checkpoint(output_record, record_log)

if __name__ == "__main__":
    main()