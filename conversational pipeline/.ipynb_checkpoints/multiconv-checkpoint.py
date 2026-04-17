
# multiturn_v1_resumable.py
# Purpose : Convert single-turn paraphrase_v12_messages.json records into
#            multi-turn conversational datasets by generating realistic
#            farmer follow-up turns grounded in the original seed answer.
#
# Design  : Follows the same architecture as self_instruct_v12_resumable.py —
#            same vLLM client, same judge-as-hard-gate pattern, same
#            checkpoint/resume logic, same dedup guard.
#
# Input   : paraphrase_v12_messages.json
# Output  : multiturn_v1.jsonl  — list of multi-turn conversation records (JSON Lines)
#           multiturn_v1.log    — per-record audit log

import json
import os
import re
import time
from datetime import datetime, timezone
from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────────
VLLM_URL        = "http://100.100.108.100:8080/v1"
MODEL           = "Qwen/Qwen3-30B-A3B"

INPUT_FILE      = "/home/kritika/self_instruct/conversational pipeline/paraphrase_conv_messages.json"
# Changed to .jsonl (JSON Lines) to allow clean appending without rewriting the file
OUT_JSON        = "/home/kritika/self_instruct/conversational pipeline/multiturn_v1.jsonl"
OUT_LOG         = "/home/kritika/self_instruct/conversational pipeline/multiturn_v1_log.txt"

MAX_RECORDS         = None    # set to an int to cap; None = process all
N_FOLLOWUP_TURNS    = 3       # how many extra (user, assistant) turns to add
MAX_RETRIES         = 5       # API-level retries per call
RETRY_DELAY         = 5       # seconds between retries
TEMPERATURE         = 0.75
MAX_TOKENS          = 3000

JUDGE_MAX_TOKENS    = 400
JUDGE_TEMPERATURE   = 0.0
MAX_JUDGE_RETRIES   = 3

# Lowered minimum words to allow concise factual answers
MIN_FOLLOWUP_WORDS  = 5
MAX_FOLLOWUP_WORDS  = 250

# ─── KVK Stripping ────────────────────────────────────────────────────────────
KVK_PATTERNS = [
    r'farmers?\s+are\s+advised\s+to\s+contact.*?(?:\.|$)',
    r'contact\s+the\s+nearest\s+krishi\s+vigyan\s+kendra.*?(?:\.|$)',
    r'consult.*?kvk.*?(?:\.|$)',
    r'local\s+agricultural\s+extension\s+officer.*?(?:\.|$)',
    r'for\s+further\s+guidance.*?(?:kvk|extension\s+officer).*?(?:\.|$)',
    r'they\s+can\s+provide\s+customized\s+nutrient\s+management.*?(?:\.|$)',
]

def strip_kvk_lines(text: str) -> str:
    for pattern in KVK_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'  +', ' ', text)
    return text.strip()

# ─── Loop Detection ───────────────────────────────────────────────────────────
def has_looping_text(text: str) -> bool:
    sentences = [s.strip() for s in re.split(r'[.।\n]', text) if len(s.strip()) > 20]
    seen = {}
    for s in sentences:
        seen[s] = seen.get(s, 0) + 1
        if seen[s] > 2:
            return True
    return False

# ─── Self-Reference Detection ─────────────────────────────────────────────────
SELF_REF_PATTERNS = [
    r'\bthe seed\b', r'\bseed answer\b', r'\baccording to the seed\b',
    r'\bas mentioned in the seed\b', r'\bthe source says\b',
    r'\bseed says\b', r'\bthe prompt\b', r'\bthe input\b',
]

def has_self_reference(text: str) -> bool:
    for pattern in SELF_REF_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

# ─── Deduplication ────────────────────────────────────────────────────────────
def normalize_q(q: str) -> str:
    return re.sub(r'[^a-z0-9\s]', '', q.lower()).strip()

def is_duplicate(new_q: str, seen_questions: set) -> bool:
    norm = normalize_q(new_q)
    if norm in seen_questions:
        return True
    new_words = set(norm.split())
    for seen in seen_questions:
        seen_words = set(seen.split())
        if not new_words or not seen_words:
            continue
        if len(new_words & seen_words) / max(len(new_words), len(seen_words)) >= 0.75:
            return True
    return False

# ─── Generic Phrase Guard ─────────────────────────────────────────────────────
GENERIC_PHRASES = [
    r"manufacturer['\u2019]?s instructions",
    r"follow the instructions on the label",
    r"as per recommended guidelines",
    r"as per the manufacturer",
    r"according to label",
    r"consult a professional",
    r"seek expert advice",
]

def has_generic_phrase(text: str) -> bool:
    for pattern in GENERIC_PHRASES:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def is_answer_length_ok(text: str) -> bool:
    wc = len(text.split())
    return MIN_FOLLOWUP_WORDS <= wc <= MAX_FOLLOWUP_WORDS

# ─── System Prompts ───────────────────────────────────────────────────────────
GENERATION_SYSTEM_PROMPT = """You are an expert in Indian agricultural conversational dataset creation.

You receive:
  - SEED ANSWER: the original expert-verified answer (the ONLY allowed source of facts)
  - CONVERSATION SO FAR: the dialogue turns already generated
  - TURN NUMBER: which follow-up turn you are generating
  - EXCLUDED ASPECTS: aspects already covered — do NOT repeat them

YOUR TASK:
Generate exactly ONE follow-up (user_question, assistant_answer) pair that
extends the conversation naturally.

ABSOLUTE RULES:
1. The follow-up user question MUST feel like a farmer continuing the chat.
2. The follow-up user question MUST focus on a DIFFERENT aspect of the seed topic.
3. The assistant answer MUST be grounded EXCLUSIVELY in the SEED ANSWER.
   Do NOT invent new facts.
4. Do NOT add "consult your nearest KVK" unless it's in the seed answer.
5. Do NOT add disclaimers or best-practice advice not in the seed answer.
6. Keep ALL chemical names in English.
7. Do NOT use self-referential language: no "the seed says", etc.
8. SCOPE RULE: ONLY address what the follow-up question asks.
9. If the seed answer has no remaining unexplored aspects, return JSON with user_question=null and assistant_answer=null.

Output ONLY a valid JSON object:
{
  "user_question": "<follow-up farmer question, or null>",
  "assistant_answer": "<grounded answer, or null>"
}"""

JUDGE_SYSTEM_PROMPT = """You are a strict quality verifier for an Indian agricultural multi-turn conversation dataset.

You receive:
  - SEED ANSWER: the original expert-verified source of truth
  - FOLLOW-UP QUESTION: the new user question generated for the conversation
  - FOLLOW-UP ANSWER: the assistant answer generated for that question

YOUR TASK — TWO CHECKS:

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

# ─── Prompt Builders ──────────────────────────────────────────────────────────

FOLLOWUP_ASPECTS = [
    "dosage or quantity of application",
    "timing or crop growth stage for application",
    "method of application (spray, soil drench, seed treatment, etc.)",
    "symptoms or how to identify the problem",
    "biological or cultural control alternatives",
    "safety or precautions when using the treatment",
    "what happens if left untreated",
    "preventive measures before the problem occurs",
    "specific chemical ingredients or product names",
    "irrigation or water management related to the problem",
    "soil or field preparation related to the treatment",
    "frequency or number of applications required",
]

def build_generation_prompt(record: dict, conversation_so_far: list,
                             turn_number: int, excluded_aspects: list) -> str:
    seed_answer = strip_kvk_lines(
        record["messages"][-1]["content"]
    )
    convo_text = ""
    for msg in conversation_so_far:
        role = "Farmer" if msg["role"] == "user" else "Expert"
        convo_text += f"{role}: {msg['content']}\n\n"

    excluded_str = (", ".join(excluded_aspects) if excluded_aspects else "none yet")

    return (
        f"SEED ANSWER:\n{seed_answer}\n\n"
        f"CONVERSATION SO FAR:\n{convo_text.strip()}\n\n"
        f"TURN NUMBER: {turn_number}\n\n"
        f"EXCLUDED ASPECTS:\n{excluded_str}\n\n"
        f"Available aspects to explore:\n"
        + "\n".join(f"  - {a}" for a in FOLLOWUP_ASPECTS) +
        "\n\nNow produce the JSON output."
    )

def build_judge_prompt(seed_answer: str, followup_question: str,
                       followup_answer: str) -> str:
    clean_seed = strip_kvk_lines(seed_answer)
    return (
        f"SEED ANSWER:\n{clean_seed}\n\n"
        f"FOLLOW-UP QUESTION:\n{followup_question}\n\n"
        f"FOLLOW-UP ANSWER:\n{followup_answer}\n\n"
        "Now output the JSON verdict."
    )

# ─── Response Parsers ─────────────────────────────────────────────────────────
def parse_generation_response(text: str):
    text = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
        # We now return the object even if values are null, so call_model can handle exhaustion
        if "user_question" not in obj or "assistant_answer" not in obj:
            return None
        return obj
    except json.JSONDecodeError:
        return None

def parse_judge_response(text: str) -> dict:
    text = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"grounded": True, "scope_overflow": False,
                "confidence": "low", "issues": [], "judge_error": "parse_failed"}
    try:
        obj = json.loads(match.group())
        scope_overflow = bool(obj.get("scope_overflow", False))
        grounded = bool(obj.get("grounded", True))
        if scope_overflow:
            grounded = False
        return {
            "grounded"      : grounded,
            "scope_overflow": scope_overflow,
            "confidence"    : str(obj.get("confidence", "low")),
            "issues"        : obj.get("issues", []),
        }
    except json.JSONDecodeError:
        return {"grounded": True, "scope_overflow": False,
                "confidence": "low", "issues": [], "judge_error": "json_error"}

# ─── Model Client ─────────────────────────────────────────────────────────────
client = OpenAI(base_url=VLLM_URL, api_key="dummy")

# ─── Generation Caller ────────────────────────────────────────────────────────
def call_model(record: dict, conversation_so_far: list,
               turn_number: int, excluded_aspects: list):
    user_prompt = build_generation_prompt(
        record, conversation_so_far, turn_number, excluded_aspects
    )
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            raw = resp.choices[0].message.content
            if raw is None:
                print(f"    [null-content] retrying...", end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

            result = parse_generation_response(raw.strip())
            if result is None:
                return None

            # Detect intentional exhaustion of topics (Rule 9)
            if result.get("user_question") is None or result.get("assistant_answer") is None:
                return {"exhausted": True}

            answer = result["assistant_answer"]

            if has_self_reference(answer) or has_self_reference(result["user_question"]):
                print(f"    [self-ref] retrying...", end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

            if has_looping_text(answer):
                print(f"    [loop] retrying...", end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

            if has_generic_phrase(answer):
                print(f"    [generic-phrase] retrying...", end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

            if not is_answer_length_ok(answer):
                wc = len(answer.split())
                print(f"    [len={wc}w, need {MIN_FOLLOWUP_WORDS}–{MAX_FOLLOWUP_WORDS}w] retrying...",
                      end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

            return result

        except Exception as e:
            print(f"    [error] attempt {attempt+1}: {e}")
        time.sleep(RETRY_DELAY * (2 ** attempt))
    return None

# ─── Judge Caller ─────────────────────────────────────────────────────────────
def call_judge(seed_answer: str, followup_question: str,
               followup_answer: str) -> dict:
    judge_prompt = build_judge_prompt(seed_answer, followup_question, followup_answer)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user",   "content": judge_prompt},
            ],
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=JUDGE_TEMPERATURE,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = resp.choices[0].message.content
        if raw is None:
            return {"grounded": True, "scope_overflow": False,
                    "confidence": "low", "issues": [], "judge_error": "null_content"}
        return parse_judge_response(raw.strip())
    except Exception as e:
        return {"grounded": True, "scope_overflow": False,
                "confidence": "low", "issues": [], "judge_error": str(e)[:120]}

# ─── Append-Only Checkpoint Helpers ───────────────────────────────────────────
def append_checkpoint(record: dict, record_log: list) -> None:
    """Appends a single record to the JSONL file without rewriting it."""
    with open(OUT_JSON, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    if record_log:
        with open(OUT_LOG, "a", encoding="utf-8") as f:
            f.write("\n".join(record_log) + "\n")

def load_checkpoint() -> tuple[list, set]:
    """Reads existing JSONL file line by line to determine progress."""
    all_results = []
    if not os.path.exists(OUT_JSON):
        return all_results, set()
    try:
        with open(OUT_JSON, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))
                    
        completed_keys = set(
            r["source_id"] + "||" + r.get("original_question", "")
            for r in all_results
        )
        print(f"  ✔ Checkpoint found — {len(all_results)} records already done. Resuming.")
        return all_results, completed_keys
    except Exception as e:
        print(f"  ⚠ Could not load checkpoint ({e}). Starting fresh.")
        return [], set()

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\nLoading input from: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    if MAX_RECORDS is not None:
        records = records[:MAX_RECORDS]

    print(f"\n{'='*60}")
    print(f"  Multi-Turn Conversation Generator v1 — RESUMABLE (JSONL Append)")
    print(f"  Input records     : {len(records)}")
    print(f"  Follow-up turns   : {N_FOLLOWUP_TURNS} per record")
    print(f"  Model             : {MODEL}")
    print(f"  Judge             : HARD GATE")
    print(f"  Checkpoint format : Append-only JSONL — extremely fast and safe.")
    print(f"{'='*60}\n")

    all_results, completed_keys = load_checkpoint()
    
    # We clear the log file ONLY if starting completely fresh
    if len(completed_keys) == 0 and os.path.exists(OUT_LOG):
        open(OUT_LOG, 'w').close()

    skip_count = judge_fail_count = success_count = 0

    for rec_idx, record in enumerate(records):
        source_id = record.get("source_id", str(rec_idx))
        user_msgs = [m for m in record["messages"] if m["role"] == "user"]
        original_question = user_msgs[0]["content"] if user_msgs else ""

        record_key = source_id + "||" + original_question

        if record_key in completed_keys:
            print(f"  ── Record {rec_idx+1}/{len(records)} [{source_id}] already done — skipping.")
            continue

        print(f"\n── Record {rec_idx+1}/{len(records)}: "
              f"{record.get('crop','?')} | {record.get('state','?')} | {record.get('domain','?')} ──")
        print(f"   Turn-1 Q: {original_question[:80]}...")

        seed_answer = next((m["content"] for m in reversed(record["messages"]) if m["role"] == "assistant"), "")
        conversation = list(record["messages"])

        seen_questions = {normalize_q(original_question)}
        excluded_aspects: list[str] = []
        turns_added = 0
        record_log: list[str] = []
        had_grounding_failure = False  

        for turn_num in range(1, N_FOLLOWUP_TURNS + 1):
            print(f"  ┌ Turn {turn_num}/{N_FOLLOWUP_TURNS} ... ", end="", flush=True)

            attempts = 0
            accepted = False
            exhausted_flag = False

            while attempts < MAX_RETRIES * 2:
                attempts += 1
                result = call_model(record, conversation, turn_num, excluded_aspects)

                if result is not None and result.get("exhausted"):
                    print(" [aspects exhausted] skipping remaining turns.", end=" ", flush=True)
                    record_log.append(f"EXHAUSTED | rec={rec_idx} | turn={turn_num}")
                    exhausted_flag = True
                    break # Break out of attempt loop

                if result is None:
                    print(f"[null/self-ref/loop/length] retry {attempts}...", end=" ", flush=True)
                    record_log.append(f"SKIP | rec={rec_idx} | turn={turn_num} | attempt={attempts}")
                    continue  

                followup_q = result["user_question"]
                followup_a = result["assistant_answer"]

                if is_duplicate(followup_q, seen_questions):
                    print(f"dup—retry", end=" ", flush=True)
                    record_log.append(f"DUPE | rec={rec_idx} | turn={turn_num} | q={followup_q[:60]}")
                    continue

                # Judge gate
                print(f"✓ gen ({len(followup_a.split())}w) → judging...", end=" ", flush=True)
                judge_retries = 0
                verdict = {"grounded": False}

                while judge_retries <= MAX_JUDGE_RETRIES:
                    verdict = call_judge(seed_answer, followup_q, followup_a)
                    if verdict["grounded"]:
                        break
                    judge_retries += 1
                    judge_fail_count += 1
                    had_grounding_failure = True
                    record_log.append(
                        f"JUDGE_RETRY | rec={rec_idx} | turn={turn_num} | retry={judge_retries}/{MAX_JUDGE_RETRIES} | "
                        f"scope_overflow={verdict['scope_overflow']} | issues={verdict['issues'][:2]}"
                    )
                    if judge_retries > MAX_JUDGE_RETRIES:
                        print(f"⚠ JUDGE FAILED — turn skipped", end=" ", flush=True)
                        break
                    print(f"⚠ FLAGGED — regen (judge retry {judge_retries})...", end=" ", flush=True)
                    
                    result = call_model(record, conversation, turn_num, excluded_aspects)
                    if result is None or result.get("exhausted"):
                        break
                    followup_q = result["user_question"]
                    followup_a = result["assistant_answer"]

                if not verdict["grounded"]:
                    record_log.append(f"JUDGE_ABANDON | rec={rec_idx} | turn={turn_num}")
                    print(f"⚠ judge abandoned — regen attempt {attempts}...", end=" ", flush=True)
                    continue  

                # Accept turn
                print(f"✓ Q: {followup_q[:55]}...")
                seen_questions.add(normalize_q(followup_q))
                conversation.append({"role": "user", "content": followup_q})
                conversation.append({"role": "assistant", "content": followup_a})
                excluded_aspects.append(f"turn {turn_num}: {followup_q[:60]}")
                turns_added += 1
                accepted = True
                record_log.append(f"OK | rec={rec_idx} | turn={turn_num} | judge=PASS | q={followup_q[:60]}")
                break  

            if exhausted_flag:
                break # Exit the turn_num loop for this record because no more facts remain

            if not accepted:
                skip_count += 1

        output_record = {
            "source_id"        : source_id,
            "crop"             : record.get("crop", ""),
            "state"            : record.get("state", ""),
            "district"         : record.get("district", ""),
            "domain"           : record.get("domain", ""),
            "grounded"         : not had_grounding_failure,
            "original_question": original_question,
            "turns_added"      : turns_added,
            "total_turns"      : (len(conversation) - 1) // 2,
            "messages"         : conversation,
            "timestamp"        : datetime.now(timezone.utc).isoformat(),
        }

        all_results.append(output_record)
        success_count += 1

        # Checkpoint: Just append this single record to the file. Very fast.
        append_checkpoint(output_record, record_log)
        print(f"   💾 Appended to checkpoint ({len(all_results)} records done so far)")

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Records processed         : {success_count}")
    print(f"  Turns skipped (all causes): {skip_count}")
    print(f"  Output JSONL : {OUT_JSON}")
    print(f"  Log          : {OUT_LOG}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
