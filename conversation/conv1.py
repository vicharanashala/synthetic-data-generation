# self_instruct_conversational.py
# Purpose : Generate ONLY conversational-style Q&A pairs from 200 seeds.
# Supports: batched processing + per-batch checkpoints + merge utility.
# Run     : python self_instruct_conversational.py

import json
import os
import time
import re
import uuid
import csv
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────────
VLLM_URL        = "http://100.100.108.100:8080/v1"
MODEL           = "Qwen/Qwen3-30B-A3B"
INPUT_FILE      = "/home/kritika/self_instruct/random_200_qa_1.csv"

# ── Output paths ──────────────────────────────────────────────────────────────
# Final merged outputs (produced by merge_checkpoints.py or end of run)
OUT_JSON        = "/home/kritika/self_instruct/conversation/paraphrase_conv.json"
OUT_CSV         = "/home/kritika/self_instruct/conversation/paraphrase_conv.csv"
OUT_LOG         = "/home/kritika/self_instruct/conversation/paraphrase_conv_log.txt"

# Per-batch checkpoint directory
CHECKPOINT_DIR  = "/home/kritika/self_instruct/conversation/checkpoints"

# ── Batch config ──────────────────────────────────────────────────────────────
MAX_SEEDS           = 200
BATCH_SIZE          = 20     # seeds per batch  → 10 batches for 200 seeds
TARGET_PER_STYLE    = 5      # 5 conversational pairs per seed → 1000 total pairs

MAX_STYLE_FAILURES  = 6
MAX_RETRIES         = 5
RETRY_DELAY         = 1
TEMPERATURE         = 0.75
MAX_TOKENS          = 1200

JUDGE_MAX_TOKENS    = 500
JUDGE_TEMPERATURE   = 0.0
MAX_JUDGE_RETRIES   = 3

MIN_ANSWER_WORDS    = 30
MAX_ANSWER_WORDS    = 200

# ─── CSV Column Names ─────────────────────────────────────────────────────────
COL_QUESTION    = "Question Text"
COL_ANSWER      = "Answer Text"
COL_CROP        = "Crop"
COL_STATE       = "State"
COL_DISTRICT    = "District"
COL_SEASON      = "Season"
COL_DOMAIN      = "Domain"
COL_ID          = "Answer ID"

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
MAX_SENTENCE_REPEATS = 2

def has_looping_text(text: str) -> bool:
    sentences = [s.strip() for s in re.split(r'[.।\n]', text) if len(s.strip()) > 20]
    seen = {}
    for s in sentences:
        seen[s] = seen.get(s, 0) + 1
        if seen[s] > MAX_SENTENCE_REPEATS:
            return True
    return False

# ─── Self-Reference Detection ─────────────────────────────────────────────────
SELF_REF_PATTERNS = [
    r'\bthe seed\b', r'\bseed answer\b', r'\baccording to the seed\b',
    r'\bas mentioned in the seed\b', r'\bthe source says\b', r'\bseed says\b',
    r'\bas stated in the seed\b', r'\bthe provided seed\b',
    r'\bthe prompt\b', r'\bthe input\b',
]

def has_self_reference(text: str) -> bool:
    for pattern in SELF_REF_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

# ─── Answer Length Guard ──────────────────────────────────────────────────────
def is_answer_length_ok(text: str) -> bool:
    word_count = len(text.split())
    return MIN_ANSWER_WORDS <= word_count <= MAX_ANSWER_WORDS

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
        intersection = len(new_words & seen_words)
        if intersection < 2:
            continue
        if intersection / max(len(new_words), len(seen_words)) >= 0.75:
            return True
    return False

# ─── Metadata ─────────────────────────────────────────────────────────────────
def build_contextual_metadata(seed: dict) -> str:
    clean_answer = seed.get("answer", "").lower()
    parts = []
    if seed.get("crop"):
        parts.append(f"CROP: {seed['crop']}")
    if seed.get("domain"):
        parts.append(f"DOMAIN: {seed['domain']}")
    state = seed.get("state", "").strip()
    if state and state.lower() in clean_answer:
        parts.append(f"STATE: {state}")
    district = seed.get("district", "").strip()
    if district and district.lower() in clean_answer:
        parts.append(f"DISTRICT: {district}")
    season = seed.get("season", "").strip()
    if season and re.search(r'\b' + re.escape(season.lower()) + r'\b', clean_answer):
        parts.append(f"SEASON: {season}")
    return ", ".join(parts)

# ─── Single Style: Conversational ─────────────────────────────────────────────
STYLE = {
    "name": "paraphrase_conversational",
    "description": (
        "Rewrite the question in the natural, informal tone of a farmer asking "
        "a helpline — as if speaking out loud. Use simple everyday language, "
        "contractions, and a direct personal voice (e.g. 'My wheat leaves are "
        "turning yellow, what should I do?'). "
        "The answer should also feel like a helpful, friendly explanation — "
        "not a formal report. Use short sentences and plain words. "
        "Target 30–200 words in the answer. "
        "GROUNDING RULE: Only use facts already present in the seed answer. "
        "Do NOT add new chemicals, dosages, or recommendations not in the seed. "
        "Do NOT add KVK referral sentences. "
        "SCOPE RULE: Your answer must ONLY address what your generated question asks."
    ),
}

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert in Indian agricultural Q&A dataset creation.
You receive a SEED question and answer from an Indian agricultural helpline,
a STYLE instruction, and a VARIATION HINT telling you which attempt this is.

YOUR TASK:
- Generate exactly ONE new (question, answer) pair in the requested style.
- The answer MUST be grounded EXCLUSIVELY in the seed answer content.

ABSOLUTE GROUNDING RULES:
1. Do NOT invent new facts, chemicals, dosages, crop names, percentages,
   or recommendations not explicitly stated in the seed answer.
2. Do NOT add "consult your nearest KVK" or "contact local extension officer"
   UNLESS this exact phrase already appears in the seed answer.
3. Do NOT add standard disclaimers, safety warnings, or generic best-practice
   advice that is not in the seed.
4. Keep ALL chemical names, fertilizer codes, and crop names in English.
5. Use the VARIATION HINT to ensure your output is lexically different from
   previous attempts — change the entry point, sentence structure, or focus angle.
6. NEVER use self-referential language: no "the seed says", "according to the seed", etc.
7. LOCATION RULE: Do NOT mention any state, district, or region name UNLESS
   that location is explicitly present in the SEED ANSWER text.
8. SCOPE RULE: Your generated_answer must ONLY address what your generated_question asks.

Output ONLY a valid JSON object. No markdown, no preamble, no extra keys:
{
  "generated_question": "<new question>",
  "generated_answer": "<new answer>"
}"""

# ─── Judge System Prompt ──────────────────────────────────────────────────────
JUDGE_SYSTEM_PROMPT = """You are a strict quality verifier for an Indian agricultural Q&A dataset.

You will receive:
  - GENERATED QUESTION: the question produced by the pipeline
  - SEED ANSWER: the original expert-verified source of truth
  - GENERATED ANSWER: a paraphrase that should be grounded in the seed

YOUR TASK — TWO CHECKS:

CHECK 1 — GROUNDING:
Does the GENERATED ANSWER contain any claim, fact, chemical name, dosage,
quantity, location, timing, or recommendation NOT explicitly present in the SEED ANSWER?
- Paraphrasing and rewording are ALLOWED. Only flag NEW facts not in the seed.
- Minor reformatting (e.g. "two or three weeks" to "2-3 weeks") is NOT a flag.
- Omitting seed content is NOT a flag — only addition of new facts matters.

CHECK 2 — SCOPE ALIGNMENT:
Does the GENERATED ANSWER stay within the scope of the GENERATED QUESTION?
- If the question asks about ONE specific chemical or intervention, the answer
  must discuss ONLY that item.
- Set scope_overflow=true if the answer discusses items the question did not ask about.

Set grounded=false if CHECK 1 fails OR if scope_overflow=true.

Output ONLY valid JSON:
{
  "grounded": true or false,
  "scope_overflow": true or false,
  "confidence": "high" or "medium" or "low",
  "issues": ["<specific problem found>"]
}"""

# ─── Variation Hints ──────────────────────────────────────────────────────────
VARIATION_HINTS = [
    "First attempt — write as if the farmer is describing their problem for the first time.",
    "Second attempt — use a different opening and sentence structure than before.",
    "Third attempt — focus on a different aspect or sub-topic within the seed.",
    "Fourth attempt — change the perspective: field level vs after harvest vs early season.",
    "Fifth attempt — use shorter, simpler sentences and more informal words.",
    "Sixth attempt — focus on a detail NOT highlighted in previous attempts.",
    "Seventh attempt — start from the solution rather than the problem.",
    "Eighth attempt — emphasise a different symptom, cause, or remedy.",
]

def get_variation_hint(call_count: int) -> str:
    idx = min(call_count, len(VARIATION_HINTS) - 1)
    return VARIATION_HINTS[idx]

# ─── Prompt Builder ───────────────────────────────────────────────────────────
def build_user_prompt(seed: dict, call_count: int) -> str:
    hint = get_variation_hint(call_count)
    clean_answer = strip_kvk_lines(seed.get("answer", ""))
    contextual_meta = build_contextual_metadata(seed)
    return (
        f"STYLE: {STYLE['name']}\n"
        f"STYLE INSTRUCTION: {STYLE['description']}\n"
        f"\nVARIATION HINT: {hint}\n\n"
        f"SEED QUESTION:\n{seed['question']}\n\n"
        f"SEED ANSWER (use ONLY this content — do not add anything not stated here):\n"
        f"{clean_answer}\n\n"
        f"CONTEXT (for reference only — do NOT inject into your answer "
        f"unless already in SEED ANSWER above):\n"
        f"{contextual_meta}\n\n"
        "Now produce the JSON output."
    )

# ─── Judge Prompt Builder ─────────────────────────────────────────────────────
def build_judge_prompt(seed_answer: str, generated_question: str,
                       generated_answer: str) -> str:
    clean_seed = strip_kvk_lines(seed_answer)
    return (
        f"GENERATED QUESTION:\n{generated_question}\n\n"
        f"SEED ANSWER:\n{clean_seed}\n\n"
        f"GENERATED ANSWER:\n{generated_answer}\n\n"
        "Now output the JSON verdict."
    )

# ─── Response Parsers ─────────────────────────────────────────────────────────
def parse_response(text: str):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None, "parse_failed"
    try:
        obj = json.loads(match.group())
        if obj.get("generated_question") is None:
            return None, "null_signal"
        return obj, "ok"
    except json.JSONDecodeError:
        return None, "json_error"

def parse_judge_response(text: str) -> dict:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
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

# ─── Client ───────────────────────────────────────────────────────────────────
client = OpenAI(base_url=VLLM_URL, api_key="dummy")

# ─── Generation Caller ────────────────────────────────────────────────────────
def call_model(seed: dict, call_count: int):
    user_prompt = build_user_prompt(seed, call_count)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

            raw_content = resp.choices[0].message.content
            if raw_content is None:
                print(f"    [null-content] retrying...", end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue
            text = raw_content.strip()

            result, reason = parse_response(text)
            if result is None:
                if reason == "null_signal":
                    return None
                print(f"    [parse-fail:{reason}] retrying...", end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

            answer = result.get("generated_answer", "")

            if has_self_reference(answer):
                print(f"    [self-ref] retrying...", end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

            if has_looping_text(answer):
                print(f"    [loop] retrying...", end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

            if not is_answer_length_ok(answer):
                wc = len(answer.split())
                print(f"    [len={wc}w, need {MIN_ANSWER_WORDS}–{MAX_ANSWER_WORDS}w] retrying...",
                      end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

            return result

        except Exception as e:
            print(f"    [error] attempt {attempt+1}: {e}")
            time.sleep(RETRY_DELAY * (2 ** attempt))
    return None

# ─── Judge Caller ─────────────────────────────────────────────────────────────
def call_judge(seed_answer: str, generated_question: str,
               generated_answer: str) -> dict:
    judge_prompt = build_judge_prompt(seed_answer, generated_question, generated_answer)
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
        raw_content = resp.choices[0].message.content
        if raw_content is None:
            return {"grounded": True, "scope_overflow": False,
                    "confidence": "low", "issues": [], "judge_error": "null_content"}
        return parse_judge_response(raw_content.strip())
    except Exception as e:
        return {"grounded": True, "scope_overflow": False,
                "confidence": "low", "issues": [], "judge_error": str(e)[:120]}

# ─── ID Generator ─────────────────────────────────────────────────────────────
def make_generated_id(seed: dict, idx: int) -> str:
    base = str(seed.get(COL_ID, uuid.uuid4()))[:8]
    return f"si_conv_{base}_{idx}"

# ─── CSV Loader ───────────────────────────────────────────────────────────────
def load_seeds_from_csv(filepath: str) -> list[dict]:
    seeds = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if COL_QUESTION not in row or COL_ANSWER not in row:
                raise ValueError(
                    f"CSV must have '{COL_QUESTION}' and '{COL_ANSWER}' columns. "
                    f"Found: {list(row.keys())}"
                )
            seed = {
                "question" : row[COL_QUESTION].strip(),
                "answer"   : row[COL_ANSWER].strip(),
                "crop"     : row.get(COL_CROP, "").strip(),
                "state"    : row.get(COL_STATE, "").strip(),
                "district" : row.get(COL_DISTRICT, "").strip(),
                "season"   : row.get(COL_SEASON, "").strip(),
                "domain"   : row.get(COL_DOMAIN, "").strip(),
                "id"       : row.get(COL_ID, str(i)).strip(),
            }
            if not seed["question"] or not seed["answer"]:
                print(f"  [warning] Row {i+1} skipped — empty question or answer.")
                continue
            seeds.append(seed)
    return seeds

# ─── Batch Checkpoint Helpers ─────────────────────────────────────────────────

def batch_checkpoint_path(batch_num: int) -> Path:
    """Return path like checkpoints/batch_001.json"""
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    return Path(CHECKPOINT_DIR) / f"batch_{batch_num:03d}.json"

def save_batch_checkpoint(batch_num: int, batch_results: list,
                           batch_log: list, meta: dict) -> None:
    """Save a single batch's results to its own JSON checkpoint file."""
    payload = {
        "meta"    : meta,           # batch_num, seed range, timestamp, counts
        "log"     : batch_log,
        "results" : batch_results,
    }
    path = batch_checkpoint_path(batch_num)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"   💾 Batch checkpoint saved → {path}  ({len(batch_results)} pairs)")

def load_batch_checkpoint(batch_num: int) -> dict | None:
    """Load an existing batch checkpoint, or return None if absent / corrupt."""
    path = batch_checkpoint_path(batch_num)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Validate minimal structure
        if "results" not in data or "meta" not in data:
            print(f"  ⚠ Batch {batch_num:03d} checkpoint malformed — will reprocess.")
            return None
        print(f"  ✔ Batch {batch_num:03d} checkpoint found — "
              f"{len(data['results'])} pairs, skipping.")
        return data
    except Exception as e:
        print(f"  ⚠ Could not load batch {batch_num:03d} checkpoint ({e}) — will reprocess.")
        return None

def completed_seed_ids_in_batch(checkpoint: dict) -> set:
    """Return the set of source_ids already recorded in a batch checkpoint."""
    return {r["source_id"] for r in checkpoint.get("results", [])}

# ─── Final-output Writer ──────────────────────────────────────────────────────

def write_final_outputs(all_results: list, all_log_lines: list) -> None:
    """Write the merged JSON + CSV + log to the final output paths."""
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

    with open(OUT_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(all_log_lines))

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    num_batches = (MAX_SEEDS + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\n{'='*65}")
    print(f"  self_instruct_conversational.py")
    print(f"  Style      : {STYLE['name']} only")
    print(f"  Seeds      : up to {MAX_SEEDS}  |  batch size: {BATCH_SIZE}")
    print(f"  Batches    : {num_batches}  |  target: {TARGET_PER_STYLE} pairs/seed")
    print(f"  Expected   : up to {MAX_SEEDS * TARGET_PER_STYLE} total pairs")
    print(f"  Checkpoints: {CHECKPOINT_DIR}/batch_NNN.json")
    print(f"{'='*65}\n")

    all_seeds = load_seeds_from_csv(INPUT_FILE)
    seeds = all_seeds[:MAX_SEEDS]
    print(f"  Loaded {len(seeds)} seeds from {INPUT_FILE}\n")

    # Accumulated across all batches (for final merge + summary)
    grand_results  : list = []
    grand_log      : list = []

    judge_flag_count     = 0
    scope_overflow_count = 0
    global_idx           = 0

    # ── Iterate batches ───────────────────────────────────────────────────────
    for batch_num in range(1, num_batches + 1):
        batch_start_idx = (batch_num - 1) * BATCH_SIZE
        batch_end_idx   = min(batch_num * BATCH_SIZE, len(seeds))
        batch_seeds     = seeds[batch_start_idx:batch_end_idx]

        print(f"\n{'─'*65}")
        print(f"  BATCH {batch_num:03d}/{num_batches:03d}  "
              f"seeds {batch_start_idx+1}–{batch_end_idx}")
        print(f"{'─'*65}")

        # ── Resume: load existing checkpoint if present ───────────────────────
        checkpoint = load_batch_checkpoint(batch_num)
        if checkpoint is not None:
            # Batch fully finished — collect its results and move on
            grand_results.extend(checkpoint["results"])
            grand_log.extend(checkpoint.get("log", []))
            global_idx += len(checkpoint["results"])
            continue

        # ── Process batch seeds ───────────────────────────────────────────────
        batch_results  : list = []
        batch_log      : list = []
        completed_ids  : set  = set()   # seed IDs finished so far in this batch
        batch_global_start = global_idx

        for seed_idx_in_batch, seed in enumerate(batch_seeds):
            seed_idx_global = batch_start_idx + seed_idx_in_batch
            seed_id = seed.get("id", str(seed_idx_global))

            if seed_id in completed_ids:
                print(f"  [skip] seed {seed_idx_global+1} already in this batch run.")
                continue

            print(f"\n── Seed {seed_idx_global+1}/{len(seeds)} "
                  f"| crop={seed.get('crop','?')} | id={seed_id} ──")

            seed_results   = []
            seen_questions = set()
            style_done     = 0
            style_failures = 0
            call_count     = 0
            seed_start     = time.time()

            while style_done < TARGET_PER_STYLE and style_failures < MAX_STYLE_FAILURES:

                print(f"  [{style_done+1}/{TARGET_PER_STYLE}] call #{call_count+1} ...",
                      end=" ", flush=True)

                result = call_model(seed, call_count)
                call_count += 1
                global_idx += 1

                if result is None:
                    print("skipped (null / self-ref / loop / length)")
                    batch_log.append(
                        f"SKIP | seed={seed_idx_global} | call={call_count} | (null or filter)"
                    )
                    style_failures += 1
                    continue

                new_q = result["generated_question"]

                if is_duplicate(new_q, seen_questions):
                    print("skipped (duplicate)")
                    batch_log.append(
                        f"DUPE | seed={seed_idx_global} | call={call_count} | q={new_q[:60]}"
                    )
                    style_failures += 1
                    continue

                print(f"✓ gen ({len(result['generated_answer'].split())}w) → judging...",
                      end=" ", flush=True)

                # ── Judge gate ────────────────────────────────────────────────
                judge_retry_count    = 0
                is_grounded          = False
                judge_confidence     = "low"
                judge_issues         = []
                judge_scope_overflow = False

                while judge_retry_count <= MAX_JUDGE_RETRIES:
                    verdict = call_judge(
                        seed["answer"],
                        result["generated_question"],
                        result["generated_answer"],
                    )
                    is_grounded          = verdict.get("grounded", True)
                    judge_confidence     = verdict.get("confidence", "low")
                    judge_issues         = verdict.get("issues", [])
                    judge_scope_overflow = verdict.get("scope_overflow", False)

                    if is_grounded:
                        break

                    judge_flag_count += 1
                    if judge_scope_overflow:
                        scope_overflow_count += 1

                    judge_retry_count += 1
                    batch_log.append(
                        f"JUDGE_RETRY | seed={seed_idx_global} | retry={judge_retry_count}/{MAX_JUDGE_RETRIES} | "
                        f"conf={judge_confidence} | scope_overflow={judge_scope_overflow} | "
                        f"issues={judge_issues[:2]} | q={new_q[:60]}"
                    )

                    if judge_retry_count > MAX_JUDGE_RETRIES:
                        print(f"⚠ FLAGGED ({judge_confidence}) — max judge retries exhausted, slot skipped")
                        break

                    print(f"⚠ FLAGGED ({judge_confidence}) — retrying "
                          f"(judge retry {judge_retry_count}/{MAX_JUDGE_RETRIES})...",
                          end=" ", flush=True)

                    global_idx += 1
                    result = call_model(seed, call_count)
                    call_count += 1

                    if result is None:
                        print("skipped (null during judge retry)")
                        batch_log.append(
                            f"SKIP | seed={seed_idx_global} | call={call_count} | (judge-retry)"
                        )
                        break

                    new_q = result["generated_question"]
                    print(f"✓ gen ({len(result['generated_answer'].split())}w) → judging...",
                          end=" ", flush=True)

                if not is_grounded or result is None:
                    style_failures += 1
                    batch_log.append(
                        f"JUDGE_ABANDON | seed={seed_idx_global} | conf={judge_confidence} | "
                        f"scope_overflow={judge_scope_overflow} | q={new_q[:60]}"
                    )
                    continue

                print(f"✓ grounded Q: {new_q[:55]}...")

                style_failures = 0
                seen_questions.add(normalize_q(new_q))

                record = {
                    "generated_id"        : make_generated_id(seed, global_idx),
                    "style"               : STYLE["name"],
                    "source_id"           : seed_id,
                    "batch_num"           : batch_num,
                    "crop"                : seed.get("crop", ""),
                    "state"               : seed.get("state", ""),
                    "district"            : seed.get("district", ""),
                    "season"              : seed.get("season", ""),
                    "domain"              : seed.get("domain", ""),
                    "generated_question"  : new_q,
                    "generated_answer"    : result["generated_answer"],
                    "seed_question"       : seed["question"],
                    "seed_answer"         : seed["answer"],
                    "style_call_number"   : call_count,
                    "answer_word_count"   : len(result["generated_answer"].split()),
                    "judge_grounded"      : True,
                    "judge_scope_overflow": False,
                    "judge_confidence"    : judge_confidence,
                    "judge_issues"        : "",
                    "timestamp"           : datetime.now(timezone.utc).isoformat(),
                }

                seed_results.append(record)
                batch_results.append(record)
                style_done += 1

                batch_log.append(
                    f"OK   | seed={seed_idx_global} | call={call_count} | "
                    f"judge=PASS | scope_overflow=False | q={new_q[:60]}"
                )

                time.sleep(0.3)

            elapsed = time.time() - seed_start
            print(f"\n   ✅ Seed {seed_idx_global+1} done — {style_done}/{TARGET_PER_STYLE} pairs "
                  f"in {elapsed:.1f}s")
            completed_ids.add(seed_id)

        # ── Save batch checkpoint ─────────────────────────────────────────────
        batch_meta = {
            "batch_num"     : batch_num,
            "seed_range"    : [batch_start_idx + 1, batch_end_idx],
            "pairs_produced": len(batch_results),
            "completed_at"  : datetime.now(timezone.utc).isoformat(),
        }
        save_batch_checkpoint(batch_num, batch_results, batch_log, batch_meta)

        grand_results.extend(batch_results)
        grand_log.extend(batch_log)

    # ── Write final merged outputs ────────────────────────────────────────────
    write_final_outputs(grand_results, grand_log)

    # ── Summary ───────────────────────────────────────────────────────────────
    dupe_count          = sum(1 for l in grand_log if l.startswith("DUPE"))
    skip_count          = sum(1 for l in grand_log if l.startswith("SKIP"))
    judge_retry_total   = sum(1 for l in grand_log if l.startswith("JUDGE_RETRY"))
    judge_abandon_total = sum(1 for l in grand_log if l.startswith("JUDGE_ABANDON"))

    if grand_results:
        wcs = [r["answer_word_count"] for r in grand_results]
        avg_wc = sum(wcs) / len(wcs)
        min_wc, max_wc = min(wcs), max(wcs)
    else:
        avg_wc = min_wc = max_wc = 0

    print(f"\n{'='*65}")
    print(f"  DONE — {len(grand_results)} conversational pairs generated")
    print(f"  Batches completed          : {num_batches}")
    print(f"  Duplicates rejected        : {dupe_count}")
    print(f"  Skipped (all causes)       : {skip_count}")
    print(f"  Judge retries              : {judge_retry_total}")
    print(f"  Judge abandoned            : {judge_abandon_total}")
    print(f"  Answer length: avg={avg_wc:.0f}w, min={min_wc}w, max={max_wc}w")
    print(f"\n  Batch checkpoints : {CHECKPOINT_DIR}/batch_NNN.json")
    print(f"  Merged JSON       : {OUT_JSON}")
    print(f"  Merged CSV        : {OUT_CSV}")
    print(f"  Log               : {OUT_LOG}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()