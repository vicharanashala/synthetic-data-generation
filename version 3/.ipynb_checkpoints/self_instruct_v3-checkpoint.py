# self_instruct_paraphrase_2.py  [self_instruct_v3.py on server]
# Purpose : Expand seed Q&A pairs into diverse paraphrase variants.
# Run     : python self_instruct_v3.py
#
# FIXES vs v2:
#   1. Deduplication — exact-match and near-match question filtering per seed
#   2. Variation hints — each style call gets a cycle counter so model varies output
#      + followup hints explicitly name which detail type to use each call
#   3. Hindi/farmer style — technical terms locked to English, no nonsense translations
#   4. Loop detection — answers with repeated sentences are discarded and retried
#   5. Grounding — KVK disclaimers banned in system prompt
#   A. KVK stripper — removes KVK/disclaimer lines from seed BEFORE passing to model
#   B. problem_description capped — 3-5 sentences only, no full seed dump

import json
import time
import re
import uuid
import csv
from datetime import datetime
from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────────
VLLM_URL        = "http://100.100.108.100:8080/v1"
MODEL           = "Qwen/Qwen3-30B-A3B"
INPUT_FILE      = "/home/kritika/self_instruct/self_instruct_clean.json"
OUT_JSON        = "/home/kritika/self_instruct/paraphrase_v3.json"
OUT_CSV         = "/home/kritika/self_instruct/paraphrase_v3.csv"
OUT_LOG         = "/home/kritika/self_instruct/paraphrase_v3_log.txt"

MAX_SEEDS       = 2        # set to 1247 for full run
TARGET_PER_SEED = 60       # pairs to generate per seed
MAX_RETRIES     = 3
RETRY_DELAY     = 5
TEMPERATURE     = 0.75     # slightly higher to encourage variety
MAX_TOKENS      = 1200

# ─── Fix A: Strip KVK/disclaimer lines from seed before passing to model ─────
KVK_PATTERNS = [
    r'farmers?\s+are\s+advised\s+to\s+contact.*?(?:\.|$)',
    r'contact\s+the\s+nearest\s+krishi\s+vigyan\s+kendra.*?(?:\.|$)',
    r'consult.*?kvk.*?(?:\.|$)',
    r'local\s+agricultural\s+extension\s+officer.*?(?:\.|$)',
    r'for\s+further\s+guidance.*?(?:kvk|extension\s+officer).*?(?:\.|$)',
    r'they\s+can\s+provide\s+customized\s+nutrient\s+management.*?(?:\.|$)',
]

def strip_kvk_lines(text: str) -> str:
    """Remove KVK referral and disclaimer sentences from seed answer."""
    for pattern in KVK_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    # Clean up extra whitespace/newlines left behind
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'  +', ' ', text)
    return text.strip()

# ─── Fix 4: Loop detection ────────────────────────────────────────────────────
# If any sentence appears more than this many times in an answer → discard
MAX_SENTENCE_REPEATS = 2

def has_looping_text(text: str) -> bool:
    """Return True if any sentence repeats more than MAX_SENTENCE_REPEATS times."""
    sentences = [s.strip() for s in re.split(r'[.।\n]', text) if len(s.strip()) > 20]
    seen = {}
    for s in sentences:
        seen[s] = seen.get(s, 0) + 1
        if seen[s] > MAX_SENTENCE_REPEATS:
            return True
    return False

# ─── Fix 1: Deduplication ─────────────────────────────────────────────────────
def normalize_q(q: str) -> str:
    """Lowercase, strip punctuation for near-match comparison."""
    return re.sub(r'[^a-z0-9\s]', '', q.lower()).strip()

def is_duplicate(new_q: str, seen_questions: set) -> bool:
    norm = normalize_q(new_q)
    if norm in seen_questions:
        return True
    # Near-match: if 90%+ of words overlap with any seen question
    new_words = set(norm.split())
    for seen in seen_questions:
        seen_words = set(seen.split())
        if not new_words or not seen_words:
            continue
        overlap = len(new_words & seen_words) / max(len(new_words), len(seen_words))
        if overlap >= 0.90:
            return True
    return False

# ─── Style Taxonomy ──────────────────────────────────────────────────────────
# Fix 3 applied in farmer_style and regional_hindi descriptions below.
# Fix 2 applied via variation_hint injected at call time (see build_user_prompt).

STYLES = [
    {
        "name": "paraphrase_formal",
        "description": (
            "Rewrite the question in a formal, academic or extension-officer tone. "
            "Use technical terminology correctly. Structure the answer with numbered "
            "points or clear sections where appropriate. "
            "STRICT: Only rephrase content already in the seed — do NOT add new facts, "
            "do NOT add KVK referral sentences, do NOT add standard disclaimers."
        ),
    },
    {
        "name": "paraphrase_farmer_style",
        "description": (
            "Rewrite the question as a semi-literate Indian farmer calling a Kisan helpline. "
            "Use simple conversational Hinglish — words like 'khet', 'fasal', 'bimari', "
            "'dawai', 'kab', 'kaise', 'kitna'. and other similar hindi words "
            "CRITICAL language rules: "
            "(1) Keep ALL crop names in English (e.g. 'peas', 'maize', not translated). "
            "(2) Keep ALL chemical names in English (e.g. 'Pseudomonas fluorescens', 'Thiram'). "
            "(3) Keep ALL fertilizer codes in English (e.g. 'NP(S) 20-20-13'). "
            "(4) Do NOT invent Hindi words for technical terms — use the English term as-is. "
            "(5) Only translate general words like 'crop', 'field', 'disease', 'spray', 'apply'. "
            "The answer should be in simple practical Hinglish. "
            "STRICT: Do NOT add new facts not in the seed."
        ),
    },
    {
        "name": "paraphrase_short",
        "description": (
            "Produce a very concise question (under 15 words) and a crisp answer "
            "(2-4 sentences maximum). Drop all background context, give only the "
            "core actionable information from the seed. "
            "STRICT: Do NOT add new facts not in the seed."
        ),
    },
    {
        "name": "paraphrase_problem_description",
        "description": (
            "Rewrite the question as a farmer describing what they are seeing in "
            "their field right now — lead with symptoms, do NOT name the disease/pest. "
            "E.g. 'My pea plants are wilting and roots look black — what is wrong?' "
            "The answer MUST be SHORT: 3-5 sentences only. "
            "Structure: 1 sentence diagnosis + 2-3 key management actions from the seed. "
            "Do NOT dump the full seed answer. Do NOT list all 10+ management points. "
            "Pick only the most important 2-3 actions. "
            "STRICT: Do NOT add new facts not in the seed."
        ),
    },
    {
        "name": "paraphrase_followup",
        "description": (
            "Rewrite as a specific follow-up question a farmer asks AFTER receiving "
            "a basic answer. Pick ONE specific detail from the seed answer and make "
            "it the sole focus. Available detail types to pick from (rotate through "
            "them across calls — never repeat the same type twice in a row): "
            "crop_rotation_duration, chemical_name, dosage_quantity, application_method, "
            "application_timing, soil_condition, irrigation_method, symptom_detail, "
            "pathogen_name, seed_treatment. "
            "The EXCLUDED DETAIL (already used) will be listed in the VARIATION HINT. "
            "E.g. 'You mentioned crop rotation — for how many years and with which crops?' "
            "STRICT: Do NOT add new facts not in the seed."
        ),
    },
    {
        "name": "paraphrase_yes_no",
        "description": (
            "Convert the topic into a binary yes/no question that can be directly "
            "answered from the seed. The answer MUST start with 'Yes' or 'No', "
            "then explain briefly using only seed content. "
            "Each call must ask about a DIFFERENT yes/no aspect of the seed. "
            "STRICT: Do NOT add new facts not in the seed."
        ),
    },
    {
        "name": "paraphrase_preventive",
        "description": (
            "Reframe the question to ask HOW TO PREVENT the problem, not treat it. "
            "Pull out only the preventive/cultural practices already mentioned in the seed. "
            "Each call should emphasise a DIFFERENT prevention angle (soil, seed, timing, etc). "
            "STRICT: Do NOT add new facts, do NOT add KVK referrals not in the seed."
        ),
    },
    {
        "name": "paraphrase_comparison",
        "description": (
            "If the seed mentions two or more distinct chemicals, pathogens, varieties, "
            "or management methods, write a question comparing them. "
            "Answer by drawing the comparison strictly from seed content. "
            "If the seed has only ONE item to compare, return JSON with generated_question=null. "
            "STRICT: Do NOT add new facts not in the seed."
        ),
    },
    {
        "name": "paraphrase_cause",
        "description": (
            "Rewrite the question to focus on WHY or WHAT CAUSES the problem. "
            "E.g. 'Why do pea roots turn black and rot?' "
            "Explain the causal mechanism using only seed content. "
            "Each call should approach the cause from a slightly different angle. "
            "STRICT: Do NOT add new facts not in the seed."
        ),
    },
    {
        "name": "paraphrase_quantity_focused",
        "description": (
            "Extract a quantity-related aspect (dosage, seed rate, concentration, "
            "area, duration) from the seed and make it the sole focus of the question. "
            "If the seed has NO quantities at all, return JSON with generated_question=null. "
            "If multiple quantities exist, each call should focus on a DIFFERENT one. "
            "STRICT: Do NOT add new facts, do NOT invent quantities not in the seed."
        ),
    },
    {
        "name": "paraphrase_timing_focused",
        "description": (
            "Focus the question on WHEN — crop growth stage, season, or application timing. "
            "If the seed has NO timing information at all, return JSON with generated_question=null. "
            "Each call should focus on a DIFFERENT timing aspect if multiple exist. "
            "STRICT: Do NOT add new facts not in the seed."
        ),
    },
    {
        "name": "paraphrase_regional_hindi",
        "description": (
            "Write the question in natural Hinglish (romanised Hindi mixed with English) "
            "as spoken by North Indian farmers. "
            "CRITICAL language rules: "
            "(1) Keep ALL crop names in English (e.g. 'peas', 'maize'). "
            "(2) Keep ALL chemical names in English (e.g. 'Pseudomonas fluorescens'). "
            "(3) Keep ALL fertilizer codes in English (e.g. 'NP(S) 20-20-13'). "
            "(4) Do NOT translate scientific or technical terms into Hindi. "
            "(5) Only use Hindi for connecting words: 'mein', 'ka', 'ke liye', 'kya', "
            "'kaise', 'kab', 'bimari', 'ilaj', 'khet', 'fasal'. "
            "Example: 'Peas ki fasal mein root rot ka kya ilaj hai?' "
            "The answer should be in simple English or light Hinglish. "
            "STRICT: Do NOT add new facts not in the seed."
        ),
    },
]

# ─── System Prompt ────────────────────────────────────────────────────────────
# Fix 5: Explicit ban on KVK disclaimers and ungrounded content
SYSTEM_PROMPT = """You are an expert in Indian agricultural Q&A dataset creation.
You receive a SEED question and answer from an Indian agricultural helpline,
a STYLE instruction, and a VARIATION HINT telling you which attempt this is.

YOUR TASK:
- Generate exactly ONE new (question, answer) pair in the requested style.
- The answer MUST be grounded ONLY in the seed answer content.

ABSOLUTE RULES — violating any of these makes the output unusable:
1. Do NOT invent new facts, chemicals, dosages, crop names, or percentages
   that are not explicitly present in the seed answer.
2. Do NOT add "consult your nearest KVK" or "contact local extension officer"
   UNLESS this exact phrase already appears in the seed answer.
3. Do NOT add standard disclaimers or generic advice not in the seed.
4. Keep ALL chemical names, fertilizer codes, and crop names in English
   even when writing in Hinglish or regional Hindi style.
5. If the style is impossible for this seed, return JSON with generated_question=null.
6. Use the VARIATION HINT to ensure your output is lexically different from
   previous attempts — change the entry point, sentence structure, or focus angle.

Output ONLY a valid JSON object. No markdown, no preamble:
{
  "generated_question": "<new question, or null>",
  "generated_answer": "<new answer, or null>"
}"""

# ─── Helpers ──────────────────────────────────────────────────────────────────
client = OpenAI(base_url=VLLM_URL, api_key="dummy")

# Fix 2: variation_hint injected per call
VARIATION_HINTS = [
    "First attempt — establish the core framing. For followup: focus on 'dosage_quantity'.",
    "Second attempt — use a different opening word and sentence structure. For followup: focus on 'crop_rotation_duration', NOT dosage.",
    "Third attempt — approach from a different sub-topic within the seed. For followup: focus on 'application_timing', NOT dosage or rotation.",
    "Fourth attempt — change the perspective (field level vs lab level vs farmer vs officer). For followup: focus on 'soil_condition' or 'irrigation_method'.",
    "Fifth attempt — use shorter sentences and simpler vocabulary. For followup: focus on 'symptom_detail' or 'pathogen_name'.",
    "Sixth attempt — focus on a detail NOT highlighted in previous attempts. For followup: focus on 'application_method' or 'seed_treatment'.",
    "Seventh attempt — reorder the information flow completely. For followup: focus on 'chemical_name' if not yet used.",
    "Eighth attempt — emphasise a different symptom, cause, or remedy. For followup: revisit rotation or timing with a narrower angle.",
    "Ninth attempt — make the question more specific and the answer more concise. For followup: pick the least-used detail type.",
    "Tenth attempt — take the most general angle possible, broad scope. For followup: focus on overall management approach.",
]

def get_variation_hint(style_call_count: int) -> str:
    idx = min(style_call_count, len(VARIATION_HINTS) - 1)
    return VARIATION_HINTS[idx]


def build_user_prompt(seed: dict, style: dict, style_call_count: int) -> str:
    hint = get_variation_hint(style_call_count)
    # Fix A: strip KVK/disclaimer lines before passing seed to model
    raw_answer = seed.get('original_full_answer') or seed.get('generated_answer', '')
    clean_answer = strip_kvk_lines(raw_answer)
    return (
        f"STYLE: {style['name']}\n"
        f"STYLE INSTRUCTION: {style['description']}\n\n"
        f"VARIATION HINT: {hint}\n\n"
        f"SEED QUESTION:\n{seed['generated_question']}\n\n"
        f"SEED ANSWER:\n{clean_answer}\n\n"
        f"CROP: {seed.get('crop','')}, STATE: {seed.get('state','')}, "
        f"DOMAIN: {seed.get('domain','')}, SEASON: {seed.get('season','')}\n\n"
        "Now produce the JSON output."
    )


def parse_response(text: str):
    text = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
        if obj.get("generated_question") is None:
            return None
        return obj
    except json.JSONDecodeError:
        return None


def call_model(seed: dict, style: dict, style_call_count: int):
    user_prompt = build_user_prompt(seed, style, style_call_count)
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
            text = resp.choices[0].message.content.strip()
            result = parse_response(text)
            if result is None:
                return None

            # Fix 4: loop detection on answer
            if has_looping_text(result.get("generated_answer", "")):
                print(f"    [loop detected] retrying...", end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

            return result

        except Exception as e:
            print(f"    [error] attempt {attempt+1}: {e}")
        time.sleep(RETRY_DELAY * (2 ** attempt))
    return None


def make_generated_id(seed: dict, style_name: str, idx: int) -> str:
    base = seed.get("source_answer_id", str(uuid.uuid4()))[:8]
    return f"si_v3_{style_name[:6]}_{base}_{idx}"


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    with open(INPUT_FILE) as f:
        all_seeds = json.load(f)

    seeds = all_seeds[:MAX_SEEDS]
    print(f"\n{'='*60}")
    print(f"  Self-Instruct Paraphrase v3 (fixed)")
    print(f"  Seeds     : {len(seeds)} (of {len(all_seeds)} total)")
    print(f"  Target    : {TARGET_PER_SEED} Q&A pairs per seed")
    print(f"  Styles    : {len(STYLES)}")
    print(f"{'='*60}\n")

    all_results = []
    log_lines   = []

    for seed_idx, seed in enumerate(seeds):
        print(f"\n── Seed {seed_idx+1}/{len(seeds)}: {seed.get('crop')} | "
              f"{seed.get('state')} | {seed.get('domain')} ──")
        print(f"   Original Q: {seed['generated_question'][:80]}...")

        seed_results    = []
        seen_questions  = set()          # Fix 1: dedup tracker
        style_pos       = 0
        global_idx      = 0
        # Fix 2: track how many times each style has been called this seed
        style_call_counts = {s["name"]: 0 for s in STYLES}

        while len(seed_results) < TARGET_PER_SEED:
            style = STYLES[style_pos % len(STYLES)]
            style_pos  += 1
            global_idx += 1

            call_count = style_call_counts[style["name"]]
            style_call_counts[style["name"]] += 1

            print(f"  [{len(seed_results)+1}/{TARGET_PER_SEED}] "
                  f"Style={style['name']} (call #{call_count+1}) ... ",
                  end="", flush=True)

            result = call_model(seed, style, call_count)

            if result is None:
                print("skipped (null / parse fail / loop)")
                log_lines.append(
                    f"SKIP | seed={seed_idx} | style={style['name']} | call={call_count+1}"
                )
                continue

            # Fix 1: deduplication check
            new_q = result["generated_question"]
            if is_duplicate(new_q, seen_questions):
                print(f"duplicate — skipped")
                log_lines.append(
                    f"DUPE | seed={seed_idx} | style={style['name']} | q={new_q[:60]}"
                )
                continue

            seen_questions.add(normalize_q(new_q))

            record = {
                "generated_id"      : make_generated_id(seed, style["name"], global_idx),
                "style"             : style["name"],
                "source_answer_id"  : seed.get("source_answer_id", ""),
                "source_question_id": seed.get("source_question_id", ""),
                "crop"              : seed.get("crop", ""),
                "state"             : seed.get("state", ""),
                "district"          : seed.get("district", ""),
                "season"            : seed.get("season", ""),
                "domain"            : seed.get("domain", ""),
                "generated_question": new_q,
                "generated_answer"  : result["generated_answer"],
                "seed_question"     : seed["generated_question"],
                "style_call_number" : call_count + 1,   # for traceability
                "timestamp"         : datetime.utcnow().isoformat(),
            }

            seed_results.append(record)
            all_results.append(record)

            print(f"✓ Q: {new_q[:70]}...")
            log_lines.append(
                f"OK   | seed={seed_idx} | style={style['name']} | "
                f"call={call_count+1} | q={new_q[:60]}"
            )

            time.sleep(0.5)

        print(f"\n   ✅ Seed {seed_idx+1} done — {len(seed_results)} unique pairs")

    # ── Save outputs ──────────────────────────────────────────────────────────
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

    with open(OUT_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    # ── Summary ───────────────────────────────────────────────────────────────
    style_counts = {}
    dupe_count   = sum(1 for l in log_lines if l.startswith("DUPE"))
    skip_count   = sum(1 for l in log_lines if l.startswith("SKIP"))

    for r in all_results:
        style_counts[r["style"]] = style_counts.get(r["style"], 0) + 1

    print(f"\n{'='*60}")
    print(f"  DONE — {len(all_results)} unique pairs generated")
    print(f"  Duplicates rejected : {dupe_count}")
    print(f"  Skipped (null/loop) : {skip_count}")
    print(f"  Per-style breakdown:")
    for sname, cnt in sorted(style_counts.items()):
        print(f"    {sname:<35}: {cnt}")
    print(f"\n  Output JSON : {OUT_JSON}")
    print(f"  Output CSV  : {OUT_CSV}")
    print(f"  Log         : {OUT_LOG}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()