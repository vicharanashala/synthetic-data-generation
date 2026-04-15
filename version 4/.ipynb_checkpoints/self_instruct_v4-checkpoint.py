# self_instruct_v4.py
# Purpose : Expand seed Q&A pairs into diverse paraphrase variants.
# Run     : python self_instruct_v4.py
#
# CHANGES vs v3:
#   1. Input is now a CSV file (random_200_qa_1.csv) with 200 original Q&A pairs
#   2. Removed all Hindi/Hinglish styles — English only
#   3. Strict grounding enforced: model is explicitly told to use ONLY seed content
#   4. Style comments added explaining purpose + example for each style
#   5. KVK stripping retained from v3
#
# FIXES vs previous v4 runs:
#   FIX-1: Added has_self_reference() filter — hard-rejects answers containing
#           "the seed", "seed answer", "according to the seed", etc.
#   FIX-2: Added min answer length guard — retries if answer < 15 words
#   FIX-3: Added max answer length guard — retries if answer > 300 words
#   FIX-4: paraphrase_short now rotates question starters via variation hint
#           to avoid collapsing to "How to <verb>..." every time
#   FIX-5: paraphrase_followup tracks used focus types per seed and injects
#           hard exclusion list into the prompt to enforce rotation
#
# FIXES from v4 run-3 analysis:
#   FIX-6: paraphrase_yes_no was copying seed question verbatim — added
#           is_copied_seed_question() post-gen check + banned it in style desc
#   FIX-7: paraphrase_short opener used seed-hash offset so different seeds
#           get different starting openers, not just different call counts
#   FIX-8: detect_focus_type() now checks application_timing BEFORE
#           dosage_quantity (specificity ordering), and fallback changed from
#           "dosage_quantity" → "unknown" to avoid poisoning exclusion list

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
INPUT_FILE      = "/home/kritika/self_instruct/random_200_qa_1.csv"
OUT_JSON        = "/home/kritika/self_instruct/paraphrase_v4.json"
OUT_CSV         = "/home/kritika/self_instruct/paraphrase_v4.csv"
OUT_LOG         = "/home/kritika/self_instruct/paraphrase_v4_log.txt"

MAX_SEEDS           = 20       # TEST RUN — set to 200 for full run
TARGET_PER_STYLE    = 5        # pairs per style per seed
                               # 10 styles x 5 = 50/seed, 20 seeds = 1000 pairs total
MAX_STYLE_FAILURES  = 6        # abandon a style after this many consecutive failures
MAX_RETRIES         = 5        # raised — more styles need more retries
RETRY_DELAY     = 5
TEMPERATURE     = 0.75
MAX_TOKENS      = 1200

# Answer length guards (word count)
MIN_ANSWER_WORDS = 15    # FIX-2: discard answers shorter than this
MAX_ANSWER_WORDS = 300   # FIX-3: discard answers longer than this

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

# ─── FIX-1: Self-Reference Detection ─────────────────────────────────────────
# Hard-rejects answers that contain meta-references to the prompt/seed.
# These slip through despite the system prompt ban and must be caught in code.
SELF_REF_PATTERNS = [
    r'\bthe seed\b',
    r'\bseed answer\b',
    r'\baccording to the seed\b',
    r'\bas mentioned in the seed\b',
    r'\bthe source says\b',
    r'\bseed says\b',
    r'\bas stated in the seed\b',
    r'\bthe provided seed\b',
    r'\bthe prompt\b',
    r'\bthe input\b',
]

def has_self_reference(text: str) -> bool:
    """Return True if the answer contains any meta-reference to the seed/prompt."""
    for pattern in SELF_REF_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

# ─── FIX-2 / FIX-3: Answer Length Guard ──────────────────────────────────────
def is_answer_length_ok(text: str) -> bool:
    """Return True if answer word count is within [MIN, MAX] bounds."""
    word_count = len(text.split())
    return MIN_ANSWER_WORDS <= word_count <= MAX_ANSWER_WORDS

# ─── FIX-6: Yes/No Copied Seed Question Guard ────────────────────────────────
# The yes_no style was outputting the seed question verbatim as generated_question.
# This guard catches exact copies and high-overlap near-copies (>=85% word overlap).
def is_copied_seed_question(generated_q: str, seed_q: str) -> bool:
    """Return True if generated question is a verbatim or near-verbatim copy of seed."""
    def norm(t):
        return re.sub(r'[^a-z0-9\s]', '', t.lower()).strip()
    g = norm(generated_q)
    s = norm(seed_q)
    if g == s:
        return True
    g_words = set(g.split())
    s_words = set(s.split())
    if not g_words or not s_words:
        return False
    overlap = len(g_words & s_words) / max(len(g_words), len(s_words))
    return overlap >= 0.85

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
        overlap = len(new_words & seen_words) / max(len(new_words), len(seen_words))
        if overlap >= 0.90:
            return True
    return False

# ─── Style Taxonomy (English Only) ───────────────────────────────────────────

STYLES = [

    # ── Style 1: Formal / Extension-Officer Tone ──────────────────────────────
    {
        "name": "paraphrase_formal",
        "description": (
            "Rewrite the question in a formal, academic or extension-officer tone. "
            "Use correct technical terminology. Structure the answer with numbered "
            "points or clear sections where appropriate. "
            "Keep the answer under 250 words total. "
            "GROUNDING RULE: Only rephrase content already present in the seed answer. "
            "Do NOT add new facts, chemicals, dosages, or recommendations not in the seed. "
            "Do NOT add KVK referral sentences or generic disclaimers."
        ),
    },

    # ── Style 2: Short / Concise ──────────────────────────────────────────────
    # FIX-4: Question starters are now rotated via VARIATION HINT.
    # The hint injects a REQUIRED_OPENER that the model must use,
    # preventing the model from always defaulting to "How to <verb>...".
    {
        "name": "paraphrase_short",
        "description": (
            "Produce a very concise question (under 15 words) and a crisp answer "
            "(2-4 sentences maximum). Drop all background context — give only the "
            "core actionable information that is already in the seed. "
            "IMPORTANT: You MUST start the question with the REQUIRED_OPENER word "
            "specified in the VARIATION HINT. Do NOT start with 'How to'. "
            "GROUNDING RULE: Do NOT add any new facts not present in the seed answer."
        ),
    },

    # ── Style 3: Symptom / Problem Description ────────────────────────────────
    {
        "name": "paraphrase_problem_description",
        "description": (
            "Rewrite the question as a farmer describing symptoms they observe "
            "in their field right now. Lead with visible symptoms; do NOT name "
            "the disease or pest in the question. "
            "The answer MUST be SHORT: 3-5 sentences only. "
            "Structure: 1 sentence diagnosis + 2-3 key management actions from the seed. "
            "Do NOT list all management points from the seed — pick only the top 2-3. "
            "GROUNDING RULE: Do NOT add new facts not present in the seed answer."
        ),
    },

    # ── Style 4: Follow-up Question ───────────────────────────────────────────
    # FIX-5: Used focus types are tracked in Python per seed and injected
    # as a hard EXCLUDED_FOCUS_TYPES list, overriding the model's default
    # tendency to always pick dosage_quantity.
    {
        "name": "paraphrase_followup",
        "description": (
            "Rewrite as a specific follow-up question a farmer asks AFTER receiving "
            "a basic answer. Pick ONE specific detail from the seed answer and make "
            "it the sole focus of the question. "
            "MANDATORY: The VARIATION HINT contains an EXCLUDED_FOCUS_TYPES list. "
            "You MUST NOT pick any focus type from that list. "
            "Available focus types: crop_rotation_duration, chemical_name, "
            "dosage_quantity, application_method, application_timing, soil_condition, "
            "irrigation_method, symptom_detail, pathogen_name, seed_treatment. "
            "Pick the type that is NOT in the excluded list and NOT yet overused. "
            "GROUNDING RULE: Do NOT add new facts not present in the seed answer."
        ),
    },

    # ── Style 5: Yes / No Question ────────────────────────────────────────────
    {
        "name": "paraphrase_yes_no",
        "description": (
            "Convert the topic into a binary yes/no question that can be directly "
            "answered from the seed. The answer MUST start with 'Yes' or 'No', "
            "then explain briefly in 2-3 sentences using only seed content. "
            "Each call must ask about a DIFFERENT yes/no aspect of the seed. "
            "CRITICAL — QUESTION MUST BE A REAL YES/NO QUESTION: "
            "The generated question MUST be a new binary yes/no question that is "
            "DIFFERENT from the seed question. Do NOT copy or repeat the seed "
            "question verbatim or near-verbatim. A valid yes/no question must be "
            "answerable with Yes or No. "
            "GOOD EXAMPLES: 'Can Chlorpyriphos 20% EC be used as seed treatment "
            "for termites in wheat?', 'Is deep summer ploughing effective against "
            "termite infestation?', 'Should insecticides be avoided during the "
            "flowering stage of paddy?' "
            "BAD EXAMPLE (do NOT do this): copying the seed question as-is. "
            "CRITICAL LANGUAGE RULE: After Yes/No, write the explanation as a "
            "direct factual statement — as if you are the expert speaking to the "
            "farmer. NEVER write 'the seed says', 'according to the seed', "
            "'the seed answer states', or any similar citation of the prompt. "
            "WRONG: 'No. The seed answer explicitly states, Do not spray during flowering.' "
            "RIGHT:  'No. Insecticides should not be sprayed during flowering or "
            "when beneficial insects are active.' "
            "GROUNDING RULE: Do NOT add new facts not present in the seed answer."
        ),
    },

    # ── Style 6: Preventive / How-to-Avoid ───────────────────────────────────
    {
        "name": "paraphrase_preventive",
        "description": (
            "Reframe the question to ask HOW TO PREVENT the problem, not treat it. "
            "Pull out only the preventive or cultural practices already mentioned "
            "in the seed answer. Each call should emphasise a DIFFERENT prevention "
            "angle (soil preparation, seed selection, timing, spacing, etc.). "
            "GROUNDING RULE: Do NOT add new facts or KVK referrals not in the seed."
        ),
    },

    # ── Style 7: Comparison ───────────────────────────────────────────────────
    {
        "name": "paraphrase_comparison",
        "description": (
            "If the seed mentions two or more distinct chemicals, pathogens, "
            "varieties, or management methods, write a question comparing them. "
            "Answer by drawing the comparison strictly from seed content. "
            "If the seed has only ONE comparable item, return JSON with "
            "generated_question=null. "
            "GROUNDING RULE: Do NOT add new facts not present in the seed answer."
        ),
    },

    # ── Style 8: Cause / Why ──────────────────────────────────────────────────
    {
        "name": "paraphrase_cause",
        "description": (
            "Rewrite the question to focus on WHY or WHAT CAUSES the problem. "
            "Explain the causal mechanism using only seed content. "
            "Each call should approach the cause from a slightly different angle "
            "(e.g., pathogen biology, environmental trigger, soil condition). "
            "GROUNDING RULE: Do NOT add new facts not present in the seed answer."
        ),
    },

    # ── Style 9: Quantity / Dosage Focused ────────────────────────────────────
    {
        "name": "paraphrase_quantity_focused",
        "description": (
            "Extract a quantity-related aspect (dosage, seed rate, concentration, "
            "area coverage, duration) from the seed and make it the sole focus "
            "of the question. If the seed has NO quantities at all, return JSON "
            "with generated_question=null. If multiple quantities exist, each call "
            "should focus on a DIFFERENT quantity. "
            "GROUNDING RULE: Do NOT invent quantities not explicitly stated in the seed."
        ),
    },

    # ── Style 10: Timing / When Focused ──────────────────────────────────────
    {
        "name": "paraphrase_timing_focused",
        "description": (
            "Focus the question on WHEN — crop growth stage, season, or application "
            "timing window. If the seed has NO timing information at all, return JSON "
            "with generated_question=null. Each call should focus on a DIFFERENT "
            "timing aspect if multiple exist in the seed. "
            "GROUNDING RULE: Do NOT add timing facts not present in the seed answer."
        ),
    },
]

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert in Indian agricultural Q&A dataset creation.
You receive a SEED question and answer from an Indian agricultural helpline,
a STYLE instruction, and a VARIATION HINT telling you which attempt this is.

YOUR TASK:
- Generate exactly ONE new (question, answer) pair in the requested style.
- The answer MUST be grounded EXCLUSIVELY in the seed answer content.

ABSOLUTE GROUNDING RULES — violating any of these makes the output unusable:
1. Do NOT invent new facts, chemicals, dosages, crop names, percentages,
   or recommendations that are not explicitly stated in the seed answer.
2. Do NOT add "consult your nearest KVK" or "contact local extension officer"
   UNLESS this exact phrase already appears in the seed answer.
3. Do NOT add standard disclaimers, safety warnings, or generic best-practice
   advice that is not in the seed.
4. Keep ALL chemical names, fertilizer codes, and crop names in English.
5. If the requested style is impossible for this seed (e.g. comparison style
   but only one chemical mentioned), return JSON with generated_question=null.
6. Use the VARIATION HINT to ensure your output is lexically different from
   previous attempts — change the entry point, sentence structure, or focus angle.
7. If you are uncertain whether a fact is in the seed, DO NOT include it.
   Only state what is explicitly written in the seed answer.
8. Do NOT restructure or recontextualize seed facts. If a detail appears under
   a specific heading or bullet in the seed (e.g. timing under ETL, not under
   application instructions), preserve that context — do NOT move it to a
   different section or use it to explain a different point than the seed does.
   Facts must be used with the same meaning and context as in the seed.
9. NEVER use self-referential or pipeline language in your answer. The generated
   answer is a real helpline response to a real farmer — it must NEVER contain
   phrases like "the seed answer states", "according to the seed", "the seed says",
   "as mentioned in the seed", "the source says", "the seed", "the prompt",
   or any similar meta-reference to the prompt or dataset. Write as if you are
   the expert giving the answer directly. Paraphrase the content naturally —
   do NOT quote or cite the prompt.

Output ONLY a valid JSON object. No markdown, no preamble, no extra keys:
{
  "generated_question": "<new question, or null>",
  "generated_answer": "<new answer, or null>"
}"""

# ─── Variation Hints ──────────────────────────────────────────────────────────
# FIX-4: SHORT style openers are now injected via REQUIRED_OPENER in the hint.
# FIX-5: Followup excluded focus types are injected dynamically in build_user_prompt.

SHORT_OPENERS = [
    "What",      # attempt 0 → "What controls termites in wheat?"
    "Which",     # attempt 1 → "Which chemical treats Gundhi Bug in paddy?"
    "When",      # attempt 2 → "When should I apply fungicide for downy mildew?"
    "Why",       # attempt 3 → "Why do my chickpea roots rot even with correct irrigation?"
    "Can",       # attempt 4 → "Can deep ploughing prevent termite infestation?"
    "Should",    # attempt 5 → "Should I treat wheat seeds before sowing for termites?"
    "What is",   # attempt 6 → "What is the correct Chlorpyriphos dose for wheat?"
    "Is",        # attempt 7 → "Is Malathion effective against Gundhi Bug?"
    "Does",      # attempt 8 → "Does seed treatment help prevent root rot?"
    "Are",       # attempt 9 → "Are there cultural practices to reduce termite damage?"
]

VARIATION_HINTS = [
    "First attempt — establish the core framing. REQUIRED_OPENER (for short style): '{opener}'. For followup: EXCLUDED_FOCUS_TYPES: {excluded}.",
    "Second attempt — use a different opening word and sentence structure. REQUIRED_OPENER (for short style): '{opener}'. For followup: EXCLUDED_FOCUS_TYPES: {excluded}.",
    "Third attempt — approach from a different sub-topic within the seed. REQUIRED_OPENER (for short style): '{opener}'. For followup: EXCLUDED_FOCUS_TYPES: {excluded}.",
    "Fourth attempt — change the perspective (field level vs lab level vs farmer vs officer). REQUIRED_OPENER (for short style): '{opener}'. For followup: EXCLUDED_FOCUS_TYPES: {excluded}.",
    "Fifth attempt — use shorter sentences and simpler vocabulary. REQUIRED_OPENER (for short style): '{opener}'. For followup: EXCLUDED_FOCUS_TYPES: {excluded}.",
    "Sixth attempt — focus on a detail NOT highlighted in previous attempts. REQUIRED_OPENER (for short style): '{opener}'. For followup: EXCLUDED_FOCUS_TYPES: {excluded}.",
    "Seventh attempt — reorder the information flow completely. REQUIRED_OPENER (for short style): '{opener}'. For followup: EXCLUDED_FOCUS_TYPES: {excluded}.",
    "Eighth attempt — emphasise a different symptom, cause, or remedy. REQUIRED_OPENER (for short style): '{opener}'. For followup: EXCLUDED_FOCUS_TYPES: {excluded}.",
    "Ninth attempt — make the question more specific and the answer more concise. REQUIRED_OPENER (for short style): '{opener}'. For followup: EXCLUDED_FOCUS_TYPES: {excluded}.",
    "Tenth attempt — take the most general angle possible, broad scope. REQUIRED_OPENER (for short style): '{opener}'. For followup: EXCLUDED_FOCUS_TYPES: {excluded}.",
]

# All possible followup focus types — model must rotate through these
ALL_FOCUS_TYPES = [
    "crop_rotation_duration",
    "chemical_name",
    "dosage_quantity",
    "application_method",
    "application_timing",
    "soil_condition",
    "irrigation_method",
    "symptom_detail",
    "pathogen_name",
    "seed_treatment",
]

def get_variation_hint(style_call_count: int, used_focus_types: list,
                       seed_id: str = "") -> str:
    """
    Build the variation hint string with:
    - REQUIRED_OPENER for short style (FIX-4 / FIX-7)
      opener_idx = (seed_hash_offset + call_count) % len(SHORT_OPENERS)
      This ensures different seeds start on different openers, not all on "What".
    - EXCLUDED_FOCUS_TYPES for followup style (FIX-5)
    """
    idx = min(style_call_count, len(VARIATION_HINTS) - 1)

    # FIX-7: hash seed_id to get a per-seed offset into SHORT_OPENERS
    seed_hash_offset = abs(hash(seed_id)) % len(SHORT_OPENERS)
    opener_idx = (seed_hash_offset + style_call_count) % len(SHORT_OPENERS)
    opener = SHORT_OPENERS[opener_idx]

    # Excluded = last 3 used types to force rotation without blocking all options
    excluded = ", ".join(used_focus_types[-3:]) if used_focus_types else "none"

    return VARIATION_HINTS[idx].format(opener=opener, excluded=excluded)

# ─── Prompt Builder ───────────────────────────────────────────────────────────
def build_user_prompt(seed: dict, style: dict, style_call_count: int,
                      used_focus_types: list) -> str:
    hint = get_variation_hint(style_call_count, used_focus_types, seed.get("id", ""))
    raw_answer = seed.get("answer", "")
    clean_answer = strip_kvk_lines(raw_answer)

    return (
        f"STYLE: {style['name']}\n"
        f"STYLE INSTRUCTION: {style['description']}\n\n"
        f"VARIATION HINT: {hint}\n\n"
        f"SEED QUESTION:\n{seed['question']}\n\n"
        f"SEED ANSWER (use ONLY this content — do not add anything not stated here):\n"
        f"{clean_answer}\n\n"
        f"CROP: {seed.get('crop', 'N/A')}, "
        f"STATE: {seed.get('state', 'N/A')}, "
        f"DOMAIN: {seed.get('domain', 'N/A')}, "
        f"SEASON: {seed.get('season', 'N/A')}\n\n"
        "Now produce the JSON output."
    )

# ─── Response Parser ──────────────────────────────────────────────────────────
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

# ─── Model Caller ─────────────────────────────────────────────────────────────
client = OpenAI(base_url=VLLM_URL, api_key="dummy")

def call_model(seed: dict, style: dict, style_call_count: int,
               used_focus_types: list):
    user_prompt = build_user_prompt(seed, style, style_call_count, used_focus_types)
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
                return None, None

            answer = result.get("generated_answer", "")

            # FIX-1: Reject self-referencing answers
            if has_self_reference(answer):
                print(f"    [self-ref detected] retrying...", end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

            # FIX-6: Reject yes_no outputs that copied the seed question verbatim
            if style["name"] == "paraphrase_yes_no" and is_copied_seed_question(
                result.get("generated_question", ""), seed["question"]
            ):
                print(f"    [copied seed Q] retrying...", end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

            # Reject looping outputs
            if has_looping_text(answer):
                print(f"    [loop detected] retrying...", end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

            # FIX-2 / FIX-3: Reject answers outside word count bounds
            if not is_answer_length_ok(answer):
                word_count = len(answer.split())
                print(f"    [length={word_count}w, out of [{MIN_ANSWER_WORDS},{MAX_ANSWER_WORDS}]] retrying...",
                      end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

            # FIX-5: Extract focus type used so caller can update used_focus_types
            # We detect which focus type the model chose by scanning the question
            detected_focus = detect_focus_type(result.get("generated_question", ""))

            return result, detected_focus

        except Exception as e:
            print(f"    [error] attempt {attempt+1}: {e}")
        time.sleep(RETRY_DELAY * (2 ** attempt))
    return None, None

# ─── FIX-5 / FIX-8: Focus Type Detector ─────────────────────────────────────
# Heuristically detects which followup focus type the model used,
# based on keyword presence in the generated question.
# FIX-8: Keys are ordered by specificity — application_timing is checked BEFORE
# dosage_quantity so "correct time to apply" → timing, not dosage (the old bug).
# Fallback changed from "dosage_quantity" → "unknown" so we don't poison the
# exclusion list with a false dosage entry when detection fails.
FOCUS_TYPE_KEYWORDS = {
    # ── checked first (most specific / least ambiguous) ──────────────────────
    "application_timing"     : [
        "correct time", "time to apply", "when to apply", "which stage",
        "growth stage", "at what stage", "before sowing", "after sowing",
        "flowering stage", "vegetative stage", "timing of application",
    ],
    "crop_rotation_duration" : [
        "rotation", "how many years", "which crop to rotate", "alternate crop",
        "crop rotation",
    ],
    "seed_treatment"         : [
        "seed treatment", "treat seeds", "before sowing seed",
        "seed dressing", "seed coating", "seed priming",
    ],
    "pathogen_name"          : [
        "pathogen", "scientific name", "caused by", "fungus", "bacterium",
        "virus", "nematode", "which organism",
    ],
    "soil_condition"         : [
        "soil type", "soil ph", "soil condition", "soil moisture",
        "waterlogged", "drainage", "soil texture",
    ],
    "irrigation_method"      : [
        "irrigation", "how to water", "watering schedule",
        "flood irrigation", "drip irrigation", "furrow irrigation",
    ],
    "application_method"     : [
        "how to apply", "method of application", "how should i apply",
        "spray method", "application method", "drip", "drench", "broadcast",
        "foliar spray", "soil drench",
    ],
    "symptom_detail"         : [
        "symptom", "yellowing", "wilting", "discolored", "shriveled",
        "sign", "lesion", "spot", "what does it look like",
    ],
    "chemical_name"          : [
        "which chemical", "which fungicide", "which insecticide",
        "which pesticide", "name of the chemical", "what chemical",
    ],
    # ── checked last (broadest keywords — most prone to false positives) ─────
    "dosage_quantity"        : [
        "dosage", "dose", "how much", "quantity", "concentration",
        "g/kg", "ml/l", "per kg", "per acre", "per litre", "seed rate",
        "application rate",
    ],
}

def detect_focus_type(question: str) -> str:
    """
    Return the focus type that best matches the generated question.
    Keys in FOCUS_TYPE_KEYWORDS are ordered by specificity (FIX-8):
    more specific types are checked first to avoid misclassification.
    Returns 'unknown' if no keyword matches, so the exclusion list
    stays clean (FIX-8: old fallback was 'dosage_quantity' which poisoned it).
    """
    q_lower = question.lower()
    for focus_type, keywords in FOCUS_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in q_lower:
                return focus_type
    return "unknown"  # FIX-8: neutral fallback — do NOT assume dosage

# ─── ID Generator ─────────────────────────────────────────────────────────────
def make_generated_id(seed: dict, style_name: str, idx: int) -> str:
    base = str(seed.get(COL_ID, uuid.uuid4()))[:8]
    return f"si_v4_{style_name[:6]}_{base}_{idx}"

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

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\nLoading seeds from: {INPUT_FILE}")
    all_seeds = load_seeds_from_csv(INPUT_FILE)
    seeds = all_seeds[:MAX_SEEDS]

    print(f"\n{'='*60}")
    print(f"  Self-Instruct Paraphrase v4 (English only, CSV input)")
    print(f"  Seeds     : {len(seeds)} (of {len(all_seeds)} loaded)")
    print(f"  Target    : {TARGET_PER_STYLE} pairs x {len(STYLES)} styles = {TARGET_PER_STYLE * len(STYLES)} per seed")
    print(f"  Styles    : {len(STYLES)}")
    print(f"  Est. total: ~{len(seeds) * TARGET_PER_STYLE * len(STYLES)} pairs")
    print(f"  Answer length window: [{MIN_ANSWER_WORDS}, {MAX_ANSWER_WORDS}] words")
    print(f"{'='*60}\n")

    all_results = []
    log_lines   = []

    for seed_idx, seed in enumerate(seeds):
        print(f"\n── Seed {seed_idx+1}/{len(seeds)}: "
              f"{seed.get('crop') or 'N/A'} | "
              f"{seed.get('state') or 'N/A'} | "
              f"{seed.get('domain') or 'N/A'} ──")
        print(f"   Original Q: {seed['question'][:80]}...")

        seed_results   = []
        seen_questions = set()
        global_idx     = 0

        # FIX-5: Per-seed tracking of used followup focus types
        used_focus_types: list[str] = []

        # Outer loop: iterate each style, generate TARGET_PER_STYLE pairs each
        for style in STYLES:
            style_done     = 0   # accepted pairs for this style this seed
            style_failures = 0   # consecutive failures (null/loop/length/dupe)
            call_count     = 0   # how many times we've called this style

            print(f"\n  ┌ Style: {style['name']}")

            while style_done < TARGET_PER_STYLE:
                # Abandon style if it keeps failing (e.g. comparison on a
                # seed that only mentions one chemical — will always return null)
                if style_failures >= MAX_STYLE_FAILURES:
                    print(f"  └ [{style_done}/{TARGET_PER_STYLE}] abandoned "
                          f"after {style_failures} consecutive failures")
                    log_lines.append(
                        f"ABANDON | seed={seed_idx} | style={style['name']} | "
                        f"done={style_done}/{TARGET_PER_STYLE}"
                    )
                    break

                global_idx += 1
                print(f"  │ [{style_done+1}/{TARGET_PER_STYLE}] call #{call_count+1} ... ",
                      end="", flush=True)

                result, detected_focus = call_model(
                    seed, style, call_count, used_focus_types
                )
                call_count += 1

                if result is None:
                    style_failures += 1
                    print("skipped (null / self-ref / loop / length)")
                    log_lines.append(
                        f"SKIP | seed={seed_idx} | style={style['name']} | call={call_count}"
                    )
                    continue

                # FIX-5: Track followup focus types for rotation
                if style["name"] == "paraphrase_followup" and detected_focus:
                    used_focus_types.append(detected_focus)

                # Deduplication check
                new_q = result["generated_question"]
                if is_duplicate(new_q, seen_questions):
                    style_failures += 1
                    print(f"duplicate — skipped")
                    log_lines.append(
                        f"DUPE | seed={seed_idx} | style={style['name']} | q={new_q[:60]}"
                    )
                    continue

                # Accepted — reset failure counter
                style_failures = 0
                seen_questions.add(normalize_q(new_q))

                record = {
                    "generated_id"      : make_generated_id(seed, style["name"], global_idx),
                    "style"             : style["name"],
                    "source_id"         : seed.get("id", ""),
                    "crop"              : seed.get("crop", ""),
                    "state"             : seed.get("state", ""),
                    "district"          : seed.get("district", ""),
                    "season"            : seed.get("season", ""),
                    "domain"            : seed.get("domain", ""),
                    "generated_question": new_q,
                    "generated_answer"  : result["generated_answer"],
                    "seed_question"     : seed["question"],
                    "seed_answer"       : seed["answer"],
                    "style_call_number" : call_count,
                    "followup_focus"    : detected_focus if style["name"] == "paraphrase_followup" else "",
                    "answer_word_count" : len(result["generated_answer"].split()),
                    "timestamp"         : datetime.utcnow().isoformat(),
                }

                seed_results.append(record)
                all_results.append(record)
                style_done += 1

                print(f"✓ ({len(result['generated_answer'].split())}w) Q: {new_q[:65]}...")
                log_lines.append(
                    f"OK   | seed={seed_idx} | style={style['name']} | "
                    f"call={call_count} | focus={detected_focus or '-'} | q={new_q[:60]}"
                )

                time.sleep(0.5)

            print(f"  └ {style['name']}: {style_done}/{TARGET_PER_STYLE} done")

        print(f"\n   ✅ Seed {seed_idx+1} done — {len(seed_results)} pairs across {len(STYLES)} styles")

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

    # Word count stats
    if all_results:
        word_counts = [r["answer_word_count"] for r in all_results]
        avg_wc = sum(word_counts) / len(word_counts)
        min_wc = min(word_counts)
        max_wc = max(word_counts)
    else:
        avg_wc = min_wc = max_wc = 0

    print(f"\n{'='*60}")
    print(f"  DONE — {len(all_results)} unique pairs generated")
    print(f"  Duplicates rejected : {dupe_count}")
    print(f"  Skipped (all causes): {skip_count}")
    print(f"  Answer length (words): avg={avg_wc:.0f}, min={min_wc}, max={max_wc}")
    print(f"  Per-style breakdown:")
    for sname, cnt in sorted(style_counts.items()):
        print(f"    {sname:<40}: {cnt}")
    print(f"\n  Output JSON : {OUT_JSON}")
    print(f"  Output CSV  : {OUT_CSV}")
    print(f"  Log         : {OUT_LOG}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()