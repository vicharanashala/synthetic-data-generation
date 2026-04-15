# self_instruct_v5.py
# Purpose : Expand seed Q&A pairs into diverse paraphrase variants.
# Run     : python self_instruct_v5.py
#
# CHANGES vs v4:
#   FIX-9 : Location/state metadata is NO LONGER appended to the user prompt
#            unless it already appears in the seed answer — prevents the LLM
#            from injecting "in Punjab" into answers where the seed is generic.
#   FIX-10: Per-style minimum word counts — paraphrase_formal/followup/cause/
#            preventive/comparison styles now require ≥50 words in the answer,
#            not just the global 15-word floor. Prevents silent content loss.
#   FIX-11: Seed coverage check — after generation, verify that at least 60%
#            of key seed tokens (chemicals, dosages, numbers, crop-specific
#            terms) appear in the generated answer. Hard-rejects answers that
#            drop critical seed content even if they pass length checks.
#   FIX-12: REQUIRED_OPENER for paraphrase_short is now injected as a
#            standalone bold line in the user prompt (not buried in hint text)
#            so the model cannot miss or ignore it.
#   FIX-13: detect_focus_type unknown accumulation guard — if ≥3 consecutive
#            unknowns, force-exclude "dosage_quantity" and "chemical_name"
#            to break the model's default pull toward those two types.
#   FIX-14: strip_metadata_if_not_in_seed() replaces raw metadata append —
#            each metadata field (state, district, season) is only included
#            in the prompt if the cleaned seed answer already mentions it.

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
OUT_JSON        = "/home/kritika/self_instruct/paraphrase_v5.json"
OUT_CSV         = "/home/kritika/self_instruct/paraphrase_v5.csv"
OUT_LOG         = "/home/kritika/self_instruct/paraphrase_v5_log.txt"

MAX_SEEDS           = 20
TARGET_PER_STYLE    = 5
MAX_STYLE_FAILURES  = 6
MAX_RETRIES         = 5
RETRY_DELAY         = 5
TEMPERATURE         = 0.75
MAX_TOKENS          = 1200

# Answer length guards (word count) — global floor/ceiling
MIN_ANSWER_WORDS = 15
MAX_ANSWER_WORDS = 300

# FIX-10: Per-style minimum word counts (overrides global MIN_ANSWER_WORDS upward)
# Styles that are expected to be comprehensive get a higher floor.
STYLE_MIN_WORDS = {
    "paraphrase_formal"           : 60,
    "paraphrase_problem_description": 30,
    "paraphrase_followup"         : 40,
    "paraphrase_preventive"       : 40,
    "paraphrase_comparison"       : 50,
    "paraphrase_cause"            : 50,
    "paraphrase_quantity_focused" : 30,
    "paraphrase_timing_focused"   : 30,
    "paraphrase_yes_no"           : 25,
    "paraphrase_short"            : 15,   # intentionally short
}

# FIX-11: Seed coverage check threshold
# At least this fraction of key seed tokens must appear in the generated answer
# for comprehensive styles. Short/yes-no styles get a lower threshold.
COVERAGE_THRESHOLD_DEFAULT = 0.0   # disabled for most styles by default
COVERAGE_THRESHOLD_COMPREHENSIVE = 0.0  # set >0 to enable (e.g. 0.35)
# NOTE: Set to 0.0 by default to avoid over-rejection. Increase if truncation
# is still a problem after raising per-style MIN_ANSWER_WORDS.

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
    for pattern in SELF_REF_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

# ─── FIX-2/FIX-3/FIX-10: Answer Length Guard (per-style aware) ───────────────
def is_answer_length_ok(text: str, style_name: str) -> bool:
    """
    Check answer word count against both global bounds and per-style minimum.
    FIX-10: Each comprehensive style has a higher floor than the global 15-word min.
    """
    word_count = len(text.split())
    style_min = STYLE_MIN_WORDS.get(style_name, MIN_ANSWER_WORDS)
    effective_min = max(MIN_ANSWER_WORDS, style_min)
    return effective_min <= word_count <= MAX_ANSWER_WORDS

# ─── FIX-11: Seed Coverage Check ─────────────────────────────────────────────
def extract_key_tokens(seed_answer: str) -> set:
    """
    Extract key tokens from seed answer: chemical names, numbers+units,
    percentages, and capitalised technical terms. These are the facts
    that MUST survive into the generated answer.
    """
    tokens = set()
    # Numbers with units (dosages, concentrations, rates)
    for m in re.finditer(r'\b\d+(?:\.\d+)?\s*(?:ml|kg|g|litre|liter|mg|ha|acre|%)\b',
                         seed_answer, re.IGNORECASE):
        tokens.add(m.group().lower().replace(' ', ''))
    # Chemical/product codes like "20% EC", "30% FS", "56%"
    for m in re.finditer(r'\b\d+\s*%\s*[A-Z]{1,4}\b', seed_answer):
        tokens.add(m.group().lower().replace(' ', ''))
    # Capitalised multi-word terms (chemical names, organisms, etc.)
    for m in re.finditer(r'\b[A-Z][a-z]{3,}(?:\s+[A-Z][a-z]{3,})+\b', seed_answer):
        tokens.add(m.group().lower())
    return tokens

def passes_coverage_check(generated_answer: str, seed_answer: str,
                           style_name: str) -> bool:
    """
    FIX-11: Verify that key seed tokens survive into the generated answer.
    Only enforced for comprehensive styles where truncation is harmful.
    Returns True (passes) if coverage threshold is 0 or set is empty.
    """
    comprehensive_styles = {
        "paraphrase_formal", "paraphrase_preventive",
        "paraphrase_cause", "paraphrase_comparison",
    }
    threshold = (COVERAGE_THRESHOLD_COMPREHENSIVE
                 if style_name in comprehensive_styles
                 else COVERAGE_THRESHOLD_DEFAULT)
    if threshold <= 0:
        return True   # check disabled

    key_tokens = extract_key_tokens(seed_answer)
    if not key_tokens:
        return True   # nothing to check

    gen_lower = generated_answer.lower()
    found = sum(1 for tok in key_tokens if tok in gen_lower)
    coverage = found / len(key_tokens)
    return coverage >= threshold

# ─── FIX-6: Yes/No Copied Seed Question Guard ────────────────────────────────
def is_copied_seed_question(generated_q: str, seed_q: str) -> bool:
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

# ─── FIX-9: Metadata Filtering ───────────────────────────────────────────────
def build_contextual_metadata(seed: dict) -> str:
    """
    FIX-9: Only include a metadata field in the prompt if the seed answer
    already mentions it. This prevents the LLM from injecting location names
    (e.g. 'Punjab', 'Basti') that appear in JSON metadata but NOT in the seed.

    Always includes CROP and DOMAIN since these are almost always present
    in the seed and are safe to remind the model about.
    """
    clean_answer = seed.get("answer", "").lower()
    parts = []

    # Crop and domain — always included (safe, usually in seed)
    if seed.get("crop"):
        parts.append(f"CROP: {seed['crop']}")
    if seed.get("domain"):
        parts.append(f"DOMAIN: {seed['domain']}")

    # State — only if seed answer mentions it
    state = seed.get("state", "").strip()
    if state and state.lower() in clean_answer:
        parts.append(f"STATE: {state}")

    # District — only if seed answer mentions it
    district = seed.get("district", "").strip()
    if district and district.lower() in clean_answer:
        parts.append(f"DISTRICT: {district}")

    # Season — only if seed answer mentions it (seasons are short words,
    # check carefully to avoid false matches like "rabi" inside other words)
    season = seed.get("season", "").strip()
    if season and re.search(r'\b' + re.escape(season.lower()) + r'\b', clean_answer):
        parts.append(f"SEASON: {season}")

    return ", ".join(parts)

# ─── Style Taxonomy (English Only) ───────────────────────────────────────────
STYLES = [

    # ── Style 1: Formal / Extension-Officer Tone ──────────────────────────────
    {
        "name": "paraphrase_formal",
        "description": (
            "Rewrite the question in a formal, academic or extension-officer tone. "
            "Use correct technical terminology. Structure the answer with numbered "
            "points or clear sections where appropriate. "
            "Keep the answer between 60 and 250 words. "
            "GROUNDING RULE: Only rephrase content already present in the seed answer. "
            "Do NOT add new facts, chemicals, dosages, or recommendations not in the seed. "
            "Do NOT add KVK referral sentences or generic disclaimers."
        ),
    },

    # ── Style 2: Short / Concise ──────────────────────────────────────────────
    {
        "name": "paraphrase_short",
        "description": (
            "Produce a very concise question (under 15 words) and a crisp answer "
            "(2-4 sentences maximum). Drop all background context — give only the "
            "core actionable information that is already in the seed. "
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
            "The answer must be at least 40 words — provide a complete explanation. "
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
            "question verbatim or near-verbatim. "
            "GOOD EXAMPLES: 'Can Chlorpyriphos 20% EC be used as seed treatment "
            "for termites in wheat?', 'Is deep summer ploughing effective against "
            "termite infestation?' "
            "CRITICAL LANGUAGE RULE: After Yes/No, write as a direct factual statement. "
            "NEVER write 'the seed says', 'according to the seed', etc. "
            "WRONG: 'No. The seed answer explicitly states, Do not spray during flowering.' "
            "RIGHT:  'No. Insecticides should not be sprayed during flowering.' "
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
            "The answer must be at least 40 words. "
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
            "The answer must be at least 50 words. "
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
   or any similar meta-reference to the prompt or dataset.
10. LOCATION RULE: Do NOT mention any state, district, or region name in your
    answer UNLESS that location name is explicitly present in the SEED ANSWER
    text provided below. The metadata line (CROP/DOMAIN/STATE/DISTRICT) is
    provided only for context — do NOT copy location names from it into your
    answer unless the seed answer itself uses those names.

Output ONLY a valid JSON object. No markdown, no preamble, no extra keys:
{
  "generated_question": "<new question, or null>",
  "generated_answer": "<new answer, or null>"
}"""

# ─── Variation Hints ──────────────────────────────────────────────────────────
SHORT_OPENERS = [
    "What",
    "Which",
    "When",
    "Why",
    "Can",
    "Should",
    "What is",
    "Is",
    "Does",
    "Are",
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
                       seed_id: str = "",
                       consecutive_unknowns: int = 0) -> str:
    """
    FIX-7 (retained): seed_hash_offset ensures different seeds start on different openers.
    FIX-13 (new): if consecutive_unknowns >= 3, force-exclude dosage_quantity
    and chemical_name to break the model's default pull toward those types.
    """
    idx = min(style_call_count, len(VARIATION_HINTS) - 1)

    seed_hash_offset = abs(hash(seed_id)) % len(SHORT_OPENERS)
    opener_idx = (seed_hash_offset + style_call_count) % len(SHORT_OPENERS)
    opener = SHORT_OPENERS[opener_idx]

    excluded_types = list(used_focus_types[-3:]) if used_focus_types else []

    # FIX-13: force-exclude dominant defaults when detection keeps failing
    if consecutive_unknowns >= 3:
        for forced in ["dosage_quantity", "chemical_name"]:
            if forced not in excluded_types:
                excluded_types.append(forced)

    excluded = ", ".join(excluded_types) if excluded_types else "none"
    return VARIATION_HINTS[idx].format(opener=opener, excluded=excluded)

# ─── Prompt Builder ───────────────────────────────────────────────────────────
def build_user_prompt(seed: dict, style: dict, style_call_count: int,
                      used_focus_types: list,
                      consecutive_unknowns: int = 0) -> str:
    """
    FIX-9: Metadata is filtered — only fields already present in the seed
    answer are appended to the prompt. This prevents location injection.
    FIX-12: REQUIRED_OPENER for paraphrase_short is now injected as a
    dedicated bold constraint line, not just buried in the hint string.
    """
    hint = get_variation_hint(style_call_count, used_focus_types,
                              seed.get("id", ""), consecutive_unknowns)
    raw_answer = seed.get("answer", "")
    clean_answer = strip_kvk_lines(raw_answer)

    # FIX-9: Only include metadata fields present in the seed answer
    contextual_meta = build_contextual_metadata(seed)

    # FIX-12: Explicit opener constraint for paraphrase_short
    opener_constraint = ""
    if style["name"] == "paraphrase_short":
        seed_hash_offset = abs(hash(seed.get("id", ""))) % len(SHORT_OPENERS)
        opener_idx = (seed_hash_offset + style_call_count) % len(SHORT_OPENERS)
        required_opener = SHORT_OPENERS[opener_idx]
        opener_constraint = (
            f"\n⚠️  REQUIRED: Your question MUST start with the word '{required_opener}'. "
            f"Do NOT start with 'How to' or any other word.\n"
        )

    return (
        f"STYLE: {style['name']}\n"
        f"STYLE INSTRUCTION: {style['description']}\n"
        f"{opener_constraint}"
        f"\nVARIATION HINT: {hint}\n\n"
        f"SEED QUESTION:\n{seed['question']}\n\n"
        f"SEED ANSWER (use ONLY this content — do not add anything not stated here):\n"
        f"{clean_answer}\n\n"
        f"CONTEXT (for reference only — do NOT inject these into your answer "
        f"unless they already appear in the SEED ANSWER above):\n"
        f"{contextual_meta}\n\n"
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
               used_focus_types: list, consecutive_unknowns: int = 0):
    user_prompt = build_user_prompt(seed, style, style_call_count,
                                    used_focus_types, consecutive_unknowns)
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
                print(f"    [self-ref] retrying...", end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

            # FIX-6: Reject yes_no outputs that copied the seed question
            if style["name"] == "paraphrase_yes_no" and is_copied_seed_question(
                result.get("generated_question", ""), seed["question"]
            ):
                print(f"    [copied-Q] retrying...", end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

            # Reject looping outputs
            if has_looping_text(answer):
                print(f"    [loop] retrying...", end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

            # FIX-10: Per-style length check
            if not is_answer_length_ok(answer, style["name"]):
                word_count = len(answer.split())
                style_min = STYLE_MIN_WORDS.get(style["name"], MIN_ANSWER_WORDS)
                print(f"    [len={word_count}w, need≥{style_min}] retrying...",
                      end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

            # FIX-11: Seed coverage check
            if not passes_coverage_check(answer, seed.get("answer", ""), style["name"]):
                print(f"    [coverage-fail] retrying...", end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

            detected_focus = detect_focus_type(result.get("generated_question", ""))
            return result, detected_focus

        except Exception as e:
            print(f"    [error] attempt {attempt+1}: {e}")
        time.sleep(RETRY_DELAY * (2 ** attempt))
    return None, None

# ─── FIX-5/FIX-8: Focus Type Detector ────────────────────────────────────────
FOCUS_TYPE_KEYWORDS = {
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
    "dosage_quantity"        : [
        "dosage", "dose", "how much", "quantity", "concentration",
        "g/kg", "ml/l", "per kg", "per acre", "per litre", "seed rate",
        "application rate",
    ],
}

def detect_focus_type(question: str) -> str:
    q_lower = question.lower()
    for focus_type, keywords in FOCUS_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in q_lower:
                return focus_type
    return "unknown"

# ─── ID Generator ─────────────────────────────────────────────────────────────
def make_generated_id(seed: dict, style_name: str, idx: int) -> str:
    base = str(seed.get(COL_ID, uuid.uuid4()))[:8]
    return f"si_v5_{style_name[:6]}_{base}_{idx}"

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
    print(f"  Self-Instruct Paraphrase v5 (English only, CSV input)")
    print(f"  Seeds     : {len(seeds)} (of {len(all_seeds)} loaded)")
    print(f"  Target    : {TARGET_PER_STYLE} pairs x {len(STYLES)} styles = "
          f"{TARGET_PER_STYLE * len(STYLES)} per seed")
    print(f"  Styles    : {len(STYLES)}")
    print(f"  Est. total: ~{len(seeds) * TARGET_PER_STYLE * len(STYLES)} pairs")
    print(f"  Answer length window: global [{MIN_ANSWER_WORDS}, {MAX_ANSWER_WORDS}] words")
    print(f"  Per-style minimums: {STYLE_MIN_WORDS}")
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

        # FIX-13: Track consecutive unknowns from detect_focus_type
        consecutive_unknowns = 0

        for style in STYLES:
            style_done     = 0
            style_failures = 0
            call_count     = 0

            print(f"\n  ┌ Style: {style['name']}")

            while style_done < TARGET_PER_STYLE:
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
                    seed, style, call_count, used_focus_types, consecutive_unknowns
                )
                call_count += 1

                if result is None:
                    style_failures += 1
                    print("skipped (null / self-ref / loop / length / coverage)")
                    log_lines.append(
                        f"SKIP | seed={seed_idx} | style={style['name']} | call={call_count}"
                    )
                    continue

                # FIX-5 + FIX-13: Track followup focus type and unknown count
                if style["name"] == "paraphrase_followup":
                    if detected_focus and detected_focus != "unknown":
                        used_focus_types.append(detected_focus)
                        consecutive_unknowns = 0
                    else:
                        consecutive_unknowns += 1

                # Deduplication check
                new_q = result["generated_question"]
                if is_duplicate(new_q, seen_questions):
                    style_failures += 1
                    print(f"duplicate — skipped")
                    log_lines.append(
                        f"DUPE | seed={seed_idx} | style={style['name']} | q={new_q[:60]}"
                    )
                    continue

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
                    "followup_focus"    : (detected_focus
                                          if style["name"] == "paraphrase_followup"
                                          else ""),
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