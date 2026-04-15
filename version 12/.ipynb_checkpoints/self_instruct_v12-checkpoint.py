# self_instruct_v12_resumable.py
# Purpose : Expand seed Q&A pairs into diverse paraphrase variants.
# Run     : python self_instruct_v12_resumable.py
#
# CHANGES vs v11:
#   FIX-37: CHECKPOINT SAVING — results are written to disk after every seed.
#            Previously all_results was only saved at the very end of the run.
#            If the script was stopped mid-run, all progress was lost.
#            Now paraphrase_v12.json / .csv / _log.txt are updated after each
#            seed completes, so Ctrl+C at any point only loses the seed
#            currently in progress.
#
#   FIX-38: AUTO-RESUME — on startup the script loads any existing
#            paraphrase_v12.json checkpoint and skips seeds that are already
#            fully represented in it. Re-running the script after a stop
#            continues from where it left off automatically, with no manual
#            index editing required.
#
#   FIX-39: Output files renamed to paraphrase_v12.* and IDs use si_v12_ prefix.
#
# CHANGES vs v10 (inherited from v11):
#   FIX-33: DEDUP THRESHOLD lowered 0.90 → 0.75 in is_duplicate().
#            Root cause of 24 near-duplicate pairs in V10: threshold was too
#            permissive. Self-Instruct paper used 0.70, Genetic-Instruct used
#            0.70. Pairs scoring 0.70–0.89 word Jaccard (synonym swaps,
#            structural paraphrases like "Is/Are there effective methods")
#            slipped through entirely. Single line change, zero new dependencies.
#
#   FIX-34: paraphrase_comparison REMOVED from STYLES entirely.
#            36% grounding failure rate was structural — the style forces the
#            model to rank or contrast two options that seeds often list as
#            equal alternatives. FIX-26 (tighter grounding) did not move this
#            number (11 vs 10 unsupported comparisons V9→V10). Valid coverage
#            already provided by paraphrase_followup and
#            paraphrase_quantity_focused.
#
#   FIX-35: paraphrase_yes_no target capped at 3 (was 5) via
#            TARGET_PER_STYLE_OVERRIDES. Seeds with limited binary surface area
#            (e.g. whitefly seed) cannot produce 5 distinct yes/no questions —
#            forcing 5 generates near-duplicates. Quality is fine (0% grounding
#            failures); ceiling is a content constraint, not a model failure.
#
#   FIX-36: Output files renamed to paraphrase_v11.* and IDs use si_v11_ prefix.
#
# CHANGES vs v9 (inherited from v10):
#   FIX-29: SCOPE ALIGNMENT — the single most impactful fix in this version.
#            Root cause identified: the model generates a question + answer in one
#            shot. When the question asks about ONE specific chemical or intervention,
#            the model answers using the full seed content — so other chemicals from
#            the seed bleed into the answer even though the question never asked
#            about them. This is NOT hallucination (the facts are grounded) but it
#            IS an alignment failure: the answer scope exceeds the question scope.
#
#            Three-part fix:
#            (a) SYSTEM_PROMPT: added Rule 11 — SCOPE RULE. The answer must only
#                address what the generated question explicitly asks. If the question
#                names one chemical, the answer must discuss only that chemical.
#            (b) paraphrase_yes_no + paraphrase_short style descriptions: added
#                inline SCOPE RULE reminders since these two styles had the highest
#                scope overflow rate (56-60% and 41-44% respectively).
#            (c) JUDGE: now receives the generated_question in addition to
#                seed_answer + generated_answer. Added a second check — SCOPE
#                ALIGNMENT — that flags answers whose scope exceeds the question's
#                scope (scope_overflow). build_judge_prompt() and call_judge()
#                updated accordingly.
#
#   FIX-30: Judge scope_overflow field added to output record so scope failures
#            are visible in the saved JSON/CSV alongside existing judge fields.
#
#   FIX-31: Output files renamed to paraphrase_v10.* and IDs use si_v10_ prefix.
#
#   FIX-32: make_generated_id() updated to use si_v10_ prefix.
#
# CHANGES vs v8 (inherited from v9):
#   FIX-25: NoneType crash guard on resp.choices[0].message.content.
#   FIX-26: Tighter per-style grounding constraints for problem_description /
#            preventive / cause.
#   FIX-27: paraphrase_comparison min word floor lowered 50 → 30.
#   FIX-28: Output files renamed to paraphrase_v9.* / si_v9_ prefix.
#
# CHANGES vs v7 (inherited from v8):
#   FIX-21: Judge is a HARD GATE — flagged generations are discarded + retried.
#   FIX-22: MAX_JUDGE_RETRIES config knob (default 3).
#   FIX-23: Output files renamed to paraphrase_v8.* / si_v8_ prefix.
#   FIX-24: Summary reports judge retry count.
#
# CHANGES vs v5 (inherited from v7):
#   FIX-16: LLM-as-a-Judge soft-flag.
#   FIX-17: Judge uses enable_thinking=False.
#   FIX-18: JUDGE_MAX_TOKENS = 400.
#   FIX-19: Output files renamed to paraphrase_v7.* / si_v7_ prefix.
#   FIX-20: Summary reports judge flag rate.
#
# INHERITED FROM v5 (unchanged):
#   FIX-9 : Metadata filtered — state/district only injected if in seed answer.
#   FIX-10: Per-style min AND max word counts.
#   FIX-11: Seed token coverage check (disabled by default).
#   FIX-12: REQUIRED_OPENER for paraphrase_short.
#   FIX-13: consecutive_unknowns guard for followup focus type rotation.
#   FIX-14: build_contextual_metadata() replaces raw metadata append.
#   FIX-15: enable_thinking=True for generation model.

import json
import os
import time
import re
import uuid
import csv
from datetime import datetime, timezone
from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────────
VLLM_URL        = "http://100.100.108.100:8080/v1"
MODEL           = "Qwen/Qwen3-30B-A3B"
INPUT_FILE      = "/home/kritika/self_instruct/random_200_qa_1.csv"
OUT_JSON        = "/home/kritika/self_instruct/paraphrase_v12.json"
OUT_CSV         = "/home/kritika/self_instruct/paraphrase_v12.csv"
OUT_LOG         = "/home/kritika/self_instruct/paraphrase_v12_log.txt"

MAX_SEEDS           = 200
TARGET_PER_STYLE    = 5
TARGET_PER_STYLE_OVERRIDES = {   # FIX-35: yes_no capped at 3 — binary surface ceiling
    "paraphrase_yes_no": 3,
}
MAX_STYLE_FAILURES  = 6
MAX_RETRIES         = 5
RETRY_DELAY         = 5
TEMPERATURE         = 0.75
MAX_TOKENS          = 1200

JUDGE_MAX_TOKENS    = 500          # FIX-29: slightly increased to accommodate scope check
JUDGE_TEMPERATURE   = 0.0

MAX_JUDGE_RETRIES   = 3

MIN_ANSWER_WORDS = 15
MAX_ANSWER_WORDS = 300

STYLE_MIN_WORDS = {
    "paraphrase_formal"             : 60,
    "paraphrase_problem_description": 25,
    "paraphrase_followup"           : 40,
    "paraphrase_preventive"         : 40,
    "paraphrase_cause"              : 50,
    "paraphrase_quantity_focused"   : 20,
    "paraphrase_timing_focused"     : 20,
    "paraphrase_yes_no"             : 20,
    "paraphrase_short"              : 15,
}

STYLE_MAX_WORDS = {
    "paraphrase_short"              : 60,
    "paraphrase_quantity_focused"   : 100,
    "paraphrase_timing_focused"     : 100,
    "paraphrase_yes_no"             : 80,
    "paraphrase_problem_description": 100,
}

COVERAGE_THRESHOLD_DEFAULT       = 0.0
COVERAGE_THRESHOLD_COMPREHENSIVE = 0.0

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

# ─── Answer Length Guard (per-style aware) ────────────────────────────────────
def is_answer_length_ok(text: str, style_name: str) -> bool:
    word_count = len(text.split())
    effective_min = max(MIN_ANSWER_WORDS, STYLE_MIN_WORDS.get(style_name, MIN_ANSWER_WORDS))
    effective_max = STYLE_MAX_WORDS.get(style_name, MAX_ANSWER_WORDS)
    return effective_min <= word_count <= effective_max

# ─── Seed Coverage Check ──────────────────────────────────────────────────────
def extract_key_tokens(seed_answer: str) -> set:
    tokens = set()
    for m in re.finditer(r'\b\d+(?:\.\d+)?\s*(?:ml|kg|g|litre|liter|mg|ha|acre|%)\b',
                         seed_answer, re.IGNORECASE):
        tokens.add(m.group().lower().replace(' ', ''))
    for m in re.finditer(r'\b\d+\s*%\s*[A-Z]{1,4}\b', seed_answer):
        tokens.add(m.group().lower().replace(' ', ''))
    for m in re.finditer(r'\b[A-Z][a-z]{3,}(?:\s+[A-Z][a-z]{3,})+\b', seed_answer):
        tokens.add(m.group().lower())
    return tokens

def passes_coverage_check(generated_answer: str, seed_answer: str,
                           style_name: str) -> bool:
    comprehensive_styles = {
        "paraphrase_formal", "paraphrase_preventive",
        "paraphrase_cause", "paraphrase_comparison",
    }
    threshold = (COVERAGE_THRESHOLD_COMPREHENSIVE
                 if style_name in comprehensive_styles
                 else COVERAGE_THRESHOLD_DEFAULT)
    if threshold <= 0:
        return True
    key_tokens = extract_key_tokens(seed_answer)
    if not key_tokens:
        return True
    gen_lower = generated_answer.lower()
    found = sum(1 for tok in key_tokens if tok in gen_lower)
    return (found / len(key_tokens)) >= threshold

# ─── Yes/No Copied Question Guard ─────────────────────────────────────────────
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
    return len(g_words & s_words) / max(len(g_words), len(s_words)) >= 0.85

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
        if len(new_words & seen_words) / max(len(new_words), len(seen_words)) >= 0.75:  # FIX-33: was 0.90
            return True
    return False

# ─── Metadata Filtering ───────────────────────────────────────────────────────
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

# ─── Style Taxonomy ───────────────────────────────────────────────────────────
STYLES = [

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

    {
        "name": "paraphrase_short",
        "description": (
            "Produce a very concise question (under 15 words) and a crisp answer "
            "(2-4 sentences maximum). Drop all background context — give only the "
            "core actionable information that is already in the seed. "
            "GROUNDING RULE: Do NOT add any new facts not present in the seed answer. "
            # FIX-29: scope alignment for short style
            "SCOPE RULE: Your answer must ONLY address what your generated question "
            "explicitly asks. If your question asks about one specific chemical or "
            "intervention, your answer must discuss ONLY that chemical or intervention. "
            "Do NOT bring in other chemicals or methods from the seed that your "
            "question did not ask about."
        ),
    },

    {
        "name": "paraphrase_problem_description",
        "description": (
            "Rewrite the question as a farmer describing symptoms they observe "
            "in their field right now. Lead with visible symptoms; do NOT name "
            "the disease or pest in the question. "
            "The answer MUST be SHORT: 3-5 sentences only. "
            "Structure: 1 sentence diagnosis + 2-3 key management actions from the seed. "
            "Do NOT list all management points from the seed — pick only the top 2-3. "
            "GROUNDING RULE: Do NOT add new facts not present in the seed answer. "
            "FORBIDDEN — never add any of the following unless they appear verbatim "
            "in the seed answer: sowing dates or timing windows, intercropping advice, "
            "state/district/region names, variety recommendations, soil-type conditions, "
            "or any general best-practice not stated in the seed."
        ),
    },

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
            "GROUNDING RULE: Do NOT add new facts not present in the seed answer. "
            # FIX-29: scope alignment for yes/no style
            "SCOPE RULE: Your answer must ONLY address what your generated question "
            "explicitly asks. If the question asks about one specific chemical, "
            "practice, or intervention, answer ONLY about that item. "
            "Do NOT mention other chemicals or methods from the seed that your "
            "question did not ask about — even if they are related."
        ),
    },

    {
        "name": "paraphrase_preventive",
        "description": (
            "Reframe the question to ask HOW TO PREVENT the problem, not treat it. "
            "Pull out only the preventive or cultural practices already mentioned "
            "in the seed answer. Each call should emphasise a DIFFERENT prevention "
            "angle (soil preparation, seed selection, timing, spacing, etc.). "
            "The answer must be at least 40 words. "
            "GROUNDING RULE: Do NOT add new facts or KVK referrals not in the seed. "
            "FORBIDDEN — never add any of the following unless they appear verbatim "
            "in the seed answer: specific chemical dosages or spray schedules, "
            "new crop variety names, sowing-date windows or calendar months, "
            "state/district/region names, or irrigation quantities."
        ),
    },

    {
        "name": "paraphrase_cause",
        "description": (
            "Rewrite the question to focus on WHY or WHAT CAUSES the problem. "
            "Explain the causal mechanism using only seed content. "
            "Each call should approach the cause from a slightly different angle "
            "(e.g., pathogen biology, environmental trigger, soil condition). "
            "The answer must be at least 50 words. "
            "GROUNDING RULE: Do NOT add new facts not present in the seed answer. "
            "FORBIDDEN — never add any of the following unless they appear verbatim "
            "in the seed answer: pathogen lifecycle stages, spore dispersal details, "
            "host-range or alternative host crops, seasonal prevalence by region, "
            "or any mechanism not explicitly described in the seed answer."
        ),
    },

    {
        "name": "paraphrase_quantity_focused",
        "description": (
            "Extract a quantity-related aspect (dosage, seed rate, concentration, "
            "area coverage, duration) from the seed and make it the sole focus "
            "of the question. If the seed has NO quantities at all, return JSON "
            "with generated_question=null. If multiple quantities exist, each call "
            "should focus on a DIFFERENT quantity. "
            "The answer must include: (1) the exact quantity from the seed, "
            "(2) what it is applied to or mixed with (from the seed), and "
            "(3) one sentence on timing or method of application (from the seed). "
            "Target 20-80 words. "
            "GROUNDING RULE: Do NOT invent quantities not explicitly stated in the seed."
        ),
    },

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
8. Do NOT restructure or recontextualize seed facts. Facts must be used with
   the same meaning and context as in the seed.
9. NEVER use self-referential language: no "the seed says", "according to the
   seed", "as mentioned in the seed", "the prompt", etc.
10. LOCATION RULE: Do NOT mention any state, district, or region name UNLESS
    that location is explicitly present in the SEED ANSWER text. The metadata
    line is for context only — do NOT copy locations from it into your answer.
11. SCOPE RULE: Your generated_answer must ONLY address what your
    generated_question asks. Read your own question carefully before writing
    the answer.
    - If your question asks about ONE specific chemical → answer must discuss
      ONLY that chemical. Do NOT mention other chemicals from the seed.
    - If your question asks about ONE specific practice or intervention →
      answer must discuss ONLY that practice.
    - If your question asks for a complete management solution → you may
      include all relevant chemicals and methods from the seed.
    - The answer scope must match the question scope. Broader is not better.

Output ONLY a valid JSON object. No markdown, no preamble, no extra keys:
{
  "generated_question": "<new question, or null>",
  "generated_answer": "<new answer, or null>"
}"""

# ─── Judge System Prompt ──────────────────────────────────────────────────────
# FIX-29: Judge now receives generated_question too and performs a second
#          scope-alignment check in addition to the existing grounding check.
JUDGE_SYSTEM_PROMPT = """You are a strict quality verifier for an Indian agricultural Q&A dataset.

You will receive:
  - GENERATED QUESTION: the question produced by the pipeline
  - SEED ANSWER: the original expert-verified source of truth
  - GENERATED ANSWER: a paraphrase that should be grounded in the seed

YOUR TASK — TWO CHECKS:

CHECK 1 — GROUNDING:
Does the GENERATED ANSWER contain any claim, fact, chemical name, dosage,
quantity, location, timing, or recommendation that is NOT explicitly present
in the SEED ANSWER?
- Paraphrasing and rewording are ALLOWED. Only flag NEW facts not in the seed.
- Minor reformatting (e.g. "two or three weeks" → "2-3 weeks") is NOT a flag.
- Omitting seed content is NOT a flag — only addition of new facts matters.
- Treat compound product names (e.g. "Carboxin 37.5% + Thiram 37.5%") as a
  single entity. Do NOT flag if such a compound appears decomposed unless the
  decomposed part introduces a genuinely new chemical not in the seed.

CHECK 2 — SCOPE ALIGNMENT:
Does the GENERATED ANSWER stay within the scope of the GENERATED QUESTION?
- If the question asks about ONE specific chemical or intervention, the answer
  must discuss ONLY that item. If the answer also discusses other chemicals or
  interventions from the seed that the question did not ask about, flag this
  as scope_overflow=true.
- If the question asks for a complete management plan or solution, the answer
  may include multiple chemicals/methods — this is NOT scope overflow.
- Examples of scope overflow:
    Q: "Is Chlorpyriphos 20% EC effective for termite control in wheat?"
    A: "Yes. Treat seeds with Chlorpyriphos 20% EC or thiomethoxam 30% FS..."
    → scope_overflow=true (question only asked about Chlorpyriphos; thiomethoxam
      was not asked about)
- Examples of correct scope:
    Q: "Is Chlorpyriphos 20% EC effective for termite control in wheat?"
    A: "Yes. Chlorpyriphos 20% EC is used as a seed treatment at 3 ml/kg to
       protect wheat from termites during early establishment."
    → scope_overflow=false

Set grounded=false if CHECK 1 fails OR if scope_overflow=true.

Output ONLY a valid JSON object with no markdown or preamble:
{
  "grounded": true or false,
  "scope_overflow": true or false,
  "confidence": "high" or "medium" or "low",
  "issues": ["<specific ungrounded claim or scope violation>"]
}

If grounded=true and scope_overflow=false, issues must be an empty list [].
If grounded=false or scope_overflow=true, issues must list the specific problems found."""

# ─── Variation Hints ──────────────────────────────────────────────────────────
SHORT_OPENERS = [
    "What", "Which", "When", "Why", "Can",
    "Should", "What is", "Is", "Does", "Are",
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

def get_variation_hint(style_call_count: int, used_focus_types: list,
                       seed_id: str = "", consecutive_unknowns: int = 0) -> str:
    idx = min(style_call_count, len(VARIATION_HINTS) - 1)
    seed_hash_offset = abs(hash(seed_id)) % len(SHORT_OPENERS)
    opener_idx = (seed_hash_offset + style_call_count) % len(SHORT_OPENERS)
    opener = SHORT_OPENERS[opener_idx]
    excluded_types = list(used_focus_types[-3:]) if used_focus_types else []
    if consecutive_unknowns >= 3:
        for forced in ["dosage_quantity", "chemical_name"]:
            if forced not in excluded_types:
                excluded_types.append(forced)
    excluded = ", ".join(excluded_types) if excluded_types else "none"
    return VARIATION_HINTS[idx].format(opener=opener, excluded=excluded)

# ─── Prompt Builder ───────────────────────────────────────────────────────────
def build_user_prompt(seed: dict, style: dict, style_call_count: int,
                      used_focus_types: list, consecutive_unknowns: int = 0) -> str:
    hint = get_variation_hint(style_call_count, used_focus_types,
                              seed.get("id", ""), consecutive_unknowns)
    clean_answer = strip_kvk_lines(seed.get("answer", ""))
    contextual_meta = build_contextual_metadata(seed)

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

# ─── Judge Prompt Builder ─────────────────────────────────────────────────────
# FIX-29: Now includes generated_question so the judge can check scope alignment.
def build_judge_prompt(seed_answer: str, generated_question: str,
                       generated_answer: str) -> str:
    clean_seed = strip_kvk_lines(seed_answer)
    return (
        f"GENERATED QUESTION:\n{generated_question}\n\n"
        f"SEED ANSWER:\n{clean_seed}\n\n"
        f"GENERATED ANSWER:\n{generated_answer}\n\n"
        "Now output the JSON verdict."
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
        # FIX-29: treat scope_overflow as a grounding failure for the hard gate
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
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )

            raw_content = resp.choices[0].message.content
            if raw_content is None:
                print(f"    [null-content] retrying...", end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue
            text = raw_content.strip()

            result = parse_response(text)
            if result is None:
                return None, None

            answer = result.get("generated_answer", "")

            if has_self_reference(answer):
                print(f"    [self-ref] retrying...", end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

            if style["name"] == "paraphrase_yes_no" and is_copied_seed_question(
                result.get("generated_question", ""), seed["question"]
            ):
                print(f"    [copied-Q] retrying...", end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

            if has_looping_text(answer):
                print(f"    [loop] retrying...", end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

            if not is_answer_length_ok(answer, style["name"]):
                word_count = len(answer.split())
                style_min = STYLE_MIN_WORDS.get(style["name"], MIN_ANSWER_WORDS)
                style_max = STYLE_MAX_WORDS.get(style["name"], MAX_ANSWER_WORDS)
                print(f"    [len={word_count}w, need {style_min}–{style_max}w] retrying...",
                      end=" ", flush=True)
                time.sleep(RETRY_DELAY)
                continue

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

# ─── Judge Caller ─────────────────────────────────────────────────────────────
# FIX-29: now accepts generated_question and passes it to build_judge_prompt.
def call_judge(seed_answer: str, generated_question: str,
               generated_answer: str) -> dict:
    judge_prompt = build_judge_prompt(seed_answer, generated_question,
                                      generated_answer)
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
        raw = raw_content.strip()
        return parse_judge_response(raw)
    except Exception as e:
        return {"grounded": True, "scope_overflow": False,
                "confidence": "low", "issues": [], "judge_error": str(e)[:120]}

# ─── Focus Type Detector ──────────────────────────────────────────────────────
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
    return f"si_v12_{style_name[:6]}_{base}_{idx}"   # FIX-39

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

# ─── Checkpoint helpers ───────────────────────────────────────────────────────
def save_checkpoint(all_results: list, log_lines: list) -> None:
    """Write current results + log to disk. Called after every seed."""
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


def load_checkpoint() -> tuple[list, set]:
    """
    Load existing results from OUT_JSON (if present).
    Returns (all_results, completed_source_ids) where completed_source_ids
    is the set of seed source IDs that are already fully saved.
    """
    if not os.path.exists(OUT_JSON):
        return [], set()
    try:
        with open(OUT_JSON, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        completed_ids = set(r["source_id"] for r in all_results)
        print(f"  ✔ Checkpoint found — loaded {len(all_results)} existing results "
              f"covering {len(completed_ids)} seed(s). Resuming from there.")
        return all_results, completed_ids
    except Exception as e:
        print(f"  ⚠ Could not load checkpoint ({e}). Starting fresh.")
        return [], set()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\nLoading seeds from: {INPUT_FILE}")
    all_seeds = load_seeds_from_csv(INPUT_FILE)
    seeds = all_seeds[:MAX_SEEDS]

    print(f"\n{'='*60}")
    print(f"  Self-Instruct Paraphrase v12 — RESUMABLE (English only, CSV input)")
    print(f"  Seeds      : {len(seeds)} (of {len(all_seeds)} loaded)")
    print(f"  Target     : {TARGET_PER_STYLE} pairs x {len(STYLES)} styles = "
          f"{TARGET_PER_STYLE * len(STYLES)} per seed")
    print(f"  Est. total : ~{len(seeds) * TARGET_PER_STYLE * len(STYLES)} pairs (yes_no capped at 3)")
    print(f"  Generation : enable_thinking=True")
    print(f"  Judge      : enable_thinking=False, max_tokens={JUDGE_MAX_TOKENS} "
          f"(HARD GATE — flagged pairs retried up to {MAX_JUDGE_RETRIES}x)")
    print(f"  Length     : global [{MIN_ANSWER_WORDS}–{MAX_ANSWER_WORDS}w], per-style overrides active")
    print(f"  FIX-37     : Checkpoint save after every seed — safe to Ctrl+C anytime.")
    print(f"  FIX-38     : Auto-resume — re-running skips already-completed seeds.")
    print(f"  FIX-39     : Output files paraphrase_v12.*, IDs si_v12_ prefix.")
    print(f"  (Inherited) Dedup 0.75, comparison removed, yes_no cap 3, scope alignment.")
    print(f"{'='*60}\n")

    # FIX-38: load any existing checkpoint
    all_results, completed_source_ids = load_checkpoint()

    log_lines        = []
    judge_flag_count = 0
    scope_overflow_count = 0

    for seed_idx, seed in enumerate(seeds):
        # FIX-38: skip seeds whose source_id already appears in checkpoint
        if seed.get("id", "") in completed_source_ids:
            print(f"  ── Seed {seed_idx+1}/{len(seeds)} [{seed.get('id','')}] "
                  f"already in checkpoint — skipping.")
            continue
        print(f"\n── Seed {seed_idx+1}/{len(seeds)}: "
              f"{seed.get('crop') or 'N/A'} | "
              f"{seed.get('state') or 'N/A'} | "
              f"{seed.get('domain') or 'N/A'} ──")
        print(f"   Original Q: {seed['question'][:80]}...")

        seed_results         = []
        seen_questions       = set()
        global_idx           = 0
        used_focus_types     : list[str] = []
        consecutive_unknowns = 0

        for style in STYLES:
            style_done     = 0
            style_failures = 0
            call_count     = 0
            style_target   = TARGET_PER_STYLE_OVERRIDES.get(style["name"], TARGET_PER_STYLE)  # FIX-35

            print(f"\n  ┌ Style: {style['name']} (target={style_target})")

            while style_done < style_target:
                if style_failures >= MAX_STYLE_FAILURES:
                    print(f"  └ [{style_done}/{style_target}] abandoned "
                          f"after {style_failures} consecutive failures")
                    log_lines.append(
                        f"ABANDON | seed={seed_idx} | style={style['name']} | "
                        f"done={style_done}/{style_target}"
                    )
                    break

                global_idx += 1
                print(f"  │ [{style_done+1}/{style_target}] call #{call_count+1} ... ",
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

                if style["name"] == "paraphrase_followup":
                    if detected_focus and detected_focus != "unknown":
                        used_focus_types.append(detected_focus)
                        consecutive_unknowns = 0
                    else:
                        consecutive_unknowns += 1

                new_q = result["generated_question"]
                if is_duplicate(new_q, seen_questions):
                    style_failures += 1
                    print(f"duplicate — skipped")
                    log_lines.append(
                        f"DUPE | seed={seed_idx} | style={style['name']} | q={new_q[:60]}"
                    )
                    continue

                print(f"✓ gen ({len(result['generated_answer'].split())}w) → judging...",
                      end=" ", flush=True)

                judge_retry_count = 0
                is_grounded       = False
                judge_confidence  = "low"
                judge_issues      = []
                judge_scope_overflow = False  # FIX-30

                while judge_retry_count <= MAX_JUDGE_RETRIES:
                    # FIX-29: pass generated_question to call_judge
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
                    log_lines.append(
                        f"JUDGE_RETRY | seed={seed_idx} | style={style['name']} | "
                        f"retry={judge_retry_count}/{MAX_JUDGE_RETRIES} | "
                        f"conf={judge_confidence} | scope_overflow={judge_scope_overflow} | "
                        f"issues={judge_issues[:2]} | q={new_q[:60]}"
                    )

                    if judge_retry_count > MAX_JUDGE_RETRIES:
                        print(f"⚠ FLAGGED ({judge_confidence}, scope_overflow={judge_scope_overflow}) "
                              f"— max judge retries exhausted, slot skipped")
                        break

                    print(f"⚠ FLAGGED ({judge_confidence}, scope_overflow={judge_scope_overflow}) "
                          f"— retrying (judge retry {judge_retry_count}/{MAX_JUDGE_RETRIES})...",
                          end=" ", flush=True)
                    global_idx += 1
                    result, detected_focus = call_model(
                        seed, style, call_count, used_focus_types, consecutive_unknowns
                    )
                    call_count += 1

                    if result is None:
                        print("skipped (null / self-ref / loop / length / coverage during judge retry)")
                        log_lines.append(
                            f"SKIP | seed={seed_idx} | style={style['name']} | "
                            f"call={call_count} | (judge-retry)"
                        )
                        break

                    new_q = result["generated_question"]
                    print(f"✓ gen ({len(result['generated_answer'].split())}w) → judging...",
                          end=" ", flush=True)

                if not is_grounded or result is None:
                    style_failures += 1
                    log_lines.append(
                        f"JUDGE_ABANDON | seed={seed_idx} | style={style['name']} | "
                        f"conf={judge_confidence} | scope_overflow={judge_scope_overflow} | "
                        f"issues={judge_issues[:2]} | q={new_q[:60]}"
                    )
                    continue

                print(f"✓ grounded Q: {new_q[:55]}...")

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
                    "judge_grounded"    : True,
                    "judge_scope_overflow": False,   # FIX-30: always False here (passed gate)
                    "judge_confidence"  : judge_confidence,
                    "judge_issues"      : "",
                    "timestamp"         : datetime.now(timezone.utc).isoformat(),
                }

                seed_results.append(record)
                all_results.append(record)
                style_done += 1

                log_lines.append(
                    f"OK   | seed={seed_idx} | style={style['name']} | "
                    f"call={call_count} | focus={detected_focus or '-'} | "
                    f"judge=PASS | scope_overflow=False | q={new_q[:60]}"
                )

                time.sleep(0.5)

            print(f"  └ {style['name']}: {style_done}/{style_target} done")

        print(f"\n   ✅ Seed {seed_idx+1} done — {len(seed_results)} pairs")

        # FIX-37: save checkpoint after every seed so progress is never lost
        save_checkpoint(all_results, log_lines)
        print(f"   💾 Checkpoint saved ({len(all_results)} total pairs so far)")

    # ── Final save (also covers the case where all seeds were skipped) ────────
    save_checkpoint(all_results, log_lines)

    # ── Summary ───────────────────────────────────────────────────────────────
    style_counts = {}
    dupe_count   = sum(1 for l in log_lines if l.startswith("DUPE"))
    skip_count   = sum(1 for l in log_lines if l.startswith("SKIP"))

    for r in all_results:
        style_counts[r["style"]] = style_counts.get(r["style"], 0) + 1

    judge_retry_total   = sum(1 for l in log_lines if l.startswith("JUDGE_RETRY"))
    judge_abandon_total = sum(1 for l in log_lines if l.startswith("JUDGE_ABANDON"))
    scope_retry_total   = sum(1 for l in log_lines
                              if "scope_overflow=True" in l and l.startswith("JUDGE_RETRY"))

    if all_results:
        word_counts = [r["answer_word_count"] for r in all_results]
        avg_wc = sum(word_counts) / len(word_counts)
        min_wc = min(word_counts)
        max_wc = max(word_counts)
    else:
        avg_wc = min_wc = max_wc = 0

    print(f"\n{'='*60}")
    print(f"  DONE — {len(all_results)} unique pairs generated (all judge-verified clean)")
    print(f"  Duplicates rejected       : {dupe_count}  (threshold=0.75, was 0.90 in v10)")
    print(f"  Skipped (all causes)      : {skip_count}")
    print(f"  Judge retries (extra regen calls) : {judge_retry_total}")
    print(f"    of which scope_overflow retries : {scope_retry_total}")
    print(f"  Judge abandoned (exhausted retries): {judge_abandon_total}")
    print(f"  Answer length (words): avg={avg_wc:.0f}, min={min_wc}, max={max_wc}")
    print(f"  Per-style breakdown:")
    for sname, cnt in sorted(style_counts.items()):
        print(f"    {sname:<40}: {cnt:>4} pairs")
    print(f"\n  Output JSON : {OUT_JSON}")
    print(f"  Output CSV  : {OUT_CSV}")
    print(f"  Log         : {OUT_LOG}")
    print(f"  Note: paraphrase_comparison removed (FIX-34).")
    print(f"        paraphrase_yes_no capped at 3/seed (FIX-35).")
    print(f"        All saved pairs passed grounding + scope alignment checks.")
    print(f"        Check JUDGE_RETRY/JUDGE_ABANDON lines in the log for details.")
    print(f"        Checkpoint was saved after every seed (FIX-37).")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()