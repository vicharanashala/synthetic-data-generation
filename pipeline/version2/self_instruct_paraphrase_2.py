# self_instruct_paraphrase_v2.py
# Purpose : Expand a small number of seed Q&A pairs into 50-100 diverse
#           paraphrase variants using a rich style taxonomy.
# Run     : python self_instruct_paraphrase_v2.py
# Designed for VLLM endpoint (Qwen3-30B-A3B), but works with any OpenAI-compat API.

import json
import time
import re
import uuid
import csv
import os
from datetime import datetime
from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────────
VLLM_URL   = "http://100.100.108.100:8080/v1"
MODEL      = "Qwen/Qwen3-30B-A3B"

# Input: your cleaned seed JSON (each record must have generated_question,
# original_full_answer, crop, state, district, season, domain, source_answer_id,
# source_question_id)
INPUT_FILE  = "/home/kritika/self_instruct/self_instruct_clean.json"

# Outputs
OUT_JSON    = "/home/kritika/self_instruct/paraphrase_v2.json"
OUT_CSV     = "/home/kritika/self_instruct/paraphrase_v2.csv"
OUT_LOG     = "/home/kritika/self_instruct/paraphrase_v2_log.txt"

# ── Generation controls ────────────────────────────────────────────────────────
# Number of seed rows to process (set to 1 or 2 for a test run)
MAX_SEEDS       = 2

# How many distinct paraphrase Q&A pairs to generate PER SEED
# The script will cycle through styles until this count is reached.
TARGET_PER_SEED = 60     # aim for 50-100; adjust freely

# Retry / timeout
MAX_RETRIES     = 3
RETRY_DELAY     = 5      # seconds (doubles on each retry)
TEMPERATURE     = 0.7    # higher than scorer — we want lexical diversity
MAX_TOKENS      = 1200

# ─── Style Taxonomy ──────────────────────────────────────────────────────────
# Each style entry:
#   name        : unique identifier stored in output
#   description : fed to the LLM so it knows what to produce
#   guard       : optional lambda(seed) → bool; if False, skip this style
#                 (use guards to avoid styles that make no sense for a seed)

STYLES = [
    {
        "name": "paraphrase_formal",
        "description": (
            "Rewrite the question in a formal, academic or extension-officer tone. "
            "Use technical terminology correctly. The answer should be structured "
            "with clear headings or numbered points where appropriate. "
            "Do NOT add new facts — only rephrase what is in the seed."
        ),
    },
    {
        "name": "paraphrase_farmer_style",
        "description": (
            "Rewrite the question exactly as a semi-literate Indian farmer would ask "
            "at a Kisan helpline — simple words, direct problem description, possibly "
            "mixing Hindi/Punjabi words like 'khet', 'fasal', 'dawai', 'kab daalein'. "
            "The answer should be simple, practical, avoid jargon. "
            "Do NOT add new facts."
        ),
    },
    {
        "name": "paraphrase_short",
        "description": (
            "Produce a very concise question (under 15 words) and a crisp answer "
            "(2-4 sentences maximum). Drop all background context and give only "
            "the core actionable information. "
            "Do NOT add new facts."
        ),
    },
    {
        "name": "paraphrase_problem_description",
        "description": (
            "Rewrite the question as a farmer describing a problem they are seeing "
            "in their field right now — symptoms first, no mention of the disease/pest "
            "name. E.g. 'My pea plants are wilting and the roots look black. "
            "What is wrong and how do I treat it?' "
            "The answer should diagnose the issue from the seed content and give "
            "management steps. Do NOT add new facts."
        ),
    },
    {
        "name": "paraphrase_followup",
        "description": (
            "Rewrite as a follow-up question that a farmer would ask AFTER already "
            "receiving a basic answer. E.g. 'You told me to use Thiram — at what "
            "stage should I apply it and how much per kg of seed?' "
            "Extract a specific detail from the seed answer and make it the focus. "
            "Do NOT add new facts."
        ),
    },
    {
        "name": "paraphrase_yes_no",
        "description": (
            "Convert the question into a binary yes/no question whose answer can be "
            "directly derived from the seed. E.g. 'Can I use NP(S) 20-20-0-13 as "
            "a foliar spray on maize?' Answer must start with Yes or No, then explain "
            "briefly using only the seed content. "
            "Do NOT add new facts."
        ),
    },
    {
        "name": "paraphrase_preventive",
        "description": (
            "Reframe the question as asking HOW TO PREVENT the problem described, "
            "rather than how to treat it. E.g. 'How can I prevent root rot in peas "
            "before sowing?' The answer should pull out preventive/cultural practices "
            "mentioned in the seed. "
            "Do NOT add new facts."
        ),
    },
    {
        "name": "paraphrase_comparison",
        "description": (
            "If the seed mentions two or more chemicals, pests, varieties, or "
            "management methods, write a question that asks to compare them. "
            "E.g. 'What is the difference between Fusarium and Rhizoctonia root rot "
            "in peas?' The answer should draw the comparison from the seed content. "
            "Do NOT add new facts. "
            "IMPORTANT: Only generate if the seed actually has two distinct items "
            "to compare — if not, reply with JSON where generated_question is null."
        ),
    },
    {
        "name": "paraphrase_cause",
        "description": (
            "Rewrite the question to focus on WHY or WHAT CAUSES the problem. "
            "E.g. 'Why do pea roots turn black and rot?' "
            "The answer should explain the causal mechanism from the seed. "
            "Do NOT add new facts."
        ),
    },
    {
        "name": "paraphrase_quantity_focused",
        "description": (
            "Extract a quantity-related aspect from the seed (dosage, seed rate, "
            "spray concentration, area, time duration) and make that the focus of "
            "the question. E.g. 'What is the recommended dosage of Carbendazim per "
            "hectare for pea root rot?' If no quantity is present in the seed, "
            "reply with JSON where generated_question is null. "
            "Do NOT add new facts."
        ),
    },
    {
        "name": "paraphrase_timing_focused",
        "description": (
            "Focus the question on WHEN to perform an action — stage of crop growth, "
            "season, time of day, or phase. E.g. 'At what growth stage of peas "
            "should I apply seed treatment for root rot?' If the seed has no "
            "timing information, reply with JSON where generated_question is null. "
            "Do NOT add new facts."
        ),
    },
    {
        "name": "paraphrase_regional_hindi",
        "description": (
            "Write the question in Hinglish — natural mix of Hindi and English as "
            "spoken by North Indian farmers. Use Devanagari script for Hindi words "
            "OR romanised Hindi, whichever flows naturally. "
            "E.g. 'Matar ki fasal mein jad galne ki bimari ka kya ilaj hai?' "
            "The answer should be in simple English (or Hinglish). "
            "Do NOT add new facts."
        ),
    },
]

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert in Indian agricultural Q&A dataset creation.
You will receive a SEED question and answer from an Indian agricultural helpline,
along with a STYLE instruction.

Your task:
- Generate exactly ONE new (question, answer) pair in the requested style.
- The answer MUST be grounded ONLY in the seed answer — do NOT invent new facts,
  chemicals, dosages, or recommendations not present in the seed.
- You MAY rephrase, restructure, shorten, or change the tone/perspective.
- If the style is impossible for this seed (e.g. comparison when only one item
  exists), return JSON with "generated_question": null.

Output ONLY a valid JSON object. No markdown, no preamble. Format:
{
  "generated_question": "<the new question, or null if style is impossible>",
  "generated_answer": "<the new answer, or null if style is impossible>"
}"""

# ─── Helpers ──────────────────────────────────────────────────────────────────

client = OpenAI(base_url=VLLM_URL, api_key="dummy")


def build_user_prompt(seed: dict, style: dict) -> str:
    return (
        f"STYLE: {style['name']}\n"
        f"STYLE INSTRUCTION: {style['description']}\n\n"
        f"SEED QUESTION:\n{seed['generated_question']}\n\n"
        f"SEED ANSWER:\n{seed.get('original_full_answer') or seed.get('generated_answer', '')}\n\n"
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
            return None          # style was skipped by the model
        return obj
    except json.JSONDecodeError:
        return None


def call_model(seed: dict, style: dict):
    user_prompt = build_user_prompt(seed, style)
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
            return result
        except Exception as e:
            print(f"    [error] attempt {attempt+1}: {e}")
        time.sleep(RETRY_DELAY * (2 ** attempt))
    return None


def make_generated_id(seed: dict, style_name: str, idx: int) -> str:
    base = seed.get("source_answer_id", str(uuid.uuid4()))[:8]
    return f"si_v2_{style_name[:6]}_{base}_{idx}"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── Load seeds ────────────────────────────────────────────────────────────
    with open(INPUT_FILE) as f:
        all_seeds = json.load(f)

    seeds = all_seeds[:MAX_SEEDS]
    print(f"\n{'='*60}")
    print(f"  Self-Instruct Paraphrase v2")
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

        seed_results = []
        style_cycle  = list(STYLES)   # we'll cycle through styles repeatedly
        global_idx   = 0              # unique counter per seed for ID generation
        style_pos    = 0              # position in the style cycle

        while len(seed_results) < TARGET_PER_SEED:
            style = style_cycle[style_pos % len(style_cycle)]
            style_pos += 1
            global_idx += 1

            print(f"  [{len(seed_results)+1}/{TARGET_PER_SEED}] "
                  f"Style={style['name']} ... ", end="", flush=True)

            result = call_model(seed, style)

            if result is None:
                print("skipped (null or parse fail)")
                log_lines.append(
                    f"SKIP | seed={seed_idx} | style={style['name']} | "
                    f"crop={seed.get('crop')} | q={seed['generated_question'][:60]}"
                )
                continue

            record = {
                "generated_id"    : make_generated_id(seed, style["name"], global_idx),
                "style"           : style["name"],
                "source_answer_id": seed.get("source_answer_id", ""),
                "source_question_id": seed.get("source_question_id", ""),
                "crop"            : seed.get("crop", ""),
                "state"           : seed.get("state", ""),
                "district"        : seed.get("district", ""),
                "season"          : seed.get("season", ""),
                "domain"          : seed.get("domain", ""),
                "generated_question": result["generated_question"],
                "generated_answer"  : result["generated_answer"],
                "seed_question"     : seed["generated_question"],   # traceability
                "timestamp"         : datetime.utcnow().isoformat(),
            }

            seed_results.append(record)
            all_results.append(record)

            short_q = result["generated_question"][:70]
            print(f"✓ Q: {short_q}...")
            log_lines.append(
                f"OK   | seed={seed_idx} | style={style['name']} | "
                f"crop={seed.get('crop')} | q={short_q}"
            )

            # Small courtesy delay
            time.sleep(0.5)

        print(f"\n   ✅ Seed {seed_idx+1} done — {len(seed_results)} pairs generated")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

    # ── Save log ──────────────────────────────────────────────────────────────
    with open(OUT_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    # ── Summary ───────────────────────────────────────────────────────────────
    style_counts = {}
    for r in all_results:
        style_counts[r["style"]] = style_counts.get(r["style"], 0) + 1

    print(f"\n{'='*60}")
    print(f"  DONE — {len(all_results)} total pairs generated")
    print(f"  Per-style breakdown:")
    for sname, cnt in sorted(style_counts.items()):
        print(f"    {sname:<35}: {cnt}")
    print(f"\n  Output JSON : {OUT_JSON}")
    print(f"  Output CSV  : {OUT_CSV}")
    print(f"  Log         : {OUT_LOG}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()