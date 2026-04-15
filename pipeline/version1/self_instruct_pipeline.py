"""
Self-Instruct Pipeline for Agricultural Q&A Dataset
=====================================================
Uses Qwen3-30B-A3B on VLLM to generate new Q&A pairs from seed data.
Styles: paraphrase | related | harder | localize
Output: self_instruct_output.csv + self_instruct_output.json

Usage:
  python self_instruct_pipeline.py --seeds 5 --per_seed 1 --style paraphrase
  python self_instruct_pipeline.py --crop all --style all --seeds 50 --per_seed 2
  python self_instruct_pipeline.py --crop Wheat --state "Uttar Pradesh" --style harder --seeds 10 --per_seed 2
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import urlencode
import urllib.request

# ── Config ──────────────────────────────────────────────────────────────────
VLLM_BASE_URL = "http://100.100.108.100:8080/v1"
MODEL_ID      = "Qwen/Qwen3-30B-A3B"
CSV_PATH      = "random_200_qa_1.csv"
OUT_CSV       = "self_instruct_output.csv"
OUT_JSON      = "self_instruct_output.json"
MAX_TOKENS    = 3000
TEMPERATURE   = 0.8
TIMEOUT_SEC   = 120

STYLES = ["paraphrase", "related", "harder", "localize"]

# ── Normalization ────────────────────────────────────────────────────────────
def normalize(val):
    return val.strip().title() if val else val.strip()

# ── Load & normalize CSV ─────────────────────────────────────────────────────
def load_seeds(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize messy fields
            row["Crop"]  = normalize(row.get("Crop", ""))
            row["State"] = normalize(row.get("State", ""))
            rows.append(row)
    return rows

# ── Filter seeds ─────────────────────────────────────────────────────────────
def filter_seeds(rows, crop=None, state=None):
    filtered = rows
    if crop and crop.lower() != "all":
        filtered = [r for r in filtered if r["Crop"].lower() == crop.lower()]
    if state and state.lower() != "all":
        filtered = [r for r in filtered if r["State"].lower() == state.lower()]
    return filtered

# ── Prompt builder ───────────────────────────────────────────────────────────
STYLE_PROMPTS = {
    "paraphrase": """You are an agricultural training data assistant. Your ONLY job is to rephrase — not correct, not improve, not add knowledge.

STRICT RULES — MUST FOLLOW:
1. Do NOT add any information not present in the original answer
2. Do NOT remove any information from the original answer
3. Do NOT correct or override any technical detail — if original says NPK 20:20:13, keep it as NPK 20:20:13
4. Do NOT use your own agricultural knowledge — only rephrase what is given
5. Keep ALL chemical names, product names, dosages, timings, and quantities EXACTLY as in original
6. Output ONLY a valid JSON object with keys "question" and "answer", no extra text, no markdown

Crop: {crop}
State: {state}
District: {district}
Season: {season}
Domain: {domain}

Original Question: {question}
Original Answer: {answer}

Rephrase the above into a new question and answer. Output JSON only:""",

    "related": """You are an agricultural training data assistant. Generate a RELATED question and answer strictly grounded in the reference material below.

STRICT RULES — MUST FOLLOW:
1. The new question must be on a related sub-topic within the same crop and domain
2. The answer must be derived ONLY from information present in the reference answer
3. Do NOT invent new chemicals, dosages, or recommendations not mentioned in the reference
4. Do NOT use your own agricultural knowledge beyond what is in the reference answer
5. Keep ALL chemical names, product names, and dosages EXACTLY as mentioned in the reference
6. If the reference does not contain enough info for a related question, paraphrase instead
7. Output ONLY a valid JSON object with keys "question" and "answer", no extra text, no markdown

Crop: {crop}
State: {state}
District: {district}
Season: {season}
Domain: {domain}

Reference Question: {question}
Reference Answer: {answer}

Generate a related Q&A strictly from the above. Output JSON only:""",

    "harder": """You are an agricultural training data assistant. Generate a MORE SPECIFIC version of the question and a more detailed answer — strictly grounded in the reference material.

STRICT RULES — MUST FOLLOW:
1. Make the question more specific (add scenario, timing, or condition details)
2. The answer must expand ONLY on information present in the reference answer
3. Do NOT invent new chemicals, dosages, or recommendations not in the reference
4. Do NOT use your own agricultural knowledge beyond what is in the reference answer
5. Keep ALL chemical names, product names, and dosages EXACTLY as in the reference
6. You may elaborate on mechanisms or reasoning but only if supported by the reference
7. Output ONLY a valid JSON object with keys "question" and "answer", no extra text, no markdown

Crop: {crop}
State: {state}
District: {district}
Season: {season}
Domain: {domain}

Original Question: {question}
Original Answer: {answer}

Generate JSON:""",

    "localize": """You are an agricultural training data assistant. Adapt the question and answer for a different Indian state — but stay strictly grounded in the original content.

STRICT RULES — MUST FOLLOW:
1. Only change the state/district name in the question and answer
2. Do NOT change any chemical names, dosages, product names, or technical recommendations
3. Do NOT add region-specific varieties, pests, or practices from your own knowledge
4. Do NOT remove any information from the original answer
5. Keep the answer structure and all technical content identical — only location references change
6. Output ONLY a valid JSON object with keys "question" and "answer", no extra text, no markdown

Crop: {crop}
Original State: {state}
District: {district}
Season: {season}
Domain: {domain}
TARGET STATE (only change location references to this): {target_state}

Original Question: {question}
Original Answer: {answer}

Adapt location references only. Output JSON only:""",
}

LOCALIZE_TARGET_STATES = ["Haryana", "Punjab", "Karnataka", "Tamil Nadu", "West Bengal", "Madhya Pradesh"]

def build_prompt(style, row):
    template = STYLE_PROMPTS[style]
    target_state = ""
    if style == "localize":
        candidates = [s for s in LOCALIZE_TARGET_STATES if s.lower() != row["State"].lower()]
        target_state = random.choice(candidates) if candidates else "Haryana"

    return template.format(
        crop=row["Crop"],
        state=row["State"],
        district=row.get("District", ""),
        season=row.get("Season", ""),
        domain=row.get("Domain", ""),
        question=row["Question Text"],
        answer=row["Answer Text"][:3000],  # Truncate to give model room to generate output
        target_state=target_state,
    ), target_state

# ── VLLM API call ────────────────────────────────────────────────────────────
def call_vllm(prompt, retries=3):
    url = f"{VLLM_BASE_URL}/chat/completions"
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    for attempt in range(1, retries + 1):
        try:
            req = Request(url, data=data, headers=headers, method="POST")
            with urlopen(req, timeout=TIMEOUT_SEC) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"  [Attempt {attempt}/{retries}] Error: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)
    return None

# ── Parse JSON from model output ─────────────────────────────────────────────
def parse_json_output(raw):
    # Try direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Extract JSON block from markdown fences
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    # Find first { ... } block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None

# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(args):
    print(f"\n{'='*60}")
    print(f"  Self-Instruct Pipeline — Agricultural Q&A")
    print(f"  Model : {MODEL_ID}")
    print(f"  Style : {args.style}")
    print(f"  Seeds : {args.seeds} | Per seed: {args.per_seed}")
    print(f"  Crop  : {args.crop} | State: {args.state}")
    print(f"{'='*60}\n")

    # Load data
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] CSV not found: {CSV_PATH}")
        sys.exit(1)

    all_rows = load_seeds(CSV_PATH)
    print(f"[INFO] Loaded {len(all_rows)} rows from {CSV_PATH}")

    # Filter
    seeds = filter_seeds(all_rows, crop=args.crop, state=args.state)
    if not seeds:
        print(f"[ERROR] No rows match crop='{args.crop}' state='{args.state}'")
        sys.exit(1)
    print(f"[INFO] Filtered to {len(seeds)} matching rows")

    # Sample seeds
    n_seeds = min(args.seeds, len(seeds))
    selected = random.sample(seeds, n_seeds)
    print(f"[INFO] Sampled {n_seeds} seed rows\n")

    # Determine styles
    styles_to_run = STYLES if args.style == "all" else [args.style]

    # Prepare output
    results = []
    fieldnames = [
        "generated_id", "style", "source_answer_id", "source_question_id",
        "crop", "state", "district", "season", "domain",
        "generated_question", "generated_answer", "target_state", "timestamp"
    ]

    # Load existing output to append
    existing = []
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        print(f"[INFO] Found {len(existing)} existing generated rows — will append\n")

    total = len(selected) * len(styles_to_run) * args.per_seed
    done = 0
    failed = 0

    for style in styles_to_run:
        print(f"\n── Style: {style.upper()} ─────────────────────────────")
        for seed_row in selected:
            for gen_idx in range(args.per_seed):
                done += 1
                print(f"  [{done}/{total}] Crop={seed_row['Crop']} | State={seed_row['State']} | Gen#{gen_idx+1}")

                prompt, target_state = build_prompt(style, seed_row)
                raw_output = call_vllm(prompt)

                if raw_output is None:
                    print(f"    ✗ VLLM call failed — skipping")
                    failed += 1
                    continue

                parsed = parse_json_output(raw_output)
                if not parsed or "question" not in parsed or "answer" not in parsed:
                    print(f"    ✗ JSON parse failed. Raw: {raw_output[:200]}")
                    failed += 1
                    continue

                gen_id = f"si_{style[:3]}_{seed_row.get('Answer ID','')[:8]}_{gen_idx}_{int(time.time())}"
                record = {
                    "generated_id":       gen_id,
                    "style":              style,
                    "source_answer_id":   seed_row.get("Answer ID", ""),
                    "source_question_id": seed_row.get("Question ID", ""),
                    "crop":               seed_row["Crop"],
                    "state":              seed_row["State"],
                    "district":           seed_row.get("District", ""),
                    "season":             seed_row.get("Season", ""),
                    "domain":             seed_row.get("Domain", ""),
                    "generated_question": parsed["question"].strip(),
                    "generated_answer":   parsed["answer"].strip(),
                    "target_state":       target_state,
                    "timestamp":          datetime.now().isoformat(),
                }
                results.append(record)
                print(f"    ✓ Q: {parsed['question'][:80]}...")

    # Write CSV
    all_results = existing + results
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Write JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"  DONE!")
    print(f"  Generated : {len(results)} new pairs")
    print(f"  Failed    : {failed}")
    print(f"  Total CSV : {len(all_results)} rows")
    print(f"  Output    : {OUT_CSV}")
    print(f"  Output    : {OUT_JSON}")
    print(f"{'='*60}\n")

# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Instruct pipeline for Agri Q&A")
    parser.add_argument("--crop",     default="all",        help="Crop name or 'all' (default: all)")
    parser.add_argument("--state",    default="all",        help="State name or 'all' (default: all)")
    parser.add_argument("--style",    default="paraphrase", choices=STYLES + ["all"],
                        help="Generation style (default: paraphrase)")
    parser.add_argument("--seeds",    type=int, default=5,  help="Number of seed rows to sample (default: 5)")
    parser.add_argument("--per_seed", type=int, default=1,  help="Generations per seed row (default: 1)")
    parser.add_argument("--seed_random", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed_random)
    run_pipeline(args)