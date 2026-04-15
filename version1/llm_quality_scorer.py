# llm_quality_scorer.py
# Run on astra: python llm_quality_scorer.py
# GPU 0 = VLLM engine, GPU 1 = training → scoring uses VLLM API, no extra GPU needed

import json
import time
import re
from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────────
VLLM_URL    = "http://100.100.108.100:8080/v1"
MODEL       = "Qwen/Qwen3-30B-A3B"
INPUT_FILE = "/home/kritika/self_instruct/paraphrase_v2.json"
OUT_PASS   = "/home/kritika/self_instruct/paraphrase_v2_scored.json"
OUT_FAIL    = "/home/kritika/self_instruct/low_quality.json"
OUT_ALL     = "/home/kritika/self_instruct/scored_all.json"

BATCH_SIZE      = 5       # pairs per API call (keeps prompt short, reduces timeout)
MIN_SCORE       = 3       # pairs with overall < MIN_SCORE go to low_quality
MAX_TOKENS      = 2000
TEMPERATURE     = 0.2     # low temp → consistent scoring
MAX_RETRIES     = 3
RETRY_DELAY     = 5       # seconds

client = OpenAI(base_url=VLLM_URL, api_key="dummy")

# ─── Prompt ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an agricultural Q&A quality evaluator. 
You will receive farmer Q&A pairs from an Indian agricultural helpline dataset.
Score EACH pair on these 5 criteria (scale 1–5):

1. Relevance     — Does the answer directly address the question?
2. Accuracy      — Are the technical details (chemicals, dosages, crop names) factually plausible?
3. Fluency       — Is the language clear and grammatically correct (Hindi/English/regional OK)?
4. Completeness  — Does the answer cover what a farmer actually needs to act on this?
5. Overall       — Holistic quality (1=unusable, 3=acceptable, 5=excellent)

STRICT RULES:
- Do NOT add information or correct factual claims — evaluate as-is
- Short answers (1–2 sentences) can still score 5 if they are direct and complete
- Regional language (Punjabi, Tamil) is valid — do not penalize
- Statement-style questions ("Information regarding X") are valid — do not penalize

Output ONLY a valid JSON array. No markdown, no preamble. Format:
[
  {
    "id": <seed_id>,
    "style": "<style>",
    "relevance": <1-5>,
    "accuracy": <1-5>,
    "fluency": <1-5>,
    "completeness": <1-5>,
    "overall": <1-5>,
    "reason": "<one sentence reason>"
  },
  ...
]"""

def build_user_prompt(batch):
    lines = ["Score these Q&A pairs:\n"]
    for i, item in enumerate(batch):
        lines.append(f"[{i+1}] seed_id={item.get('seed_id','?')} style={item.get('style','?')}")
        lines.append(f"Q: {item['generated_question']}")
        lines.append(f"A: {item['generated_answer'][:800]}")  # truncate long answers
        lines.append("")
    return "\n".join(lines)

def parse_scores(text, batch):
    """Extract JSON array from model response, fallback gracefully."""
    # Strip markdown fences
    text = re.sub(r"```json|```", "", text).strip()
    # Find JSON array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return None
    try:
        scores = json.loads(match.group())
        # Validate length matches batch
        if len(scores) == len(batch):
            return scores
        # Partial parse — take what we got
        return scores[:len(batch)] if scores else None
    except json.JSONDecodeError:
        return None

def score_batch(batch):
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(batch)},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            text = resp.choices[0].message.content.strip()
            scores = parse_scores(text, batch)
            if scores:
                return scores
            print(f"  [warn] Parse failed attempt {attempt+1}, raw:\n{text[:300]}")
        except Exception as e:
            print(f"  [error] Attempt {attempt+1}: {e}")
        time.sleep(RETRY_DELAY * (2 ** attempt))
    return None

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    with open(INPUT_FILE) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} pairs. Scoring in batches of {BATCH_SIZE}...")

    all_scored = []
    failed_ids = []

    batches = [data[i:i+BATCH_SIZE] for i in range(0, len(data), BATCH_SIZE)]
    total = len(batches)

    for idx, batch in enumerate(batches):
        print(f"  Batch {idx+1}/{total} ({len(batch)} pairs)...", end=" ", flush=True)
        scores = score_batch(batch)

        if scores is None:
            print("FAILED — skipping batch")
            for item in batch:
                failed_ids.append(item.get("seed_id", "?"))
            continue

        # Merge scores back into items
        for item, score in zip(batch, scores):
            merged = {**item, **score}
            all_scored.append(merged)
        
        avg = sum(s.get("overall", 0) for s in scores) / len(scores)
        print(f"done. avg overall={avg:.2f}")

        # Small delay to be nice to VLLM
        time.sleep(1)

    # ─── Save all ──────────────────────────────────────────────────────────────
    passed = [r for r in all_scored if r.get("overall", 0) >= MIN_SCORE]
    failed = [r for r in all_scored if r.get("overall", 0) < MIN_SCORE]

    with open(OUT_ALL,  "w") as f: json.dump(all_scored, f, ensure_ascii=False, indent=2)
    with open(OUT_PASS, "w") as f: json.dump(passed,     f, ensure_ascii=False, indent=2)
    with open(OUT_FAIL, "w") as f: json.dump(failed,     f, ensure_ascii=False, indent=2)

    # ─── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print(f"Total scored   : {len(all_scored)}")
    print(f"Passed (≥{MIN_SCORE})   : {len(passed)}")
    print(f"Filtered (<{MIN_SCORE}) : {len(failed)}")
    if all_scored:
        for dim in ["relevance","accuracy","fluency","completeness","overall"]:
            avg = sum(r.get(dim,0) for r in all_scored) / len(all_scored)
            print(f"  avg {dim:<14}: {avg:.2f}")
    if failed_ids:
        print(f"\nFailed batches (no score): {failed_ids}")
    print("="*50)
    print(f"Output → {OUT_PASS}")

if __name__ == "__main__":
    main()