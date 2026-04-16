# merge_v12.py
# Purpose : Merge Part A (seeds 1–150) and Part B (seeds 151–200) outputs into
#           a single paraphrase_v12_merged.json / .csv / _log.txt.
# Run     : python merge_v12.py
#
# Prerequisites:
#   paraphrase_v12_part_a.json   — output from self_instruct_v12_part_a.py
#   paraphrase_v12_part_b.json   — output from self_instruct_v12_part_b.py
#
# The merge is additive: Part A records come first, then Part B records.
# Duplicate generated_ids (should not normally occur) are silently deduplicated,
# keeping the first occurrence.

import json
import csv
import os
from datetime import datetime, timezone

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = "/home/kritika/self_instruct"

PART_A_JSON = os.path.join(BASE_DIR, "paraphrase_v12_part_a.json")
PART_B_JSON = os.path.join(BASE_DIR, "paraphrase_v12_part_b.json")
PART_A_LOG  = os.path.join(BASE_DIR, "paraphrase_v12_part_a_log.txt")
PART_B_LOG  = os.path.join(BASE_DIR, "paraphrase_v12_part_b_log.txt")

OUT_JSON    = os.path.join(BASE_DIR, "paraphrase_v12_merged.json")
OUT_CSV     = os.path.join(BASE_DIR, "paraphrase_v12_merged.csv")
OUT_LOG     = os.path.join(BASE_DIR, "paraphrase_v12_merged_log.txt")


def load_json(path: str) -> list:
    if not os.path.exists(path):
        print(f"  ⚠  File not found: {path}  — treating as empty.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  ✔  Loaded {len(data):>5} records from {os.path.basename(path)}")
    return data


def load_log(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def merge():
    print(f"\n{'='*60}")
    print(f"  Self-Instruct v12 — Merge Part A + Part B")
    print(f"{'='*60}\n")

    part_a = load_json(PART_A_JSON)
    part_b = load_json(PART_B_JSON)

    # Concatenate, then deduplicate by generated_id (keep first occurrence)
    seen_ids = set()
    merged   = []
    for record in part_a + part_b:
        gid = record.get("generated_id", "")
        if gid in seen_ids:
            print(f"  ⚠  Duplicate generated_id skipped: {gid}")
            continue
        seen_ids.add(gid)
        merged.append(record)

    dupe_count = (len(part_a) + len(part_b)) - len(merged)

    # Save merged JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\n  ✔  Merged JSON written  → {OUT_JSON}")

    # Save merged CSV
    if merged:
        fieldnames = list(merged[0].keys())
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged)
        print(f"  ✔  Merged CSV written   → {OUT_CSV}")
    else:
        print(f"  ⚠  No records to write — CSV skipped.")

    # Save merged log
    log_a = load_log(PART_A_LOG)
    log_b = load_log(PART_B_LOG)
    merged_log = (
        ["# ── Part A log ─────────────────────────────────────────────"] +
        log_a +
        ["", "# ── Part B log ─────────────────────────────────────────────"] +
        log_b
    )
    with open(OUT_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(merged_log))
    print(f"  ✔  Merged log written   → {OUT_LOG}")

    # Summary
    style_counts: dict = {}
    for r in merged:
        style_counts[r.get("style", "unknown")] = style_counts.get(r.get("style", "unknown"), 0) + 1

    print(f"\n{'='*60}")
    print(f"  MERGE SUMMARY")
    print(f"  Part A records          : {len(part_a)}")
    print(f"  Part B records          : {len(part_b)}")
    print(f"  Duplicates removed      : {dupe_count}")
    print(f"  Total merged records    : {len(merged)}")
    print(f"\n  Per-style breakdown:")
    for sname, cnt in sorted(style_counts.items()):
        print(f"    {sname:<40}: {cnt:>4} pairs")
    print(f"\n  Output JSON : {OUT_JSON}")
    print(f"  Output CSV  : {OUT_CSV}")
    print(f"  Log         : {OUT_LOG}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    merge()