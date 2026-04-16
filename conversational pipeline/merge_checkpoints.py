# merge_checkpoints.py
# Purpose : Merge all per-batch checkpoint files in CHECKPOINT_DIR into a
#           single combined JSON + CSV + log file.
#
# Usage   : python merge_checkpoints.py
#           python merge_checkpoints.py --checkpoint-dir /custom/path --out-json merged.json
#
# Notes   :
#   - Batches are merged in numeric order (batch_001, batch_002, …).
#   - A batch file is skipped with a warning if it is missing or malformed.
#   - Duplicate source_id records across batches are preserved — dedup was
#     already handled inside the generator; this script is a pure merge.

import json
import csv
import argparse
from pathlib import Path
from datetime import datetime, timezone

# ─── Defaults (must match self_instruct_conversational.py) ───────────────────
DEFAULT_CHECKPOINT_DIR = "/home/kritika/self_instruct/conversation/checkpoints"
DEFAULT_OUT_JSON       = "/home/kritika/self_instruct/conversation/paraphrase_conv_merged.json"
DEFAULT_OUT_CSV        = "/home/kritika/self_instruct/conversation/paraphrase_conv_merged.csv"
DEFAULT_OUT_LOG        = "/home/kritika/self_instruct/conversation/paraphrase_conv_merged_log.txt"


def discover_batch_files(checkpoint_dir: Path) -> list[Path]:
    """Return all batch_NNN.json files, sorted numerically."""
    files = sorted(
        checkpoint_dir.glob("batch_*.json"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    return files


def load_batch(path: Path) -> dict | None:
    """Load and validate a single batch checkpoint. Returns None on failure."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "results" not in data or "meta" not in data:
            print(f"  ⚠  {path.name} — missing 'results' or 'meta' key, skipping.")
            return None
        return data
    except json.JSONDecodeError as e:
        print(f"  ⚠  {path.name} — JSON decode error ({e}), skipping.")
        return None
    except Exception as e:
        print(f"  ⚠  {path.name} — could not read ({e}), skipping.")
        return None


def merge(checkpoint_dir: Path, out_json: Path, out_csv: Path, out_log: Path) -> None:
    batch_files = discover_batch_files(checkpoint_dir)

    if not batch_files:
        print(f"  No batch_*.json files found in {checkpoint_dir}.")
        return

    print(f"\n  Found {len(batch_files)} batch file(s) in {checkpoint_dir}")

    all_results : list[dict] = []
    all_log     : list[str]  = []
    loaded       = 0
    skipped      = 0

    for path in batch_files:
        data = load_batch(path)
        if data is None:
            skipped += 1
            continue

        batch_results = data["results"]
        batch_log     = data.get("log", [])
        meta          = data["meta"]

        all_results.extend(batch_results)
        all_log.extend(batch_log)
        loaded += 1

        print(f"  ✔  {path.name}  — batch {meta.get('batch_num','?'):>3}  "
              f"seeds {meta.get('seed_range', ['?','?'])[0]}–{meta.get('seed_range', ['?','?'])[1]}  "
              f"pairs: {len(batch_results)}")

    # ── Write merged JSON ─────────────────────────────────────────────────────
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # ── Write merged CSV ──────────────────────────────────────────────────────
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

    # ── Write merged log ──────────────────────────────────────────────────────
    merge_header = (
        f"# Merged at {datetime.now(timezone.utc).isoformat()}\n"
        f"# Batches loaded: {loaded}  skipped: {skipped}\n"
        f"# Total records : {len(all_results)}\n"
        f"{'─'*60}"
    )
    with open(out_log, "w", encoding="utf-8") as f:
        f.write(merge_header + "\n" + "\n".join(all_log))

    # ── Summary ───────────────────────────────────────────────────────────────
    ok_count    = sum(1 for l in all_log if l.startswith("OK"))
    dupe_count  = sum(1 for l in all_log if l.startswith("DUPE"))
    skip_count  = sum(1 for l in all_log if l.startswith("SKIP"))
    retry_count = sum(1 for l in all_log if l.startswith("JUDGE_RETRY"))
    abandon_count = sum(1 for l in all_log if l.startswith("JUDGE_ABANDON"))

    if all_results:
        wcs = [r.get("answer_word_count", 0) for r in all_results]
        avg_wc = sum(wcs) / len(wcs)
        min_wc, max_wc = min(wcs), max(wcs)
    else:
        avg_wc = min_wc = max_wc = 0

    print(f"\n  ── Merge summary ──────────────────────────────────────")
    print(f"  Batch files loaded   : {loaded}  (skipped: {skipped})")
    print(f"  Total pairs merged   : {len(all_results)}")
    print(f"  OK log entries       : {ok_count}")
    print(f"  Duplicates rejected  : {dupe_count}")
    print(f"  Skipped entries      : {skip_count}")
    print(f"  Judge retries        : {retry_count}")
    print(f"  Judge abandoned      : {abandon_count}")
    print(f"  Answer length        : avg={avg_wc:.0f}w  min={min_wc}w  max={max_wc}w")
    print(f"\n  Merged JSON  → {out_json}")
    print(f"  Merged CSV   → {out_csv}")
    print(f"  Merged log   → {out_log}\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Merge per-batch checkpoint JSON files into one combined output."
    )
    p.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR,
                   help=f"Directory containing batch_NNN.json files (default: {DEFAULT_CHECKPOINT_DIR})")
    p.add_argument("--out-json", default=DEFAULT_OUT_JSON,
                   help=f"Path for merged JSON output (default: {DEFAULT_OUT_JSON})")
    p.add_argument("--out-csv",  default=DEFAULT_OUT_CSV,
                   help=f"Path for merged CSV output (default: {DEFAULT_OUT_CSV})")
    p.add_argument("--out-log",  default=DEFAULT_OUT_LOG,
                   help=f"Path for merged log output (default: {DEFAULT_OUT_LOG})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge(
        checkpoint_dir = Path(args.checkpoint_dir),
        out_json       = Path(args.out_json),
        out_csv        = Path(args.out_csv),
        out_log        = Path(args.out_log),
    )