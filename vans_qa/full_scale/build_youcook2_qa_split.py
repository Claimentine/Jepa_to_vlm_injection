"""Builds a YouCook2-ONLY 2-choice QA eval split, held out as its own
benchmark rather than merged into build_qa_pairs_full.py's combined
COIN+YouCook2 train/val/test split -- same role as
build_mvbench_qa_split.py/build_tempcompass_qa_split.py/
extract_crosstask_qa.py: a distinct benchmark row, not more COIN training
data.

Mirrors build_qa_pairs_full.py's own pairing logic exactly (same CSV
schema, same clip_exists() check, same real-caption-from-elsewhere
distractor convention) but reads only VANS-DATA_YouCook.csv and writes
ALL matched items to the "test" key (train/val empty), matching how
TempCompass/MVBench/CrossTask are structured -- this is an eval-only
benchmark, not additional training data.

Run with VANS_WORK_ROOT pointed at job-88's own output
(/data/vans_work_youcook, whose clips/ holds step1_youcook_fixed.py's
output directly -- no merge with COIN's clips/ tree needed, since this
script never reads VANS-DATA_COIN.csv at all).
"""
import argparse
import csv
import json
import os
import random

BASE = os.environ.get("VANS_ROOT", "/data")
WORK_BASE = os.environ.get("VANS_WORK_ROOT", "/data/vans_work_youcook")
CLIPS_ROOT = os.path.join(WORK_BASE, "clips")
CSV_PATH = os.path.join(BASE, "hf_data/VANS-DATA_YouCook.csv")
OUT_PATH = os.path.join(BASE, "raw_data/qa_split_youcook2.json")


def clip_exists(rel_path):
    p = os.path.join(CLIPS_ROOT, rel_path.lstrip("/"))
    return os.path.exists(p) and os.path.getsize(p) > 1000


def load_rows():
    rows = []
    with open(CSV_PATH, encoding="utf-8-sig", errors="replace") as f:
        for r in csv.DictReader(f):
            q = (r.get("ENG_Instruction") or "").strip()
            a = (r.get("ENG_GT_Caption") or "").strip()
            in_path = (r.get("input_video_path") or "").strip()
            out_path = (r.get("output_video_path") or "").strip()
            if not (q and a and in_path and out_path):
                continue
            rows.append({"pid": "youcook2_" + in_path.lstrip("/").replace(".mp4", "").replace("/", "__"),
                         "in_path": in_path, "out_path": out_path,
                         "question": q, "answer": a})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = load_rows()
    print(f"[INFO] {len(rows)} YouCook2 CSV rows with non-empty QA text", flush=True)

    matched = [r for r in rows if clip_exists(r["in_path"]) and clip_exists(r["out_path"])]
    print(f"[INFO] {len(matched)} rows with BOTH input and output clips actually extracted "
          f"(CLIPS_ROOT={CLIPS_ROOT})", flush=True)

    all_answers = [r["answer"] for r in matched]
    items = []
    for i, r in enumerate(matched):
        distractor = rng.choice(all_answers[:i] + all_answers[i + 1:]) if len(all_answers) > 1 else r["answer"]
        items.append({
            "pid": r["pid"],
            "in_clip": os.path.join(CLIPS_ROOT, r["in_path"].lstrip("/")),
            "out_clip": os.path.join(CLIPS_ROOT, r["out_path"].lstrip("/")),
            "question": r["question"],
            "correct_caption": r["answer"],
            "distractor_caption": distractor,
        })

    rng.shuffle(items)
    split = {"train": [], "val": [], "test": items}
    print(f"[INFO] wrote {len(items)} items to test (eval-only benchmark)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(split, f, indent=2)
    print(f"[DONE] wrote -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
