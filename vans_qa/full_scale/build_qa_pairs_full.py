"""
Full-scale version of build_qa_pairs.py: scans BOTH VANS-DATA_COIN.csv and
VANS-DATA_YouCook.csv (28,226 rows total) instead of just the 500-item demo
bundle, and keeps only rows whose input_video_path/output_video_path clips
were actually extracted by step1_coin_fixed.py / step1_youcook_fixed.py
(i.e. whose source video made it through the still-partial raw download).

Same paired 2-choice QA design and same directional-pilot framing as the
demo version -- just at whatever scale the current download+split state
supports (~20K clips / a five-digit number of matched QA pairs, versus the
demo's 499).
"""
import argparse
import csv
import json
import os
import random

BASE = os.environ.get("VANS_ROOT", "/projects/bhay/william/ruixin/vans_world_model")
WORK_BASE = os.environ.get("VANS_WORK_ROOT", "/work/nvme/bdqf/yli8/vans_raw_data")  # relocated off the near-full /projects/bhay allocation
CLIPS_ROOT = os.path.join(WORK_BASE, "clips")
CSV_PATHS = [
    os.path.join(BASE, "hf_data/VANS-DATA_COIN.csv"),
    os.path.join(BASE, "hf_data/VANS-DATA_YouCook.csv"),
]
OUT_PATH = os.path.join(BASE, "raw_data/qa_split_full.json")


def clip_exists(rel_path):
    # rel_path like "/-0X2mXPy3Mc/603.mp4"
    p = os.path.join(CLIPS_ROOT, rel_path.lstrip("/"))
    return os.path.exists(p) and os.path.getsize(p) > 1000


def load_rows():
    rows = []
    for csv_path in CSV_PATHS:
        with open(csv_path, encoding="utf-8-sig", errors="replace") as f:
            for r in csv.DictReader(f):
                q = (r.get("ENG_Instruction") or "").strip()
                a = (r.get("ENG_GT_Caption") or "").strip()
                in_path = (r.get("input_video_path") or "").strip()
                out_path = (r.get("output_video_path") or "").strip()
                if not (q and a and in_path and out_path):
                    continue
                rows.append({"pid": in_path.lstrip("/").replace(".mp4", "").replace("/", "__"),
                             "in_path": in_path, "out_path": out_path,
                             "question": q, "answer": a})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = load_rows()
    print(f"[INFO] {len(rows)} CSV rows with non-empty QA text")

    matched = [r for r in rows if clip_exists(r["in_path"]) and clip_exists(r["out_path"])]
    print(f"[INFO] {len(matched)} rows with BOTH input and output clips actually extracted")

    all_answers = [r["answer"] for r in matched]
    items = []
    for i, r in enumerate(matched):
        distractor_pool_idx = i
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
    n = len(items)
    n_train = int(round(n * args.train_frac))
    n_val = int(round(n * args.val_frac))
    split = {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:],
    }
    print(f"[INFO] split: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")

    with open(args.out, "w") as f:
        json.dump(split, f, indent=2)
    print(f"[INFO] wrote -> {args.out}")


if __name__ == "__main__":
    main()
