"""Temporally-hard variant of build_youcook2_qa_split.py, mirroring
build_qa_pairs_temporal.py's exact scheme (see that script's own docstring
for the full rationale) -- a new file, not an edit, since the underlying
design changes (this project's established convention when that happens).

build_youcook2_qa_split.py's own distractor sampling draws the wrong-answer
caption uniformly from the ENTIRE matched-row pool, with no constraint tying
it to the anchor's source video -- so the 2-choice task reduces to "does
this caption's topic match this video's topic," solvable from coarse
scene/topic content alone (confirmed project-wide: this is the same shortcut
qa_split_temporal.json was built to eliminate on the VANS side, and the
reason job-155's own joint-training run showed near-chance-or-below accuracy
specifically on "easy"-tagged items while "hard" items stayed meaningfully
above chance -- see this session's own diagnosis).

This script instead draws the distractor from a DIFFERENT clip of the SAME
source YouTube video wherever at least one such alternative exists (YouCook2
has 5.9 rows/video on average per this session's own survey -- comfortably
dense enough). Anchors whose source video has no other matched row (or whose
only other rows happen to share the identical caption text) are dropped
entirely, keeping every row uniformly "hard" rather than silently diluting
train/eval with easy rows.

train: one row per anchor, distractor_caption = the same-video ("hard")
       alternative only -- forces training itself off the shortcut.
test (held-out): TWO rows per anchor -- difficulty="easy" (the original
       random-any-video distractor) and difficulty="hard" (the same-video
       distractor) -- so a checkpoint can be scored on both tiers directly,
       the same easy/hard breakdown already used for qa_split_temporal.json.

Split stays at the VIDEO level (not the pair/anchor level) -- same reasoning
as build_youcook2_qa_split.py's own docstring: consecutive-segment pairs
from the same video share scene/context, so a pair-level split would leak
context between train and held-out eval.
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
OUT_PATH = os.path.join(BASE, "raw_data/qa_split_youcook2_temporal.json")


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
            video_id = in_path.lstrip("/").split("/")[0]
            rows.append({"pid": "youcook2_" + in_path.lstrip("/").replace(".mp4", "").replace("/", "__"),
                         "in_path": in_path, "out_path": out_path,
                         "question": q, "answer": a, "video_id": video_id})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.85,
                     help="fraction of distinct source VIDEOS (not pairs) assigned to train -- "
                          "same video-level split convention as build_youcook2_qa_split.py")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = load_rows()
    print(f"[INFO] {len(rows)} YouCook2 CSV rows with non-empty QA text", flush=True)

    matched = [r for r in rows if clip_exists(r["in_path"]) and clip_exists(r["out_path"])]
    print(f"[INFO] {len(matched)} rows with BOTH input and output clips actually extracted "
          f"(CLIPS_ROOT={CLIPS_ROOT})", flush=True)

    all_answers = [r["answer"] for r in matched]
    by_video = {}
    for r in matched:
        by_video.setdefault(r["video_id"], []).append(r)

    anchors = []
    n_dropped_no_hard = 0
    for i, r in enumerate(matched):
        siblings = [s for s in by_video[r["video_id"]] if s is not r and s["answer"] != r["answer"]]
        if not siblings:
            n_dropped_no_hard += 1
            continue
        hard_distractor = rng.choice(siblings)["answer"]
        easy_distractor = rng.choice(all_answers[:i] + all_answers[i + 1:]) if len(all_answers) > 1 else r["answer"]
        anchors.append({
            "pid": r["pid"], "video_id": r["video_id"],
            "in_clip": os.path.join(CLIPS_ROOT, r["in_path"].lstrip("/")),
            "out_clip": os.path.join(CLIPS_ROOT, r["out_path"].lstrip("/")),
            "question": r["question"], "correct_caption": r["answer"],
            "hard_distractor": hard_distractor, "easy_distractor": easy_distractor,
        })
    print(f"[INFO] {len(anchors)} anchors kept, {n_dropped_no_hard} dropped "
          f"(no other matched clip with a distinct caption from the same source video)", flush=True)

    video_ids = sorted({a["video_id"] for a in anchors})
    rng.shuffle(video_ids)
    n_train_videos = int(round(len(video_ids) * args.train_frac))
    train_video_ids = set(video_ids[:n_train_videos])
    print(f"[INFO] video-level split: {len(train_video_ids)} train / "
          f"{len(video_ids) - len(train_video_ids)} held-out videos "
          f"(train_frac={args.train_frac}, seed={args.seed}, {len(video_ids)} total)", flush=True)

    def train_row(a):
        return {"pid": a["pid"], "in_clip": a["in_clip"], "out_clip": a["out_clip"],
                "question": a["question"], "correct_caption": a["correct_caption"],
                "distractor_caption": a["hard_distractor"], "difficulty": "hard"}

    def eval_rows(a):
        base = {"pid": a["pid"], "in_clip": a["in_clip"], "out_clip": a["out_clip"],
                "question": a["question"], "correct_caption": a["correct_caption"]}
        return [
            {**base, "distractor_caption": a["easy_distractor"], "difficulty": "easy"},
            {**base, "distractor_caption": a["hard_distractor"], "difficulty": "hard"},
        ]

    train_items, test_items = [], []
    for a in anchors:
        if a["video_id"] in train_video_ids:
            train_items.append(train_row(a))
        else:
            test_items.extend(eval_rows(a))

    rng.shuffle(train_items)
    rng.shuffle(test_items)
    split = {"train": train_items, "val": [], "test": test_items}
    print(f"[INFO] wrote {len(train_items)} train rows (hard only), "
          f"{len(test_items)} held-out test rows (easy+hard per anchor)", flush=True)

    with open(args.out, "w") as f:
        json.dump(split, f, indent=2)
    print(f"[INFO] wrote -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
