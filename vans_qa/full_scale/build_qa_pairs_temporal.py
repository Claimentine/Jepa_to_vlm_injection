"""
Temporally-hard variant of build_qa_pairs_full.py.

The original distractor sampling (build_qa_pairs_full.py:70-71) draws the
wrong-answer caption uniformly from the ENTIRE ~20K-row answer pool, with no
constraint tying it to the anchor's video. In practice this means the
distractor is almost always a different scene/topic entirely (a different
video, different objects, different setting), so the 2-choice QA task
reduces to "does this caption's topic match this video's topic" -- a task
solvable from coarse scene/topic content alone, never requiring the
video-to-video *temporal* transition JEPA's predictive features are meant to
capture. (Live evidence: the job-05 jepa x cross_attn run hit val_acc=1.0000
and near-zero loss within 200 steps -- suspiciously fast for a real 2-choice
reasoning task, consistent with a shortcut rather than genuine learning.)

This script instead draws the distractor from a DIFFERENT clip of the SAME
source video (grouped by the video-id prefix of `pid`, e.g. "ggO4G0iEcnk" in
"ggO4G0iEcnk__714") wherever at least one such alternative exists. Same
scene, same objects, same actors -- only the specific moment differs, so
solving it requires identifying which continuation actually follows this
clip, not just recognizing the video's general topic.

Anchors whose source video has no other extracted/matched clip (so no
same-video distractor is possible) are dropped entirely, rather than falling
back to a cross-video distractor -- this keeps every row in the output
uniformly "hard" instead of silently mixing in easy rows that would dilute
train and confound the val/test difficulty split.

train: one row per anchor, distractor_caption = the same-video ("hard")
       alternative. Forces training itself off the scene-matching shortcut.
val/test: TWO rows per anchor -- one tagged difficulty="easy" (the original
       random-any-video distractor, for comparability with the old split)
       and one tagged difficulty="hard" (the same-video distractor) -- so a
       trained checkpoint can be scored on both tiers. A model that's
       actually doing scene-matching should land near chance on the hard
       tier while still acing easy; a model that's learned real temporal
       structure should do reasonably on both.

Split happens at the ANCHOR level (before val/test get expanded into their
easy/hard pairs), so a given anchor's rows never cross the train/val/test
boundary.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_qa_pairs_full import CLIPS_ROOT, clip_exists, load_rows  # noqa: E402

BASE = os.environ.get("VANS_ROOT", "/projects/bhay/william/ruixin/vans_world_model")
OUT_PATH = os.path.join(BASE, "raw_data/qa_split_temporal.json")


def video_id_of(pid):
    # pid is "<video_id>__<clip_index>" (see build_qa_pairs_full.load_rows).
    # rsplit on the LAST "__" since a YouTube video id can itself contain
    # "__" as a substring (base64url alphabet includes '_').
    return pid.rsplit("__", 1)[0]


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

    by_video = {}
    for r in matched:
        by_video.setdefault(video_id_of(r["pid"]), []).append(r)

    anchors = []
    n_dropped_no_hard = 0
    for i, r in enumerate(matched):
        siblings = [s for s in by_video[video_id_of(r["pid"])] if s is not r and s["answer"] != r["answer"]]
        if not siblings:
            n_dropped_no_hard += 1
            continue
        hard_distractor = rng.choice(siblings)["answer"]
        easy_distractor = rng.choice(all_answers[:i] + all_answers[i + 1:]) if len(all_answers) > 1 else r["answer"]
        anchors.append({
            "pid": r["pid"],
            "in_clip": os.path.join(CLIPS_ROOT, r["in_path"].lstrip("/")),
            "out_clip": os.path.join(CLIPS_ROOT, r["out_path"].lstrip("/")),
            "question": r["question"],
            "correct_caption": r["answer"],
            "hard_distractor": hard_distractor,
            "easy_distractor": easy_distractor,
        })
    print(f"[INFO] {len(anchors)} anchors kept, {n_dropped_no_hard} dropped "
          f"(no other matched clip from the same source video)")

    rng.shuffle(anchors)
    n = len(anchors)
    n_train = int(round(n * args.train_frac))
    n_val = int(round(n * args.val_frac))
    anchor_split = {
        "train": anchors[:n_train],
        "val": anchors[n_train:n_train + n_val],
        "test": anchors[n_train + n_val:],
    }

    def train_row(a):
        return {
            "pid": a["pid"], "in_clip": a["in_clip"], "out_clip": a["out_clip"],
            "question": a["question"], "correct_caption": a["correct_caption"],
            "distractor_caption": a["hard_distractor"], "difficulty": "hard",
        }

    def eval_rows(a):
        base = {
            "pid": a["pid"], "in_clip": a["in_clip"], "out_clip": a["out_clip"],
            "question": a["question"], "correct_caption": a["correct_caption"],
        }
        return [
            {**base, "distractor_caption": a["easy_distractor"], "difficulty": "easy"},
            {**base, "distractor_caption": a["hard_distractor"], "difficulty": "hard"},
        ]

    split = {"train": [train_row(a) for a in anchor_split["train"]]}
    for part in ("val", "test"):
        rows_out = []
        for a in anchor_split[part]:
            rows_out.extend(eval_rows(a))
        split[part] = rows_out

    print(f"[INFO] split (anchors -> rows): "
          f"train={len(anchor_split['train'])}->{len(split['train'])} "
          f"val={len(anchor_split['val'])}->{len(split['val'])} "
          f"test={len(anchor_split['test'])}->{len(split['test'])}")

    with open(args.out, "w") as f:
        json.dump(split, f, indent=2)
    print(f"[INFO] wrote -> {args.out}")


if __name__ == "__main__":
    main()
