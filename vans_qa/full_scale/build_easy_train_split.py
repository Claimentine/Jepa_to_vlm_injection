"""Builds easy-only and easy+hard TRAINING variants of a temporal QA split.

Every temporal split's train side is hard-only (distractor = a different
clip of the SAME video), while val/test carry both an easy and a hard row
per anchor. A model trained only on hard rows can learn a shortcut that is
inverted on easy rows (e.g. "avoid the caption that matches the current
scene"), which shows up as easy val accuracy collapsing far below chance
(job-72 step 4500: easy 0.16; job-172 step 300: easy 0.22).

For each hard train row this makes an easy twin with the same anchor and a
distractor drawn from a DIFFERENT video's correct_caption -- the same
"random other anchor" rule build_qa_pairs_temporal.py uses for its easy
val/test rows, restricted to other videos so a twin can't silently turn
hard. val/test are copied unchanged, so every variant is scored on the
identical held-out rows.

Outputs (next to --in_split unless --out_dir is given):
  <stem>_easy.json   train = easy twins only
  <stem>_mixed.json  train = hard rows + easy twins, shuffled
"""
import argparse
import json
import os
import random


def video_id_of(pid):
    return pid.rsplit("__", 1)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_split", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.in_split) as f:
        split = json.load(f)
    train = split["train"]
    kinds = {it.get("difficulty") for it in train}
    if kinds != {"hard"}:
        raise ValueError(f"expected a hard-only train side, got difficulties {kinds}")

    rng = random.Random(args.seed)
    n = len(train)
    easy = []
    for i, it in enumerate(train):
        vid = video_id_of(it["pid"])
        while True:
            j = rng.randrange(n)
            if j != i and video_id_of(train[j]["pid"]) != vid:
                break
        easy.append({**it, "distractor_caption": train[j]["correct_caption"], "difficulty": "easy"})

    mixed = train + easy
    rng.shuffle(mixed)

    stem = os.path.splitext(os.path.basename(args.in_split))[0]
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.in_split))
    for suffix, rows in (("easy", easy), ("mixed", mixed)):
        out = {**split, "train": rows}
        path = os.path.join(out_dir, f"{stem}_{suffix}.json")
        with open(path + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(path + ".tmp", path)
        print(f"[OK] {path}: train={len(rows)} val={len(split.get('val', []))} "
              f"test={len(split.get('test', []))}", flush=True)


if __name__ == "__main__":
    main()
