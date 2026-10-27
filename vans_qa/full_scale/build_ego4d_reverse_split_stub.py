"""Builds a qa_split-shaped stub file ({"train"/"val"/"test": [{"pid": ...}, ...]})
listing every pid cached by extract_ego4d_reverse_guided.py (job-109), all
under "train" -- per explicit decision (2026-10-27): the new Ego4D reverse
pairs exist to grow the reverse training pool, not to add a second held-out
set (EgoDex's own test split and the dedicated EgoDex probe jobs, e.g.
job-113/116, already cover reverse-direction evaluation).

Only the "pid" field is populated -- load_qa_split_membership() (train_
latent_world_model_full.py) only ever reads item["pid"], and merge_qa_
splits.py only reads item["pid"] for its cross-file dupe check, so no other
fields are needed for this file's one purpose: extending --rev_qa_split's
train/val/test membership to cover these new pids.

The actual .npz files stay in their own directory (extract_ego4d_reverse_
guided.py's --cache_dir) -- this script only lists pids, it does not move
or symlink any cache files (see the job that calls this for that step).
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache_dir", required=True, help="dir of .npz files, e.g. job-109's cache_narration")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pids = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(args.cache_dir, "*.npz"))
        if not os.path.basename(p).startswith(".")
    )
    stub = {"train": [{"pid": pid} for pid in pids], "val": [], "test": []}
    with open(args.out, "w") as f:
        json.dump(stub, f, indent=2)
    print(f"[DONE] {len(pids)} pid(s) from {args.cache_dir} -> {args.out} (all under 'train')", flush=True)


if __name__ == "__main__":
    main()
