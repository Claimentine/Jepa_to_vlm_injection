"""
Stratified train/val/test split for the JEPA-injection classification
experiment. Unlike the zero-shot probe (which needed no split at all,
nothing was fit to the data), here we're training adapter parameters, so a
proper held-out test set is required.

Only keeps episodes that have BOTH:
  - the downloaded npz cache (for vjepa_feats)          -> data/cache/part2
  - the raw mp4 (fed live to the frozen VLM)             -> hf_staging/.../videos

Split is stratified per category (not the original ThinkJEPA global 90/10
split) so small categories like fold_unfold_paper_origami (23 episodes) get
a non-trivial, evenly-covered test slice instead of ~2 episodes.
"""
import argparse
import glob
import json
import os
import random

from task_categories import CATEGORIES

NPZ_ROOT = os.environ.get("EGODEX_NPZ_ROOT", "/projects/bhay/william/ruixin/data/cache/part2")
VIDEO_ROOT = os.environ.get(
    "EGODEX_VIDEO_ROOT",
    "/work/nvme/bdqf/william/charles/data/hf_staging/egodex_part2_video_cache_subset2000_ratio0.9_seed42/videos/egodex/part2",
)
OUT_DIR = os.environ.get("EGODEX_PROBE_OUT_DIR", os.path.dirname(os.path.abspath(__file__)))


def find_npz_for_episode(category, episode_id):
    matches = glob.glob(os.path.join(NPZ_ROOT, category, f"{episode_id}_*.npz"))
    return matches[0] if matches else None


def list_usable_episodes(category):
    mp4s = sorted(glob.glob(os.path.join(VIDEO_ROOT, category, "*.mp4")))
    out = []
    for mp4_path in mp4s:
        eid = os.path.splitext(os.path.basename(mp4_path))[0]
        npz_path = find_npz_for_episode(category, eid)
        if npz_path is not None:
            out.append({"category": category, "episode_id": eid, "mp4_path": mp4_path, "npz_path": npz_path})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    # test_frac is whatever remains
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "classification_split.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    split = {"train": [], "val": [], "test": []}

    for cat in CATEGORIES:
        eps = list_usable_episodes(cat)
        rng.shuffle(eps)
        n = len(eps)
        n_train = int(round(n * args.train_frac))
        n_val = int(round(n * args.val_frac))
        train, val, test = eps[:n_train], eps[n_train : n_train + n_val], eps[n_train + n_val :]
        split["train"] += train
        split["val"] += val
        split["test"] += test
        print(f"[OK] {cat:55s} usable={n:4d}  train={len(train):3d} val={len(val):3d} test={len(test):3d}")

    for part in ("train", "val", "test"):
        rng.shuffle(split[part])

    with open(args.out, "w") as f:
        json.dump(split, f, indent=2)
    print(f"\nWrote train={len(split['train'])} val={len(split['val'])} test={len(split['test'])} -> {args.out}")


if __name__ == "__main__":
    main()
