"""
Stratified sampling of episodes for the "task recognition already known"
probe (see conversation design: zero-shot Qwen3-VL-Thinking, no training,
just checking whether it already knows the task from raw video).

No train/test-split concern here: nothing is fit to this data, it's a pure
frozen-model eval, so we sample from the full local mirror rather than
restricting to the held-out trajectory-eval split (some categories, e.g.
fold_unfold_paper_origami, only have 23 episodes total -- restricting to the
~10% test slice would leave a handful of episodes and a noisy macro-accuracy
estimate).

Writes a JSON manifest: [{category, episode_id, hdf5_path, mp4_path}, ...]
"""
import argparse
import glob
import json
import os
import random

from task_categories import CATEGORIES

DATA_ROOT = os.environ.get(
    "EGODEX_DATA_ROOT",
    "/work/nvme/bdqf/william/charles/data/hf_staging/egodex_part2_video_cache_subset2000_ratio0.9_seed42",
)
HDF5_ROOT = os.path.join(DATA_ROOT, "hdf5", "egodex", "part2")
VIDEO_ROOT = os.path.join(DATA_ROOT, "videos", "egodex", "part2")
OUT_DIR = os.environ.get("EGODEX_PROBE_OUT_DIR", os.path.dirname(os.path.abspath(__file__)))


def list_episode_ids(category):
    hdf5_files = sorted(glob.glob(os.path.join(HDF5_ROOT, category, "*.hdf5")))
    ids = [os.path.splitext(os.path.basename(p))[0] for p in hdf5_files]
    # keep only ids that also have a matching mp4
    ids = [i for i in ids if os.path.exists(os.path.join(VIDEO_ROOT, category, f"{i}.mp4"))]
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k_per_category", type=int, default=20,
                    help="Stratified sample size per category (uses all available if fewer).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=os.path.join(OUT_DIR, "probe_manifest.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    manifest = []
    for cat in CATEGORIES:
        ids = list_episode_ids(cat)
        if not ids:
            print(f"[WARN] no usable episodes for category={cat}")
            continue
        k = min(args.k_per_category, len(ids))
        chosen = rng.sample(ids, k)
        for eid in chosen:
            manifest.append({
                "category": cat,
                "episode_id": eid,
                "hdf5_path": os.path.join(HDF5_ROOT, cat, f"{eid}.hdf5"),
                "mp4_path": os.path.join(VIDEO_ROOT, cat, f"{eid}.mp4"),
            })
        print(f"[OK] {cat:55s} available={len(ids):4d}  sampled={k}")

    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(manifest)} episodes -> {args.out}")


if __name__ == "__main__":
    main()
