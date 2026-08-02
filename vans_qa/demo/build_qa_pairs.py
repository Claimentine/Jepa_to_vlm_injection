"""
Builds a paired binary-choice "what happens next" QA set from the VANS
demo's own bundled VANS_DATA_EXAMPLE.csv (matches Instruction/GT_Caption
text to the exact 500 in/out clip pairs, no COIN/YouCook2 download needed).

Design (see conversation): with only ~500 items, open-ended
generation+judging is too noisy to say anything reliable. A VLEP-style
paired 2-choice task (true ENG_GT_Caption vs a distractor caption borrowed
from a different item) is far more sample-efficient -- same items get
scored under jepa / random / none, so per-item difficulty cancels out in a
paired comparison instead of adding to between-condition noise.

This is explicitly a *directional pilot*, not a scaled result: see
README_pilot_caveats.md written alongside the split for the framing to keep
when reporting numbers from this.
"""
import argparse
import csv
import json
import os
import random

VANS_ROOT = os.environ.get("VANS_ROOT", "/projects/bhay/william/ruixin/vans_world_model")
CSV_PATH = os.path.join(VANS_ROOT, "hf_data/demo_sample/VANS_DATA_EXAMPLE.csv")
VJEPA_CACHE = os.path.join(VANS_ROOT, "vjepa_cache")
VLM_CACHE = os.path.join(VANS_ROOT, "vlm_guidance_cache")
OUT_PATH = os.path.join(VANS_ROOT, "qa_split.json")


def load_rows():
    with open(CSV_PATH, encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            pid = r["input_video_path"].replace("_in.mp4", "").strip()
            q = (r.get("ENG_Instruction") or "").strip()
            a = (r.get("ENG_GT_Caption") or "").strip()
            if not pid or not q or not a:
                continue
            rows.append({"pid": pid, "question": q, "answer": a})
    return rows


def has_features(pid):
    npz = os.path.join(VJEPA_CACHE, f"{pid}.npz")
    vlm_matches = [f for f in os.listdir(VLM_CACHE) if f.startswith(f"{pid}_") and f.endswith(".npz")]
    return os.path.exists(npz) and len(vlm_matches) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = load_rows()
    rows = [r for r in rows if has_features(r["pid"])]
    print(f"[INFO] {len(rows)} items with both QA text and precomputed features")

    # distractor: another item's true answer, sampled without replacement
    # against itself; simple random pick is fine for a first pilot (no
    # semantic near-duplicate filtering -- noted as a caveat, not a bug).
    all_answers = [r["answer"] for r in rows]
    items = []
    for i, r in enumerate(rows):
        distractor_pool = [a for j, a in enumerate(all_answers) if j != i]
        distractor = rng.choice(distractor_pool)
        items.append({
            "pid": r["pid"],
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

    caveats_path = os.path.join(os.path.dirname(args.out), "README_pilot_caveats.md")
    with open(caveats_path, "w") as f:
        f.write(
            "# VANS QA pilot -- read before reporting numbers\n\n"
            f"- N={n} items total (test set ~{len(split['test'])}). This is a "
            "directional pilot on the VANS demo bundle, not a powered study.\n"
            "- Distractors are unfiltered random picks from other items' true "
            "answers -- occasionally too easy or (rarely) too plausible; not "
            "curated.\n"
            "- Report test accuracy with a Wilson confidence interval, and "
            "prefer the *paired* jepa-vs-random comparison (same test items) "
            "over comparing independent point estimates.\n"
            "- Treat agreement/disagreement in *direction* with the EgoDex "
            "classification result as the main signal this pilot can support, "
            "not the exact accuracy numbers.\n"
        )


if __name__ == "__main__":
    main()
