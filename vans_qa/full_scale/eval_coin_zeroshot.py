"""Zero-shot baseline for the COIN-mixed-training ablation (see
train_latent_world_model_mixed.py's module docstring): evaluates an existing
VANS-only-trained checkpoint (job-15's best.pt) on COIN's held-out long-axis
test split, WITHOUT any further training. This is the "before" number the
mixed-training run's own coin_test metric needs to be compared against to
answer the actual question -- does training on COIN's long-horizon pairs
improve the model's future-latent prediction there, on top of what VANS-only
training already gets for free (if anything)?

Uses the exact same seeded COIN train/val/test split as
train_latent_world_model_mixed.py (same --seed default, same split
fractions) so this script's test set is identical to that run's coin_test --
otherwise the two numbers wouldn't be comparable.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_latent_world_model_full as base  # noqa: E402
from train_latent_world_model_mixed import (  # noqa: E402
    list_coin_ids, split_coin_ids, evaluate_source,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="a VANS-only best.pt/step_*.pt to evaluate")
    ap.add_argument("--coin_cache_dir", default=os.path.join(base.BASE, "raw_data/coin_longaxis"))
    ap.add_argument("--coin_val_frac", type=float, default=0.05)
    ap.add_argument("--coin_test_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_path", default=None, help="optional path to dump result JSON")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coin_ids = list_coin_ids(args.coin_cache_dir)
    _, _, coin_test = split_coin_ids(coin_ids, args.coin_val_frac, args.coin_test_frac, args.seed)
    print(f"[INFO] COIN test split: n={len(coin_test)} (of {len(coin_ids)} total, seed={args.seed})", flush=True)

    predictor = base.build_predictor(device, "crossattn")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt["predictor_state"])
    print(f"[INFO] loaded {args.checkpoint} (step={ckpt.get('step')})", flush=True)

    metrics, n = evaluate_source(predictor, args.coin_cache_dir, coin_test, "coin", device)
    print(f"[RESULT] zero-shot VANS-only checkpoint on COIN test: n={n} {metrics}", flush=True)

    if args.out_path:
        with open(args.out_path, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "n_test": n, **metrics}, f, indent=2)


if __name__ == "__main__":
    main()
