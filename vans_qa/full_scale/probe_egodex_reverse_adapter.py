"""Stage 2 of the EgoDex plan: does OUR reverse-direction predictor (never
trained on EgoDex -- only VANS/COIN/Ego4D self-supervised pairs) already
carry useful trajectory signal, without any EgoDex-specific training?

Freezes a given checkpoint's CortexGuidedVideoPredictor entirely. Trains
ONLY a small linear probe (predicted future latent -> 3D joint positions)
on EgoDex's train split, then reports ADE/FDE on the held-out test split
using ThinkJEPA's own compute_trajectory_loss_and_accuracy -- same formula,
same units as job-61's own reported numbers (ADE=0.0683, FDE=0.0733 at
epoch 20, their own officially-trained-on-EgoDex reference point), so the
two are directly comparable: "how good is our predictor's latent space as
a feature extractor for a real downstream task it never saw," vs. "how
good is a model actually trained end-to-end on that task."

Run with --inspect_batch_only first (no checkpoint, no GPU needed) to
print the real batch structure build_egodex_dataloaders returns before
trusting the field indices this script assumes -- the NpzCacheDataset
docstring's own stated tuple order was not independently verified against
a live batch before this script was written.
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

THINKJEPA_ROOT = os.environ.get("THINKJEPA_ROOT", "/home/jovyan/ThinkJEPA")
if THINKJEPA_ROOT not in sys.path:
    sys.path.insert(0, THINKJEPA_ROOT)

from common.jepa_injection_model import resolve_decoder_layers  # noqa: E402,F401 (kept for parity w/ eval_forward_checkpoint.py imports)
from common.cross_modal_bridge import ForwardingAdapter  # noqa: E402
import train_latent_world_model_full as rev  # noqa: E402
from egodex.trajectory_dataset import build_egodex_dataloaders, WRISTS  # noqa: E402
from cache_train.thinker_train import compute_trajectory_loss_and_accuracy  # noqa: E402

BASE = os.environ.get("VANS_ROOT", "/data")
EGODEX_BUNDLE = os.environ.get("EGODEX_BUNDLE", "/data/raw_data/thinkjepa_egodex_iso")

CHECKPOINTS = {
    "job15_independent": "/data/raw_data/latent_world_model_runs/crossattn/best.pt",
    "job42_bridge_mlp2048": "/data/raw_data/joint_coupled_bridge_mlp_runs/best_reverse.pt",
    "job49_coin_mix": "/data/raw_data/joint_coupled_mixed_normalized_nobridge_runs/best_reverse.pt",
}


def build_dataloaders(args):
    # dataset_path must be the supervision_hdf5 directory itself -- matches
    # thinker_train.py's own call (its first positional arg is args.data_dir,
    # which scripts/train.sh sets to "${BUNDLE}/supervision_hdf5"). There is
    # no separate supervision_cache_root override in the real training call;
    # build_egodex_dataloaders derives supervision_root from dataset_path
    # itself via _expand_roots_for_sidecar_lookup when use_npz_cache=True,
    # so dataset_path has to BE that root, not the bundle's own top level
    # (confirmed the hard way: passing the bundle root produced
    # "schema-v2 join requires source_video_relpath and explicit
    # supervision_root" / "cam_ext is None" on every sample).
    return build_egodex_dataloaders(
        dataset_path=os.path.join(EGODEX_BUNDLE, "supervision_hdf5"),
        query_tfs=WRISTS,
        if_return_path=True,
        train_batch=args.batch_size,
        test_batch=args.batch_size,
        train_manifest=os.path.join(EGODEX_BUNDLE, "manifests/portable_v1/train_cache.txt"),
        test_manifest=os.path.join(EGODEX_BUNDLE, "manifests/portable_v1/test_cache.txt"),
        use_npz_cache=True,
        cache_dir=os.path.join(EGODEX_BUNDLE, "cache"),
        num_workers=0,
        pin_memory=False,
        load_cache_images=False,
        camera_mode="egodex",
    )


def describe(obj, name, depth=0):
    pad = "  " * depth
    if torch.is_tensor(obj):
        print(f"{pad}{name}: Tensor shape={tuple(obj.shape)} dtype={obj.dtype}", flush=True)
    elif isinstance(obj, dict):
        print(f"{pad}{name}: dict keys={list(obj.keys())}", flush=True)
        for k, v in obj.items():
            describe(v, f"[{k!r}]", depth + 1)
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{name}: {type(obj).__name__} len={len(obj)}", flush=True)
        for i, v in enumerate(obj):
            describe(v, f"[{i}]", depth + 1)
    else:
        print(f"{pad}{name}: {type(obj).__name__} = {str(obj)[:120]}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inspect_batch_only", action="store_true")
    ap.add_argument("--checkpoint", default=None, help="label (see CHECKPOINTS) or label=path")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--probe_epochs", type=int, default=10)
    ap.add_argument("--probe_lr", type=float, default=1e-3)
    args = ap.parse_args()

    train_loader, test_loader = build_dataloaders(args)
    print(f"[INFO] train_loader batches~{len(train_loader)} test_loader batches~{len(test_loader)}", flush=True)
    ds = train_loader.dataset
    for attr in ("query_tfs", "joint_names", "tfs_names", "query_joint_names", "camera_mode"):
        if hasattr(ds, attr):
            print(f"[INFO] dataset.{attr} = {getattr(ds, attr)}", flush=True)
    print(f"[INFO] WRISTS constant = {WRISTS}", flush=True)

    batch = next(iter(train_loader))
    print("[INFO] first train batch structure:", flush=True)
    describe(batch, "batch")

    if args.inspect_batch_only:
        print("[DONE] inspect_batch_only -- stopping here", flush=True)
        return

    raise NotImplementedError(
        "probe training not wired up yet -- run --inspect_batch_only first, "
        "confirm the structure printed above matches what this script assumes, "
        "then implement the probe loop against the REAL field layout"
    )


if __name__ == "__main__":
    main()
