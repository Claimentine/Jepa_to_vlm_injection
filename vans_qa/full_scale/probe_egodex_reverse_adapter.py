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

Batch structure (verified live via --inspect_batch_only on 2026-10-05): a
13-tuple (xyz_cam, R_cam, xyz_world, R_world, tfs_in_cam, tfs, cam_ext,
cam_int, img, lang_instruct, confs, extras, paths); extras carries
vlm_old/vlm_new (+ _mask/_len), vjepa_input_feats/vjepa_target_feats
(B,16,256,1024, matching this project's own in_feats/out_feats convention
exactly), and vjepa_input/target_frame_indices (B,32) -- absolute frame numbers
from which the window-relative trajectory target positions are derived.
"""
import argparse
import json
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

import train_latent_world_model_full as rev  # noqa: E402
from egodex.trajectory_dataset import build_egodex_dataloaders, WRISTS  # noqa: E402

# compute_trajectory_loss_and_accuracy's exact formula (confirmed by reading
# cache_train/thinker_train.py directly):
#   err = torch.linalg.norm(pred - target, dim=-1)       # (B,T,J)
#   avg_dist (ADE) = err.mean(dim=(1,2))
#   final_dist (FDE) = err[:, -1, :].mean(dim=1)
#   acc = (err < thr).float().mean()
# Reimplemented inline below (ade_fde()) rather than importing the function
# itself, since its exact return type/loss_fn argument wiring for the
# dict-call-site wasn't independently confirmed -- the 4-line formula is
# simple enough to not need importing, and reusing the identical formula is
# what makes these numbers comparable to job-61's, not the call signature.

# WRISTS query gets silently ignored by the NPZ-cache path (use_npz_cache=True
# always returns the fixed 52-joint NPZ_TARGET_QUERY_TFS order regardless of
# query_tfs) -- confirmed by reading NpzCacheDataset's source, which has no
# query_tfs parameter at all. NPZ_TARGET_QUERY_TFS = RIGHT_FINGERS (20) +
# ["rightHand","rightForearm"] + LEFT_FINGERS (20) + ["leftHand","leftForearm"],
# so the two wrist joints sit at fixed indices 24 (rightHand) and 50 (leftHand).
WRIST_JOINT_IDX = [24, 50]
N_WRIST_JOINTS = len(WRIST_JOINT_IDX)

# extras['vjepa_input_frame_indices'] (past, 32) followed by
# ['vjepa_target_frame_indices'] (future, 32) are absolute source-video frame
# numbers sampled at a stride (~5), forming one 64-entry sequence that maps onto
# xyz_world's 64-frame axis. Future trajectory = xyz_world[:, 32:64]; the
# combined indices are checked to be strictly increasing in batch_to_samples().
FUTURE_FRAMES = slice(32, 64)

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
        num_workers=args.num_workers,
        pin_memory=False,
        load_cache_images=False,
        camera_mode="egodex",
    )


def load_frozen_predictor(label, device):
    path = CHECKPOINTS[label] if "=" not in label else label.split("=", 1)[1]
    predictor = rev.build_predictor(device, "crossattn")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # job15_independent's format (plain {"predictor_state": ...}), confirmed
    # by reading train_latent_world_model_full.py's own torch.save call sites
    # directly -- no bridge/ForwardingAdapter composition involved for this
    # checkpoint. job42/job49's best_reverse.pt format is NOT yet confirmed to
    # match -- only job15_independent is wired up so far.
    state = ckpt["predictor_state"] if "predictor_state" in ckpt else ckpt
    predictor.load_state_dict(state)
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad_(False)
    print(f"[INFO] loaded frozen predictor from {path} (step={ckpt.get('step')})", flush=True)
    return predictor


def predict_future_latent(predictor, in_feats, extras, device):
    """Re-implements forward_step()'s context-assembly + predictor call, but
    returns the raw predicted latent y_future instead of forward_step's own
    compute_predicted_latent_metrics() output -- we need the latent itself as
    the probe's input feature, not a loss against a target we don't have
    (EgoDex's "out_feats" slot is a placeholder since forward_step only
    reads it for masks_y's index range / concatenation shape, never its
    values -- x_ctxt is gathered purely from the first T_PER_CLIP positions,
    i.e. from in_feats; confirmed by reading forward_step's own source)."""
    B = in_feats.shape[0]
    dummy_out = torch.zeros_like(in_feats)
    feats_total = torch.cat([in_feats, dummy_out], dim=1)
    x_seq = rev.flatten_temporal_patch_tokens(feats_total)
    idx_ctx_1d = rev.build_temporal_patch_indices(rev.P_PATCHES, 0, rev.T_PER_CLIP)
    idx_tgt_1d = rev.build_temporal_patch_indices(rev.P_PATCHES, rev.T_PER_CLIP, 2 * rev.T_PER_CLIP)
    masks_x = rev.repeat_indices_for_batch(idx_ctx_1d.long(), B, device=x_seq.device)
    masks_y = rev.repeat_indices_for_batch(idx_tgt_1d.long(), B, device=x_seq.device)
    x_ctxt = x_seq.gather(dim=1, index=masks_x.unsqueeze(-1).expand(-1, -1, rev.D_EMBED))
    ext = rev.build_thinkjepa_guidance_inputs(extras=extras, args=rev.GUIDANCE_ARGS, device=device)
    y_future_seq = predictor(x_ctxt, masks_x, masks_y, ext=ext)
    y_future = y_future_seq.view(B, rev.T_PER_CLIP, rev.P_PATCHES, rev.D_EMBED)
    return y_future


class TrajectoryProbe(nn.Module):
    """Small trained head: pooled predicted future latent -> future WRIST xyz
    trajectory. Everything upstream of this (the predictor) stays frozen --
    this is the only part allowed to learn, so a good result means the
    FROZEN predictor's own latent space already carries real trajectory
    signal, not that we re-trained our way to a good answer."""

    def __init__(self, in_dim, hidden_dim, out_frames, out_joints):
        super().__init__()
        self.out_frames = out_frames
        self.out_joints = out_joints
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_frames * out_joints * 3),
        )

    def forward(self, pooled_feat):
        out = self.net(pooled_feat)
        return out.view(out.shape[0], self.out_frames, self.out_joints, 3)


def batch_to_samples(batch, device):
    """De-batches one EgoDex DataLoader batch into a list of per-sample dicts
    with everything this script needs, in the unbatched [L,S,D] convention
    build_thinkjepa_guidance_inputs/load_pair already use elsewhere in this
    project. Padding is stripped using the batch's own *_len fields rather
    than passing the mask through, since load_pair's own convention has no
    mask field at all -- slicing to the real length is the closer match."""
    (xyz_cam, R_cam, xyz_world, R_world, tfs_in_cam, tfs, cam_ext, cam_int,
     img, lang_instruct, confs, extras, paths) = batch
    B = xyz_world.shape[0]
    vjepa_in = extras["vjepa_input_feats"]
    vlm_old = extras["vlm_old"]
    vlm_new = extras["vlm_new"]
    vlm_old_len = extras["vlm_old_len"]
    vlm_new_len = extras["vlm_new_len"]
    input_frame_idx = extras["vjepa_input_frame_indices"]    # (B,32), absolute source-video frame numbers
    target_frame_idx = extras["vjepa_target_frame_indices"]  # (B,32), absolute source-video frame numbers

    samples = []
    for i in range(B):
        old_len = int(vlm_old_len[i])
        new_len = int(vlm_new_len[i])
        combined = torch.cat([input_frame_idx[i], target_frame_idx[i]])
        if not bool((combined[1:] > combined[:-1]).all()):
            raise ValueError(f"past+future frame indices are not strictly increasing: {combined.tolist()}")
        target = xyz_world[i, FUTURE_FRAMES][:, WRIST_JOINT_IDX, :]  # (32, 2, 3)
        samples.append({
            "in_feats": vjepa_in[i:i + 1].to(device),               # (1,16,256,1024)
            "vlm_old": vlm_old[i, :, :old_len, :].to(device),       # (L,S_real,2048)
            "vlm_new": vlm_new[i, :, :new_len, :].to(device),       # (L,S_real,2048)
            "target": target.to(device),                             # (32,2,3)
        })
    return samples


def ade_fde(pred, target, thr=0.05):
    """pred/target: (B,T,J,3). Returns per-batch-mean ADE, FDE, thr-accuracy,
    using the identical formula to ThinkJEPA's own
    compute_trajectory_loss_and_accuracy (see import-site comment above)."""
    err = torch.linalg.norm(pred - target, dim=-1)       # (B,T,J)
    avg_dist = err.mean(dim=(1, 2))                       # (B,)
    final_dist = err[:, -1, :].mean(dim=1)                # (B,)
    acc = (err < thr).float().mean()
    return float(avg_dist.mean().item()), float(final_dist.mean().item()), float(acc.item())


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


def run_probe(label, predictor, train_loader, test_loader, device, args):
    probe = TrajectoryProbe(
        in_dim=rev.D_EMBED, hidden_dim=args.probe_hidden_dim,
        out_frames=32, out_joints=N_WRIST_JOINTS,
    ).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr)

    def run_epoch(loader, train, max_batches=None):
        probe.train(train)
        total_loss, n_samples = 0.0, 0
        for bi, batch in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            samples = batch_to_samples(batch, device)
            preds, targets = [], []
            for s in samples:
                with torch.no_grad():
                    y_future = predict_future_latent(
                        predictor, s["in_feats"], {"vlm_old": s["vlm_old"], "vlm_new": s["vlm_new"]}, device,
                    )  # (1,16,256,1024), frozen -- no grad needed through the predictor itself
                pooled = y_future.mean(dim=(1, 2))  # (1,1024)
                pred_traj = probe(pooled)[0]        # (32,2,3)
                preds.append(pred_traj)
                targets.append(s["target"])
            pred_stack = torch.stack(preds)    # (B,32,2,3)
            target_stack = torch.stack(targets)
            loss = nn.functional.mse_loss(pred_stack, target_stack)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += float(loss.item()) * len(samples)
            n_samples += len(samples)
        return total_loss / max(n_samples, 1)

    for epoch in range(args.probe_epochs):
        train_loss = run_epoch(train_loader, train=True, max_batches=args.max_train_batches)
        print(f"[{label}] epoch {epoch+1}/{args.probe_epochs} probe_train_mse={train_loss:.6f}", flush=True)

    probe.eval()
    all_pred, all_target = [], []
    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            if args.max_test_batches and bi >= args.max_test_batches:
                break
            samples = batch_to_samples(batch, device)
            for s in samples:
                y_future = predict_future_latent(
                    predictor, s["in_feats"], {"vlm_old": s["vlm_old"], "vlm_new": s["vlm_new"]}, device,
                )
                pooled = y_future.mean(dim=(1, 2))
                pred_traj = probe(pooled)[0]
                all_pred.append(pred_traj)
                all_target.append(s["target"])
    pred_stack = torch.stack(all_pred)
    target_stack = torch.stack(all_target)
    ade, fde, acc = ade_fde(pred_stack.unsqueeze(0), target_stack.unsqueeze(0))
    # NOTE: ade_fde expects (B,T,J,3); stacking all test samples into one
    # pseudo-batch (unsqueeze(0)) and averaging over all of them at once
    # matches "mean over the whole test set" exactly as job-61's own
    # per-epoch test-set mean does (its batch-mean IS the test-set mean
    # whenever it evaluates in one pass), so these numbers are directly
    # comparable to job-61's ADE=0.0683/FDE=0.0733.
    print(f"[{label}] TEST n={len(all_pred)} ADE={ade:.6f} FDE={fde:.6f} acc@0.05={acc:.4f}", flush=True)
    return {"label": label, "n_test": len(all_pred), "ADE": ade, "FDE": fde, "acc_at_0.05": acc}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inspect_batch_only", action="store_true")
    ap.add_argument("--checkpoints", nargs="+", default=["job15_independent"],
                     help="one or more labels from CHECKPOINTS (or label=path)")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4,
                     help="DataLoader worker count -- matches job-61's own NUM_WORKERS=4 for the "
                          "real EgoDex training recipe; only affects per-batch iteration speed, "
                          "not the one-time dataset/manifest construction inside "
                          "build_egodex_dataloaders() itself")
    ap.add_argument("--probe_epochs", type=int, default=10)
    ap.add_argument("--probe_lr", type=float, default=1e-3)
    ap.add_argument("--probe_hidden_dim", type=int, default=512)
    ap.add_argument("--max_train_batches", type=int, default=None, help="smoke-test cap")
    ap.add_argument("--max_test_batches", type=int, default=None, help="smoke-test cap")
    ap.add_argument("--out_json", default=None)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for label in args.checkpoints:
        predictor = load_frozen_predictor(label, device)
        results.append(run_probe(label, predictor, train_loader, test_loader, device, args))
        del predictor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("[SUMMARY]", flush=True)
    for r in results:
        print(f"  {r['label']}: ADE={r['ADE']:.6f} FDE={r['FDE']:.6f} acc@0.05={r['acc_at_0.05']:.4f} n={r['n_test']}", flush=True)
    print("  job61_thinkjepa_official_recipe (reference): ADE=0.068294 FDE=0.073280", flush=True)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
