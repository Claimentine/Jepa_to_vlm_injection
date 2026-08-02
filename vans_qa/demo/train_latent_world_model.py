"""
VLM-guided JEPA latent-prediction training on VANS demo (in, out) pairs --
i.e. the original ThinkJEPA direction (VLM guides the JEPA predictor), but
with the trajectory-readout head removed entirely and the training target
switched to V-JEPA2's own native self-supervised objective: predict the
future clip's patch-token latents from the past clip's latents + VLM
guidance, scored by MSE / L2 / cosine distance against the *actual* encoded
future latents. This is required because VANS (COIN/YouCook2) has no
hand-skeleton ground truth the way EgoDex does, so the stock
thinker_train.py trajectory path (which always requires xyz_world/xyz_cam)
cannot run on it unmodified.

Reuses, via direct import, the exact tested building blocks from
cache_train.thinker_train / cache_train.thinker_predictor:
  - CortexGuidedVideoPredictor      (unmodified)
  - compute_predicted_latent_metrics (unmodified)
  - flatten_temporal_patch_tokens / build_temporal_patch_indices /
    repeat_indices_for_batch         (unmodified)
  - build_thinkjepa_guidance_inputs  (unmodified)

Data: extract_vjepa_features.py -> vjepa_cache/{pid}.npz (in_feats, out_feats)
      extract_vlm_guidance.py   -> vlm_guidance_cache/{pid}_*.npz (vlm_old, vlm_new)
"""
import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(os.environ.get("THINKJEPA_ROOT", "/projects/bhay/william/ruixin/ThinkJEPA"))
for _p in (REPO_ROOT, REPO_ROOT / "cache_train", REPO_ROOT / "vjepa2", REPO_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from cache_train.thinker_train import (
    compute_predicted_latent_metrics,
    flatten_temporal_patch_tokens,
    build_temporal_patch_indices,
    repeat_indices_for_batch,
    build_thinkjepa_guidance_inputs,
)
from cache_train.thinker_predictor import CortexGuidedVideoPredictor

VANS_ROOT = os.environ.get("VANS_ROOT", "/projects/bhay/william/ruixin/vans_world_model")
VJEPA_CACHE = os.path.join(VANS_ROOT, "vjepa_cache")
VLM_CACHE = os.path.join(VANS_ROOT, "vlm_guidance_cache")
OUT_DIR = os.path.join(VANS_ROOT, "runs")

P_PATCHES = 128       # vjepa_feats patch-tokens per frame
D_EMBED = 1024        # vjepa_feats embed dim
T_PER_CLIP = 64        # frames per clip (in and out each)
VLM_OLD_DIM = 2048
VLM_NEW_DIM = 2048

GUIDANCE_ARGS = SimpleNamespace(
    thinkjepa_use_vlm_merge=True,
    thinkjepa_use_cache_ext=True,
    thinkjepa_vlm_source="both",
    thinkjepa_vlm_layer_selector="last",
    thinkjepa_vlm_cond_mode="film",
)


def find_vlm_npz(pid):
    matches = glob.glob(os.path.join(VLM_CACHE, f"{pid}_*.npz"))
    return matches[0] if matches else None


def list_available_pairs():
    vjepa_ids = {
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(VJEPA_CACHE, "*.npz"))
    }
    pairs = []
    for pid in sorted(vjepa_ids):
        if find_vlm_npz(pid) is not None:
            pairs.append(pid)
    return pairs


def load_pair(pid, device):
    d = np.load(os.path.join(VJEPA_CACHE, f"{pid}.npz"))
    in_feats = torch.from_numpy(d["in_feats"].astype(np.float32)).unsqueeze(0).to(device)   # (1,64,128,1024)
    out_feats = torch.from_numpy(d["out_feats"].astype(np.float32)).unsqueeze(0).to(device)  # (1,64,128,1024)

    vlm_path = find_vlm_npz(pid)
    v = np.load(vlm_path)
    extras = {
        "vlm_old": torch.from_numpy(v["vlm_old"].astype(np.float32)).to(device),
        "vlm_new": torch.from_numpy(v["vlm_new"].astype(np.float32)).to(device),
    }
    return in_feats, out_feats, extras


def build_predictor(device):
    total_frames = 2 * T_PER_CLIP
    predictor = CortexGuidedVideoPredictor(
        img_size=(P_PATCHES, 1),
        patch_size=1,
        num_frames=total_frames,
        tubelet_size=1,
        embed_dim=D_EMBED,
        predictor_embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_std=0.02,
        uniform_power=False,
        use_mask_tokens=True,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
        return_all_tokens=False,
        chop_last_n_tokens=0,
        use_rope=True,
        use_vlm_merge=True,
        vlm_cond_mode="film",
        vlm_old_dim=VLM_OLD_DIM,
        vlm_new_dim=VLM_NEW_DIM,
    ).to(device)
    return predictor


def forward_step(predictor, in_feats, out_feats, extras, device):
    B = in_feats.shape[0]
    feats_total = torch.cat([in_feats, out_feats], dim=1)  # (B, 128, P, D)
    x_seq = flatten_temporal_patch_tokens(feats_total)      # (B, 128*P, D)

    idx_ctx_1d = build_temporal_patch_indices(P_PATCHES, 0, T_PER_CLIP)
    idx_tgt_1d = build_temporal_patch_indices(P_PATCHES, T_PER_CLIP, 2 * T_PER_CLIP)
    masks_x = repeat_indices_for_batch(idx_ctx_1d.long(), B, device=x_seq.device)
    masks_y = repeat_indices_for_batch(idx_tgt_1d.long(), B, device=x_seq.device)

    x_ctxt = x_seq.gather(dim=1, index=masks_x.unsqueeze(-1).expand(-1, -1, D_EMBED))
    ext = build_thinkjepa_guidance_inputs(extras=extras, args=GUIDANCE_ARGS, device=device)

    y_future_seq = predictor(x_ctxt, masks_x, masks_y, ext=ext)  # (B, T_tgt*P, D)
    y_future = y_future_seq.view(B, T_PER_CLIP, P_PATCHES, D_EMBED)

    metrics = compute_predicted_latent_metrics(y_future, out_feats)
    return metrics


def evaluate(predictor, pair_ids, device):
    predictor.eval()
    totals = {"pred_loss": 0.0, "pred_latent_dist": 0.0, "pred_latent_cosine_distance": 0.0}
    n = 0
    with torch.no_grad():
        for pid in pair_ids:
            in_feats, out_feats, extras = load_pair(pid, device)
            m = forward_step(predictor, in_feats, out_feats, extras, device)
            for k in totals:
                totals[k] += float(m[k].item())
            n += 1
    predictor.train()
    return {k: v / max(n, 1) for k, v in totals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="cap total pairs used (smoke test)")
    ap.add_argument("--run_name", default="vans_demo_latent_world_model")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(OUT_DIR, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    pairs = list_available_pairs()
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"[INFO] {len(pairs)} pairs with both vjepa + vlm guidance features available")
    if len(pairs) < 5:
        raise RuntimeError("too few pairs with both feature caches present -- run extraction first")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_frac))
    n_test = max(1, int(len(pairs) * args.test_frac))
    test_ids = pairs[:n_test]
    val_ids = pairs[n_test : n_test + n_val]
    train_ids = pairs[n_test + n_val :]
    print(f"[INFO] split: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    predictor = build_predictor(device)
    n_params = sum(p.numel() for p in predictor.parameters())
    print(f"[INFO] predictor params: {n_params:,}")
    opt = torch.optim.AdamW(predictor.parameters(), lr=args.lr)

    log_path = os.path.join(run_dir, "train_log.jsonl")
    log_f = open(log_path, "w")
    best_val = float("inf")

    for epoch in range(args.epochs):
        rng.shuffle(train_ids)
        epoch_loss = 0.0
        for pid in train_ids:
            in_feats, out_feats, extras = load_pair(pid, device)
            metrics = forward_step(predictor, in_feats, out_feats, extras, device)
            loss = metrics["pred_loss"]

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())

        val_metrics = evaluate(predictor, val_ids, device)
        train_avg = epoch_loss / max(len(train_ids), 1)
        print(
            f"[epoch {epoch}] train_mse={train_avg:.4f}  "
            f"val_mse={val_metrics['pred_loss']:.4f}  "
            f"val_latent_dist={val_metrics['pred_latent_dist']:.4f}  "
            f"val_cosine_dist={val_metrics['pred_latent_cosine_distance']:.4f}"
        )
        log_f.write(json.dumps({"epoch": epoch, "train_mse": train_avg, **{f"val_{k}": v for k, v in val_metrics.items()}}) + "\n")
        log_f.flush()

        if val_metrics["pred_loss"] < best_val:
            best_val = val_metrics["pred_loss"]
            torch.save(predictor.state_dict(), os.path.join(run_dir, "best.pt"))

    log_f.close()

    predictor.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device))
    test_metrics = evaluate(predictor, test_ids, device)
    print(f"[TEST] {test_metrics}")
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
