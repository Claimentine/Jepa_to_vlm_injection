"""First experiment wiring COIN's long-axis data (extract_coin_longaxis.py,
validated end to end by job-23/24/25 -- 2256 saved pairs so far) into actual
training, per the extraction script's own deferred docstring: "using it ...
requires guidance='none' for these items specifically. Wiring that up is a
follow-up step once this extraction itself is validated."

The motivating question: VANS's own reverse-direction pairs (used by job-15,
extract_vlm_guidance_paired.py's output) have a short past/future span (median
~3-4 clip-index gap). COIN's long-axis pairs are genuinely longer-horizon
(median 71s, p90 154s between first and last annotated step) but carry no VLM
guidance (no vlm_old/vlm_new -- COIN's step labels are short controlled-
vocabulary text, not scene captions, and extracting real VLM guidance for them
would need a whole separate Qwen3-VL pass this project hasn't built). Does
mixing COIN's long-horizon self-supervised examples into training help
CortexGuidedVideoPredictor generalize to longer-horizon future-latent
prediction, on top of (not instead of) VANS's guided short-horizon pairs?

Key finding this design relies on (confirmed 2026-09-13 by reading
cache_train/thinker_predictor.py directly on the cluster): a predictor built
with use_vlm_merge=True (guidance="crossattn") does NOT require guidance on
every forward call -- _build_layerwise_guidance returns None immediately when
ext is None, and _inject_guidance is a no-op whenever the guidance bank is
None (falls through to `return rollout_stream` unchanged). So a SINGLE
predictor instance can be trained on a mixed batch stream: VANS items get a
real ext built from vlm_old/vlm_new, COIN items get ext=None and fall back to
pure self-supervised prediction for that step -- no architecture change,
no second predictor, no dummy/zero guidance tensors needed.

Also confirmed (same date, inspecting an actual saved .npz): COIN's
vjepa_input_feats/vjepa_target_feats are (16, 256, 1024) float16 -- exactly
T_PER_CLIP/P_PATCHES/D_EMBED, because extract_coin_longaxis.py reuses the
identical encode_vjepa_clip() call VANS's own extraction uses. No shape
adapter needed.

Reuses train_latent_world_model_full.py's build_predictor/load_pair/
GUIDANCE_ARGS unmodified (imported, not copied) -- that script itself stays
the single-source-of-truth VANS-only baseline (job-15) this experiment is
compared against. Only forward_step is duplicated-with-a-change (forward_step
here decides ext from whether `extras` is None, not from a global
--guidance string, since a single run now needs per-item guidance) --
everything else (loss/metrics computation, predictor architecture) is the
exact same ThinkJEPA code both scripts call into.

COIN's own 2256 saved items are split train/val/test locally (90/5/5, seeded)
since qa_split_full.json has no knowledge of COIN video_ids. VANS's
train/val/test membership is inherited from --qa_split exactly as
train_latent_world_model_full.py does, so this run's VANS-side numbers stay
directly comparable to job-15's.

Validation/test report BOTH sources' metrics separately (never blended into
one number) -- the real test of the hypothesis is whether COIN-mixed training
changes the COIN-test metric (a number job-15's VANS-only baseline can also be
evaluated against zero-shot, for the actual ablation), not whether it changes
the VANS-test metric, which is the existing job-15 comparison.
"""
import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_latent_world_model_full as base  # noqa: E402

T_PER_CLIP = base.T_PER_CLIP
P_PATCHES = base.P_PATCHES
D_EMBED = base.D_EMBED
GUIDANCE_ARGS = base.GUIDANCE_ARGS

from cache_train.thinker_train import (  # noqa: E402
    compute_predicted_latent_metrics,
    flatten_temporal_patch_tokens,
    build_temporal_patch_indices,
    repeat_indices_for_batch,
    build_thinkjepa_guidance_inputs,
)


def load_coin_pair(cache_dir, video_id, device):
    d = np.load(os.path.join(cache_dir, f"{video_id}.npz"))
    in_feats = torch.from_numpy(d["vjepa_input_feats"].astype(np.float32)).unsqueeze(0).to(device)
    out_feats = torch.from_numpy(d["vjepa_target_feats"].astype(np.float32)).unsqueeze(0).to(device)
    return in_feats, out_feats, None  # no vlm_old/vlm_new -- ext stays None for this item


def list_coin_ids(cache_dir):
    return sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(cache_dir, "*.npz"))
        if not os.path.basename(p).startswith(".")
    )


def split_coin_ids(ids, val_frac, test_frac, seed):
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_val = max(1, int(n * val_frac))
    n_test = max(1, int(n * test_frac))
    test_ids = ids[:n_test]
    val_ids = ids[n_test:n_test + n_val]
    train_ids = ids[n_test + n_val:]
    return train_ids, val_ids, test_ids


def load_item(source, cache_dir, item_id, device):
    if source == "coin":
        return load_coin_pair(cache_dir, item_id, device)
    return base.load_pair(cache_dir, item_id, device)


def forward_step_mixed(predictor, in_feats, out_feats, extras, device):
    """Same computation as train_latent_world_model_full.forward_step, except
    ext is decided from whether `extras` is present (per-item), not from a
    global --guidance string (a single mixed run needs both per step)."""
    B = in_feats.shape[0]
    feats_total = torch.cat([in_feats, out_feats], dim=1)
    x_seq = flatten_temporal_patch_tokens(feats_total)

    idx_ctx_1d = build_temporal_patch_indices(P_PATCHES, 0, T_PER_CLIP)
    idx_tgt_1d = build_temporal_patch_indices(P_PATCHES, T_PER_CLIP, 2 * T_PER_CLIP)
    masks_x = repeat_indices_for_batch(idx_ctx_1d.long(), B, device=x_seq.device)
    masks_y = repeat_indices_for_batch(idx_tgt_1d.long(), B, device=x_seq.device)

    x_ctxt = x_seq.gather(dim=1, index=masks_x.unsqueeze(-1).expand(-1, -1, D_EMBED))
    ext = build_thinkjepa_guidance_inputs(extras=extras, args=GUIDANCE_ARGS, device=device) if extras is not None else None

    y_future_seq = predictor(x_ctxt, masks_x, masks_y, ext=ext)
    y_future = y_future_seq.view(B, T_PER_CLIP, P_PATCHES, D_EMBED)
    return compute_predicted_latent_metrics(y_future, out_feats)


def evaluate_source(predictor, cache_dir, ids, source, device, max_items=None):
    predictor.eval()
    totals = {"pred_loss": 0.0, "pred_latent_dist": 0.0, "pred_latent_cosine_distance": 0.0}
    n = 0
    use_ids = ids[:max_items] if max_items else ids
    with torch.no_grad():
        for item_id in use_ids:
            in_feats, out_feats, extras = load_item(source, cache_dir, item_id, device)
            m = forward_step_mixed(predictor, in_feats, out_feats, extras, device)
            for k in totals:
                totals[k] += float(m[k].item())
            n += 1
    predictor.train()
    return {k: v / max(n, 1) for k, v in totals.items()}, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vans_cache_dir", default=os.path.join(base.WORK_BASE, "vlm_guidance_cache_paired"))
    ap.add_argument("--coin_cache_dir", default=os.path.join(base.BASE, "raw_data/coin_longaxis"))
    ap.add_argument("--qa_split", default=os.path.join(base.BASE, "raw_data/qa_split_full.json"))
    ap.add_argument("--coin_val_frac", type=float, default=0.05)
    ap.add_argument("--coin_test_frac", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_every_steps", type=int, default=500)
    ap.add_argument("--save_every_steps", type=int, default=1000)
    ap.add_argument("--max_val_items", type=int, default=200)
    ap.add_argument("--max_train_items", type=int, default=None,
                     help="smoke-test cap, applied to the VANS and COIN train pools independently")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=os.path.join(base.BASE, "raw_data/latent_world_model_runs_mixed_coin"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    membership = base.load_qa_split_membership(args.qa_split)
    vans_pids = base.list_cached_pids(args.vans_cache_dir)
    vans_by_split = {"train": [], "val": [], "test": []}
    n_unmatched = 0
    for pid in vans_pids:
        part = membership.get(pid)
        if part in vans_by_split:
            vans_by_split[part].append(pid)
        else:
            n_unmatched += 1
    if n_unmatched:
        print(f"[WARN] {n_unmatched} cached VANS pids not found in {args.qa_split}, skipped", flush=True)

    coin_ids = list_coin_ids(args.coin_cache_dir)
    coin_train, coin_val, coin_test = split_coin_ids(coin_ids, args.coin_val_frac, args.coin_test_frac, args.seed)

    rng = random.Random(args.seed)
    for part in vans_by_split.values():
        rng.shuffle(part)
    vans_train, vans_val, vans_test = vans_by_split["train"], vans_by_split["val"], vans_by_split["test"]

    if args.max_train_items:
        vans_train = vans_train[: args.max_train_items]
        coin_train = coin_train[: args.max_train_items]

    print(f"[INFO] VANS split: train={len(vans_train)} val={len(vans_val)} test={len(vans_test)}", flush=True)
    print(f"[INFO] COIN split: train={len(coin_train)} val={len(coin_val)} test={len(coin_test)} "
          f"(of {len(coin_ids)} total)", flush=True)
    if len(vans_train) < 5:
        raise RuntimeError("too few cached VANS train pairs -- run extract_vlm_guidance_paired.py first")
    if len(coin_train) < 5:
        raise RuntimeError("too few cached COIN train pairs -- run extract_coin_longaxis.py first")

    predictor = base.build_predictor(device, "crossattn")
    n_params = sum(p.numel() for p in predictor.parameters())
    print(f"[INFO] mixed VANS+COIN training, predictor params: {n_params:,}", flush=True)
    opt = torch.optim.AdamW(predictor.parameters(), lr=args.lr)

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    log_f = open(log_path, "a")
    best_val_vans = float("inf")
    step = 0

    for epoch in range(args.epochs):
        combined = [("vans", pid) for pid in vans_train] + [("coin", vid) for vid in coin_train]
        rng.shuffle(combined)
        for source, item_id in combined:
            step += 1
            cache_dir = args.vans_cache_dir if source == "vans" else args.coin_cache_dir
            in_feats, out_feats, extras = load_item(source, cache_dir, item_id, device)
            metrics = forward_step_mixed(predictor, in_feats, out_feats, extras, device)
            loss = metrics["pred_loss"]

            opt.zero_grad()
            loss.backward()
            opt.step()

            log_f.write(json.dumps({"step": step, "epoch": epoch, "source": source, "train_loss": float(loss.item())}) + "\n")

            if step % args.val_every_steps == 0:
                vans_val_m, n_vv = evaluate_source(predictor, args.vans_cache_dir, vans_val, "vans", device, args.max_val_items)
                coin_val_m, n_cv = evaluate_source(predictor, args.coin_cache_dir, coin_val, "coin", device, args.max_val_items)
                print(
                    f"[step {step}] train_loss={loss.item():.4f} src={source}  "
                    f"vans_val_mse={vans_val_m['pred_loss']:.4f} (n={n_vv})  "
                    f"coin_val_mse={coin_val_m['pred_loss']:.4f} (n={n_cv})",
                    flush=True,
                )
                log_f.write(json.dumps({
                    "step": step, "epoch": epoch,
                    **{f"vans_val_{k}": v for k, v in vans_val_m.items()},
                    **{f"coin_val_{k}": v for k, v in coin_val_m.items()},
                }) + "\n")
                if vans_val_m["pred_loss"] < best_val_vans:
                    best_val_vans = vans_val_m["pred_loss"]
                    torch.save({"step": step, "best_vans_val_mse": best_val_vans,
                                "predictor_state": predictor.state_dict()},
                               os.path.join(args.out_dir, "best.pt"))
            if step % args.save_every_steps == 0:
                torch.save({"step": step, "predictor_state": predictor.state_dict()},
                           os.path.join(args.out_dir, f"step_{step}.pt"))
            log_f.flush()

    best_path = os.path.join(args.out_dir, "best.pt")
    if os.path.exists(best_path):
        predictor.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["predictor_state"])

    vans_test_m, n_vt = evaluate_source(predictor, args.vans_cache_dir, vans_test, "vans", device)
    coin_test_m, n_ct = evaluate_source(predictor, args.coin_cache_dir, coin_test, "coin", device)
    print(f"[TEST] vans n={n_vt} {vans_test_m}", flush=True)
    print(f"[TEST] coin n={n_ct} {coin_test_m}", flush=True)
    with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
        json.dump({
            "vans": {"n_test": n_vt, **vans_test_m},
            "coin": {"n_test": n_ct, **coin_test_m},
        }, f, indent=2)

    log_f.write(json.dumps({"done": True, "steps_completed": step,
                             "best_vans_val_mse": best_val_vans}) + "\n")
    log_f.close()


if __name__ == "__main__":
    main()
