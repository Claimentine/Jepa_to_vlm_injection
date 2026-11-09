"""Full-scale VLM-guides-JEPA training: does conditioning on real VLM
guidance (vlm_old/vlm_new from extract_vlm_guidance_paired.py) actually
improve V-JEPA2's own self-supervised future-latent prediction, compared to
a no-guidance control with the identical predictor otherwise?

This experiment has never been run before at scale. The only prior script,
vans_qa/demo/train_latent_world_model.py, is demo-scoped in three ways that
block reuse as-is (hence a new file, not an edit -- this project's
established convention when the data/design changes underneath a script):

1. Wrong cache schema/shapes. The demo reads TWO separate cache dirs
   (vjepa_cache/{pid}.npz with in_feats/out_feats, vlm_guidance_cache/
   {pid}_*.npz with vlm_old/vlm_new) with hardcoded T_PER_CLIP=64,
   P_PATCHES=128. extract_vlm_guidance_paired.py's real output is ONE merged
   npz per pid (vlm_guidance_cache_paired/{pid}.npz) and the real encoded
   shape is (16, 256, 1024), not (64, 128, 1024) -- confirmed 2026-09-06 by
   inspecting an actual extracted file on the cluster.
2. No no-guidance control. The demo always sets use_vlm_merge=True, so it
   can only report "loss with guidance" -- never "loss without", which is
   the actual comparison this question needs.
3. film mode. The demo hardcodes vlm_cond_mode="film". The sibling forward-
   direction experiment (vans_qa/full_scale/train_qa_full.py's
   LayerWiseJEPAInjector) diagnosed FiLM as architecturally bandwidth-
   limited there (gate_lr_multiplier=10 only moved gates to ~0.01 magnitude,
   val_acc gain within noise -- see job-11/README). CortexGuidedVideoPredictor
   is a separate, unmodified ThinkJEPA implementation -- its own film/adaln
   path does NOT share that specific double-zero-init bug (only
   guidance_layer_scale is zero-init; guidance_fusion_mlps keeps its default
   init, confirmed by reading cache_train/thinker_predictor.py directly) --
   but the same architectural argument (a single global scale/shift per
   layer vs. per-token cross-attention) still favors crossattn as the
   stronger prior, and this run should stay consistent with the other
   direction's now-established preference. So --guidance defaults the
   "with guidance" condition to crossattn, not film.

Reuses, via import, the exact tested building blocks from
cache_train.thinker_train / cache_train.thinker_predictor (unmodified):
  - CortexGuidedVideoPredictor
  - compute_predicted_latent_metrics
  - flatten_temporal_patch_tokens / build_temporal_patch_indices /
    repeat_indices_for_batch
  - build_thinkjepa_guidance_inputs

Train/val/test membership is inherited from --qa_split (qa_split_full.json)
by pid, rather than a fresh random split of whatever cache files happen to
exist -- so results are reported on the same held-out pids the forward
(JEPA->VLM QA) direction's eval already uses, keeping the two directions'
"test set" comparable for the eventual warm-start-coupling ablation.

Data: extract_vlm_guidance_paired.py -> vlm_guidance_cache_paired/{pid}.npz
      (vjepa_input_feats, vjepa_target_feats, vlm_old, vlm_new, pid, ...)
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

# Needed for --live_vlm's `from common.live_vlm_guidance import ...` /
# `from common.lora import ...` (deferred imports inside main()) -- this
# script never previously imported anything from the project's own common/
# package, only from cache_train.*, so repo root was never added to
# sys.path (confirmed by a real run: "ModuleNotFoundError: No module named
# 'common'" the first time --live_vlm hit that import). Script's own
# directory is vans_qa/full_scale/, so repo root is 2 levels up.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

BASE = os.environ.get("VANS_ROOT", "/data")
WORK_BASE = os.environ.get("VANS_WORK_ROOT", "/data/vans_work")
THINKJEPA_ROOT = os.environ.get("THINKJEPA_ROOT", "/home/jovyan/ThinkJEPA")
for _p in (THINKJEPA_ROOT, os.path.join(THINKJEPA_ROOT, "cache_train"), os.path.join(THINKJEPA_ROOT, "vjepa2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cache_train.thinker_train import (  # noqa: E402
    compute_predicted_latent_metrics,
    flatten_temporal_patch_tokens,
    build_temporal_patch_indices,
    repeat_indices_for_batch,
    build_thinkjepa_guidance_inputs,
)
from cache_train.thinker_predictor import CortexGuidedVideoPredictor  # noqa: E402

# Real shapes of extract_vlm_guidance_paired.py's output, confirmed against
# an actual extracted file on 2026-09-06 -- NOT the demo script's (64, 128)
# guess, which was carried over from a different/older extraction path.
T_PER_CLIP = 16       # vjepa_input_feats/vjepa_target_feats temporal tokens per clip
P_PATCHES = 256       # spatial patch tokens per temporal position
D_EMBED = 1024        # vjepa embed dim
VLM_OLD_DIM = 2048    # Qwen3-VL-2B-Thinking hidden size
VLM_NEW_DIM = 2048

GUIDANCE_ARGS = SimpleNamespace(
    thinkjepa_use_vlm_merge=True,
    thinkjepa_use_cache_ext=True,
    thinkjepa_vlm_source="both",
    thinkjepa_vlm_layer_selector="last",
)


def load_qa_split_membership(qa_split_path):
    """pid -> 'train'/'val'/'test', inherited from the forward direction's split."""
    split = json.load(open(qa_split_path))
    membership = {}
    for part_name, items in split.items():
        for item in items:
            membership[item["pid"]] = part_name
    return membership


def list_cached_pids(cache_dir):
    return sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(cache_dir, "*.npz"))
        if not os.path.basename(p).startswith(".")
    )


def load_pair(cache_dir, pid, device):
    d = np.load(os.path.join(cache_dir, f"{pid}.npz"))
    in_feats = torch.from_numpy(d["vjepa_input_feats"].astype(np.float32)).unsqueeze(0).to(device)   # (1,16,256,1024)
    out_feats = torch.from_numpy(d["vjepa_target_feats"].astype(np.float32)).unsqueeze(0).to(device)  # (1,16,256,1024)
    extras = {
        # [L,S,D], single-sample (no batch axis) -- CortexGuidedVideoPredictor's
        # _normalize_guidance_stream unsqueezes this to [1,L,S,D] itself and
        # maps its own depth onto our L=8 cached layers internally
        # (_project_guidance_layer / _mapped_level_index), so no manual layer
        # selection is needed here despite GUIDANCE_ARGS.thinkjepa_vlm_layer_selector
        # (that flag affects reasoning-token filtering, not layer choice --
        # confirmed by reading apply_guidance_policy/build_thinkjepa_guidance_inputs).
        "vlm_old": torch.from_numpy(d["vlm_old"].astype(np.float32)).to(device),
        "vlm_new": torch.from_numpy(d["vlm_new"].astype(np.float32)).to(device),
    }
    return in_feats, out_feats, extras


def load_pair_live(cache_dir, pid, device, qwen_model, qwen_processor, qwen_saved, qwen_args):
    """--live_vlm variant of load_pair: vjepa_input_feats/vjepa_target_feats
    still come from the cache (V-JEPA2's encoder is frozen and deterministic
    given the input frames, so caching it is still exactly correct -- the
    user's own "only JEPA's encoder stays frozen" scope does not touch this
    side). vlm_old/vlm_new, instead of being read from the cache's own frozen
    arrays, are recomputed LIVE by re-running Qwen3-VL's teacher-forced
    inference on this pid's actual (in_clip, question, correct_caption) --
    all three already stored in the SAME npz by extract_vlm_guidance_paired.py
    (see its own payload dict), so no separate raw-data lookup is needed.
    Returns the same (in_feats, out_feats, extras) shape as load_pair, so
    forward_step needs no changes -- only vlm_old/vlm_new now carry gradients
    back into qwen_model's LoRA parameters."""
    from cache_train.rebuild_causal_cache import NUM_PAST_FRAMES
    from common.live_vlm_guidance import decode_clip_with_timeout, infer_qwen_teacher_forced_live

    d = np.load(os.path.join(cache_dir, f"{pid}.npz"))
    in_feats = torch.from_numpy(d["vjepa_input_feats"].astype(np.float32)).unsqueeze(0).to(device)
    out_feats = torch.from_numpy(d["vjepa_target_feats"].astype(np.float32)).unsqueeze(0).to(device)

    in_clip_relpath = str(d["in_clip_relpath"])
    question = str(d["question"])
    correct_caption = str(d["correct_caption"])
    past_frames, past_times, in_fps = decode_clip_with_timeout(in_clip_relpath, NUM_PAST_FRAMES)
    vlm_old, vlm_new = infer_qwen_teacher_forced_live(
        qwen_args, qwen_model, qwen_processor, qwen_saved, device,
        past_frames, past_times, in_fps, question, correct_caption,
    )
    extras = {"vlm_old": vlm_old.float(), "vlm_new": vlm_new.float()}
    return in_feats, out_feats, extras


def build_predictor(device, guidance):
    total_frames = 2 * T_PER_CLIP
    use_guidance = guidance != "none"
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
        use_vlm_merge=use_guidance,
        vlm_cond_mode=guidance if use_guidance else "film",
        vlm_old_dim=VLM_OLD_DIM,
        vlm_new_dim=VLM_NEW_DIM,
    ).to(device)
    return predictor


def forward_step(predictor, in_feats, out_feats, extras, guidance, device):
    B = in_feats.shape[0]
    feats_total = torch.cat([in_feats, out_feats], dim=1)  # (B, 32, P, D)
    x_seq = flatten_temporal_patch_tokens(feats_total)      # (B, 32*P, D)

    idx_ctx_1d = build_temporal_patch_indices(P_PATCHES, 0, T_PER_CLIP)
    idx_tgt_1d = build_temporal_patch_indices(P_PATCHES, T_PER_CLIP, 2 * T_PER_CLIP)
    masks_x = repeat_indices_for_batch(idx_ctx_1d.long(), B, device=x_seq.device)
    masks_y = repeat_indices_for_batch(idx_tgt_1d.long(), B, device=x_seq.device)

    x_ctxt = x_seq.gather(dim=1, index=masks_x.unsqueeze(-1).expand(-1, -1, D_EMBED))
    ext = build_thinkjepa_guidance_inputs(extras=extras, args=GUIDANCE_ARGS, device=device) if guidance != "none" else None

    y_future_seq = predictor(x_ctxt, masks_x, masks_y, ext=ext)  # (B, T_tgt*P, D)
    y_future = y_future_seq.view(B, T_PER_CLIP, P_PATCHES, D_EMBED)

    metrics = compute_predicted_latent_metrics(y_future, out_feats)
    return metrics


def evaluate(predictor, get_pair, pids, guidance, device, max_items=None):
    predictor.eval()
    totals = {"pred_loss": 0.0, "pred_latent_dist": 0.0, "pred_latent_cosine_distance": 0.0}
    n = 0
    use_pids = pids[:max_items] if max_items else pids
    with torch.no_grad():
        for pid in use_pids:
            in_feats, out_feats, extras = get_pair(pid)
            m = forward_step(predictor, in_feats, out_feats, extras, guidance, device)
            for k in totals:
                totals[k] += float(m[k].item())
            n += 1
    predictor.train()
    return {k: v / max(n, 1) for k, v in totals.items()}, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache_dir", default=os.path.join(WORK_BASE, "vlm_guidance_cache_paired"))
    ap.add_argument("--qa_split", default=os.path.join(BASE, "raw_data/qa_split_full.json"),
                     help="only used to inherit train/val/test membership by pid")
    ap.add_argument("--guidance", choices=["crossattn", "film", "adaln", "none"], required=True,
                     help="'none' is the no-guidance control -- identical predictor, use_vlm_merge=False")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_every_steps", type=int, default=500)
    ap.add_argument("--save_every_steps", type=int, default=1000)
    ap.add_argument("--max_val_items", type=int, default=200, help="cap for periodic in-training eval")
    ap.add_argument("--max_train_items", type=int, default=None, help="smoke-test cap on the train split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=os.path.join(BASE, "raw_data/latent_world_model_runs"))
    ap.add_argument("--warmstart_alignment", default=None,
                     help="path to a fit_warmstart_alignment.py output .pt file -- if given (and "
                          "--guidance != none), initializes guidance_old_adapter/guidance_new_adapter "
                          "from the fitted VLM->JEPA alignment (composed through CortexGuidedVideoPredictor's "
                          "own context_adapter) instead of random init.")
    ap.add_argument("--live_vlm", action="store_true",
                     help="FULL-STACK FINE-TUNING (2026-11-07/08): instead of reading this pid's "
                          "vlm_old/vlm_new from the (frozen, pre-extracted) cache, load Qwen3-VL live, "
                          "LoRA-wrap it (common/lora.py), and recompute vlm_old/vlm_new via a "
                          "gradient-carrying teacher-forced forward pass on this pid's actual "
                          "(in_clip, question, correct_caption) every time -- so the predictor's "
                          "pred_loss gradient reaches the VLM's own LoRA params too, not just the "
                          "predictor. V-JEPA2's encoder output (in_feats/out_feats) still comes from "
                          "the cache unchanged -- that encoder stays frozen regardless (see "
                          "load_pair_live's docstring).")
    ap.add_argument("--pretrained", default="Qwen/Qwen3-VL-2B-Thinking", help="only used with --live_vlm")
    ap.add_argument("--qwen_revision", default="main", help="only used with --live_vlm")
    ap.add_argument("--qwen_checkpoint_sha", default="", help="only used with --live_vlm")
    ap.add_argument("--qwen_res", type=int, default=256, help="only used with --live_vlm")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 12, 16, 20, 24, 27],
                     help="which Qwen3-VL decoder layers to hook for vlm_old/vlm_new -- only used with "
                          "--live_vlm; MUST match whatever extract_vlm_guidance_paired.py used to build "
                          "the cache this run inherits its train/val/test pid split from, since "
                          "guidance_old_adapter/guidance_new_adapter's expected layer count (L) is fixed "
                          "at predictor-construction time by the cache's OWN L (see load_pair's comment).")
    ap.add_argument("--lora_vlm_r", type=int, default=8, help="only used with --live_vlm")
    ap.add_argument("--lora_vlm_alpha", type=float, default=16.0, help="only used with --live_vlm")
    ap.add_argument("--lora_vlm_dropout", type=float, default=0.0, help="only used with --live_vlm")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(args.out_dir, args.guidance)
    os.makedirs(run_dir, exist_ok=True)

    membership = load_qa_split_membership(args.qa_split)
    cached_pids = list_cached_pids(args.cache_dir)
    print(f"[INFO] {len(cached_pids)} cached pairs in {args.cache_dir}", flush=True)

    by_split = {"train": [], "val": [], "test": []}
    n_unmatched = 0
    for pid in cached_pids:
        part = membership.get(pid)
        if part in by_split:
            by_split[part].append(pid)
        else:
            n_unmatched += 1
    if n_unmatched:
        print(f"[WARN] {n_unmatched} cached pids not found in {args.qa_split}, skipped", flush=True)

    rng = random.Random(args.seed)
    for part in by_split.values():
        rng.shuffle(part)
    train_ids, val_ids, test_ids = by_split["train"], by_split["val"], by_split["test"]
    if args.max_train_items:
        train_ids = train_ids[: args.max_train_items]
    print(f"[INFO] split: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}", flush=True)
    if len(train_ids) < 5:
        raise RuntimeError("too few cached train pairs -- run extract_vlm_guidance_paired.py first")

    predictor = build_predictor(device, args.guidance)
    n_params = sum(p.numel() for p in predictor.parameters())
    print(f"[INFO] guidance={args.guidance} predictor params: {n_params:,}", flush=True)

    vlm_lora_params = []
    if args.live_vlm:
        from common.live_vlm_guidance import load_qwen_with_lora
        qwen_model, qwen_processor, qwen_saved, _qwen_sha, _wrapped = load_qwen_with_lora(
            args, device, lora_r=args.lora_vlm_r, lora_alpha=args.lora_vlm_alpha, lora_dropout=args.lora_vlm_dropout,
        )
        vlm_lora_params = [p for p in qwen_model.parameters() if p.requires_grad]

        def get_pair(pid):
            return load_pair_live(args.cache_dir, pid, device, qwen_model, qwen_processor, qwen_saved, args)
    else:
        def get_pair(pid):
            return load_pair(args.cache_dir, pid, device)

    if args.warmstart_alignment and args.guidance != "none":
        # See fit_warmstart_alignment.py's module docstring for the full
        # rationale -- guidance_old_adapter/guidance_new_adapter only exist
        # when use_vlm_merge=True, i.e. args.guidance != "none".
        align = torch.load(args.warmstart_alignment, map_location=device, weights_only=False)
        for attr, key in [
            ("guidance_old_adapter", "rev_guidance_old_adapter_weight"),
            ("guidance_new_adapter", "rev_guidance_new_adapter_weight"),
        ]:
            layer = getattr(predictor, attr)
            expected_shape = tuple(layer.weight.shape)
            loaded_shape = tuple(align[key].shape)
            assert loaded_shape == expected_shape, (
                f"warmstart alignment shape {loaded_shape} != {attr}.weight shape {expected_shape}"
            )
            with torch.no_grad():
                layer.weight.copy_(align[key].to(device))
        print(f"[INFO] warm-started guidance_old_adapter/guidance_new_adapter from "
              f"{args.warmstart_alignment} (fit on {align['n_pairs']} pairs)", flush=True)
    opt = torch.optim.AdamW(list(predictor.parameters()) + vlm_lora_params, lr=args.lr)

    log_path = os.path.join(run_dir, "train_log.jsonl")
    log_f = open(log_path, "a")
    best_val = float("inf")
    step = 0

    for epoch in range(args.epochs):
        rng.shuffle(train_ids)
        for pid in train_ids:
            step += 1
            in_feats, out_feats, extras = get_pair(pid)
            metrics = forward_step(predictor, in_feats, out_feats, extras, args.guidance, device)
            loss = metrics["pred_loss"]

            opt.zero_grad()
            loss.backward()
            if args.live_vlm and step == 1:
                # One-time diagnostic: confirm the reverse direction's loss
                # actually reaches the VLM's own LoRA params, not just the
                # predictor's -- the earlier version of this path silently
                # didn't (register_thinker_decoder_hooks detaches+CPUs the
                # captured hidden state, severing autograd right there; see
                # common/live_vlm_guidance.py's register_decoder_hooks_live
                # docstring). A zero/None total here means that regressed.
                vlm_grad_norm = sum(
                    p.grad.norm().item() for p in vlm_lora_params if p.grad is not None
                )
                n_with_grad = sum(1 for p in vlm_lora_params if p.grad is not None)
                print(f"[DIAG --live_vlm step 1] VLM LoRA params with grad: "
                      f"{n_with_grad}/{len(vlm_lora_params)}  total_grad_norm={vlm_grad_norm:.6f}", flush=True)
                if n_with_grad == 0 or vlm_grad_norm == 0.0:
                    raise RuntimeError(
                        "--live_vlm: no gradient reached the VLM's LoRA params -- "
                        "the reverse direction is not actually training the VLM"
                    )
            opt.step()

            log_f.write(json.dumps({"step": step, "epoch": epoch, "train_loss": float(loss.item())}) + "\n")

            if step % args.val_every_steps == 0:
                val_metrics, n_val = evaluate(predictor, get_pair, val_ids, args.guidance, device, args.max_val_items)
                print(
                    f"[step {step}] train_loss={loss.item():.4f}  "
                    f"val_mse={val_metrics['pred_loss']:.4f}  "
                    f"val_latent_dist={val_metrics['pred_latent_dist']:.4f}  "
                    f"val_cosine_dist={val_metrics['pred_latent_cosine_distance']:.4f}  (n={n_val})",
                    flush=True,
                )
                log_f.write(json.dumps({"step": step, "epoch": epoch, **{f"val_{k}": v for k, v in val_metrics.items()}}) + "\n")
                if val_metrics["pred_loss"] < best_val:
                    best_val = val_metrics["pred_loss"]
                    ckpt = {"step": step, "guidance": args.guidance, "best_val_mse": best_val,
                            "predictor_state": predictor.state_dict()}
                    if args.live_vlm:
                        from common.lora import lora_state_dict
                        ckpt["vlm_lora_state"] = lora_state_dict(qwen_model)
                    torch.save(ckpt, os.path.join(run_dir, "best.pt"))
            if step % args.save_every_steps == 0:
                ckpt = {"step": step, "guidance": args.guidance, "predictor_state": predictor.state_dict()}
                if args.live_vlm:
                    from common.lora import lora_state_dict
                    ckpt["vlm_lora_state"] = lora_state_dict(qwen_model)
                torch.save(ckpt, os.path.join(run_dir, f"step_{step}.pt"))
            log_f.flush()

    log_f.close()

    best_path = os.path.join(run_dir, "best.pt")
    if os.path.exists(best_path):
        # NOTE: with --live_vlm, this restores the predictor's own best
        # snapshot but NOT qwen_model's VLM LoRA weights back to that same
        # step (they're left at whatever they were at the end of training) --
        # best.pt's "vlm_lora_state" is saved for inspection/reuse elsewhere,
        # not auto-reloaded here. Fine for a smoke test; a real run wanting a
        # true best-checkpoint test score would need to also reload it here.
        predictor.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["predictor_state"])
    test_metrics, n_test = evaluate(predictor, get_pair, test_ids, args.guidance, device)
    print(f"[TEST] guidance={args.guidance} n={n_test} {test_metrics}", flush=True)
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump({"n_test": n_test, **test_metrics}, f, indent=2)


if __name__ == "__main__":
    main()
