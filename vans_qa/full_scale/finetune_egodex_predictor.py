"""Stage 3 of the EgoDex plan: actually FINE-TUNE our predictor (+ its
CrossModalBridge/guidance-injection modules, when present) on EgoDex,
instead of freezing it and only training a small xyz probe head
(probe_egodex_reverse_adapter.py). Motivation: ThinkJEPA's own officially-
reported latent-space metrics (FD/SL1/CD, see
cache_train/thinker_train.py's compute_predicted_latent_metrics) come from
a predictor that was trained end-to-end on EgoDex with

    total_loss = lambda_task * task_loss(xyz) + lambda_pred * pred_loss
    pred_loss  = F.mse_loss(pred_latent, target_latent)        (lambda_pred=1.0 default)

i.e. their predictor gets a direct latent-MSE gradient on EgoDex every
step. Our own job-101 FD/SL1/CD numbers, by contrast, came from a
predictor that has NEVER seen EgoDex at all -- a pure zero-shot
generalization test, not a fair same-training-regime comparison (the
same class of confound C4/--target_mode absolute fixed for ADE/FDE).

This script closes that gap on our side: fine-tunes a given checkpoint's
predictor (unfrozen, not wrapped in requires_grad_(False) the way
load_frozen_predictor does) on EgoDex's own train split, using the
IDENTICAL loss ThinkJEPA's own pred_loss term uses (plain
F.mse_loss(pred_latent, target_latent), no probe head, no xyz decode at
all) -- so the resulting FD/SL1/CD numbers are judged under the same
objective their own reference number was optimized for. The VLM-guidance
injection pathway (vlm_old/vlm_new -> predictor.guidance_old_adapter/
guidance_new_adapter, routed through the checkpoint's own CrossModalBridge
for bridge-trained checkpoints) stays active throughout -- this is
ThinkJEPA's own predictor architecture's one structural difference from
ours, and the whole point of this experiment is to see whether fine-tuning
WITH that pathway active beats their number, not just whether more EgoDex
exposure alone does (a job_none_guidance-vs-guided ablation under this
exact same recipe is the natural follow-up once a first result is in,
deliberately NOT run by this invocation).

Starts from an already-trained self-supervised checkpoint (VANS/COIN/
Ego4D pairs, e.g. job72_step4000) rather than from scratch -- this is
domain-adaptation fine-tuning on top of real pretraining, not a from-zero
run, so it should need far fewer steps than ThinkJEPA's own from-scratch
EgoDex training.

Reuses probe_egodex_reverse_adapter.py's EgoDex dataloaders, batch_to_
samples (already extracts target_latent), predict_future_latent, and
latent_metrics (FD/SL1/CD) directly -- only the freeze/eval-vs-train
distinction and the loss function differ.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import probe_egodex_reverse_adapter as probe  # noqa: E402
import train_latent_world_model_full as rev  # noqa: E402
from common.cross_modal_bridge import CrossModalBridge, ForwardingAdapter  # noqa: E402


def load_predictor_for_finetune(label, device):
    """Same checkpoint-loading logic as probe_egodex_reverse_adapter.py's
    load_frozen_predictor (bridge rewiring for joint-trained checkpoints,
    etc.) -- but leaves everything UNFROZEN (train mode, requires_grad=True)
    and returns (predictor, bridge_or_None) separately so the caller can
    build an optimizer over both."""
    path = probe.CHECKPOINTS[label] if "=" not in label else label.split("=", 1)[1]
    ckpt = torch.load(path, map_location=device, weights_only=False)
    guidance_mode = ckpt.get("guidance", "crossattn")
    predictor = rev.build_predictor(device, guidance_mode)
    bridge = None
    if "bridge_state" in ckpt:
        bridge_state = ckpt["bridge_state"]
        if "jepa_to_vlm.weight" in bridge_state:
            hidden_dim = None
        else:
            hidden_dim = bridge_state["jepa_to_vlm.0.weight"].shape[0]
        print(f"[INFO] {path}: has a shared CrossModalBridge (hidden_dim={hidden_dim}) -- "
              "rewiring ForwardingAdapter before loading state dicts", flush=True)
        bridge = CrossModalBridge(jepa_dim=1024, vlm_dim=2048, hidden_dim=hidden_dim).to(device)
        predictor.guidance_old_adapter = ForwardingAdapter(bridge.vlm_to_jepa, predictor.context_adapter)
        predictor.guidance_new_adapter = ForwardingAdapter(bridge.vlm_to_jepa, predictor.context_adapter)
        bridge.load_state_dict(bridge_state)
        bridge.train()
        for p in bridge.parameters():
            p.requires_grad_(True)
    state = ckpt["predictor_state"] if "predictor_state" in ckpt else ckpt
    predictor.load_state_dict(state)
    predictor.train()
    for p in predictor.parameters():
        p.requires_grad_(True)
    print(f"[INFO] loaded (UNFROZEN) predictor from {path} (step={ckpt.get('step')}) "
          f"for fine-tuning", flush=True)
    return predictor, bridge


def train_epoch(predictor, loader, device, optimizer, max_batches=None):
    predictor.train()
    total_loss, n_samples = 0.0, 0
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        samples = probe.batch_to_samples(batch, device, probe.ALL_JOINT_IDX)
        pred_list, target_list = [], []
        for s in samples:
            y_future = probe.predict_future_latent(
                predictor, s["in_feats"], {"vlm_old": s["vlm_old"], "vlm_new": s["vlm_new"]}, device,
            )
            pred_list.append(y_future)
            target_list.append(s["target_latent"])
        pred_stack = torch.cat(pred_list, dim=0)
        target_stack = torch.cat(target_list, dim=0)
        # Exactly ThinkJEPA's own pred_loss term (thinker_train.py:
        # F.mse_loss(pred_latent, target_latent)) -- no SL1/cosine weighting
        # here, so the resulting FD/SL1/CD are judged under the identical
        # objective their own reference number was optimized for.
        loss = F.mse_loss(pred_stack, target_stack)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(samples)
        n_samples += len(samples)
    return total_loss / max(n_samples, 1)


def evaluate_latent_metrics(predictor, loader, device, max_batches=None):
    predictor.eval()
    all_pred_latent, all_target_latent = [], []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            samples = probe.batch_to_samples(batch, device, probe.ALL_JOINT_IDX)
            for s in samples:
                y_future = probe.predict_future_latent(
                    predictor, s["in_feats"], {"vlm_old": s["vlm_old"], "vlm_new": s["vlm_new"]}, device,
                )
                all_pred_latent.append(y_future)
                all_target_latent.append(s["target_latent"])
    fd, sl1, cd = probe.latent_metrics(torch.cat(all_pred_latent, dim=0), torch.cat(all_target_latent, dim=0))
    return fd, sl1, cd, len(all_pred_latent)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="job72_step4000",
                     help="one label from probe_egodex_reverse_adapter.CHECKPOINTS (or label=path) "
                          "to initialize from -- fine-tunes ONE checkpoint per invocation, unlike the "
                          "probe script's --checkpoints list, since each run mutates its own optimizer "
                          "state over real weights, not just a small disposable probe head.")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5,
                     help="fine-tuning epochs over EgoDex train split -- kept small since this "
                          "continues from an already-trained self-supervised checkpoint, not a "
                          "from-scratch run (see module docstring).")
    ap.add_argument("--lr", type=float, default=1e-5,
                     help="small LR: fine-tuning a pretrained predictor, not training one from "
                          "random init -- a probe-head-scale LR (1e-3, see "
                          "probe_egodex_reverse_adapter.py's --probe_lr) would likely destroy the "
                          "pretrained weights in a few steps.")
    ap.add_argument("--max_train_batches", type=int, default=None, help="smoke-test cap")
    ap.add_argument("--max_test_batches", type=int, default=None, help="smoke-test cap")
    ap.add_argument("--save_path", required=True,
                     help="where to write the fine-tuned {'predictor_state', 'bridge_state'?} "
                          "checkpoint -- NEVER overwrites --checkpoint's own original file.")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = probe.build_dataloaders(args)
    print(f"[INFO] train_loader batches~{len(train_loader)} test_loader batches~{len(test_loader)}", flush=True)

    predictor, bridge = load_predictor_for_finetune(args.checkpoint, device)
    params = list(predictor.parameters())
    if bridge is not None:
        params += list(bridge.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    fd0, sl10, cd0, n0 = evaluate_latent_metrics(predictor, test_loader, device, max_batches=args.max_test_batches)
    print(f"[epoch 0/before fine-tune] TEST n={n0} FD={fd0:.6f} SL1={sl10:.6f} CD={cd0:.6f}", flush=True)
    history = [{"epoch": 0, "FD": fd0, "SL1": sl10, "CD": cd0, "n_test": n0}]

    for epoch in range(args.epochs):
        train_loss = train_epoch(predictor, train_loader, device, optimizer,
                                  max_batches=args.max_train_batches)
        fd, sl1, cd, n_test = evaluate_latent_metrics(predictor, test_loader, device,
                                                        max_batches=args.max_test_batches)
        print(f"[epoch {epoch+1}/{args.epochs}] train_pred_mse={train_loss:.6f} "
              f"TEST n={n_test} FD={fd:.6f} SL1={sl1:.6f} CD={cd:.6f}", flush=True)
        history.append({"epoch": epoch + 1, "train_pred_mse": train_loss,
                         "FD": fd, "SL1": sl1, "CD": cd, "n_test": n_test})

    save_payload = {"predictor_state": predictor.state_dict(), "checkpoint_source": args.checkpoint}
    if bridge is not None:
        save_payload["bridge_state"] = bridge.state_dict()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(save_payload, args.save_path)
    print(f"[INFO] saved fine-tuned checkpoint to {args.save_path}", flush=True)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(history, f, indent=2)
    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
