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
from common.lora import (  # noqa: E402
    apply_lora_to_all_layers, freeze_all_except_lora, lora_trainable_param_count,
)


def load_predictor_for_finetune(label, device, lora_cfg=None):
    """Same checkpoint-loading logic as probe_egodex_reverse_adapter.py's
    load_frozen_predictor (bridge rewiring for joint-trained checkpoints,
    etc.). Two modes, selected by lora_cfg:
      - lora_cfg is None (default): leaves everything UNFROZEN (train mode,
        requires_grad=True) -- the original full-fine-tune behavior.
      - lora_cfg is a dict {"r", "alpha", "dropout"}: freezes the predictor's
        (and, if present, the bridge's) own pretrained weights entirely, then
        wraps every nn.Linear/nn.MultiheadAttention in both with a LoRA
        adapter (see common/lora.py) and leaves only the LoRA A/B factors
        trainable. The VJEPA encoder itself is never loaded by this script
        at all (predict_future_latent only ever consumes precomputed
        vjepa_input_feats/vjepa_target_feats -- the encoder's own forward
        pass never runs here), so "freeze the encoder" is automatically true
        in both modes; lora_cfg controls how the PREDICTOR (and bridge) are
        trained instead of full fine-tuning.
    Returns (predictor, bridge_or_None) separately so the caller can build
    an optimizer over both."""
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
    state = ckpt["predictor_state"] if "predictor_state" in ckpt else ckpt
    predictor.load_state_dict(state)

    if lora_cfg is None:
        predictor.train()
        for p in predictor.parameters():
            p.requires_grad_(True)
        if bridge is not None:
            bridge.train()
            for p in bridge.parameters():
                p.requires_grad_(True)
        print(f"[INFO] loaded (UNFROZEN) predictor from {path} (step={ckpt.get('step')}) "
              f"for fine-tuning", flush=True)
        return predictor, bridge

    # LoRA mode: apply AFTER load_state_dict (so the pretrained weights land
    # in the plain nn.Linear/nn.MultiheadAttention modules first, exactly as
    # saved) -- wrapping first would change the state_dict's key names
    # (base.weight vs weight) and break loading.
    for p in predictor.parameters():
        p.requires_grad_(False)
    wrapped = apply_lora_to_all_layers(predictor, r=lora_cfg["r"], alpha=lora_cfg["alpha"],
                                        dropout=lora_cfg["dropout"])
    freeze_all_except_lora(predictor)
    predictor.train()
    n_trainable, n_total = lora_trainable_param_count(predictor), sum(p.numel() for p in predictor.parameters())
    print(f"[INFO] loaded predictor from {path} (step={ckpt.get('step')}) with LoRA on "
          f"{len(wrapped)} layers (r={lora_cfg['r']} alpha={lora_cfg['alpha']}): "
          f"trainable={n_trainable:,}/{n_total:,} ({100 * n_trainable / n_total:.2f}%)", flush=True)
    if bridge is not None:
        for p in bridge.parameters():
            p.requires_grad_(False)
        b_wrapped = apply_lora_to_all_layers(bridge, r=lora_cfg["r"], alpha=lora_cfg["alpha"],
                                              dropout=lora_cfg["dropout"])
        freeze_all_except_lora(bridge)
        bridge.train()
        bn_trainable, bn_total = lora_trainable_param_count(bridge), sum(p.numel() for p in bridge.parameters())
        print(f"[INFO] bridge also got LoRA on {len(b_wrapped)} layers: "
              f"trainable={bn_trainable:,}/{bn_total:,} ({100 * bn_trainable / max(bn_total, 1):.2f}%)", flush=True)
    return predictor, bridge


def train_epoch(predictor, loader, device, optimizer, max_batches=None, grad_accum_steps=1,
                 sl1_weight=0.0, cd_weight=0.0):
    """One DataLoader batch (--batch_size samples) is the true effective
    batch size for a single optimizer step -- all samples' forward passes
    are concatenated into one pred_stack/target_stack before ONE loss/
    backward/step, not accumulated per-sample. So --batch_size alone
    already controls the effective batch size for gradient averaging
    (ThinkJEPA's own recipe uses batch=16/GPU -- --batch_size 16 here is
    the direct analogue). --grad_accum_steps > 1 is an ADDITIONAL lever on
    top of that for when GPU memory can't hold --batch_size samples'
    activations simultaneously: it accumulates (sums, scaled by
    1/grad_accum_steps) gradients over multiple DataLoader batches before
    stepping, reaching a larger EFFECTIVE batch (--batch_size *
    grad_accum_steps) without that peak-memory cost."""
    predictor.train()
    total_loss, n_samples = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        samples = probe.batch_to_samples(batch, device, probe.ALL_JOINT_IDX)
        pred_list, target_list = [], []
        # bf16 autocast: the first real attempt OOM'd at --batch_size 4 (22.10/
        # 22.11GiB used) -- unlike the frozen-predictor eval path (no_grad,
        # no graph retained), fine-tuning keeps the full unfrozen predictor's
        # activation graph (every rollout/attention layer) for every sample in
        # the batch until one combined backward() call, far heavier than a
        # plain forward pass. No GradScaler needed -- bf16 has fp32's exponent
        # range, just less mantissa precision, so it doesn't underflow the way
        # fp16 can.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            for s in samples:
                y_future = probe.predict_future_latent(
                    predictor, s["in_feats"], {"vlm_old": s["vlm_old"], "vlm_new": s["vlm_new"]}, device,
                )
                pred_list.append(y_future)
                target_list.append(s["target_latent"])
            pred_stack = torch.cat(pred_list, dim=0)
            target_stack = torch.cat(target_list, dim=0)
        # Base term is exactly ThinkJEPA's own pred_loss (thinker_train.py:
        # F.mse_loss(pred_latent, target_latent)) -- rounds 1-2 used this
        # alone, matching their training objective exactly. Round 3 (per
        # user request, chasing a bigger drop in FD/SL1/CD rather than a
        # clean apples-to-apples match to their formula) adds SL1 and
        # cosine-distance directly into the loss at small weight: MSE's
        # gradient geometry isn't identical to smooth-L1's (L1-like in its
        # large-residual regime, unlike MSE's quadratic) or to cosine
        # distance's (direction-only, magnitude-invariant), so optimizing
        # MSE alone leaves some SL1/CD improvement on the table that
        # directly including them can capture. Both default to 0.0 (off,
        # reproduces rounds 1-2's exact loss) unless passed on the CLI.
        mse = F.mse_loss(pred_stack, target_stack)
        loss = mse
        if sl1_weight:
            loss = loss + sl1_weight * F.smooth_l1_loss(pred_stack, target_stack)
        if cd_weight:
            cos_sim = F.cosine_similarity(pred_stack, target_stack, dim=-1, eps=1e-8).mean()
            loss = loss + cd_weight * (1.0 - cos_sim)
        (loss / grad_accum_steps).backward()
        if (bi + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.item()) * len(samples)
        n_samples += len(samples)
    if (bi + 1) % grad_accum_steps != 0:
        optimizer.step()  # flush a trailing partial accumulation window
        optimizer.zero_grad(set_to_none=True)
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
    ap.add_argument("--batch_size", type=int, default=4,
                     help="the true effective batch size for one optimizer step (all samples in a "
                          "DataLoader batch are concatenated into one loss/backward/step, not "
                          "accumulated per-sample) -- ThinkJEPA's own recipe uses batch=16/GPU, so "
                          "--batch_size 16 is the direct analogue. Limited by GPU memory, since all "
                          "of a batch's forward activations are held simultaneously until one "
                          "backward() call; see --grad_accum_steps for a memory-cheaper way to reach "
                          "a larger EFFECTIVE batch.")
    ap.add_argument("--grad_accum_steps", type=int, default=1,
                     help="accumulate gradients over this many DataLoader batches before each "
                          "optimizer step, reaching an effective batch of --batch_size * "
                          "--grad_accum_steps without needing that many samples' activations in "
                          "memory at once. 1 (default) = no accumulation, matches the original "
                          "per-DataLoader-batch step behavior.")
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
    ap.add_argument("--lr_min", type=float, default=None,
                     help="when set, applies a CosineAnnealingLR schedule over --epochs from --lr "
                          "down to --lr_min (stepped once per epoch). Rounds 1-2 used a constant "
                          "--lr the whole time; per-epoch improvement visibly decelerated by the "
                          "end of round 2 (consistent with a fixed step size overshooting near a "
                          "local optimum), so round 3 decays it instead. Default None = constant LR, "
                          "reproduces rounds 1-2's behavior exactly.")
    ap.add_argument("--sl1_weight", type=float, default=0.0,
                     help="weight on an added F.smooth_l1_loss(pred,target) term in the training "
                          "loss, on top of the base MSE term -- see train_epoch's own comment for "
                          "why. 0.0 (default) reproduces rounds 1-2's pure-MSE loss exactly.")
    ap.add_argument("--cd_weight", type=float, default=0.0,
                     help="weight on an added (1 - cosine_similarity) term in the training loss, "
                          "on top of the base MSE term -- see train_epoch's own comment for why. "
                          "0.0 (default) reproduces rounds 1-2's pure-MSE loss exactly.")
    ap.add_argument("--lora", action="store_true",
                     help="freeze the predictor's (and bridge's, if the checkpoint has one) own "
                          "pretrained weights entirely and train LoRA adapters on every "
                          "nn.Linear/nn.MultiheadAttention layer instead of full fine-tuning. The "
                          "VJEPA encoder is never loaded by this script regardless (see "
                          "load_predictor_for_finetune's docstring), so this is the "
                          "'freeze JEPA's encoder, LoRA all other layers' recipe end to end.")
    ap.add_argument("--lora_r", type=int, default=8, help="LoRA rank, only used with --lora.")
    ap.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA scaling alpha, only used with --lora.")
    ap.add_argument("--lora_dropout", type=float, default=0.0, help="dropout on LoRA's input, only used with --lora.")
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

    lora_cfg = {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout} if args.lora else None
    predictor, bridge = load_predictor_for_finetune(args.checkpoint, device, lora_cfg=lora_cfg)
    params = [p for p in predictor.parameters() if p.requires_grad]
    if bridge is not None:
        params += [p for p in bridge.parameters() if p.requires_grad]
    print(f"[INFO] optimizer over {sum(p.numel() for p in params):,} trainable params", flush=True)
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scheduler = None
    if args.lr_min is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min)

    fd0, sl10, cd0, n0 = evaluate_latent_metrics(predictor, test_loader, device, max_batches=args.max_test_batches)
    print(f"[epoch 0/before fine-tune] TEST n={n0} FD={fd0:.6f} SL1={sl10:.6f} CD={cd0:.6f}", flush=True)
    history = [{"epoch": 0, "FD": fd0, "SL1": sl10, "CD": cd0, "n_test": n0}]

    for epoch in range(args.epochs):
        cur_lr = optimizer.param_groups[0]["lr"]
        train_loss = train_epoch(predictor, train_loader, device, optimizer,
                                  max_batches=args.max_train_batches,
                                  grad_accum_steps=args.grad_accum_steps,
                                  sl1_weight=args.sl1_weight, cd_weight=args.cd_weight)
        if scheduler is not None:
            scheduler.step()
        fd, sl1, cd, n_test = evaluate_latent_metrics(predictor, test_loader, device,
                                                        max_batches=args.max_test_batches)
        print(f"[epoch {epoch+1}/{args.epochs}] lr={cur_lr:.3e} train_loss={train_loss:.6f} "
              f"TEST n={n_test} FD={fd:.6f} SL1={sl1:.6f} CD={cd:.6f}", flush=True)
        history.append({"epoch": epoch + 1, "lr": cur_lr, "train_loss": train_loss,
                         "FD": fd, "SL1": sl1, "CD": cd, "n_test": n_test})

    # NOTE: when --lora is on, predictor.state_dict()'s keys reflect the
    # LoRA-wrapped structure (e.g. "...attn.qkv.base.weight" / "...lora_A.weight"),
    # not the original flat keys -- this checkpoint is NOT directly loadable
    # by load_frozen_predictor/load_predictor_for_finetune(lora_cfg=None)
    # without first re-applying the same wrapping. lora_cfg is saved
    # alongside so a future loader can detect this and re-wrap before
    # load_state_dict.
    save_payload = {"predictor_state": predictor.state_dict(), "checkpoint_source": args.checkpoint,
                     "lora_cfg": lora_cfg}
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
