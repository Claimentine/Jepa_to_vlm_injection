"""Joint/multi-task training: a shared CrossModalBridge trained
simultaneously by the forward (JEPA->VLM QA injection) and reverse
(VLM->JEPA future-latent prediction) directions.

Replaces the abandoned warm-start-initialization approach
(fit_warmstart_alignment.py): a one-shot closed-form ridge-regression fit
before training turned out to be fitting pure noise for the reverse
direction (held-out R^2 of -0.25 and -1.21 via --holdout_frac -- worse than
predicting the mean, from regressing a 2048-dim input against ~1463
samples, more free parameters than data points) and only a modest real
signal for the forward direction (held-out R^2=0.33). Ablation runs
(job-18/job-19) confirmed no measurable training benefit from that
initialization either way. See common/cross_modal_bridge.py's module
docstring for the full rationale of what replaces it here: instead of
fitting the cross-modal mapping once before either model has trained, this
script makes it a LIVE, jointly-trained parameter, pulled every step by
both directions' real task losses plus an explicit cycle-consistency term
-- genuine gradient-based mutual influence instead of a one-time init.

Reuses, via import, every already-tested building block from both existing
single-direction scripts -- train_qa_full.py (imported as `fwd`) and
train_latent_world_model_full.py (imported as `rev`), both left completely
unmodified; they remain the independent-training baselines this script's
result gets compared against (job-13/a clean rerun of it, and job-15).

Each step: one QA item (forward, from --qa_split) + one latent-prediction
item (reverse, from --cache_dir) -> total_loss = fwd_loss + rev_loss +
--cycle_loss_weight * cycle_loss -> one optimizer.step() over
injector.parameters() + predictor.parameters() + bridge.parameters(), with
no double-counted tensors (see common/cross_modal_bridge.py's
ForwardingAdapter for how the wiring avoids that).

GPU memory: the frozen Qwen3-VL-2B alone needs ~10.5GB (same floor as
train_qa_full.py's cross_attn jobs). The reverse direction adds no other
live frozen backbone -- vlm_old/vlm_new and V-JEPA2 features are
precomputed and cached on disk already (train_latent_world_model_full.py
never loads V-JEPA2 or Qwen3-VL live) -- so the added pressure here is just
CortexGuidedVideoPredictor's own modest footprint (~30M params) plus both
directions' backward-pass activations in the same process. Needs more
headroom than train_qa_full.py's existing job templates request.
"""
import argparse
import json
import os
import random
import sys
import traceback

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from common.jepa_injection_model import (  # noqa: E402
    DecoderLayerInjectionHook,
    LayerWiseJEPAInjector,
    resolve_decoder_layers,
    select_layer_indices,
)
from common.cross_modal_bridge import CrossModalBridge, ForwardingAdapter  # noqa: E402

import train_qa_full as fwd  # noqa: E402
import train_latent_world_model_full as rev  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qa_split", default=os.path.join(fwd.BASE, "raw_data/qa_split_temporal.json"),
                     help="forward direction's QA split -- matches job-13/job-18's choice")
    ap.add_argument("--layer_strategy", default="middle4", choices=["middle4", "last4", "uniform4", "all"])
    ap.add_argument("--cache_dir", default=os.path.join(rev.WORK_BASE, "vlm_guidance_cache_paired"))
    ap.add_argument("--rev_qa_split", default=os.path.join(rev.BASE, "raw_data/qa_split_full.json"),
                     help="pid train/val/test membership source for the reverse direction")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--cycle_loss_weight", type=float, default=0.1,
                     help="weight on the bridge's cycle-consistency term relative to the two "
                          "task losses -- small enough not to dominate either task, large enough "
                          "to give the shared bridge a real coupling gradient. Tunable.")
    ap.add_argument("--val_every_steps", type=int, default=500)
    ap.add_argument("--save_every_steps", type=int, default=1000)
    ap.add_argument("--max_val_items", type=int, default=100)
    ap.add_argument("--max_train_items", type=int, default=None,
                     help="smoke-test cap, applied to BOTH directions' train split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=os.path.join(fwd.BASE, "raw_data/joint_coupled_runs"))
    args = ap.parse_args()

    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    run_dir = args.out_dir
    os.makedirs(run_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- forward direction setup ----------------
    with open(args.qa_split) as f:
        fwd_split = json.load(f)
    fwd_train = [it for it in fwd_split["train"] if fwd.has_features(it)]
    fwd_val = [it for it in fwd_split["val"] if fwd.has_features(it)]
    if args.max_train_items:
        fwd_train = fwd_train[: args.max_train_items]
    fwd_val = fwd_val[: args.max_val_items]
    print(f"[INFO] forward usable: train={len(fwd_train)} val={len(fwd_val)}", flush=True)

    print(f"[INFO] loading {fwd.MODEL_ID} (frozen) ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(fwd.MODEL_ID, dtype="auto").to(device)
    processor = AutoProcessor.from_pretrained(fwd.MODEL_ID)
    model.requires_grad_(False)
    model.eval()
    hidden_size = model.config.text_config.hidden_size
    n_tokens = 64
    placeholder_id = processor.tokenizer.pad_token_id

    layer_indices = select_layer_indices(len(resolve_decoder_layers(model)), args.layer_strategy)
    injector = LayerWiseJEPAInjector(
        hidden_size=hidden_size, layer_indices=layer_indices, condition_mode="jepa",
        seed=args.seed, mode="cross_attn",
    ).to(device)
    hook = DecoderLayerInjectionHook(model, injector)

    # ---------------- reverse direction setup ----------------
    membership = rev.load_qa_split_membership(args.rev_qa_split)
    cached_pids = rev.list_cached_pids(args.cache_dir)
    by_split = {"train": [], "val": [], "test": []}
    for pid in cached_pids:
        part = membership.get(pid)
        if part in by_split:
            by_split[part].append(pid)
    split_rng = random.Random(args.seed)
    for part in by_split.values():
        split_rng.shuffle(part)
    rev_train, rev_val, rev_test = by_split["train"], by_split["val"], by_split["test"]
    if args.max_train_items:
        rev_train = rev_train[: args.max_train_items]
    rev_val = rev_val[: args.max_val_items]
    print(f"[INFO] reverse usable: train={len(rev_train)} val={len(rev_val)} test={len(rev_test)}", flush=True)

    predictor = rev.build_predictor(device, guidance="crossattn")

    # ---------------- shared bridge, wired into both ----------------
    # See common/cross_modal_bridge.py: ForwardingAdapter holds no parameters
    # of its own, so bridge stays the sole registration point for both heads
    # -- the optimizer below is exactly injector+predictor+bridge params,
    # no double-counted tensors.
    bridge = CrossModalBridge(jepa_dim=1024, vlm_dim=2048).to(device)
    injector.conditioner.adapter.mlp[0] = ForwardingAdapter(bridge.jepa_to_vlm)
    predictor.guidance_old_adapter = ForwardingAdapter(bridge.vlm_to_jepa, predictor.context_adapter)
    predictor.guidance_new_adapter = ForwardingAdapter(bridge.vlm_to_jepa, predictor.context_adapter)

    opt = torch.optim.AdamW(
        list(injector.parameters()) + list(predictor.parameters()) + list(bridge.parameters()),
        lr=args.lr,
    )
    n_fwd_params = sum(p.numel() for p in injector.parameters())
    n_rev_params = sum(p.numel() for p in predictor.parameters())
    n_bridge_params = sum(p.numel() for p in bridge.parameters())
    print(f"[INFO] trainable params: forward_injector={n_fwd_params:,} "
          f"reverse_predictor={n_rev_params:,} shared_bridge={n_bridge_params:,} "
          f"cycle_loss_weight={args.cycle_loss_weight}", flush=True)

    log_f = open(os.path.join(run_dir, "train_log.jsonl"), "w")
    rng = random.Random(args.seed)
    step = 0
    n_fwd_skipped = 0
    n_rev_skipped = 0
    seen_exc_types = set()
    best_fwd_val_acc = -1.0
    best_rev_val_mse = float("inf")

    n_steps_per_epoch = min(len(fwd_train), len(rev_train))
    if n_steps_per_epoch < 5:
        raise RuntimeError(
            f"too few paired train steps available (forward={len(fwd_train)}, reverse={len(rev_train)}) "
            "-- check --qa_split/--cache_dir/--rev_qa_split point at real extracted data"
        )

    for epoch in range(args.epochs):
        rng.shuffle(fwd_train)
        rng.shuffle(rev_train)
        for i in range(n_steps_per_epoch):
            fwd_item = fwd_train[i]
            rev_pid = rev_train[i]

            opt.zero_grad()
            total_loss = torch.zeros((), device=device)
            log_entry = {"step": step + 1, "epoch": epoch}

            fwd_ok = False
            pooled_jepa = None
            try:
                option_lines, correct_letter = fwd.build_option_block(fwd_item, rng)
                prompt_text = fwd.build_prompt_text(fwd_item["question"], option_lines)
                inputs = fwd.build_inputs(processor, fwd_item["in_clip"], prompt_text)
                model_inputs = fwd.prepare_model_inputs(inputs, "cross_attn", n_tokens, placeholder_id)
                model_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in model_inputs.items()}
                jepa_feats = torch.from_numpy(fwd.load_vjepa_in_feats(fwd_item["in_clip"])).unsqueeze(0).to(device)
                pooled_jepa = jepa_feats.mean(dim=(1, 2))  # (1, 1024)
                condition = injector.condition(jepa_feats)
                hook.set(condition)
                scores = torch.stack([
                    fwd.candidate_logprob(model, model_inputs, processor.tokenizer, "A"),
                    fwd.candidate_logprob(model, model_inputs, processor.tokenizer, "B"),
                ]).unsqueeze(0)
                hook.clear()
                target = torch.tensor([0 if correct_letter == "A" else 1], device=device)
                fwd_loss = nn.functional.cross_entropy(scores, target)
                total_loss = total_loss + fwd_loss
                log_entry["fwd_loss"] = float(fwd_loss.item())
                fwd_ok = True
            except Exception as e:
                n_fwd_skipped += 1
                exc_name = type(e).__name__
                print(f"[WARN] fwd skip {fwd_item.get('pid')}: {exc_name}: {e}", flush=True)
                if exc_name not in seen_exc_types:
                    seen_exc_types.add(exc_name)
                    traceback.print_exc()

            rev_ok = False
            pooled_vlm = None
            try:
                in_feats, out_feats, extras = rev.load_pair(args.cache_dir, rev_pid, device)
                pooled_vlm = extras["vlm_old"].mean(dim=(0, 1), keepdim=True)  # (1, 2048)
                metrics = rev.forward_step(predictor, in_feats, out_feats, extras, "crossattn", device)
                rev_loss = metrics["pred_loss"]
                total_loss = total_loss + rev_loss
                log_entry["rev_loss"] = float(rev_loss.item())
                rev_ok = True
            except Exception as e:
                n_rev_skipped += 1
                exc_name = type(e).__name__
                print(f"[WARN] rev skip {rev_pid}: {exc_name}: {e}", flush=True)
                if exc_name not in seen_exc_types:
                    seen_exc_types.add(exc_name)
                    traceback.print_exc()

            if not (fwd_ok or rev_ok):
                continue

            cyc_loss = bridge.cycle_loss(pooled_jepa, pooled_vlm)
            total_loss = total_loss + args.cycle_loss_weight * cyc_loss
            log_entry["cycle_loss"] = float(cyc_loss.item())

            total_loss.backward()
            opt.step()
            step += 1

            log_f.write(json.dumps(log_entry) + "\n")
            log_f.flush()

            if step % 50 == 0:
                print(f"[epoch {epoch} step {step}] {log_entry}", flush=True)

            if step % args.val_every_steps == 0:
                fwd_val_tally = fwd.evaluate_logprob(
                    model, processor, hook, injector, fwd_val, "cross_attn", n_tokens, placeholder_id, device, rng,
                )
                rev_val_metrics, n_rev_val = rev.evaluate(predictor, args.cache_dir, rev_val, "crossattn", device)
                print(f"[epoch {epoch} step {step}] fwd_val_acc={fwd_val_tally.summary_str()}  "
                      f"rev_val_mse={rev_val_metrics['pred_loss']:.4f} (n={n_rev_val})", flush=True)
                log_f.write(json.dumps({
                    "step": step,
                    "fwd_val_acc": fwd_val_tally.acc, "fwd_val_acc_by_difficulty": fwd_val_tally.acc_by_difficulty,
                    "rev_val_mse": rev_val_metrics["pred_loss"],
                    "rev_val_latent_dist": rev_val_metrics["pred_latent_dist"],
                    "rev_val_cosine_dist": rev_val_metrics["pred_latent_cosine_distance"],
                }) + "\n")
                log_f.flush()

                if fwd_val_tally.acc > best_fwd_val_acc:
                    best_fwd_val_acc = fwd_val_tally.acc
                    torch.save({"step": step, "val_acc": fwd_val_tally.acc,
                                "injector_state": injector.state_dict(), "bridge_state": bridge.state_dict()},
                               os.path.join(run_dir, "best_forward.pt"))
                if rev_val_metrics["pred_loss"] < best_rev_val_mse:
                    best_rev_val_mse = rev_val_metrics["pred_loss"]
                    torch.save({"step": step, "val_mse": rev_val_metrics["pred_loss"],
                                "predictor_state": predictor.state_dict(), "bridge_state": bridge.state_dict()},
                               os.path.join(run_dir, "best_reverse.pt"))

            if args.save_every_steps and step % args.save_every_steps == 0:
                torch.save({
                    "step": step, "injector_state": injector.state_dict(),
                    "predictor_state": predictor.state_dict(), "bridge_state": bridge.state_dict(),
                }, os.path.join(run_dir, f"step_{step}.pt"))
                print(f"[epoch {epoch} step {step}] saved checkpoint -> step_{step}.pt", flush=True)

    # Reverse direction's own established convention: a final test-set eval
    # (train_latent_world_model_full.py does this; train_qa_full.py doesn't
    # have a test-split concept at all, so the forward side has no analogous
    # step here -- each side just mirrors its own baseline's protocol).
    best_reverse_path = os.path.join(run_dir, "best_reverse.pt")
    if os.path.exists(best_reverse_path):
        predictor.load_state_dict(torch.load(best_reverse_path, map_location=device, weights_only=False)["predictor_state"])
    rev_test_metrics, n_rev_test = rev.evaluate(predictor, args.cache_dir, rev_test, "crossattn", device)
    print(f"[TEST] reverse n={n_rev_test} {rev_test_metrics}", flush=True)
    with open(os.path.join(run_dir, "rev_test_metrics.json"), "w") as f:
        json.dump({"n_test": n_rev_test, **rev_test_metrics}, f, indent=2)

    summary = {
        "done": True, "steps_completed": step,
        "fwd_items_skipped": n_fwd_skipped, "rev_items_skipped": n_rev_skipped,
        "best_fwd_val_acc": best_fwd_val_acc if step else None,
        "best_rev_val_mse": best_rev_val_mse if step else None,
    }
    print(f"[DONE] {summary}", flush=True)
    log_f.write(json.dumps(summary) + "\n")
    log_f.flush()
    log_f.close()


if __name__ == "__main__":
    main()
