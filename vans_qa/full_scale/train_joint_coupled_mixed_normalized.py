"""train_joint_coupled_normalized.py (job-34's script), with one change:
the reverse direction's training pool mixes in COIN long-axis pairs
(train_joint_coupled_mixed.py's job-30 mechanism) alongside VANS pairs,
instead of VANS alone.

Motivation: job-30 (COIN mixing on top of train_joint_coupled.py's
UN-normalized math) found no measurable benefit over job-22 -- but job-22's
own baseline was later shown to be broken by a raw loss-magnitude imbalance
(see train_joint_coupled_normalized.py's own docstring), not by the
shared-bridge architecture itself. job-30's null result may reflect that
broken baseline rather than COIN data being genuinely unhelpful. This run
retests COIN mixing on top of job-34's fix (EMA loss normalization) instead,
with --cycle_loss_weight/--disable_norm/--bridge_hidden_dim all defaulting
to job-34's own exact values -- so the ONLY variable that differs from
job-34 is whether the reverse pool includes COIN pairs. If job-34's forward
accuracy improves (or reverse's degrades/improves) with COIN mixed in, that
isolates COIN's own contribution, cleanly separated from the normalization
fix job-30 never had.

Reverse pool: VANS pairs (rev.load_pair, has real vlm_old/vlm_new -- cycle
loss gets a real pooled_vlm term) and COIN pairs (load_coin_pair, no
vlm_old/vlm_new -- pooled_vlm stays None for that step, so cycle_loss's
"vlm->jepa->vlm" term is skipped for COIN steps; see CrossModalBridge.
cycle_loss's own docstring: either argument can be omitted). Both sources
flow through forward_step_mixed (train_latent_world_model_mixed.py) instead
of rev.forward_step -- same computation, just decides its ext input from
whether extras is present rather than from a fixed --guidance string, which
handles both sources uniformly without a per-source branch in the training
loop itself.

Step budget: n_steps_per_epoch = min(my_fwd_train, my_rev_train_combined)
per rank -- the SAME min-based philosophy job-34 uses (not job-30's
max-based "cycle the shorter pool to use every item" design), specifically
so this run makes the exact same number of forward-direction gradient
updates as job-34 (forward remains the bottleneck either way, since VANS
alone already exceeds forward's per-rank count -- adding COIN only changes
what reverse's per-step draw can be, not the total step count). This avoids
the step-count confound train_joint_coupled_mixed.py's own docstring flags
in its comparison against job-22.

Everything else (DDP manual gradient all-reduce, LossNormalizer,
--bridge_hidden_dim, barrier placement/NCCL timeout) is unchanged from
train_joint_coupled_normalized.py -- see that script's own docstring for
the full rationale on each of those.
"""
import argparse
import datetime
import json
import os
import random
import sys
import traceback

import torch
import torch.distributed as dist
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
from train_latent_world_model_mixed import (  # noqa: E402
    load_coin_pair, list_coin_ids, split_coin_ids, forward_step_mixed, evaluate_source,
)


def is_ddp():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_ddp():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=3))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def sync_gradients(params, world_size):
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


class LossNormalizer:
    def __init__(self, decay=0.98, eps=1e-4, disabled=False):
        self.decay = decay
        self.eps = eps
        self.ema = {}
        self.disabled = disabled

    def normalize(self, name, raw_loss_tensor):
        raw_value = float(raw_loss_tensor.item())
        if name not in self.ema:
            self.ema[name] = raw_value
        else:
            self.ema[name] = self.decay * self.ema[name] + (1 - self.decay) * raw_value
        if self.disabled:
            return raw_loss_tensor, self.ema[name]
        return raw_loss_tensor / (self.ema[name] + self.eps), self.ema[name]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qa_split", default=os.path.join(fwd.BASE, "raw_data/qa_split_temporal.json"))
    ap.add_argument("--layer_strategy", default="middle4", choices=["middle4", "last4", "uniform4", "all"])
    ap.add_argument("--cache_dir", default=os.path.join(rev.WORK_BASE, "vlm_guidance_cache_paired"))
    ap.add_argument("--rev_qa_split", default=os.path.join(rev.BASE, "raw_data/qa_split_full.json"))
    ap.add_argument("--coin_cache_dir", default=os.path.join(rev.BASE, "raw_data/coin_longaxis"))
    ap.add_argument("--coin_val_frac", type=float, default=0.05)
    ap.add_argument("--coin_test_frac", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--cycle_loss_weight", type=float, default=0.1)
    ap.add_argument("--loss_norm_decay", type=float, default=0.98)
    ap.add_argument("--disable_norm", action="store_true")
    ap.add_argument("--bridge_hidden_dim", type=int, default=None)
    ap.add_argument("--val_every_steps", type=int, default=500)
    ap.add_argument("--save_every_steps", type=int, default=1000)
    ap.add_argument("--max_val_items", type=int, default=100)
    ap.add_argument("--max_train_items", type=int, default=None,
                     help="smoke-test cap, applied to forward and to the VANS/COIN reverse pools independently")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=os.path.join(fwd.BASE, "raw_data/joint_coupled_mixed_normalized_runs"))
    args = ap.parse_args()

    if is_ddp():
        rank, world_size, local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    run_dir = args.out_dir
    if is_main:
        os.makedirs(run_dir, exist_ok=True)

    # ---------------- forward direction setup (unchanged from job-34) ----------------
    with open(args.qa_split) as f:
        fwd_split = json.load(f)
    fwd_train = [it for it in fwd_split["train"] if fwd.has_features(it)]
    fwd_val = [it for it in fwd_split["val"] if fwd.has_features(it)]
    if args.max_train_items:
        fwd_train = fwd_train[: args.max_train_items]
    fwd_val = fwd_val[: args.max_val_items]
    if is_main:
        print(f"[INFO] forward usable: train={len(fwd_train)} val={len(fwd_val)}", flush=True)

    if is_main:
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

    # ---------------- reverse direction setup: VANS + COIN mixed ----------------
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
    rev_vans_train, rev_vans_val, rev_vans_test = by_split["train"], by_split["val"], by_split["test"]

    coin_ids = list_coin_ids(args.coin_cache_dir)
    coin_train, coin_val, coin_test = split_coin_ids(coin_ids, args.coin_val_frac, args.coin_test_frac, args.seed)

    if args.max_train_items:
        rev_vans_train = rev_vans_train[: args.max_train_items]
        coin_train = coin_train[: args.max_train_items]
    rev_vans_val = rev_vans_val[: args.max_val_items]
    coin_val = coin_val[: args.max_val_items]

    rev_train_combined = [("vans", pid) for pid in rev_vans_train] + [("coin", vid) for vid in coin_train]
    if is_main:
        print(f"[INFO] reverse usable: vans_train={len(rev_vans_train)} coin_train={len(coin_train)} "
              f"combined={len(rev_train_combined)} vans_val={len(rev_vans_val)} coin_val={len(coin_val)} "
              f"vans_test={len(rev_vans_test)} coin_test={len(coin_test)}", flush=True)

    predictor = rev.build_predictor(device, guidance="crossattn")

    # ---------------- shared bridge, wired into both (unchanged from job-34) ----------------
    bridge = CrossModalBridge(jepa_dim=1024, vlm_dim=2048, hidden_dim=args.bridge_hidden_dim).to(device)
    injector.conditioner.adapter.mlp[0] = ForwardingAdapter(bridge.jepa_to_vlm)
    predictor.guidance_old_adapter = ForwardingAdapter(bridge.vlm_to_jepa, predictor.context_adapter)
    predictor.guidance_new_adapter = ForwardingAdapter(bridge.vlm_to_jepa, predictor.context_adapter)

    all_params = list(injector.parameters()) + list(predictor.parameters()) + list(bridge.parameters())
    opt = torch.optim.AdamW(all_params, lr=args.lr)
    n_fwd_params = sum(p.numel() for p in injector.parameters())
    n_rev_params = sum(p.numel() for p in predictor.parameters())
    n_bridge_params = sum(p.numel() for p in bridge.parameters())
    if is_main:
        print(f"[INFO] trainable params: forward_injector={n_fwd_params:,} "
              f"reverse_predictor={n_rev_params:,} shared_bridge={n_bridge_params:,} "
              f"cycle_loss_weight={args.cycle_loss_weight} disable_norm={args.disable_norm} "
              f"loss_norm_decay={args.loss_norm_decay} world_size={world_size}", flush=True)

    normalizer = LossNormalizer(decay=args.loss_norm_decay, disabled=args.disable_norm)

    log_f = open(os.path.join(run_dir, "train_log.jsonl"), "w") if is_main else None
    rng = random.Random(args.seed)
    step = 0
    n_fwd_skipped = 0
    n_rev_skipped = 0
    seen_exc_types = set()
    best_fwd_val_acc = -1.0
    best_rev_val_mse = float("inf")

    split_shuffle_rng = random.Random(args.seed)
    split_shuffle_rng.shuffle(fwd_train)
    split_shuffle_rng.shuffle(rev_train_combined)
    my_fwd_train = fwd_train[rank::world_size]
    my_rev_train = rev_train_combined[rank::world_size]
    n_steps_per_epoch = min(len(my_fwd_train), len(my_rev_train))
    if n_steps_per_epoch < 5:
        raise RuntimeError(
            f"too few paired train steps available per rank (forward={len(my_fwd_train)}, "
            f"reverse_combined={len(my_rev_train)}, world_size={world_size})"
        )
    if is_main:
        print(f"[INFO] n_steps_per_epoch={n_steps_per_epoch} per rank "
              f"(total forward={len(fwd_train)} reverse_combined={len(rev_train_combined)})", flush=True)

    for epoch in range(args.epochs):
        rng.shuffle(my_fwd_train)
        rng.shuffle(my_rev_train)
        for i in range(n_steps_per_epoch):
            fwd_item = my_fwd_train[i]
            rev_source, rev_id = my_rev_train[i]

            opt.zero_grad()
            total_loss = torch.zeros((), device=device)
            log_entry = {"step": step + 1, "epoch": epoch, "rev_source": rev_source}

            fwd_ok = False
            pooled_jepa = None
            try:
                option_lines, correct_letter = fwd.build_option_block(fwd_item, rng)
                prompt_text = fwd.build_prompt_text(fwd_item["question"], option_lines)
                inputs = fwd.build_inputs(processor, fwd_item["in_clip"], prompt_text)
                model_inputs = fwd.prepare_model_inputs(inputs, "cross_attn", n_tokens, placeholder_id)
                model_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in model_inputs.items()}
                jepa_feats = torch.from_numpy(fwd.load_vjepa_in_feats(fwd_item["in_clip"])).unsqueeze(0).to(device)
                pooled_jepa = jepa_feats.mean(dim=(1, 2))
                condition = injector.condition(jepa_feats)
                hook.set(condition)
                scores = torch.stack([
                    fwd.candidate_logprob(model, model_inputs, processor.tokenizer, "A"),
                    fwd.candidate_logprob(model, model_inputs, processor.tokenizer, "B"),
                ]).unsqueeze(0)
                hook.clear()
                target = torch.tensor([0 if correct_letter == "A" else 1], device=device)
                fwd_loss = nn.functional.cross_entropy(scores, target)
                fwd_loss_norm, fwd_ema = normalizer.normalize("fwd", fwd_loss)
                total_loss = total_loss + fwd_loss_norm
                log_entry["fwd_loss"] = float(fwd_loss.item())
                log_entry["fwd_loss_ema"] = fwd_ema
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
                if rev_source == "vans":
                    in_feats, out_feats, extras = rev.load_pair(args.cache_dir, rev_id, device)
                    pooled_vlm = extras["vlm_old"].mean(dim=(0, 1), keepdim=True)
                else:
                    in_feats, out_feats, extras = load_coin_pair(args.coin_cache_dir, rev_id, device)
                    pooled_vlm = None  # no vlm_old for COIN items -- cycle_loss handles this as None
                metrics = forward_step_mixed(predictor, in_feats, out_feats, extras, device)
                rev_loss = metrics["pred_loss"]
                rev_loss_norm, rev_ema = normalizer.normalize("rev", rev_loss)
                total_loss = total_loss + rev_loss_norm
                log_entry["rev_loss"] = float(rev_loss.item())
                log_entry["rev_loss_ema"] = rev_ema
                rev_ok = True
            except Exception as e:
                n_rev_skipped += 1
                exc_name = type(e).__name__
                print(f"[WARN] rev skip ({rev_source}) {rev_id}: {exc_name}: {e}", flush=True)
                if exc_name not in seen_exc_types:
                    seen_exc_types.add(exc_name)
                    traceback.print_exc()

            if not (fwd_ok or rev_ok):
                continue

            cyc_loss = bridge.cycle_loss(pooled_jepa, pooled_vlm)
            cyc_loss_norm, cyc_ema = normalizer.normalize("cyc", cyc_loss)
            total_loss = total_loss + args.cycle_loss_weight * cyc_loss_norm
            log_entry["cycle_loss"] = float(cyc_loss.item())
            log_entry["cycle_loss_ema"] = cyc_ema

            total_loss.backward()
            if is_ddp():
                sync_gradients(all_params, world_size)
            opt.step()
            step += 1

            if is_main:
                log_f.write(json.dumps(log_entry) + "\n")
                log_f.flush()

                if step % 50 == 0:
                    print(f"[epoch {epoch} step {step}] {log_entry}", flush=True)

            if is_main and step % args.val_every_steps == 0:
                fwd_val_tally = fwd.evaluate_logprob(
                    model, processor, hook, injector, fwd_val, "cross_attn", n_tokens, placeholder_id, device, rng,
                )
                rev_val_metrics, n_rev_val = evaluate_source(predictor, args.cache_dir, rev_vans_val, "vans", device)
                coin_val_metrics, n_coin_val = evaluate_source(predictor, args.coin_cache_dir, coin_val, "coin", device)
                print(f"[epoch {epoch} step {step}] fwd_val_acc={fwd_val_tally.summary_str()}  "
                      f"rev_val_mse(vans)={rev_val_metrics['pred_loss']:.4f} (n={n_rev_val})  "
                      f"rev_val_mse(coin)={coin_val_metrics['pred_loss']:.4f} (n={n_coin_val})", flush=True)
                log_f.write(json.dumps({
                    "step": step,
                    "fwd_val_acc": fwd_val_tally.acc, "fwd_val_acc_by_difficulty": fwd_val_tally.acc_by_difficulty,
                    "rev_val_mse_vans": rev_val_metrics["pred_loss"],
                    "rev_val_latent_dist_vans": rev_val_metrics["pred_latent_dist"],
                    "rev_val_cosine_dist_vans": rev_val_metrics["pred_latent_cosine_distance"],
                    "rev_val_mse_coin": coin_val_metrics["pred_loss"],
                    "rev_val_latent_dist_coin": coin_val_metrics["pred_latent_dist"],
                    "rev_val_cosine_dist_coin": coin_val_metrics["pred_latent_cosine_distance"],
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

            if is_main and args.save_every_steps and step % args.save_every_steps == 0:
                torch.save({
                    "step": step, "injector_state": injector.state_dict(),
                    "predictor_state": predictor.state_dict(), "bridge_state": bridge.state_dict(),
                }, os.path.join(run_dir, f"step_{step}.pt"))
                print(f"[epoch {epoch} step {step}] saved checkpoint -> step_{step}.pt", flush=True)

            did_rank0_only_work = (step % args.val_every_steps == 0) or (
                args.save_every_steps and step % args.save_every_steps == 0
            )
            if is_ddp() and did_rank0_only_work:
                dist.barrier()

    if is_main:
        best_reverse_path = os.path.join(run_dir, "best_reverse.pt")
        if os.path.exists(best_reverse_path):
            predictor.load_state_dict(torch.load(best_reverse_path, map_location=device, weights_only=False)["predictor_state"])
        rev_test_metrics, n_rev_test = evaluate_source(predictor, args.cache_dir, rev_vans_test, "vans", device)
        coin_test_metrics, n_coin_test = evaluate_source(predictor, args.coin_cache_dir, coin_test, "coin", device)
        print(f"[TEST] reverse (vans) n={n_rev_test} {rev_test_metrics}", flush=True)
        print(f"[TEST] reverse (coin) n={n_coin_test} {coin_test_metrics}", flush=True)
        with open(os.path.join(run_dir, "rev_test_metrics.json"), "w") as f:
            json.dump({
                "vans": {"n_test": n_rev_test, **rev_test_metrics},
                "coin": {"n_test": n_coin_test, **coin_test_metrics},
            }, f, indent=2)

        summary = {
            "done": True, "steps_completed": step,
            "fwd_items_skipped": n_fwd_skipped, "rev_items_skipped": n_rev_skipped,
            "best_fwd_val_acc": best_fwd_val_acc if step else None,
            "best_rev_val_mse_vans": best_rev_val_mse if step else None,
        }
        print(f"[DONE] {summary}", flush=True)
        log_f.write(json.dumps(summary) + "\n")
        log_f.flush()
        log_f.close()

    if is_ddp():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
