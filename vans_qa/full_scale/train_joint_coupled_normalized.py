"""Joint/multi-task training (see train_joint_coupled.py), with one change:
adaptive EMA-based loss normalization before summing fwd_loss + rev_loss +
cycle_loss_weight * cycle_loss, to test whether job-22's forward-accuracy
regression (best_fwd_val_acc=0.72 on a 100-item snapshot, and confirmed
0.6811 on the FULL 2854-item test split -- clearly below job-13's
independent-training 0.7477) is caused by a raw loss-magnitude imbalance
rather than (or in addition to) the shared-bottleneck architecture itself.

Motivation (from this session's diagnosis): rev_loss (a JEPA future-latent
MSE) sits chronically around 7-9 across every reverse-direction run this
project has ever logged (job-15/16/22/28/30's own train_log.jsonl all show
this), while fwd_loss (a 2-way cross-entropy) is usually well under 1 once
training progresses, dropping toward 0 for confident-correct predictions.
Summed 1:1 (train_joint_coupled.py's own total_loss = fwd_loss + rev_loss +
cycle_loss_weight * cyc_loss), rev_loss's gradient into the SHARED bridge
parameters numerically dominates fwd_loss's, especially as forward improves
and its own loss shrinks while reverse's stays chronically high -- a
plausible mechanism for the shared bridge being pulled toward what serves
reverse, at forward's expense. cyc_loss's own raw magnitude is also
observed to be spiky (0.15-25+ in job-30's own step logs), an second,
independent source of imbalance.

Normalization: each loss is divided by an exponential moving average of its
own recent magnitude (decay=0.98 default) BEFORE being combined -- a
simple, common multi-task-learning trick (loss-magnitude/uncertainty
weighting; see e.g. Kendall et al. 2018 for the more principled version).
The EMA is computed from detached .item() values (no gradient flows through
the normalizer itself), so this only rescales each task's contribution to
the shared parameters' gradient at each step, not the underlying loss
computation. --cycle_loss_weight is still applied on top of the normalized
cyc term, preserving the original design's intent that cycle-consistency
stays a secondary regularizer, not an equal-weight third task.

Everything else -- data, architecture, shared-bridge wiring, step budget
(n_steps_per_epoch = min(fwd_train, rev_train), unchanged from job-22, no
COIN data here) -- is identical to train_joint_coupled.py, so this run is
directly comparable to job-22: if forward accuracy recovers toward job-13's
independent-training level, the loss-imbalance hypothesis is the primary
driver; if it's still degraded, the shared-bottleneck architecture itself
is more likely the real constraint.

Reuses train_qa_full.py/train_latent_world_model_full.py unmodified
(imported as fwd/rev, same as train_joint_coupled.py); that script itself
also stays unmodified as the VANS-only-joint, un-normalized baseline this
run is compared against.

DDP: single-node, manual gradient all-reduce rather than
nn.parallel.DistributedDataParallel module-wrapping. Regular DDP tracks
parameter usage via hooks attached to a wrapped module's own forward()
call, but this script's forward direction calls injector.condition(...)
directly and drives the frozen VLM through DecoderLayerInjectionHook's
forward-pre-hooks (see common/jepa_injection_model.py) -- neither goes
through a single wrapped forward() the way train_latent_world_model_ddp.py's
plain predictor(...) call does, so DDP's automatic bucketing assumptions
don't cleanly apply here. Instead: each rank runs a normal local
forward+backward on its own data shard (regular autograd, no wrapper), then
every trainable parameter's .grad is manually all-reduced (averaged) across
ranks before optimizer.step() -- functionally identical to what DDP does
internally, just done explicitly so it doesn't depend on which specific
method call produced the gradient. Every rank always all-reduces the SAME
fixed, ordered parameter list (injector + predictor + bridge) each step,
substituting a zero tensor for any parameter this rank's own item(s) didn't
happen to touch (e.g. an unused film/adaln branch, or one direction's item
failing to load) -- required so the collective call sequence matches across
ranks regardless of which sub-branches fired on which rank this step.
Falls back to plain single-process behavior when WORLD_SIZE is unset/1.

Validation/checkpointing are rank-0-only (real VLM inference over
--max_val_items items -- redundant and slow to duplicate on every rank).
Confirmed 2026-09-20 the hard way (job-33's first DDP smoke test) that
without an explicit dist.barrier() right after that block, non-main ranks
race ahead to the next step's backward()+sync_gradients() while rank 0 is
still validating, and hang on an all_reduce rank 0 hasn't reached yet --
NCCL's watchdog then kills the whole job with a collective-timeout error
after its default 10 minutes. Fixed with a barrier every rank hits right
after the val/checkpoint block, plus a longer (30 min) process-group
timeout for headroom on a full-scale run's larger validation set.
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


def is_ddp():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_ddp():
    # Default NCCL collective timeout (10 min) was too tight: rank 0's
    # validation pass (real VLM inference over --max_val_items items) is
    # rank-0-only, unguarded work that other ranks don't wait for except at
    # the explicit dist.barrier() this script now adds right after it --
    # confirmed 2026-09-20 that job-33's first DDP smoke test's validation
    # alone exceeded 10 minutes. 30 minutes gives real headroom for a
    # full-scale run's larger --max_val_items without needing to reason
    # precisely about how long validation takes.
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def sync_gradients(params, world_size):
    """Manual DDP: all-reduce (average) every trainable parameter's grad,
    across a FIXED, identically-ordered list on every rank. Any parameter
    this rank's step didn't touch gets a zero placeholder first, so every
    rank calls all_reduce on the exact same sequence of tensors regardless
    of which sub-branches (film/adaln, forward vs. reverse) fired locally."""
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


class LossNormalizer:
    """Tracks an EMA of each named loss's raw magnitude and returns
    loss / (ema + eps) -- a cheap per-task loss-scale balancer. The EMA is
    seeded with the first raw value seen for that name (not 0), so the very
    first step of each loss doesn't produce a huge/undefined normalized
    spike from a near-zero denominator."""

    def __init__(self, decay=0.98, eps=1e-4):
        self.decay = decay
        self.eps = eps
        self.ema = {}

    def normalize(self, name, raw_loss_tensor):
        raw_value = float(raw_loss_tensor.item())
        if name not in self.ema:
            self.ema[name] = raw_value
        else:
            self.ema[name] = self.decay * self.ema[name] + (1 - self.decay) * raw_value
        return raw_loss_tensor / (self.ema[name] + self.eps), self.ema[name]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qa_split", default=os.path.join(fwd.BASE, "raw_data/qa_split_temporal.json"))
    ap.add_argument("--layer_strategy", default="middle4", choices=["middle4", "last4", "uniform4", "all"])
    ap.add_argument("--cache_dir", default=os.path.join(rev.WORK_BASE, "vlm_guidance_cache_paired"))
    ap.add_argument("--rev_qa_split", default=os.path.join(rev.BASE, "raw_data/qa_split_full.json"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--cycle_loss_weight", type=float, default=0.1)
    ap.add_argument("--loss_norm_decay", type=float, default=0.98,
                     help="EMA decay for the per-loss magnitude normalizer -- higher means "
                          "slower-adapting/more stable, lower reacts faster to loss-scale drift")
    ap.add_argument("--val_every_steps", type=int, default=500)
    ap.add_argument("--save_every_steps", type=int, default=1000)
    ap.add_argument("--max_val_items", type=int, default=100)
    ap.add_argument("--max_train_items", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=os.path.join(fwd.BASE, "raw_data/joint_coupled_normalized_runs"))
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

    # ---------------- forward direction setup ----------------
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
    if is_main:
        print(f"[INFO] reverse usable: train={len(rev_train)} val={len(rev_val)} test={len(rev_test)}", flush=True)

    predictor = rev.build_predictor(device, guidance="crossattn")

    # ---------------- shared bridge, wired into both (unchanged from job-22) ----------------
    bridge = CrossModalBridge(jepa_dim=1024, vlm_dim=2048).to(device)
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
              f"cycle_loss_weight={args.cycle_loss_weight} loss_norm_decay={args.loss_norm_decay} "
              f"world_size={world_size}", flush=True)

    normalizer = LossNormalizer(decay=args.loss_norm_decay)

    log_f = open(os.path.join(run_dir, "train_log.jsonl"), "w") if is_main else None
    rng = random.Random(args.seed)
    step = 0
    n_fwd_skipped = 0
    n_rev_skipped = 0
    seen_exc_types = set()
    best_fwd_val_acc = -1.0
    best_rev_val_mse = float("inf")

    # Same seed on every rank so the pre-shard order matches, then each rank
    # takes a disjoint [rank::world_size] slice, truncated to a common
    # length -- guarantees identical step counts across ranks (required so
    # every rank calls sync_gradients the same number of times).
    split_shuffle_rng = random.Random(args.seed)
    split_shuffle_rng.shuffle(fwd_train)
    split_shuffle_rng.shuffle(rev_train)
    my_fwd_train = fwd_train[rank::world_size]
    my_rev_train = rev_train[rank::world_size]
    n_steps_per_epoch = min(len(my_fwd_train), len(my_rev_train))
    if n_steps_per_epoch < 5:
        raise RuntimeError(
            f"too few paired train steps available per rank (forward={len(my_fwd_train)}, "
            f"reverse={len(my_rev_train)}, world_size={world_size})"
        )
    if is_main:
        print(f"[INFO] n_steps_per_epoch={n_steps_per_epoch} per rank "
              f"(total forward={len(fwd_train)} reverse={len(rev_train)})", flush=True)

    for epoch in range(args.epochs):
        rng.shuffle(my_fwd_train)
        rng.shuffle(my_rev_train)
        for i in range(n_steps_per_epoch):
            fwd_item = my_fwd_train[i]
            rev_pid = my_rev_train[i]

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
                in_feats, out_feats, extras = rev.load_pair(args.cache_dir, rev_pid, device)
                pooled_vlm = extras["vlm_old"].mean(dim=(0, 1), keepdim=True)
                metrics = rev.forward_step(predictor, in_feats, out_feats, extras, "crossattn", device)
                rev_loss = metrics["pred_loss"]
                rev_loss_norm, rev_ema = normalizer.normalize("rev", rev_loss)
                total_loss = total_loss + rev_loss_norm
                log_entry["rev_loss"] = float(rev_loss.item())
                log_entry["rev_loss_ema"] = rev_ema
                rev_ok = True
            except Exception as e:
                n_rev_skipped += 1
                exc_name = type(e).__name__
                print(f"[WARN] rev skip {rev_pid}: {exc_name}: {e}", flush=True)
                if exc_name not in seen_exc_types:
                    seen_exc_types.add(exc_name)
                    traceback.print_exc()

            if not (fwd_ok or rev_ok):
                # Known, accepted DDP edge case: if this happens on only SOME
                # ranks in the same step (both fwd and rev failing on the
                # same item, on this rank specifically), those ranks skip
                # this iteration's collectives while others proceed to
                # sync_gradients -- a call-count mismatch. Not fixed here:
                # empirically both directions failing on the identical step
                # simultaneously is far rarer than either failing alone
                # (single-digit occurrences across thousands of steps in
                # every non-DDP run this project has logged), and NCCL
                # surfaces a mismatch as a clear timeout/error rather than a
                # silent hang, so the failure mode is "resubmit", not "wasted
                # GPU-days undetected".
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

            if is_main and args.save_every_steps and step % args.save_every_steps == 0:
                torch.save({
                    "step": step, "injector_state": injector.state_dict(),
                    "predictor_state": predictor.state_dict(), "bridge_state": bridge.state_dict(),
                }, os.path.join(run_dir, f"step_{step}.pt"))
                print(f"[epoch {epoch} step {step}] saved checkpoint -> step_{step}.pt", flush=True)

            # `step` is incremented identically on every rank (all ranks run
            # the same n_steps_per_epoch in lockstep -- see the known,
            # accepted continue-on-total-failure edge case above), so every
            # rank evaluates step % args.val_every_steps identically -- but
            # only rank 0 actually RUNS the validation/checkpoint block above
            # (real VLM inference over --max_val_items items, genuinely slow,
            # confirmed 2026-09-20 to exceed NCCL's default 10-minute
            # collective timeout on job-33's first DDP smoke test). Without
            # this barrier, non-main ranks race ahead to the NEXT step's
            # backward()+sync_gradients() while rank 0 is still validating,
            # and hang on an all_reduce rank 0 hasn't reached yet ->
            # "Watchdog caught collective operation timeout". Every rank
            # must wait here so nobody starts the next step's collectives
            # before rank 0 is done.
            did_rank0_only_work = (step % args.val_every_steps == 0) or (
                args.save_every_steps and step % args.save_every_steps == 0
            )
            if is_ddp() and did_rank0_only_work:
                dist.barrier()

    if is_main:
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

    if is_ddp():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
