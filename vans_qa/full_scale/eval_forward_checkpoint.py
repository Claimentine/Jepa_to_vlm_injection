"""Evaluates a saved forward-direction (JEPA->VLM QA) injector checkpoint on
the FULL held-out val/test split, not the small --max_val_items sample
(typically n=100) used for periodic in-training validation.

Why this matters: every forward val_acc number reported so far in this
project's training logs (job-13, job-22, job-30, ...) came from a 100-item
snapshot taken every --val_every_steps -- and this direction's trajectory is
consistently noisy (see job-13's own 0.75->0.51->0.46 swings). A single
"best" checkpoint picked by the highest of those noisy 100-item snapshots
(e.g. job-30's best_forward.pt, val_acc=0.75 at step 3000) needs a real,
low-variance number before it means anything -- qa_split_temporal.json's
actual test split has 2866 items, ~29x the sample size.

Handles two checkpoint shapes, auto-detected by whether "bridge_state" is
present:
  - Plain forward-only checkpoints (train_qa_full.py, e.g. job-13's best.pt):
    just {"injector_state": ...}.
  - Joint-training checkpoints with a shared CrossModalBridge
    (train_joint_coupled.py / train_joint_coupled_mixed.py, e.g. job-22/
    job-30's best_forward.pt): {"injector_state": ..., "bridge_state": ...}.
    The injector's own state_dict does NOT include the bridge's actual
    weights -- injector.conditioner.adapter.mlp[0] was replaced with a
    ForwardingAdapter(bridge.jepa_to_vlm) during training (see
    common/cross_modal_bridge.py: ForwardingAdapter is deliberately
    parameter-free, so it never shows up in injector.state_dict()). Loading
    only injector_state here would silently evaluate with an
    UNTRAINED substitute in that slot. This script rewires the same
    ForwardingAdapter substitution before loading both state dicts, so the
    reconstructed model is bit-for-bit what was actually checkpointed.

Reuses train_qa_full.py's evaluate_logprob/build_option_block/etc.
unmodified (imported, not copied).

Accepts one or more checkpoints via repeated --checkpoint label=path
(evaluated in one process, sharing the single loaded Qwen3-VL-2B -- avoids
paying that load cost, and the GPU-scheduling wait to get a pod at all on
this cluster, once per checkpoint) so a job-13 vs job-22 vs job-30 style
comparison runs as a single job.
"""
import argparse
import collections
import json
import os
import random
import sys
import traceback

import torch
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

NONE_SENTINEL = "NONE"


def evaluate_logprob_none(model, processor, items, device, rng, prefetch_depth=4, precompute_cache_dir=None):
    """Zero-shot frozen-VLM baseline over the FULL split: same prompt/option/
    decode pipeline as evaluate_logprob, but with no hook and no injector at
    all -- not even a placeholder/condition tensor. This is the "injection=none"
    control from train_qa_full.py (there only ever run on the 100-item
    val snapshot of one split), reusing its exact per-item construction so
    the comparison against a real checkpoint's full-test-set accuracy is
    apples-to-apples.
    """
    model.eval()
    tally = fwd.AccTally()
    seen_exc_types = set()
    next_to_start = 0
    queue = collections.deque()

    def _top_up():
        nonlocal next_to_start
        while next_to_start < len(items) and len(queue) < prefetch_depth:
            queue.append(fwd._start_item_decode(items[next_to_start], rng, precompute_cache_dir))
            next_to_start += 1

    _top_up()
    with torch.no_grad():
        for item in items:
            cur_ok, cur_payload = queue.popleft()
            _top_up()
            try:
                if not cur_ok:
                    raise cur_payload
                if cur_payload[0] == "cached":
                    _, correct_letter, inputs = cur_payload
                else:
                    option_lines, correct_letter, prompt_text, handle = cur_payload
                    decoded = fwd.finish_decode(handle)
                    inputs = fwd.build_inputs_from_decoded(processor, handle.abs_path, prompt_text, decoded)
                model_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
                scores = torch.stack([
                    fwd.candidate_logprob(model, model_inputs, processor.tokenizer, "A"),
                    fwd.candidate_logprob(model, model_inputs, processor.tokenizer, "B"),
                ])
                tally.add(item, ("A" if scores[0] > scores[1] else "B") == correct_letter)
            except Exception as e:
                exc_name = type(e).__name__
                print(f"[WARN] eval(none): skipping item {item.get('pid')}: {exc_name}: {e}", flush=True)
                if exc_name not in seen_exc_types:
                    seen_exc_types.add(exc_name)
                    traceback.print_exc()
    if tally.n_total == 0 and items:
        print(f"[WARN] eval(none): 0/{len(items)} items succeeded -- val_acc is meaningless", flush=True)
    return tally


def load_checkpoint_into_fresh_injector(checkpoint_path, model, hidden_size, default_layer_strategy, seed, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Older checkpoints (job-22 through job-79) predate these fields -- they
    # were all cross_attn/middle4, so that's the correct fallback, not just
    # an arbitrary default.
    injection_mode = ckpt.get("injection_mode", "cross_attn")
    layer_strategy = ckpt.get("layer_strategy", default_layer_strategy)
    layer_indices = select_layer_indices(len(resolve_decoder_layers(model)), layer_strategy)
    injector = LayerWiseJEPAInjector(
        hidden_size=hidden_size, layer_indices=layer_indices, condition_mode="jepa",
        seed=seed, mode=injection_mode,
    ).to(device)
    print(f"[INFO] {checkpoint_path}: injection_mode={injection_mode} layer_strategy={layer_strategy} "
          f"({'from checkpoint' if 'injection_mode' in ckpt else 'default -- not recorded in this checkpoint'})",
          flush=True)
    if "bridge_state" in ckpt:
        # jepa_to_vlm.weight (single Linear) vs jepa_to_vlm.0.weight (2-layer
        # MLP, --bridge_hidden_dim run) tells us which architecture to build;
        # the MLP's hidden width is just that layer's own out_features.
        bridge_state = ckpt["bridge_state"]
        if "jepa_to_vlm.weight" in bridge_state:
            hidden_dim = None
        else:
            hidden_dim = bridge_state["jepa_to_vlm.0.weight"].shape[0]
        print(f"[INFO] {checkpoint_path}: has a shared CrossModalBridge (hidden_dim={hidden_dim}) -- "
              "rewiring ForwardingAdapter before loading state dicts", flush=True)
        bridge = CrossModalBridge(jepa_dim=1024, vlm_dim=2048, hidden_dim=hidden_dim).to(device)
        injector.conditioner.adapter.mlp[0] = ForwardingAdapter(bridge.jepa_to_vlm)
        bridge.load_state_dict(bridge_state)
    injector.load_state_dict(ckpt["injector_state"])
    injector.eval()
    print(f"[INFO] loaded {checkpoint_path} (step={ckpt.get('step')}, "
          f"checkpoint's own val_acc={ckpt.get('val_acc')})", flush=True)
    return injector


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", action="append", required=True,
                     help="label=path, repeatable -- e.g. --checkpoint job13=/data/.../best.pt "
                          "--checkpoint job22=/data/.../best_forward.pt -- path may also be the "
                          f"sentinel {NONE_SENTINEL!r} for a true zero-shot frozen-VLM baseline "
                          "(no injector, no hook at all), e.g. --checkpoint zeroshot=NONE")
    ap.add_argument("--model_id", default=None,
                     help="HF model id for the frozen VLM, overriding fwd.MODEL_ID "
                          "(Qwen/Qwen3-VL-2B-Thinking) -- for A2's additional zero-shot VLM "
                          "baselines (e.g. Qwen3.5-VL). Only meaningful with --checkpoint "
                          f"label={NONE_SENTINEL!r} (zero-shot); a real checkpoint's injector was "
                          "trained against the ORIGINAL model's hidden_size/tokenizer, so loading "
                          "it onto a different --model_id is unsupported and will likely error or "
                          "silently produce meaningless results.")
    ap.add_argument("--qa_split", action="append",
                     default=None,
                     help="repeatable -- one or more qa_split JSON files to evaluate in the SAME "
                          "process, so the frozen VLM is loaded from disk only ONCE regardless of "
                          "how many benchmarks run. Added after job-90 spent ~30-40min per "
                          "benchmark reloading the IDENTICAL checkpoint from a slow CephFS mount, "
                          "at ~0% GPU utilization each time -- the same idle-GPU pattern job-54's "
                          "own incident (see build_eval_input_cache.py's docstring) was about, just "
                          "from repeated weight loads instead of live CPU decode work. Defaults to "
                          "a single-element list (qa_split_temporal.json) if omitted, preserving "
                          "old single-benchmark-per-invocation behavior.")
    ap.add_argument("--split_part", default="test", choices=["val", "test", "train"],
                     help="applies to every --qa_split in this run")
    ap.add_argument("--layer_strategy", default="middle4", choices=["middle4", "last4", "uniform4", "all"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_path", action="append", default=None,
                     help="repeatable, one per --qa_split IN THE SAME ORDER -- pass this flag the "
                          "same number of times as --qa_split, or omit entirely")
    ap.add_argument("--limit", type=int, default=None,
                     help="cap on the number of items evaluated per checkpoint PER qa_split -- for "
                          "a quick smoke check (e.g. validating a change to evaluate_logprob's "
                          "decode pipeline) before committing to a full multi-thousand-item run")
    ap.add_argument("--precompute_cache_dir", action="append", default=None,
                     help="repeatable, one per --qa_split IN THE SAME ORDER (pass '' for a "
                          "qa_split with no cache) -- directory of build_eval_input_cache.py "
                          "output; if given for a split, evaluate_logprob loads each item's "
                          "already-decoded-and-processed model inputs from here instead of doing "
                          "that CPU-bound work live, falling back to live building on any "
                          "per-item cache miss")
    ap.add_argument("--resume", action="store_true",
                     help="if --out_path already exists, load it first and skip any checkpoint "
                          "label already present with the same split_part -- lets a multi-checkpoint "
                          "run recover from a crash (node eviction, CUDA fault, etc.) without "
                          "re-evaluating checkpoints that already finished. Off by default so a "
                          "plain rerun with the same --out_path always re-evaluates everything.")
    args = ap.parse_args()

    checkpoints = []
    for spec in args.checkpoint:
        label, _, path = spec.partition("=")
        if not path:
            raise ValueError(f"--checkpoint must be label=path, got {spec!r}")
        checkpoints.append((label, path))

    qa_splits = args.qa_split or [os.path.join(fwd.BASE, "raw_data/qa_split_temporal.json")]
    out_paths = args.out_path or [None] * len(qa_splits)
    cache_dirs = args.precompute_cache_dir or [None] * len(qa_splits)
    if len(out_paths) != len(qa_splits):
        ap.error(f"--out_path given {len(out_paths)} time(s) but --qa_split given {len(qa_splits)} "
                  "time(s) -- pass one --out_path per --qa_split, in the same order, or omit both")
    if len(cache_dirs) != len(qa_splits):
        ap.error(f"--precompute_cache_dir given {len(cache_dirs)} time(s) but --qa_split given "
                  f"{len(qa_splits)} time(s) -- pass one per --qa_split (use '' for none), in the "
                  "same order, or omit entirely")
    cache_dirs = [c if c else None for c in cache_dirs]
    benchmarks = list(zip(qa_splits, out_paths, cache_dirs))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_id = args.model_id or fwd.MODEL_ID
    print(f"[INFO] loading {model_id} (frozen) ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype="auto").to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    model.requires_grad_(False)
    model.eval()
    hidden_size = model.config.text_config.hidden_size
    n_tokens = 64
    placeholder_id = processor.tokenizer.pad_token_id

    for qa_split_path, out_path, cache_dir in benchmarks:
        print(f"[INFO] === benchmark: {qa_split_path} ===", flush=True)
        with open(qa_split_path) as f:
            split = json.load(f)
        items = [it for it in split[args.split_part] if fwd.has_features(it)]
        print(f"[INFO] {args.split_part} split: {len(items)} usable items "
              f"(of {len(split[args.split_part])} total)", flush=True)
        if args.limit:
            items = items[: args.limit]
            print(f"[INFO] --limit {args.limit}: evaluating {len(items)} items", flush=True)

        results = {}
        if args.resume and out_path and os.path.exists(out_path):
            with open(out_path) as f:
                results = json.load(f)
            print(f"[INFO] --resume: loaded {len(results)} existing result(s) from {out_path}", flush=True)

        for label, checkpoint_path in checkpoints:
            existing = results.get(label)
            if args.resume and existing is not None and existing.get("split_part") == args.split_part:
                print(f"[SKIP] {label} {args.split_part}: already in {out_path} "
                      f"(n={existing.get('n_total')} acc={existing.get('acc')})", flush=True)
                continue
            if checkpoint_path == NONE_SENTINEL:
                rng = random.Random(args.seed)
                tally = evaluate_logprob_none(
                    model, processor, items, device, rng,
                    precompute_cache_dir=cache_dir,
                )
            else:
                injector = load_checkpoint_into_fresh_injector(checkpoint_path, model, hidden_size, args.layer_strategy, args.seed, device)
                # A fresh hook per checkpoint (its pre_hook closures bake in this
                # specific injector's own trained weights via self.injector.apply) --
                # must remove() it before the next checkpoint's hook is created, or
                # the old one's forward-pre-hooks stay registered on the same frozen
                # decoder layers and stack on top of the new one.
                hook = DecoderLayerInjectionHook(model, injector)
                try:
                    rng = random.Random(args.seed)
                    tally = fwd.evaluate_logprob(
                        model, processor, hook, injector, items, "cross_attn", n_tokens, placeholder_id, device, rng,
                        precompute_cache_dir=cache_dir,
                    )
                finally:
                    hook.remove()
            print(f"[RESULT] {label} {args.split_part} n={tally.n_total} val_acc(logprob)={tally.summary_str()}", flush=True)
            results[label] = {
                "checkpoint": checkpoint_path, "split_part": args.split_part,
                "n_total": tally.n_total, "acc": tally.acc,
                "acc_by_difficulty": tally.acc_by_difficulty,
            }
            # Write after EVERY checkpoint, not just at the end -- a crash
            # (node eviction, CUDA fault) partway through a multi-checkpoint run
            # must not lose already-finished results, or --resume above would
            # have nothing to skip on the next attempt.
            if out_path:
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)

        print(f"[SUMMARY] {qa_split_path}: {json.dumps(results, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
