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
import json
import os
import random
import sys

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


def load_checkpoint_into_fresh_injector(checkpoint_path, hidden_size, layer_indices, seed, device):
    injector = LayerWiseJEPAInjector(
        hidden_size=hidden_size, layer_indices=layer_indices, condition_mode="jepa",
        seed=seed, mode="cross_attn",
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
                          "--checkpoint job22=/data/.../best_forward.pt")
    ap.add_argument("--qa_split", default=os.path.join(fwd.BASE, "raw_data/qa_split_temporal.json"))
    ap.add_argument("--split_part", default="test", choices=["val", "test", "train"])
    ap.add_argument("--layer_strategy", default="middle4", choices=["middle4", "last4", "uniform4", "all"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_path", default=None)
    args = ap.parse_args()

    checkpoints = []
    for spec in args.checkpoint:
        label, _, path = spec.partition("=")
        if not path:
            raise ValueError(f"--checkpoint must be label=path, got {spec!r}")
        checkpoints.append((label, path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.qa_split) as f:
        split = json.load(f)
    items = [it for it in split[args.split_part] if fwd.has_features(it)]
    print(f"[INFO] {args.split_part} split: {len(items)} usable items "
          f"(of {len(split[args.split_part])} total)", flush=True)

    print(f"[INFO] loading {fwd.MODEL_ID} (frozen) ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(fwd.MODEL_ID, dtype="auto").to(device)
    processor = AutoProcessor.from_pretrained(fwd.MODEL_ID)
    model.requires_grad_(False)
    model.eval()
    hidden_size = model.config.text_config.hidden_size
    n_tokens = 64
    placeholder_id = processor.tokenizer.pad_token_id
    layer_indices = select_layer_indices(len(resolve_decoder_layers(model)), args.layer_strategy)

    results = {}
    for label, checkpoint_path in checkpoints:
        injector = load_checkpoint_into_fresh_injector(checkpoint_path, hidden_size, layer_indices, args.seed, device)
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
            )
        finally:
            hook.remove()
        print(f"[RESULT] {label} {args.split_part} n={tally.n_total} val_acc(logprob)={tally.summary_str()}", flush=True)
        results[label] = {
            "checkpoint": checkpoint_path, "split_part": args.split_part,
            "n_total": tally.n_total, "acc": tally.acc,
            "acc_by_difficulty": tally.acc_by_difficulty,
        }

    print(f"[SUMMARY] {json.dumps(results, indent=2)}", flush=True)
    if args.out_path:
        with open(args.out_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
