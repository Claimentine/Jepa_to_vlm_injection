"""
Score several checkpoints from the same run_dir against the same held-out
test items in one pass, instead of one eval_qa_full.py invocation per
checkpoint. Each item's video is decoded and its A/B option order is chosen
exactly once, then every checkpoint scores that same input -- decode is the
dominant per-item cost throughout this project (seconds to tens of seconds,
vs. a frozen-model forward pass that's cheap by comparison), so this avoids
paying it once per checkpoint for no reason.

Motivating case: train_qa_full.py's best.pt only ever holds the single
snapshot that first reached a run's high-water mark, so a run whose val_acc
is noisy (e.g. random x cross_attn on qa_split_temporal.json crashing to
easy=0.08 at step 5000 before recovering) leaves no way to tell whether
best.pt's number was a fluke without checking neighboring checkpoints too.
--save_every_steps (added alongside this script) writes those out as
step_<N>.pt; this script scores several of them together.
"""
import argparse
import json
import os
import random
import sys
import traceback

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from common.jepa_injection_model import (  # noqa: E402
    DecoderLayerInjectionHook,
    JepaInjector,
    LanguageModelInjectionHook,
    LayerWiseJEPAInjector,
    resolve_decoder_layers,
    select_layer_indices,
)
from transformers import AutoModelForImageTextToText, AutoProcessor

from train_qa_full import (
    MODEL_ID, build_option_block, build_prompt_text, build_inputs, load_vjepa_in_feats, has_features, BASE,
    candidate_logprob, prepare_model_inputs,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=os.path.join(BASE, "raw_data/qa_split_full.json"))
    ap.add_argument("--injection", choices=["jepa", "random", "constant"], required=True)
    ap.add_argument("--injection_site", choices=["prefix", "film", "cross_attn"], default="prefix")
    ap.add_argument("--layer_strategy", choices=["middle4", "last4", "uniform4", "all"], default="middle4")
    ap.add_argument("--run_dir", required=True, help="directory holding the .pt checkpoint files")
    ap.add_argument("--checkpoint_names", required=True,
                    help="comma-separated filenames within --run_dir, e.g. 'best.pt,step_4000.pt,step_5000.pt'")
    ap.add_argument("--max_test_items", type=int, default=300, help="cap test set for cost control")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    n_tokens = 64
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    ckpt_names = [c.strip() for c in args.checkpoint_names.split(",") if c.strip()]
    out_path = args.out or os.path.join(BASE, f"raw_data/qa_eval_multi_{args.injection}.jsonl")

    with open(args.split) as f:
        split = json.load(f)
    test_items = [it for it in split["test"] if has_features(it)][: args.max_test_items]
    print(f"[INFO] {len(test_items)} usable test items, checkpoints={ckpt_names}")

    print(f"[INFO] loading {MODEL_ID} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="auto").to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.requires_grad_(False)
    model.eval()
    hidden_size = model.config.text_config.hidden_size
    placeholder_id = processor.tokenizer.pad_token_id

    # One injector/hook pair, reused for every checkpoint -- registering a
    # separate DecoderLayerInjectionHook per checkpoint would stack multiple
    # forward-pre-hooks on the same decoder layers, all firing on every call.
    # Swapping load_state_dict() in place between checkpoints keeps exactly
    # one hook active at a time.
    if args.injection_site == "prefix":
        injector = JepaInjector(jepa_dim=1024, hidden_size=hidden_size, mode=args.injection, pooling="temporal")
        hook = LanguageModelInjectionHook(model, n_tokens)
        layer_indices = []
    else:
        layer_indices = select_layer_indices(len(resolve_decoder_layers(model)), args.layer_strategy)
        injector = LayerWiseJEPAInjector(
            hidden_size=hidden_size, layer_indices=layer_indices, condition_mode=args.injection,
            mode=args.injection_site,
        )
        hook = DecoderLayerInjectionHook(model, injector)
    injector.to(device).eval()
    print(f"[INFO] injection={args.injection} site={args.injection_site} layers={layer_indices}")

    # Load every checkpoint's state dict up front (small, tens of MB each) so
    # swapping between them per item is just an in-memory load_state_dict(),
    # not a disk read.
    states = {}
    for name in ckpt_names:
        path = os.path.join(args.run_dir, name)
        ckpt = torch.load(path, map_location=device)
        saved_args = ckpt.get("args", {})
        checkpoint_site = saved_args.get("injection_site", "prefix")
        if checkpoint_site != args.injection_site:
            raise ValueError(f"{name}: checkpoint site={checkpoint_site}, requested {args.injection_site}")
        checkpoint_injection = saved_args.get("injection")
        if checkpoint_injection is not None and checkpoint_injection != args.injection:
            raise ValueError(f"{name}: checkpoint injection={checkpoint_injection}, requested {args.injection}")
        states[name] = ckpt["injector_state"]
        print(f"[INFO] loaded {name} (step={ckpt.get('step', 'n/a')})")

    rng = random.Random(args.seed)
    seen_exc_types = set()

    with open(out_path, "w") as out_f, torch.no_grad():
        for i, item in enumerate(test_items):
            option_lines, correct_letter = build_option_block(item, rng)
            prompt_text = build_prompt_text(item["question"], option_lines)
            try:
                inputs = build_inputs(processor, item["in_clip"], prompt_text)
                model_inputs = prepare_model_inputs(inputs, args.injection_site, n_tokens, placeholder_id)
                model_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in model_inputs.items()}
                feats = torch.from_numpy(load_vjepa_in_feats(item["in_clip"])).unsqueeze(0).to(device)

                for name, state in states.items():
                    injector.load_state_dict(state)
                    condition = injector(feats) if args.injection_site == "prefix" else injector.condition(feats)
                    hook.set(condition)
                    scores = torch.stack([
                        candidate_logprob(model, model_inputs, processor.tokenizer, "A"),
                        candidate_logprob(model, model_inputs, processor.tokenizer, "B"),
                    ])
                    hook.clear()
                    pred_letter = "A" if scores[0] > scores[1] else "B"
                    record = {
                        "pid": item["pid"], "difficulty": item.get("difficulty"), "checkpoint": name,
                        "correct_letter": correct_letter, "pred_letter": pred_letter,
                        "correct": pred_letter == correct_letter,
                        "logprob_A": float(scores[0].item()), "logprob_B": float(scores[1].item()),
                    }
                    out_f.write(json.dumps(record) + "\n")
            except Exception as e:
                exc_name = type(e).__name__
                for name in states:
                    out_f.write(json.dumps({
                        "pid": item["pid"], "difficulty": item.get("difficulty"), "checkpoint": name,
                        "error": f"{exc_name}: {e}",
                    }) + "\n")
                if exc_name not in seen_exc_types:
                    seen_exc_types.add(exc_name)
                    traceback.print_exc()

            out_f.flush()
            if (i + 1) % 25 == 0:
                print(f"[{i+1}/{len(test_items)}] done")

    print(f"[DONE] -> {out_path}")


if __name__ == "__main__":
    main()
