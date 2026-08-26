"""
Full-scale held-out evaluation using paired A/B continuation log-probabilities.
This deliberately avoids autoregressive generation and answer parsing, so
training, validation, and held-out evaluation use the same decision rule.
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
    ap.add_argument("--injection", choices=["jepa", "random", "constant", "none"], required=True)
    ap.add_argument("--injection_site", choices=["prefix", "film", "cross_attn"], default="prefix")
    ap.add_argument("--layer_strategy", choices=["middle4", "last4", "uniform4", "all"], default="middle4")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--max_test_items", type=int, default=300, help="cap test set for cost control")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    n_tokens = 64
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    out_path = args.out or os.path.join(BASE, f"raw_data/qa_eval_full_{args.injection}.jsonl")

    with open(args.split) as f:
        split = json.load(f)
    test_items = [it for it in split["test"] if has_features(it)][: args.max_test_items]
    print(f"[INFO] {len(test_items)} usable test items")

    print(f"[INFO] loading {MODEL_ID} ...")
    # See train_qa_full.py's model-loading comment: device_map="auto" can
    # silently CPU-offload a decoder layer, which breaks
    # DecoderLayerInjectionHook's film/cross_attn injection with a
    # CPU/CUDA device-mismatch RuntimeError on every item. Pin to one device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="auto").to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.requires_grad_(False)
    model.eval()
    hidden_size = model.config.text_config.hidden_size

    injector, hook, layer_indices = None, None, []
    if args.injection != "none":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required unless --injection none")
        ckpt = torch.load(args.checkpoint, map_location=device)
        saved_args = ckpt.get("args", {})
        checkpoint_site = saved_args.get("injection_site", "prefix")
        if checkpoint_site != args.injection_site:
            raise ValueError(f"checkpoint site={checkpoint_site}, requested {args.injection_site}")
        # film/cross_attn checkpoints for jepa/random/constant conditions share
        # identical parameter shapes, so a mismatched --injection here would
        # load_state_dict() successfully and silently score this checkpoint
        # under the wrong condition instead of failing loudly.
        checkpoint_injection = saved_args.get("injection")
        if checkpoint_injection is not None and checkpoint_injection != args.injection:
            raise ValueError(f"checkpoint injection={checkpoint_injection}, requested {args.injection}")
        if args.injection_site == "prefix":
            injector = JepaInjector(
                jepa_dim=1024, hidden_size=hidden_size, mode=args.injection,
                seed=saved_args.get("seed", 0), pooling="temporal",
            )
            hook = LanguageModelInjectionHook(model, n_tokens)
        else:
            strategy = saved_args.get("layer_strategy", args.layer_strategy)
            layer_indices = select_layer_indices(len(resolve_decoder_layers(model)), strategy)
            injector = LayerWiseJEPAInjector(
                hidden_size=hidden_size, layer_indices=layer_indices, condition_mode=args.injection,
                seed=saved_args.get("seed", 0), mode=args.injection_site,
            )
            hook = DecoderLayerInjectionHook(model, injector)
        injector.load_state_dict(ckpt["injector_state"])
        injector.to(device).eval()
        placeholder_id = processor.tokenizer.pad_token_id
        print(f"[INFO] injection={args.injection} site={args.injection_site} layers={layer_indices}")

    rng = random.Random(args.seed)
    seen_exc_types = set()

    with open(out_path, "w") as out_f, torch.no_grad():
        for i, item in enumerate(test_items):
            option_lines, correct_letter = build_option_block(item, rng)
            prompt_text = build_prompt_text(item["question"], option_lines)
            try:
                inputs = build_inputs(processor, item["in_clip"], prompt_text)
                site = args.injection_site if injector is not None else "none"
                model_inputs = prepare_model_inputs(inputs, site, n_tokens, placeholder_id if injector else None)
                model_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in model_inputs.items()}

                if injector is not None:
                    feats = torch.from_numpy(load_vjepa_in_feats(item["in_clip"])).unsqueeze(0).to(device)
                    condition = injector(feats) if args.injection_site == "prefix" else injector.condition(feats)
                    hook.set(condition)
                scores = torch.stack([
                    candidate_logprob(model, model_inputs, processor.tokenizer, "A"),
                    candidate_logprob(model, model_inputs, processor.tokenizer, "B"),
                ])
                if hook is not None:
                    hook.clear()
                pred_letter = "A" if scores[0] > scores[1] else "B"

                record = {
                    "pid": item["pid"], "difficulty": item.get("difficulty"),
                    "correct_letter": correct_letter, "pred_letter": pred_letter,
                    "correct": pred_letter == correct_letter,
                    "logprob_A": float(scores[0].item()), "logprob_B": float(scores[1].item()),
                }
            except Exception as e:
                exc_name = type(e).__name__
                record = {"pid": item["pid"], "difficulty": item.get("difficulty"), "error": f"{exc_name}: {e}"}
                if exc_name not in seen_exc_types:
                    seen_exc_types.add(exc_name)
                    traceback.print_exc()

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            if (i + 1) % 25 == 0:
                print(f"[{i+1}/{len(test_items)}] done")

    print(f"[DONE] -> {out_path}")


if __name__ == "__main__":
    main()
