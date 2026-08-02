"""
Final held-out-test evaluation for the JEPA-injection classification
experiment, using real autoregressive generate() (not teacher forcing), so
numbers are directly comparable to the zero-shot baseline in
probe_results.jsonl (same past_window condition, same prompt format, same
"Final answer: <LETTER>" parsing).

Output schema matches probe_results.jsonl so score_task_recognition_probe.py
can be reused unchanged:
  python score_task_recognition_probe.py --results injection_eval_jepa.jsonl
"""
import sys
import argparse
import json
import os
import random
import re

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from task_categories import CATEGORIES
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from common.jepa_injection_model import JepaInjector, LanguageModelInjectionHook, prepend_placeholder_tokens  # noqa: E402
from train_jepa_injection_probe import (
    MODEL_ID, build_option_block, build_prompt_text, build_inputs, load_vjepa_feats,
)

ANSWER_RE = re.compile(r"final answer\s*[:\-]?\s*([A-M])\b", re.IGNORECASE)
FALLBACK_LETTER_RE = re.compile(r"\b([A-M])\b")


def parse_answer_letter(text, valid_letters):
    m = ANSWER_RE.search(text)
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper()
    matches = [m.group(1).upper() for m in FALLBACK_LETTER_RE.finditer(text) if m.group(1).upper() in valid_letters]
    return matches[-1] if matches else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="classification_split.json")
    ap.add_argument("--injection", choices=["jepa", "random", "linear_probe", "none"], required=True)
    ap.add_argument("--checkpoint", default=None, help="required unless --injection none")
    ap.add_argument("--pooling", choices=["temporal", "global", "auto"], default="auto",
                    help="auto (default): read from the checkpoint's saved training args")
    ap.add_argument("--n_tokens", type=int, default=8, help="only used when resolved pooling=global")
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=123)  # different seed than training's option shuffles
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    out_path = args.out or f"injection_eval_{args.injection}.jsonl"

    with open(args.split) as f:
        split = json.load(f)
    test_eps = split["test"]

    print(f"[INFO] loading {MODEL_ID} ...")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.requires_grad_(False)
    model.eval()
    device = next(model.parameters()).device
    hidden_size = model.config.text_config.hidden_size

    injector, hook, linear_head = None, None, None
    if args.injection in ("jepa", "random"):
        ckpt = torch.load(args.checkpoint, map_location=device)
        pooling = args.pooling
        if pooling == "auto":
            pooling = ckpt.get("args", {}).get("pooling", "global")  # old checkpoints predate --pooling
        n_tokens = 64 if pooling == "temporal" else args.n_tokens
        print(f"[INFO] resolved pooling={pooling} n_tokens={n_tokens} (from checkpoint)")
        injector = JepaInjector(jepa_dim=1024, hidden_size=hidden_size, n_tokens=n_tokens, mode=args.injection, pooling=pooling)
        injector.load_state_dict(ckpt["injector_state"])
        injector.to(device).eval()
        hook = LanguageModelInjectionHook(model, n_tokens)
        placeholder_id = processor.tokenizer.pad_token_id
    elif args.injection == "linear_probe":
        import torch.nn as nn
        ckpt = torch.load(args.checkpoint, map_location=device)
        linear_head = nn.Linear(hidden_size, len(CATEGORIES)).to(device)
        linear_head.load_state_dict(ckpt["linear_head_state"])
        linear_head.eval()

    rng = random.Random(args.seed)

    with open(out_path, "w") as out_f, torch.no_grad():
        for i, ep in enumerate(test_eps):
            option_lines, letter_to_category, category_to_letter = build_option_block(rng)
            prompt_text = build_prompt_text(option_lines)
            valid_letters = set(letter_to_category.keys())

            try:
                inputs = build_inputs(processor, ep["mp4_path"], prompt_text)
                model_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

                if args.injection == "linear_probe":
                    out = model(**model_inputs, output_hidden_states=True)
                    last_hidden = out.hidden_states[-1][:, -1, :]
                    logits = linear_head(last_hidden.float())
                    pred_category = CATEGORIES[logits.argmax(dim=-1).item()]
                    record = {
                        "episode_key": f"{ep['category']}/{ep['episode_id']}",
                        "condition": "past_window",
                        "true_category": ep["category"],
                        "pred_category": pred_category,
                        "correct": pred_category == ep["category"],
                        "raw_output": "<linear_probe: no generation>",
                    }
                else:
                    if args.injection in ("jepa", "random"):
                        feats = torch.from_numpy(load_vjepa_feats(ep["npz_path"])).unsqueeze(0).to(device)
                        soft_prompt = injector(feats)
                        input_ids, attn_mask = prepend_placeholder_tokens(
                            model_inputs["input_ids"], model_inputs["attention_mask"], n_tokens, placeholder_id
                        )
                        model_inputs["input_ids"] = input_ids
                        model_inputs["attention_mask"] = attn_mask
                        hook.set(soft_prompt)

                    generated_ids = model.generate(**model_inputs, max_new_tokens=args.max_new_tokens)
                    if hook is not None:
                        hook.clear()

                    trimmed = generated_ids[:, model_inputs["input_ids"].shape[1]:]
                    out_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
                    pred_letter = parse_answer_letter(out_text, valid_letters)
                    pred_category = letter_to_category.get(pred_letter) if pred_letter else None

                    record = {
                        "episode_key": f"{ep['category']}/{ep['episode_id']}",
                        "condition": "past_window",
                        "true_category": ep["category"],
                        "true_letter": category_to_letter[ep["category"]],
                        "pred_category": pred_category,
                        "pred_letter": pred_letter,
                        "correct": pred_category == ep["category"],
                        "raw_output": out_text,
                    }
            except Exception as e:
                record = {
                    "episode_key": f"{ep['category']}/{ep['episode_id']}",
                    "condition": "past_window",
                    "true_category": ep["category"],
                    "error": f"{type(e).__name__}: {e}",
                }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            if (i + 1) % 25 == 0:
                print(f"[{i+1}/{len(test_eps)}] done")

    print(f"[DONE] -> {out_path}")


if __name__ == "__main__":
    main()
