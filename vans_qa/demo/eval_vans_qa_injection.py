"""
Final held-out test evaluation for the VANS 2-choice QA pilot, using real
autoregressive generate() (matches the classification experiments'
methodology, not teacher forcing) so jepa/random/none are on equal footing.

`none` = frozen baseline, no injection, no training -- added here since the
training script only covers jepa/random (nothing to train for the baseline).
"""
import argparse
import json
import os
import random
import re
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from common.jepa_injection_model import JepaInjector, LanguageModelInjectionHook, prepend_placeholder_tokens  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor

from train_vans_qa_injection import (
    MODEL_ID, build_option_block, build_prompt_text, build_inputs, load_vjepa_in_feats,
)

ANSWER_RE = re.compile(r"final answer\s*[:\-]?\s*([AB])\b", re.IGNORECASE)
FALLBACK_RE = re.compile(r"\b([AB])\b")


def parse_letter(text):
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    matches = [m.group(1).upper() for m in FALLBACK_RE.finditer(text)]
    return matches[-1] if matches else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="qa_split.json")
    ap.add_argument("--injection", choices=["jepa", "random", "none"], required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=123)  # different from training's option-shuffle seed
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    n_tokens = 64
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    out_path = args.out or f"qa_eval_{args.injection}.jsonl"

    with open(args.split) as f:
        split = json.load(f)
    test_items = split["test"]

    print(f"[INFO] loading {MODEL_ID} ...")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.requires_grad_(False)
    model.eval()
    device = next(model.parameters()).device
    hidden_size = model.config.text_config.hidden_size

    injector, hook = None, None
    if args.injection in ("jepa", "random"):
        ckpt = torch.load(args.checkpoint, map_location=device)
        injector = JepaInjector(jepa_dim=1024, hidden_size=hidden_size, mode=args.injection, pooling="temporal")
        injector.load_state_dict(ckpt["injector_state"])
        injector.to(device).eval()
        hook = LanguageModelInjectionHook(model, n_tokens)
        placeholder_id = processor.tokenizer.pad_token_id

    rng = random.Random(args.seed)

    with open(out_path, "w") as out_f, torch.no_grad():
        for i, item in enumerate(test_items):
            option_lines, correct_letter = build_option_block(item, rng)
            prompt_text = build_prompt_text(item["question"], option_lines)

            try:
                inputs = build_inputs(processor, item["pid"], prompt_text)
                model_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

                if args.injection in ("jepa", "random"):
                    feats = torch.from_numpy(load_vjepa_in_feats(item["pid"])).unsqueeze(0).to(device)
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
                pred_letter = parse_letter(out_text)

                record = {
                    "pid": item["pid"],
                    "correct_letter": correct_letter,
                    "pred_letter": pred_letter,
                    "correct": pred_letter == correct_letter,
                    "raw_output": out_text,
                }
            except Exception as e:
                record = {"pid": item["pid"], "error": f"{type(e).__name__}: {e}"}

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(test_items)}] done")

    print(f"[DONE] -> {out_path}")


if __name__ == "__main__":
    main()
