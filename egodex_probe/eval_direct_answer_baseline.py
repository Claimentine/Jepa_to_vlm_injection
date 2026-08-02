"""
Control experiment: how much of the jepa/random accuracy jump is just
"skip the chain-of-thought, force a direct answer" rather than anything to
do with the injected content?

No adapter, no soft-prompt, no training at all -- literally the frozen,
untouched VLM, prompt + "Final answer:" appended directly, reading off the
argmax over the 13 candidate letter tokens at that one position. This
isolates the "direct answer" measurement methodology from any injection
effect, using the exact same held-out test split / prompt / past_window
video condition as the other conditions for a fair comparison.
"""
import argparse
import json
import os
import random

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from task_categories import CATEGORIES
from train_jepa_injection_probe import MODEL_ID, build_option_block, build_prompt_text, build_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="classification_split.json")
    ap.add_argument("--seed", type=int, default=123)  # match eval_jepa_injection_probe.py's default
    ap.add_argument("--out", default="injection_eval_no_injection_direct_answer.jsonl")
    args = ap.parse_args()

    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

    with open(args.split) as f:
        split = json.load(f)
    test_eps = split["test"]

    print(f"[INFO] loading {MODEL_ID} ...")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.requires_grad_(False)
    model.eval()
    device = next(model.parameters()).device

    rng = random.Random(args.seed)

    with open(args.out, "w") as out_f, torch.no_grad():
        for i, ep in enumerate(test_eps):
            option_lines, letter_to_category, category_to_letter = build_option_block(rng)
            prompt_text = build_prompt_text(option_lines)
            true_letter = category_to_letter[ep["category"]]

            try:
                inputs = build_inputs(processor, ep["mp4_path"], prompt_text)
                suffix_ids = processor.tokenizer(" Final answer:", add_special_tokens=False, return_tensors="pt")["input_ids"]
                input_ids = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
                attn_mask = torch.cat([inputs["attention_mask"], torch.ones_like(suffix_ids)], dim=1)

                model_inputs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
                model_inputs["input_ids"] = input_ids
                model_inputs["attention_mask"] = attn_mask
                model_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in model_inputs.items()}

                out = model(**model_inputs)
                next_token_logits = out.logits[0, -1]

                # restrict to the valid letter tokens actually in play for this question
                letter_token_ids = {}
                for letter in letter_to_category:
                    ids = processor.tokenizer(f" {letter}", add_special_tokens=False)["input_ids"]
                    letter_token_ids[letter] = ids[-1]
                best_letter = max(letter_token_ids, key=lambda l: next_token_logits[letter_token_ids[l]].item())
                pred_category = letter_to_category[best_letter]

                record = {
                    "episode_key": f"{ep['category']}/{ep['episode_id']}",
                    "condition": "past_window",
                    "true_category": ep["category"],
                    "true_letter": true_letter,
                    "pred_category": pred_category,
                    "pred_letter": best_letter,
                    "correct": pred_category == ep["category"],
                    "raw_output": "<direct-answer, no generation, no injection>",
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

    print(f"[DONE] -> {args.out}")


if __name__ == "__main__":
    main()
