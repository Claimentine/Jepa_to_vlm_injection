"""
Zero-shot "does the VLM already know the task" probe.

Runs frozen Qwen3-VL-2B-Thinking (no JEPA, no fine-tuning) on a 13-way
multiple-choice task-recognition question, under three input-granularity
conditions:

  A. full_clip   - the whole episode, uniformly sampled to --max_frames
  B. past_window - only the first --past_t frames (matches the past_T=32
                   window used by the trajectory task / semantic QA probe,
                   so this condition tells us directly whether task identity
                   already leaks at the same granularity our other
                   experiments operate on)
  C. single_frame - a single still frame near the start of the episode

Must run inside the `qwen3vl` conda env (has transformers + qwen_vl_utils).
Requires a GPU node (this repo's login node has none).

Usage:
  conda activate qwen3vl
  python run_task_recognition_probe.py \
      --manifest probe_manifest.json \
      --out probe_results.jsonl
"""
import argparse
import json
import os
import random
import re

import decord
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from task_categories import CATEGORIES, CATEGORY_PHRASES

MODEL_ID = "Qwen/Qwen3-VL-2B-Thinking"  # same checkpoint already used for JEPA guidance features
CONDITIONS = ["full_clip", "past_window", "single_frame"]

ANSWER_RE = re.compile(r"final answer\s*[:\-]?\s*([A-M])\b", re.IGNORECASE)
FALLBACK_LETTER_RE = re.compile(r"\b([A-M])\b")


def build_option_block(rng):
    """Shuffle option order per-question so the model can't learn a position
    prior; returns (prompt_lines, letter_to_category, category_to_letter)."""
    cats = list(CATEGORIES)
    rng.shuffle(cats)
    letters = [chr(ord("A") + i) for i in range(len(cats))]
    letter_to_category = dict(zip(letters, cats))
    category_to_letter = dict(zip(cats, letters))
    lines = [f"{letters[i]}. {CATEGORY_PHRASES[cats[i]]}" for i in range(len(cats))]
    return lines, letter_to_category, category_to_letter


def build_prompt_text(option_lines):
    return (
        "You will be shown footage of someone performing a manual task, filmed from a "
        "first-person (egocentric) viewpoint.\n"
        "Which of the following best describes what is happening?\n\n"
        + "\n".join(option_lines)
        + "\n\nThink briefly, then answer with exactly one line in the form:\n"
        "Final answer: <LETTER>"
    )


def parse_answer_letter(text, valid_letters):
    m = ANSWER_RE.search(text)
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper(), True
    # fallback: last standalone A-M letter mentioned anywhere
    matches = [m.group(1).upper() for m in FALLBACK_LETTER_RE.finditer(text) if m.group(1).upper() in valid_letters]
    if matches:
        return matches[-1], False
    return None, False


def extract_single_frame(mp4_path, frame_idx=16):
    vr = decord.VideoReader(mp4_path)
    idx = min(frame_idx, len(vr) - 1)
    frame = vr[idx].asnumpy()
    return Image.fromarray(frame)


def build_messages(condition, mp4_path, prompt_text, past_t, max_frames):
    if condition == "single_frame":
        img = extract_single_frame(mp4_path, frame_idx=past_t // 2)
        content = [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt_text},
        ]
        return [{"role": "user", "content": content}]

    abs_path = os.path.abspath(mp4_path)
    file_url = f"file://{abs_path}"
    video_content = {
        "type": "video",
        "video": file_url,
        "resized_height": 256,
        "resized_width": 256,
    }

    if condition == "past_window":
        vr = decord.VideoReader(mp4_path)
        fps = max(float(vr.get_avg_fps()), 1e-6)
        n_avail = min(past_t, len(vr))
        video_content["nframes"] = max(2, (n_avail // 2) * 2)
        video_content["video_start"] = 0.0
        video_content["video_end"] = n_avail / fps
    else:  # full_clip
        video_content["nframes"] = max_frames

    return [{"role": "user", "content": [video_content, {"type": "text", "text": prompt_text}]}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="probe_manifest.json")
    ap.add_argument("--out", default="probe_results.jsonl")
    ap.add_argument("--max_frames", type=int, default=32, help="uniform frame count for full_clip condition")
    ap.add_argument("--past_t", type=int, default=32, help="window length (frames) for past_window condition")
    ap.add_argument("--max_new_tokens", type=int, default=160, help="thinking + answer budget")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

    with open(args.manifest) as f:
        manifest = json.load(f)

    print(f"[INFO] loading {MODEL_ID} ...")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()

    rng = random.Random(args.seed)

    with open(args.out, "w") as out_f:
        for ep_i, ep in enumerate(manifest):
            for condition in CONDITIONS:
                option_lines, letter_to_category, category_to_letter = build_option_block(rng)
                prompt_text = build_prompt_text(option_lines)
                valid_letters = set(letter_to_category.keys())

                try:
                    messages = build_messages(
                        condition, ep["mp4_path"], prompt_text, args.past_t, args.max_frames
                    )
                    images, videos, video_kwargs = process_vision_info(
                        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True,
                    )
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    video_metadatas = None
                    if videos is not None:
                        videos, video_metadatas = zip(*videos)
                        videos, video_metadatas = list(videos), list(video_metadatas)

                    inputs = processor(
                        text=text, images=images, videos=videos,
                        video_metadata=video_metadatas, return_tensors="pt",
                        do_resize=False, **(video_kwargs or {}),
                    )
                    inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

                    with torch.inference_mode():
                        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], generated_ids)]
                    out_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

                    pred_letter, matched_format = parse_answer_letter(out_text, valid_letters)
                    pred_category = letter_to_category.get(pred_letter) if pred_letter else None

                    record = {
                        "episode_key": f"{ep['category']}/{ep['episode_id']}",
                        "condition": condition,
                        "true_category": ep["category"],
                        "true_letter": category_to_letter[ep["category"]],
                        "pred_category": pred_category,
                        "pred_letter": pred_letter,
                        "matched_expected_format": matched_format,
                        "correct": pred_category == ep["category"],
                        "raw_output": out_text,
                    }
                except Exception as e:
                    record = {
                        "episode_key": f"{ep['category']}/{ep['episode_id']}",
                        "condition": condition,
                        "true_category": ep["category"],
                        "error": f"{type(e).__name__}: {e}",
                    }

                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

            if (ep_i + 1) % 10 == 0:
                print(f"[{ep_i+1}/{len(manifest)}] episodes done")

    print(f"\nDone. Results -> {args.out}")


if __name__ == "__main__":
    main()
