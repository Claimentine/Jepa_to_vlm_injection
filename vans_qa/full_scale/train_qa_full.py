"""
Full-scale (~12K matched QA pairs, vs the demo's 499) version of
train_vans_qa_injection.py. Same task (2-choice "what happens next"),
same temporal-pooling injection architecture validated on EgoDex, same
training recipe -- only the data source changed: qa_split_full.json +
vjepa_cache_full/ + vlm_guidance_cache_full/ instead of the 500-pair demo
bundle.

Run extract_vjepa_features_full.py and extract_vlm_guidance_full.py first.
"""
import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from common.jepa_injection_model import JepaInjector, LanguageModelInjectionHook, prepend_placeholder_tokens  # noqa: E402

BASE = os.environ.get("VANS_ROOT", "/projects/bhay/william/ruixin/vans_world_model")
WORK_BASE = os.environ.get("VANS_WORK_ROOT", "/work/nvme/bdqf/yli8/vans_raw_data")  # relocated off the near-full /projects/bhay allocation
MODEL_ID = "Qwen/Qwen3-VL-2B-Thinking"
VJEPA_CACHE = os.path.join(WORK_BASE, "vjepa_cache_full")
VLM_CACHE = os.path.join(WORK_BASE, "vlm_guidance_cache_full")
CLIPS_ROOT = os.path.join(WORK_BASE, "clips")


def safe_name(clip_path):
    rel = os.path.relpath(clip_path, CLIPS_ROOT)
    return rel.replace("/", "__").replace(".mp4", "")


def find_vlm_npz(clip_path):
    name = safe_name(clip_path)
    matches = glob.glob(os.path.join(VLM_CACHE, f"{name}_*.npz"))
    return matches[0] if matches else None


def load_vjepa_in_feats(clip_path):
    name = safe_name(clip_path)
    p = os.path.join(VJEPA_CACHE, f"{name}.npz")
    d = np.load(p)
    return d["feats"].astype(np.float32)


def has_features(item):
    vjepa_ok = os.path.exists(os.path.join(VJEPA_CACHE, f"{safe_name(item['in_clip'])}.npz"))
    vlm_ok = find_vlm_npz(item["in_clip"]) is not None
    return vjepa_ok and vlm_ok


def build_option_block(item, rng):
    options = [item["correct_caption"], item["distractor_caption"]]
    order = [0, 1]
    rng.shuffle(order)
    letters = ["A", "B"]
    letter_to_is_correct = {}
    lines = []
    for i, letter in enumerate(letters):
        opt_idx = order[i]
        lines.append(f"{letter}. {options[opt_idx]}")
        letter_to_is_correct[letter] = (opt_idx == 0)
    correct_letter = "A" if letter_to_is_correct["A"] else "B"
    return lines, correct_letter


def build_prompt_text(question, option_lines):
    return (
        "You are shown footage filmed from a first-person viewpoint.\n"
        f"{question}\n\n"
        + "\n".join(option_lines)
        + "\n\nAnswer with exactly one line in the form:\n"
        "Final answer: <LETTER>"
    )


def clamp_frame_count(mp4_path, requested):
    import cv2
    cap = cv2.VideoCapture(mp4_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0:
        return requested
    nf = min(requested, total)
    nf = max(2, nf)
    return (nf // 2) * 2


def build_inputs(processor, in_clip_path, prompt_text, max_frames=32):
    abs_path = os.path.abspath(in_clip_path)
    nframes = clamp_frame_count(abs_path, max_frames)
    video_content = {
        "type": "video", "video": f"file://{abs_path}",
        "resized_height": 256, "resized_width": 256, "nframes": nframes,
    }
    messages = [{"role": "user", "content": [video_content, {"type": "text", "text": prompt_text}]}]
    images, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True,
    )
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    video_metadatas = None
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    inputs = processor(
        text=text, images=images, videos=videos, video_metadata=video_metadatas,
        return_tensors="pt", do_resize=False, **(video_kwargs or {}),
    )
    return inputs


def append_answer_tokens(tokenizer, input_ids, attention_mask, true_letter):
    suffix_text = f" Final answer: {true_letter}"
    suffix_ids = tokenizer(suffix_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    full_ids = torch.cat([input_ids, suffix_ids], dim=1)
    full_mask = torch.cat([attention_mask, torch.ones_like(suffix_ids)], dim=1)
    labels = torch.full_like(full_ids, -100)
    labels[:, -1] = full_ids[:, -1]
    return full_ids, full_mask, labels


def evaluate_teacher_forced(model, processor, hook, injector, items, n_tokens, placeholder_id, device, rng):
    model.eval()
    n_correct, n_total = 0, 0
    with torch.no_grad():
        for item in items:
            try:
                feats_np = load_vjepa_in_feats(item["in_clip"])
                option_lines, correct_letter = build_option_block(item, rng)
                prompt_text = build_prompt_text(item["question"], option_lines)
                inputs = build_inputs(processor, item["in_clip"], prompt_text)
                input_ids, attn_mask, labels = append_answer_tokens(
                    processor.tokenizer, inputs["input_ids"], inputs["attention_mask"], correct_letter
                )
                input_ids, attn_mask = prepend_placeholder_tokens(input_ids, attn_mask, n_tokens, placeholder_id)
                labels = torch.cat([torch.full((1, n_tokens), -100, dtype=labels.dtype), labels], dim=1)

                model_inputs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
                model_inputs["input_ids"] = input_ids
                model_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in model_inputs.items()}
                model_inputs["attention_mask"] = attn_mask.to(device)
                labels = labels.to(device)

                feats = torch.from_numpy(feats_np).unsqueeze(0).to(device)
                soft_prompt = injector(feats)
                hook.set(soft_prompt)
                out = model(**model_inputs)
                hook.clear()

                target_pos = (labels != -100).nonzero(as_tuple=True)[1].item()
                pred_id = out.logits[0, target_pos - 1].argmax().item()
                true_id = labels[0, target_pos].item()
                n_total += 1
                n_correct += int(pred_id == true_id)
            except Exception as e:
                # matches the training loop's per-item try/except: a small fraction of
                # clips are corrupt/too-short (bad feature cache or torchvision frame-count
                # errors) -- skip rather than crash a multi-hour job on one item
                print(f"[WARN] eval: skipping item {item.get('pid')}: {type(e).__name__}: {e}", flush=True)
    model.train()
    return n_correct / max(n_total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=os.path.join(BASE, "raw_data/qa_split_full.json"))
    ap.add_argument("--injection", choices=["jepa", "random"], required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_every_steps", type=int, default=500)
    ap.add_argument("--max_val_items", type=int, default=200, help="cap for periodic in-training eval")
    ap.add_argument("--max_train_items", type=int, default=None, help="smoke-test cap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=os.path.join(BASE, "raw_data/qa_full_runs"))
    args = ap.parse_args()

    n_tokens = 64
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    run_dir = os.path.join(args.out_dir, args.injection)
    os.makedirs(run_dir, exist_ok=True)

    with open(args.split) as f:
        split = json.load(f)
    train_items = [it for it in split["train"] if has_features(it)]
    val_items = [it for it in split["val"] if has_features(it)]
    print(f"[INFO] usable (features present): train={len(train_items)} val={len(val_items)}")
    if args.max_train_items:
        train_items = train_items[: args.max_train_items]
    val_items = val_items[: args.max_val_items]

    rng = random.Random(args.seed)

    print(f"[INFO] loading {MODEL_ID} (frozen) ...")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.requires_grad_(False)
    model.eval()
    device = next(model.parameters()).device
    hidden_size = model.config.text_config.hidden_size

    video_token_id = model.config.video_token_id
    image_token_id = model.config.image_token_id
    placeholder_id = processor.tokenizer.pad_token_id
    assert placeholder_id not in (video_token_id, image_token_id)

    injector = JepaInjector(
        jepa_dim=1024, hidden_size=hidden_size, mode=args.injection, seed=args.seed, pooling="temporal",
    ).to(device)
    hook = LanguageModelInjectionHook(model, n_tokens)
    opt = torch.optim.AdamW(injector.parameters(), lr=args.lr)
    print(f"[INFO] injection={args.injection}  trainable_params={sum(p.numel() for p in injector.parameters()):,}")

    step = 0
    best_val_acc = -1.0
    log_f = open(os.path.join(run_dir, "train_log.jsonl"), "w")

    for epoch in range(args.epochs):
        rng.shuffle(train_items)
        for item in train_items:
            try:
                option_lines, correct_letter = build_option_block(item, rng)
                prompt_text = build_prompt_text(item["question"], option_lines)
                inputs = build_inputs(processor, item["in_clip"], prompt_text)

                input_ids, attn_mask, labels = append_answer_tokens(
                    processor.tokenizer, inputs["input_ids"], inputs["attention_mask"], correct_letter
                )
                input_ids, attn_mask = prepend_placeholder_tokens(input_ids, attn_mask, n_tokens, placeholder_id)
                labels = torch.cat([torch.full((1, n_tokens), -100, dtype=labels.dtype), labels], dim=1)

                model_inputs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
                model_inputs["input_ids"] = input_ids
                model_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in model_inputs.items()}
                model_inputs["attention_mask"] = attn_mask.to(device)
                labels = labels.to(device)

                feats = torch.from_numpy(load_vjepa_in_feats(item["in_clip"])).unsqueeze(0).to(device)
                soft_prompt = injector(feats)
                hook.set(soft_prompt)
                out = model(**model_inputs)
                hook.clear()

                shift_logits = out.logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                loss = nn.functional.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)).float(),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
            except Exception as e:
                print(f"[WARN] skipping item {item.get('pid')}: {type(e).__name__}: {e}")
                continue

            step += 1
            if step % 50 == 0:
                print(f"[epoch {epoch} step {step}] loss={loss.item():.4f}")
            log_f.write(json.dumps({"step": step, "epoch": epoch, "loss": loss.item()}) + "\n")
            log_f.flush()

            if step % args.val_every_steps == 0:
                val_acc = evaluate_teacher_forced(model, processor, hook, injector, val_items, n_tokens, placeholder_id, device, rng)
                print(f"[epoch {epoch} step {step}] val_acc(teacher_forced)={val_acc:.4f}")
                log_f.write(json.dumps({"step": step, "val_acc": val_acc}) + "\n")
                log_f.flush()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save({"args": vars(args), "injector_state": injector.state_dict()},
                               os.path.join(run_dir, "best.pt"))

    log_f.close()
    print(f"[DONE] injection={args.injection}  best_val_acc(teacher_forced)={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
