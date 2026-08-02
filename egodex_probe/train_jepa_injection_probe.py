"""
Trains a small adapter that injects cached V-JEPA2 features into a FROZEN
Qwen3-VL-2B-Thinking, to see whether it improves 13-way task classification
accuracy -- especially on the 6 assemble_disassemble_furniture_bench_*
sub-categories where the zero-shot baseline was near chance (see
probe_results.jsonl from the earlier zero-shot run: macro acc 0.05-0.15 on
those 6 categories).

Three `--injection` modes, all using the *same* frozen-VLM + same prompt +
same past_window video condition (32 frames), so numbers are comparable:

  jepa         - soft-prompt from real cached vjepa_feats (the experiment)
  random       - soft-prompt from fixed-seed random noise, identical adapter
                 architecture/param count (control: rules out "just extra
                 trainable capacity" as the explanation for any gain)
  linear_probe - no soft-prompt at all; a plain linear head on the frozen
                 VLM's own last-token hidden state (lower-bound baseline:
                 how much do you get from *just* making the existing head
                 trainable, with no JEPA signal at all)

Only the adapter (+ linear_probe head, if that mode) is trained. The VLM
backbone is frozen throughout.

Must run in the qwen3vl conda env, on a GPU node.
"""
import sys
import argparse
import json
import os
import random

import decord
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from task_categories import CATEGORIES, CATEGORY_PHRASES
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from common.jepa_injection_model import JepaInjector, LanguageModelInjectionHook, prepend_placeholder_tokens  # noqa: E402

MODEL_ID = "Qwen/Qwen3-VL-2B-Thinking"
CAT_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}
LETTERS = [chr(ord("A") + i) for i in range(len(CATEGORIES))]


def build_option_block(rng):
    cats = list(CATEGORIES)
    rng.shuffle(cats)
    letters = LETTERS[: len(cats)]
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
        + "\n\nAnswer with exactly one line in the form:\n"
        "Final answer: <LETTER>"
    )


def load_vjepa_feats(npz_path):
    with np.load(npz_path) as d:
        return d["vjepa_feats"].astype(np.float32)  # (T, N, D)


def build_video_message(mp4_path, past_t=32):
    abs_path = os.path.abspath(mp4_path)
    vr = decord.VideoReader(mp4_path)
    fps = max(float(vr.get_avg_fps()), 1e-6)
    n_avail = min(past_t, len(vr))
    return {
        "type": "video",
        "video": f"file://{abs_path}",
        "resized_height": 256,
        "resized_width": 256,
        "nframes": max(2, (n_avail // 2) * 2),
        "video_start": 0.0,
        "video_end": n_avail / fps,
    }


def build_inputs(processor, mp4_path, prompt_text):
    video_content = build_video_message(mp4_path)
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
    """Appends ' Final answer: <LETTER>' and returns (input_ids, attn_mask,
    labels) with labels=-100 everywhere except the single letter token."""
    suffix_text = f" Final answer: {true_letter}"
    suffix_ids = tokenizer(suffix_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    full_ids = torch.cat([input_ids, suffix_ids], dim=1)
    full_mask = torch.cat([attention_mask, torch.ones_like(suffix_ids)], dim=1)
    labels = torch.full_like(full_ids, -100)
    labels[:, -1] = full_ids[:, -1]  # supervise only the final letter token
    return full_ids, full_mask, labels


def evaluate_teacher_forced(model, processor, hook, injector, episodes, n_tokens, placeholder_id,
                             injection_mode, device, rng):
    """Fast validation: teacher-forced argmax at the answer-letter position,
    no autoregressive generation. Used for periodic val accuracy during
    training; final reported numbers should use eval_jepa_injection_probe.py
    (real generate()-based decoding) for apples-to-apples comparison with
    the zero-shot baseline."""
    model.eval()
    n_correct, n_total = 0, 0
    with torch.no_grad():
        for ep in episodes:
            option_lines, letter_to_category, category_to_letter = build_option_block(rng)
            prompt_text = build_prompt_text(option_lines)
            true_letter = category_to_letter[ep["category"]]

            inputs = build_inputs(processor, ep["mp4_path"], prompt_text)
            input_ids, attn_mask, labels = append_answer_tokens(
                processor.tokenizer, inputs["input_ids"], inputs["attention_mask"], true_letter
            )

            if injection_mode in ("jepa", "random"):
                input_ids, attn_mask = prepend_placeholder_tokens(input_ids, attn_mask, n_tokens, placeholder_id)
                labels = torch.cat([torch.full((1, n_tokens), -100, dtype=labels.dtype), labels], dim=1)

            model_inputs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
            model_inputs["input_ids"] = input_ids.to(device)
            model_inputs["attention_mask"] = attn_mask.to(device)
            model_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in model_inputs.items()}

            if injection_mode in ("jepa", "random"):
                feats = torch.from_numpy(load_vjepa_feats(ep["npz_path"])).unsqueeze(0).to(device)
                soft_prompt = injector(feats)
                hook.set(soft_prompt)

            out = model(**model_inputs)
            hook.clear() if injection_mode in ("jepa", "random") else None

            target_pos = (labels != -100).nonzero(as_tuple=True)[1].item()
            pred_id = out.logits[0, target_pos - 1].argmax().item()
            true_id = labels[0, target_pos].item()

            n_total += 1
            n_correct += int(pred_id == true_id)

    model.train()
    return n_correct / max(n_total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="classification_split.json")
    ap.add_argument("--injection", choices=["jepa", "random", "linear_probe"], required=True)
    ap.add_argument("--pooling", choices=["temporal", "global"], default="temporal",
                    help="temporal (default): spatial-only pooling, one soft-prompt token per "
                         "frame (T=64), fixes the over-pooling diagnosed after v1. "
                         "global: legacy single-vector pooling from the first run, kept for "
                         "direct comparison.")
    ap.add_argument("--n_tokens", type=int, default=8, help="only used when --pooling global")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_every_steps", type=int, default=200)
    ap.add_argument("--max_train_episodes", type=int, default=None, help="cap for smoke tests")
    ap.add_argument("--max_val_episodes", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="jepa_injection_runs")
    ap.add_argument("--balance_classes", action="store_true", default=True)
    ap.add_argument("--no_balance_classes", dest="balance_classes", action="store_false",
                    help="disable inverse-frequency weighted sampling (legacy v1 behavior "
                         "iterated the raw, imbalanced train split)")
    args = ap.parse_args()

    n_tokens = 64 if args.pooling == "temporal" else args.n_tokens

    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    run_dir = os.path.join(args.out_dir, args.injection)
    os.makedirs(run_dir, exist_ok=True)

    with open(args.split) as f:
        split = json.load(f)
    train_eps, val_eps = split["train"], split["val"]
    if args.max_train_episodes:
        train_eps = train_eps[: args.max_train_episodes]
    val_eps = val_eps[: args.max_val_episodes]

    rng = random.Random(args.seed)

    # inverse-frequency sampling weights: v1 iterated the raw imbalanced split
    # (basic_pick_place=433 train vs fold_unfold_paper_origami=16) uniformly,
    # which can bias the adapter toward majority-class shortcuts.
    cat_counts = {}
    for ep in train_eps:
        cat_counts[ep["category"]] = cat_counts.get(ep["category"], 0) + 1
    train_weights = [1.0 / cat_counts[ep["category"]] for ep in train_eps]

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
    assert placeholder_id not in (video_token_id, image_token_id), \
        "pad_token_id collides with a vision token id -- pick a different placeholder id"

    trainable_params = []
    hook = None
    injector = None
    linear_head = None

    if args.injection in ("jepa", "random"):
        injector = JepaInjector(
            jepa_dim=1024, hidden_size=hidden_size, n_tokens=n_tokens,
            mode=args.injection, seed=args.seed, pooling=args.pooling,
        ).to(device)
        trainable_params += list(injector.parameters())
        hook = LanguageModelInjectionHook(model, n_tokens)
    else:  # linear_probe
        linear_head = nn.Linear(hidden_size, len(CATEGORIES)).to(device)
        trainable_params += list(linear_head.parameters())

    opt = torch.optim.AdamW(trainable_params, lr=args.lr)
    print(f"[INFO] injection={args.injection}  pooling={args.pooling}  n_tokens={n_tokens}  "
          f"balance_classes={args.balance_classes}  "
          f"trainable_params={sum(p.numel() for p in trainable_params):,}")

    step = 0
    best_val_acc = -1.0
    log_path = os.path.join(run_dir, "train_log.jsonl")
    log_f = open(log_path, "w")

    for epoch in range(args.epochs):
        if args.balance_classes:
            epoch_eps = rng.choices(train_eps, weights=train_weights, k=len(train_eps))
        else:
            rng.shuffle(train_eps)
            epoch_eps = train_eps
        for ep in epoch_eps:
            option_lines, letter_to_category, category_to_letter = build_option_block(rng)
            prompt_text = build_prompt_text(option_lines)
            true_letter = category_to_letter[ep["category"]]
            true_idx = CAT_TO_IDX[ep["category"]]

            inputs = build_inputs(processor, ep["mp4_path"], prompt_text)

            if args.injection == "linear_probe":
                model_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
                out = model(**model_inputs, output_hidden_states=True)
                last_hidden = out.hidden_states[-1][:, -1, :]  # last token, last layer
                logits = linear_head(last_hidden.float())
                loss = nn.functional.cross_entropy(logits, torch.tensor([true_idx], device=device))
            else:
                input_ids, attn_mask, labels = append_answer_tokens(
                    processor.tokenizer, inputs["input_ids"], inputs["attention_mask"], true_letter
                )
                input_ids, attn_mask = prepend_placeholder_tokens(input_ids, attn_mask, n_tokens, placeholder_id)
                labels = torch.cat([torch.full((1, n_tokens), -100, dtype=labels.dtype), labels], dim=1)

                model_inputs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
                model_inputs["input_ids"] = input_ids
                model_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in model_inputs.items()}
                model_inputs["attention_mask"] = attn_mask.to(device)
                labels = labels.to(device)

                feats = torch.from_numpy(load_vjepa_feats(ep["npz_path"])).unsqueeze(0).to(device)
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

            step += 1
            if step % 20 == 0:
                print(f"[epoch {epoch} step {step}] loss={loss.item():.4f}")
            log_f.write(json.dumps({"step": step, "epoch": epoch, "loss": loss.item()}) + "\n")
            log_f.flush()

            if step % args.val_every_steps == 0:
                if args.injection == "linear_probe":
                    val_acc = evaluate_linear_probe(model, processor, linear_head, val_eps, device)
                else:
                    val_acc = evaluate_teacher_forced(
                        model, processor, hook, injector, val_eps, n_tokens,
                        placeholder_id, args.injection, device, rng,
                    )
                print(f"[epoch {epoch} step {step}] val_acc(teacher_forced)={val_acc:.4f}")
                log_f.write(json.dumps({"step": step, "val_acc": val_acc}) + "\n")
                log_f.flush()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    ckpt = {"args": vars(args)}
                    if injector is not None:
                        ckpt["injector_state"] = injector.state_dict()
                    if linear_head is not None:
                        ckpt["linear_head_state"] = linear_head.state_dict()
                    torch.save(ckpt, os.path.join(run_dir, "best.pt"))

    log_f.close()
    print(f"[DONE] injection={args.injection}  best_val_acc(teacher_forced)={best_val_acc:.4f}")
    print(f"        checkpoint -> {os.path.join(run_dir, 'best.pt')}")


def evaluate_linear_probe(model, processor, linear_head, episodes, device):
    model.eval()
    n_correct, n_total = 0, 0
    rng = random.Random(0)
    with torch.no_grad():
        for ep in episodes:
            option_lines, _, _ = build_option_block(rng)
            prompt_text = build_prompt_text(option_lines)
            inputs = build_inputs(processor, ep["mp4_path"], prompt_text)
            model_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
            out = model(**model_inputs, output_hidden_states=True)
            last_hidden = out.hidden_states[-1][:, -1, :]
            logits = linear_head(last_hidden.float())
            pred_idx = logits.argmax(dim=-1).item()
            n_total += 1
            n_correct += int(CATEGORIES[pred_idx] == ep["category"])
    model.train()
    return n_correct / max(n_total, 1)


if __name__ == "__main__":
    main()
