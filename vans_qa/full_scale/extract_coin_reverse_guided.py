"""COIN analogue of extract_ego4d_reverse_guided.py -- see that script's
own docstring for the full rationale (train_joint_coupled_normalized.py's
--cache_dir/--rev_qa_split need per-pid vlm_old/vlm_new + vjepa features,
which coin_longaxis's existing 2927 npz pairs don't have: no VLM guidance
was ever run on them, and their saved past_imgs/target_imgs are the
VJEPA-preprocessed version, not raw frames a fresh Qwen call could reuse
-- so the source video has to be (re-)downloaded regardless, and this
does the VJEPA encoding AND Qwen guidance captioning in that same pass).

Reuses extract_coin_longaxis.py's own annotation loading/pair selection/
yt-dlp download (COIN videos are YouTube, much faster/more reliable to
(re-)download than Ego4D's S3-CLI path has been this session) and
extract_vlm_guidance_paired.py's infer_qwen_teacher_forced, exactly like
the Ego4D version does.
"""
import argparse
import json
import os
import sys
import traceback
import uuid
from pathlib import Path

import numpy as np

BASE = os.environ.get("VANS_ROOT", "/data")
WORK_BASE = os.environ.get("VANS_WORK_ROOT", "/data/vans_work_coin_reverse_guided")
THINKJEPA_ROOT = os.environ.get("THINKJEPA_ROOT", "/home/jovyan/ThinkJEPA")
if THINKJEPA_ROOT not in sys.path:
    sys.path.insert(0, THINKJEPA_ROOT)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from extract_coin_longaxis import (  # noqa: E402
    load_coin_annotations, select_longaxis_pairs, download_video, extract_window_frames,
)
from extract_ego4d_longaxis import split_video_keys  # noqa: E402
from extract_vlm_guidance_paired import infer_qwen_teacher_forced  # noqa: E402


def build_question(coin_class):
    activity = (coin_class or "an everyday activity").lower()
    return (f"This is video of someone doing {activity}. "
            f"Based on the video I just sent, what should I do next?")


def is_valid_output(path):
    if not path.exists():
        return False
    try:
        with np.load(path) as data:
            required = {"pid", "vlm_old", "vlm_new", "vjepa_input_feats", "vjepa_target_feats"}
            return required.issubset(set(data.files))
    except Exception:
        return False


def atomic_write_npz(output_path, payload):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}.npz"
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, output_path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coin_json", default=None, help="path to COIN.json; downloads it if omitted")
    ap.add_argument("--min_span_seconds", type=float, default=0.0)
    ap.add_argument("--min_steps", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="cap candidate pairs")
    ap.add_argument("--eval_frac", type=float, default=0.1, help="video-level held-out fraction -> val split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--video_cache_dir", default=os.path.join(WORK_BASE, "raw_videos"))
    ap.add_argument("--cache_dir", default=os.path.join(WORK_BASE, "cache"))
    ap.add_argument("--split_out", default=os.path.join(BASE, "raw_data/qa_split_coin_reverse_guided.json"))
    ap.add_argument("--pretrained", default="Qwen/Qwen3-VL-2B-Thinking")
    ap.add_argument("--qwen_revision", default="main")
    ap.add_argument("--qwen_checkpoint_sha", default="")
    ap.add_argument("--vjepa_checkpoint", default=os.environ.get("VJEPA2_CKPT", "/data/checkpoints/vjepa2/vitl.pt"))
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 12, 16, 20, 24, 27])
    ap.add_argument("--qwen_res", type=int, default=256)
    ap.add_argument("--save_dtype", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()

    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    from cache_train.rebuild_causal_cache import (
        NUM_PAST_FRAMES, NUM_TARGET_FRAMES, load_qwen, load_vjepa_encoder,
        preprocess_vjepa_frames, encode_vjepa_clip, tensor_to_numpy, sha256_file,
    )

    database = load_coin_annotations(args.coin_json)
    all_video_keys = list(database.keys())
    pairs = select_longaxis_pairs(database, args.min_span_seconds, args.min_steps, args.limit)
    print(f"[INFO] {len(pairs)} candidate longaxis pairs across "
          f"{len({p['video_id'] for p in pairs})} videos", flush=True)

    train_keys, val_keys = split_video_keys(all_video_keys, args.eval_frac, args.seed)
    video_part = {}
    for k in train_keys:
        video_part[k] = "train"
    for k in val_keys:
        video_part[k] = "val"

    video_cache_root = Path(args.video_cache_dir)
    video_cache_root.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] loading {args.pretrained} + hooks on layers {args.layers} ...", flush=True)
    qwen_model, qwen_processor, qwen_saved, qwen_sha = load_qwen(args, device)
    print(f"[INFO] loading V-JEPA2 checkpoint {args.vjepa_checkpoint} ...", flush=True)
    vjepa_checkpoint_sha = sha256_file(Path(args.vjepa_checkpoint))
    vjepa_model = load_vjepa_encoder(Path(args.vjepa_checkpoint), device)

    n = len(pairs)
    saved_count = skipped_count = failed_count = 0
    seen_exc_types = set()
    split_membership = {"train": [], "val": [], "test": []}
    with torch.no_grad():
        for i, pair in enumerate(pairs, start=1):
            video_id = pair["video_id"]
            pid = f"coin_longaxis_guided_{video_id}"
            output_path = cache_root / f"{pid}.npz"
            part = video_part.get(video_id, "train")
            if is_valid_output(output_path):
                skipped_count += 1
                split_membership[part].append({"pid": pid})
                print(f"[SKIP {i}/{n}] valid {output_path}", flush=True)
                continue
            video_path = video_cache_root / f"{video_id}.mp4"
            try:
                if not video_path.exists():
                    download_video(video_id, video_path)

                past_frames, past_times = extract_window_frames(
                    video_path, pair["past_segment"][0], pair["past_segment"][1], NUM_PAST_FRAMES,
                )
                target_frames, target_times = extract_window_frames(
                    video_path, pair["future_segment"][0], pair["future_segment"][1], NUM_TARGET_FRAMES,
                )

                question = build_question(pair["class"])
                correct_caption = pair["future_label"] or "the next step of this activity"
                span = max(1e-6, float(past_times[-1] - past_times[0]))
                in_fps = max(1.0, (len(past_times) - 1) / span)
                qwen_result = infer_qwen_teacher_forced(
                    args, qwen_model, qwen_processor, qwen_saved, device,
                    past_frames, past_times, in_fps, question, correct_caption,
                )

                past_clip, _ = preprocess_vjepa_frames(past_frames)
                target_clip, _ = preprocess_vjepa_frames(target_frames)
                vjepa_input_feats = encode_vjepa_clip(vjepa_model, past_clip, device)
                vjepa_target_feats = encode_vjepa_clip(vjepa_model, target_clip, device)

                def to_dtype(value):
                    array = np.asarray(value)
                    return array.astype(np.float16 if args.save_dtype == "fp16" else np.float32, copy=False)

                payload = {
                    "schema_name": np.asarray("coin_reverse_guided_v1"),
                    "pid": np.asarray(pid),
                    "video_id": np.asarray(video_id),
                    "coin_class": np.asarray(pair["class"] or ""),
                    "question": np.asarray(question),
                    "correct_caption": np.asarray(correct_caption),
                    "past_frame_times_seconds": past_times.astype(np.float64),
                    "target_frame_times_seconds": target_times.astype(np.float64),
                    "vjepa_input_feats": to_dtype(vjepa_input_feats),
                    "vjepa_target_feats": to_dtype(vjepa_target_feats),
                    "vjepa_model": np.asarray("vjepa2_vit_large_rope"),
                    "vjepa_checkpoint_sha256": np.asarray(vjepa_checkpoint_sha),
                    "qwen_model": np.asarray(args.pretrained),
                    "qwen_checkpoint_sha": np.asarray(qwen_sha),
                    "layers": np.asarray(args.layers, dtype=np.int32),
                    "prompt_token_ids": qwen_result["prompt_token_ids"],
                    "answer_token_ids": qwen_result["answer_token_ids"],
                    "vlm_old": tensor_to_numpy(qwen_result["vlm_old_tensor"], args.save_dtype),
                    "vlm_new": tensor_to_numpy(qwen_result["vlm_new_tensor"], args.save_dtype),
                }
                atomic_write_npz(output_path, payload)
                saved_count += 1
                split_membership[part].append({"pid": pid})
                print(f"[SAVED {i}/{n}] {output_path}", flush=True)
            except Exception as exc:
                failed_count += 1
                exc_name = type(exc).__name__
                print(f"[FAIL {i}/{n}] video_id={video_id}: {exc_name}: {exc}",
                      file=sys.stderr, flush=True)
                if exc_name not in seen_exc_types:
                    seen_exc_types.add(exc_name)
                    traceback.print_exc()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            finally:
                # One pair per video (longaxis) -- always the last/only pair
                # for this video_id, safe to delete unconditionally.
                try:
                    video_path.unlink()
                except FileNotFoundError:
                    pass
            if i % 20 == 0:
                print(f"[{i}/{n}] saved={saved_count} skipped={skipped_count} failed={failed_count}", flush=True)

    out_path = Path(args.split_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(split_membership, f, indent=2)
    print(f"[DONE] saved={saved_count} skipped_valid={skipped_count} failed={failed_count} "
          f"cache_dir={cache_root} split_out={out_path} "
          f"(train={len(split_membership['train'])} val={len(split_membership['val'])})", flush=True)
    return 1 if (failed_count and not saved_count and not skipped_count) else 0


if __name__ == "__main__":
    sys.exit(main())
