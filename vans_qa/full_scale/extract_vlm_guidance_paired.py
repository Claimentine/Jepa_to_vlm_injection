"""VLM guidance extraction using VANS's *actual* in_clip -> out_clip pairing
and *actual* correct_caption text, instead of
cache_train/rebuild_causal_cache.py's generic single-file T//2 self-split
plus free-form generation.

Two departures from the original ThinkJEPA design, both because VANS gives
us real supervision that raw, caption-free video doesn't have:

1. Pairing: rebuild_causal_cache.py bisects one file into a pseudo-past/
   pseudo-future -- the "future" it predicts never reaches past the
   midpoint of in_clip. VANS's in_clip/out_clip are real, dataset-provided
   sequential segments of the same source video (pid convention
   "<video_id>__<n>" confirms this -- and this is about the *true* pairing,
   not the cross-video shortcut distractors that motivated
   qa_split_temporal.json). Using in_clip as the full past and out_clip as
   the full target predicts a genuinely longer-horizon, dataset-intended
   future.

2. vlm_new content: the original design has Qwen3-VL freely generate() a
   continuation of what it just watched -- a guess, possibly hallucinated,
   about content it never saw. VANS gives us `correct_caption`, a real,
   dataset-authored description of what out_clip actually shows. Teacher-
   forcing that known text (tokenize prompt and answer *separately*, then
   concatenate token ids and run one forward pass -- exactly
   train_qa_full.py's append_answer_tokens/candidate_logprob pattern, which
   keeps the prompt/answer boundary exact without re-tokenizing a combined
   string) grounds vlm_new in the true future instead of a guess, and is
   cheaper besides: one forward pass instead of an autoregressive
   generate() loop.

Reuses (via import, not copy-paste) the proven pieces of
cache_train/rebuild_causal_cache.py that are agnostic to both of the above:
Qwen3-VL loading + decoder-hook registration (load_qwen), V-JEPA2 loading
(load_vjepa_encoder), frame preprocessing (preprocess_vjepa_frames), and
V-JEPA2 encoding (encode_vjepa_clip). The Qwen inference call itself is new
here (infer_qwen_teacher_forced below) since it needs the item's real
question/correct_caption, not rebuild_causal_cache.py's generic
--prompt + generate().
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
WORK_BASE = os.environ.get("VANS_WORK_ROOT", "/data/vans_work")
THINKJEPA_ROOT = os.environ.get("THINKJEPA_ROOT", "/home/jovyan/ThinkJEPA")
if THINKJEPA_ROOT not in sys.path:
    sys.path.insert(0, THINKJEPA_ROOT)

DECODE_TIMEOUT_S = 60  # same threshold as train_qa_full.py's DECODE_TIMEOUT_S

# spawn, not fork: both the Qwen3-VL and V-JEPA2 models are already loaded
# onto a CUDA context in the parent process by the time any clip is decoded.
# Forking after CUDA init inherits driver state a child can't safely reuse
# and hangs at child startup rather than doing real decode work -- confirmed
# earlier this session on this exact project's forward-direction pipeline
# (vans_qa/full_scale/train_qa_full.py's _MP_CTX/DECODE_TIMEOUT_S).
import multiprocessing as mp  # noqa: E402
_MP_CTX = mp.get_context("spawn")


def _decode_clip_worker(conn, video_path_str, num_frames):
    try:
        from decord import VideoReader, cpu
        from cache_train.rebuild_causal_cache import uniform_indices

        reader = VideoReader(video_path_str, ctx=cpu(0), num_threads=1)
        total = int(len(reader))
        if total < 1:
            raise ValueError(f"no frames in {video_path_str}")
        fps = float(reader.get_avg_fps())
        indices = uniform_indices(0, total, num_frames)
        frames = np.asarray(reader.get_batch(indices).asnumpy())
        frames = np.ascontiguousarray(frames[..., :3].astype(np.uint8, copy=False))
        try:
            timestamps = np.asarray(reader.get_frame_timestamp(indices), dtype=np.float64)
            centers = (timestamps[:, 0] + timestamps[:, 1]) * 0.5
        except Exception:
            fps_safe = fps if fps > 0 else 30.0
            centers = indices.astype(np.float64) / fps_safe
        conn.send(("ok", (frames, centers, fps)))
    except Exception as e:
        conn.send(("err", e))
    finally:
        conn.close()


def decode_clip_with_timeout(video_path, num_frames):
    """Returns (frames uint8 [num_frames,H,W,3], frame_center_times, source_fps)."""
    parent_conn, child_conn = _MP_CTX.Pipe(duplex=False)
    proc = _MP_CTX.Process(target=_decode_clip_worker, args=(child_conn, str(video_path), num_frames))
    proc.start()
    child_conn.close()
    if parent_conn.poll(DECODE_TIMEOUT_S):
        status, payload = parent_conn.recv()
    else:
        status, payload = "timeout", None
    parent_conn.close()
    proc.join(5)
    if proc.is_alive():
        proc.terminate()
        proc.join()
    if status == "timeout":
        raise TimeoutError(f"video decode exceeded {DECODE_TIMEOUT_S}s on {video_path}")
    if status == "err":
        raise payload
    return payload


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
        # A failed write (e.g. disk quota exceeded mid-write) otherwise
        # leaves the half-written temp file behind forever -- confirmed
        # 2026-09-03: a disk-full run left 10,517 orphaned .tmp.*.npz files
        # on disk, matching the failure count almost exactly, both wasting
        # space and making `find -name "*.npz"` wildly overcount real
        # output. os.replace() above already consumed tmp_path on success,
        # so this is a no-op then.
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def collect_pairs(split_path):
    split = json.load(open(split_path))
    seen = {}
    for part in split.values():
        for item in part:
            pid = item["pid"]
            if pid not in seen:
                seen[pid] = {
                    "in_clip": item["in_clip"],
                    "out_clip": item["out_clip"],
                    "question": item["question"],
                    "correct_caption": item["correct_caption"],
                }
    return seen  # pid -> {in_clip, out_clip, question, correct_caption}


def infer_qwen_teacher_forced(args, model, processor, saved, device, past_frames, past_times, in_fps, question, correct_caption):
    """Like rebuild_causal_cache.py's infer_qwen_past_only, but the "new"
    half of the sequence is the real correct_caption (teacher-forced via one
    forward pass), not a freely generate()'d guess -- see module docstring.

    `saved` is the dict register_thinker_decoder_hooks() (inside load_qwen)
    populates on every forward call; a hook fires exactly once here since
    there is exactly one forward call, so each layer's list holds exactly
    one [1, full_len, D] tensor, no generation-step stacking needed.
    """
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from cache_train.rebuild_causal_cache import _effective_sample_fps

    for values in saved.values():
        values.clear()

    # Qwen3-VL uses sample_fps/raw_fps to place the video's temporal/RoPE
    # position embeddings -- passing the true rate the 32 frames were
    # actually sampled at (not left to some processor default) keeps this
    # consistent with what rebuild_causal_cache.py does for its own past-only
    # inference call.
    effective_fps = _effective_sample_fps(past_times, in_fps)
    exact_past_pil = [Image.fromarray(frame) for frame in past_frames]
    video_content = {
        "type": "video",
        "video": exact_past_pil,
        "resized_height": int(args.qwen_res),
        "resized_width": int(args.qwen_res),
        "sample_fps": effective_fps,
        "raw_fps": effective_fps,
    }
    messages = [{"role": "user", "content": [video_content, {"type": "text", "text": str(question)}]}]
    images, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True,
    )
    if not videos or len(videos) != 1:
        raise RuntimeError("Qwen vision processing did not return exactly one video")
    video_tensors, video_metadata = [], []
    for value in videos:
        if isinstance(value, tuple):
            video_tensors.append(value[0])
            video_metadata.append(value[1])
        else:
            video_tensors.append(value)
    if int(video_tensors[0].shape[0]) != past_frames.shape[0]:
        raise AssertionError(
            f"Qwen vision processing changed the exact past-frame count: {tuple(video_tensors[0].shape)}"
        )

    rendered_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processor_kwargs = {
        "text": [rendered_text], "videos": video_tensors, "return_tensors": "pt",
        "padding": True, "do_resize": False,
    }
    if images:
        processor_kwargs["images"] = images
    if video_metadata:
        processor_kwargs["video_metadata"] = video_metadata
    processor_kwargs.update(video_kwargs or {})
    prompt_inputs = processor(**processor_kwargs)
    prefix_len = prompt_inputs["input_ids"].shape[1]

    # Tokenized separately from the prompt (not appended-then-retokenized as
    # one string) so the prompt/answer boundary is exact regardless of BPE
    # merge behavior at the join point -- same reasoning as
    # train_qa_full.py's append_answer_tokens.
    answer_ids = processor.tokenizer(
        f" {correct_caption}", add_special_tokens=False, return_tensors="pt",
    )["input_ids"]
    full_ids = torch.cat([prompt_inputs["input_ids"], answer_ids], dim=1)
    full_mask = torch.cat([prompt_inputs["attention_mask"], torch.ones_like(answer_ids)], dim=1)

    call_inputs = {k: v for k, v in prompt_inputs.items() if k not in ("input_ids", "attention_mask")}
    call_inputs["input_ids"] = full_ids
    call_inputs["attention_mask"] = full_mask
    call_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in call_inputs.items()}

    with torch.inference_mode():
        model(**call_inputs)

    layer_states = []
    for layer_idx in args.layers:
        calls = saved[f"dec_{layer_idx}"]
        if len(calls) != 1:
            raise RuntimeError(f"expected exactly one forward call per item, got {len(calls)} for layer {layer_idx}")
        layer_states.append(calls[0][0])  # [1, full_len, D] -> [full_len, D]
    all_states = torch.stack(layer_states, dim=0)  # [L, full_len, D]
    vlm_old = all_states[:, :prefix_len, :]
    vlm_new = all_states[:, prefix_len:, :]
    return {
        "vlm_old_tensor": vlm_old,
        "vlm_new_tensor": vlm_new,
        "prompt_token_ids": prompt_inputs["input_ids"][0].cpu().numpy().astype(np.int32),
        "answer_token_ids": answer_ids[0].cpu().numpy().astype(np.int32),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=os.path.join(BASE, "raw_data/qa_split_full.json"))
    ap.add_argument("--output_dir", default=os.path.join(WORK_BASE, "vlm_guidance_cache_paired"))
    ap.add_argument("--pretrained", default="Qwen/Qwen3-VL-2B-Thinking")
    ap.add_argument("--qwen_revision", default="main")
    ap.add_argument("--qwen_checkpoint_sha", default="")
    ap.add_argument("--vjepa_checkpoint", default=os.environ.get("VJEPA2_CKPT", "/data/checkpoints/vjepa2/vitl.pt"))
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 12, 16, 20, 24, 27])
    ap.add_argument("--qwen_res", type=int, default=256)
    ap.add_argument("--save_dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--limit", type=int, default=None, help="cap number of pairs (smoke test)")
    args = ap.parse_args()

    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    from cache_train.rebuild_causal_cache import (
        NUM_PAST_FRAMES, NUM_TARGET_FRAMES,
        load_qwen, load_vjepa_encoder, preprocess_vjepa_frames,
        encode_vjepa_clip, tensor_to_numpy, sha256_file,
    )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(args.split)
    items = sorted(pairs.items())  # deterministic order, matches this project's other extraction scripts
    if args.limit:
        items = items[: args.limit]
    print(f"[INFO] {len(items)} unique (in_clip, out_clip) pairs from {args.split}", flush=True)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] loading {args.pretrained} + hooks on layers {args.layers} ...", flush=True)
    qwen_model, qwen_processor, qwen_saved, qwen_sha = load_qwen(args, device)
    print(f"[INFO] loading V-JEPA2 checkpoint {args.vjepa_checkpoint} ...", flush=True)
    vjepa_checkpoint_sha = sha256_file(Path(args.vjepa_checkpoint))
    vjepa_model = load_vjepa_encoder(Path(args.vjepa_checkpoint), device)

    saved_count = skipped_count = failed_count = 0
    seen_exc_types = set()
    with torch.no_grad():
        for i, (pid, pair) in enumerate(items, start=1):
            in_clip, out_clip = pair["in_clip"], pair["out_clip"]
            output_path = output_root / f"{pid}.npz"
            if is_valid_output(output_path):
                skipped_count += 1
                print(f"[SKIP {i}/{len(items)}] valid {output_path}", flush=True)
                continue
            try:
                past_frames, past_times, in_fps = decode_clip_with_timeout(in_clip, NUM_PAST_FRAMES)
                target_frames, target_times, _out_fps = decode_clip_with_timeout(out_clip, NUM_TARGET_FRAMES)

                qwen_result = infer_qwen_teacher_forced(
                    args, qwen_model, qwen_processor, qwen_saved, device,
                    past_frames, past_times, in_fps, pair["question"], pair["correct_caption"],
                )

                past_clip, past_imgs = preprocess_vjepa_frames(past_frames)
                target_clip, target_imgs = preprocess_vjepa_frames(target_frames)
                vjepa_input_feats = encode_vjepa_clip(vjepa_model, past_clip, device)
                vjepa_target_feats = encode_vjepa_clip(vjepa_model, target_clip, device)

                payload = {
                    "schema_name": np.asarray("vlm_guidance_paired_v2_teacher_forced"),
                    "pid": np.asarray(pid),
                    "in_clip_relpath": np.asarray(str(in_clip)),
                    "out_clip_relpath": np.asarray(str(out_clip)),
                    "question": np.asarray(pair["question"]),
                    "correct_caption": np.asarray(pair["correct_caption"]),
                    "past_frame_times_seconds": past_times.astype(np.float64),
                    "target_frame_times_seconds": target_times.astype(np.float64),
                    "past_imgs": past_imgs,
                    "target_imgs": target_imgs,
                    "vjepa_input_feats": tensor_or_array_to_dtype(vjepa_input_feats, args.save_dtype),
                    "vjepa_target_feats": tensor_or_array_to_dtype(vjepa_target_feats, args.save_dtype),
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
                print(f"[SAVED {i}/{len(items)}] {output_path}", flush=True)
            except Exception as exc:
                failed_count += 1
                exc_name = type(exc).__name__
                print(f"[FAIL {i}/{len(items)}] pid={pid}: {exc_name}: {exc}", file=sys.stderr, flush=True)
                if exc_name not in seen_exc_types:
                    seen_exc_types.add(exc_name)
                    traceback.print_exc()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    print(f"[DONE] saved={saved_count} skipped_valid={skipped_count} failed={failed_count} "
          f"output={output_root}", flush=True)
    # Matches rebuild_causal_cache.py's own convention (return 1 if
    # failed_count else 0) -- without this, a run that failed on nearly
    # every item (e.g. a disk-full stretch) still exits 0, so job-12's
    # outer watchdog treats it as a clean success and never retries the
    # items that never got a valid output. Confirmed 2026-09-03: exactly
    # this happened, 10772/12032 failed on one run and it was reported
    # "completed successfully".
    return 1 if failed_count else 0


def tensor_or_array_to_dtype(value, save_dtype):
    array = np.asarray(value)
    if save_dtype == "fp16":
        return array.astype(np.float16, copy=False)
    return array.astype(np.float32, copy=False)


if __name__ == "__main__":
    sys.exit(main())
