"""VLM guidance extraction using VANS's *actual* in_clip -> out_clip pairing,
instead of cache_train/rebuild_causal_cache.py's generic single-file T//2
self-split.

Why this is a separate file, not a change to extract_vlm_guidance_full.py:
that script feeds only in_clip into rebuild_causal_cache.py, which then
bisects in_clip itself into a pseudo-past/pseudo-future -- the "future" it
predicts is just the second half of in_clip, never actually reaching
out_clip, which is VANS's real, dataset-provided continuation of the same
source video (pid convention "<video_id>__<n>" confirms in_clip/out_clip
are sequential segments of one source video, not the shortcut-diagnosis
cross-video distractors that motivated qa_split_temporal.json -- that
diagnosis was about *distractor* out_clips sampled from other videos, not
about the true (in_clip, out_clip) pairing itself). Using in_clip as the
full past and out_clip as the full target instead means predicting a
genuinely longer-horizon, dataset-intended future.

Reuses (via import, not copy-paste) the proven pieces of
cache_train/rebuild_causal_cache.py: Qwen3-VL loading + decoder-hook
registration (load_qwen), V-JEPA2 loading (load_vjepa_encoder), frame
preprocessing (preprocess_vjepa_frames), V-JEPA2 encoding (encode_vjepa_clip),
and the past-only Qwen inference call (infer_qwen_past_only) -- all of
these are file-agnostic single-clip utilities, they don't care whether the
32 "past" frames came from bisecting one file or from a whole separate
in_clip file. Only the top-level orchestration (which two files to decode,
how to sample frames from each, the output schema) is new here.
"""
import argparse
import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace

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
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, output_path)


def collect_pairs(split_path):
    split = json.load(open(split_path))
    seen = {}
    for part in split.values():
        for item in part:
            pid = item["pid"]
            if pid not in seen:
                seen[pid] = (item["in_clip"], item["out_clip"])
    return seen  # pid -> (in_clip, out_clip)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=os.path.join(BASE, "raw_data/qa_split_full.json"))
    ap.add_argument("--output_dir", default=os.path.join(WORK_BASE, "vlm_guidance_cache_paired"))
    ap.add_argument("--pretrained", default="Qwen/Qwen3-VL-2B-Thinking")
    ap.add_argument("--qwen_revision", default="main")
    ap.add_argument("--qwen_checkpoint_sha", default="")
    ap.add_argument("--vjepa_checkpoint", default=os.environ.get("VJEPA2_CKPT", "/data/checkpoints/vjepa2/vitl.pt"))
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 12, 16, 20, 24, 27])
    ap.add_argument("--prompt", default="Describe this video.")
    ap.add_argument("--max_new_token_num", type=int, default=16)
    ap.add_argument("--qwen_res", type=int, default=256)
    ap.add_argument("--save_dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--limit", type=int, default=None, help="cap number of pairs (smoke test)")
    args = ap.parse_args()

    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    from cache_train.rebuild_causal_cache import (
        NUM_PAST_FRAMES, NUM_TARGET_FRAMES,
        load_qwen, load_vjepa_encoder, preprocess_vjepa_frames,
        encode_vjepa_clip, infer_qwen_past_only, tensor_to_numpy,
        sha256_file,
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
        for i, (pid, (in_clip, out_clip)) in enumerate(items, start=1):
            output_path = output_root / f"{pid}.npz"
            if is_valid_output(output_path):
                skipped_count += 1
                print(f"[SKIP {i}/{len(items)}] valid {output_path}", flush=True)
                continue
            try:
                past_frames, past_times, in_fps = decode_clip_with_timeout(in_clip, NUM_PAST_FRAMES)
                target_frames, target_times, _out_fps = decode_clip_with_timeout(out_clip, NUM_TARGET_FRAMES)

                selection_like = SimpleNamespace(past_times=past_times, source_fps=in_fps)
                qwen_result = infer_qwen_past_only(
                    args, qwen_model, qwen_processor, qwen_saved, device, past_frames, selection_like,
                )

                past_clip, past_imgs = preprocess_vjepa_frames(past_frames)
                target_clip, target_imgs = preprocess_vjepa_frames(target_frames)
                vjepa_input_feats = encode_vjepa_clip(vjepa_model, past_clip, device)
                vjepa_target_feats = encode_vjepa_clip(vjepa_model, target_clip, device)

                payload = {
                    "schema_name": np.asarray("vlm_guidance_paired_v1"),
                    "pid": np.asarray(pid),
                    "in_clip_relpath": np.asarray(str(in_clip)),
                    "out_clip_relpath": np.asarray(str(out_clip)),
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
                    "text": qwen_result["text"],
                    "token_ids": qwen_result["token_ids"],
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


def tensor_or_array_to_dtype(value, save_dtype):
    array = np.asarray(value)
    if save_dtype == "fp16":
        return array.astype(np.float16, copy=False)
    return array.astype(np.float32, copy=False)


if __name__ == "__main__":
    main()
