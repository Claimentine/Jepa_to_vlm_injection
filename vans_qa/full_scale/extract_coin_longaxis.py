"""Extracts genuinely long-horizon (past, future) V-JEPA2 feature pairs from
COIN's own step annotations, for the reverse direction's self-supervised
leg -- the "long axis" data source the user asked to investigate after
VANS's own short-step annotations turned out to only support small spans
(median ~3-4 clip-index gap, see fit_warmstart_alignment.py-era feasibility
checks) and EgoDex turned out to have no step-level QA/captions at all.

Why COIN, not a fresh raw-video re-extraction of VANS's own COIN/YouCook2
source clips: COIN's public annotation file (COIN.json, no license needed
-- see coin-dataset/annotations on GitHub; the license agreement is only
for Tsinghua's own pre-packaged video archive, not the annotations or a
self-download) already gives every video's step boundaries directly, with
a real temporal span between its first and last annotated step -- median
71s, p90 154s, max 733s (confirmed 2026-09-10 by loading the actual
COIN.json), a genuinely longer horizon than VANS's own adjacent-clip pairs.
A 60-video random sample of COIN's YouTube IDs came back 49/60 (~82%)
still live via YouTube's oEmbed endpoint -- healthier than VANS's own
48.8% survival rate for its already-downloaded clip references.

Unlike VANS, COIN gives ONE video per entry (not pre-cut clip files), so
this script downloads the whole video once via yt-dlp (COIN's own videos
average ~142s -- short enough that downloading the whole thing is simpler
and more robust than yt-dlp's section-download flags, which would need two
disjoint ranges per video here) and extracts two local time-windows itself
via decord: one around the first annotated step (the "past"), one around
the last (the "future").

No VLM/Qwen3-VL step here at all -- this is pure V-JEPA2 self-supervised
feature extraction (no caption injection, since COIN's step `label` fields
are short controlled-vocabulary text like "take out the laptop CD drive",
not free-form scene captions, and this data's role is the reverse
direction's self-supervised leg, which doesn't need VLM guidance -- see
train_latent_world_model_full.py's --guidance none path). Reuses, via
import, the same tested V-JEPA2 loading/encoding pieces
extract_vlm_guidance_paired.py already uses from
cache_train.rebuild_causal_cache: load_vjepa_encoder, preprocess_vjepa_frames,
encode_vjepa_clip, uniform_indices, sha256_file.

Output schema deliberately has no vlm_old/vlm_new fields (there is no VLM
guidance for this data) -- this cache is NOT a drop-in replacement for
vlm_guidance_cache_paired/*.npz in scripts that assume guidance is always
present; using it in train_latent_world_model_full.py or
train_joint_coupled.py's reverse leg requires guidance="none" for these
items specifically. Wiring that up is a follow-up step once this
extraction itself is validated (smoke test first, same discipline as every
other extraction script in this project).
"""
import argparse
import json
import os
import subprocess
import sys
import traceback
import uuid
import urllib.request
from pathlib import Path

import numpy as np

BASE = os.environ.get("VANS_ROOT", "/data")
WORK_BASE = os.environ.get("VANS_WORK_ROOT", "/data/vans_work")
THINKJEPA_ROOT = os.environ.get("THINKJEPA_ROOT", "/home/jovyan/ThinkJEPA")
if THINKJEPA_ROOT not in sys.path:
    sys.path.insert(0, THINKJEPA_ROOT)

COIN_JSON_URL = "https://raw.githubusercontent.com/coin-dataset/annotations/master/COIN.json"

DOWNLOAD_TIMEOUT_S = 180   # a single yt-dlp call, network-bound like everything else this
                           # project has had to timeout-wrap this session (CephFS reads,
                           # video decode) -- a stalled download otherwise blocks forever
DECODE_TIMEOUT_S = 60     # matches extract_vlm_guidance_paired.py's own constant

import multiprocessing as mp  # noqa: E402
_MP_CTX = mp.get_context("spawn")


def load_coin_annotations(coin_json_path):
    if coin_json_path and os.path.exists(coin_json_path):
        with open(coin_json_path) as f:
            return json.load(f)["database"]
    print(f"[INFO] downloading {COIN_JSON_URL} ...", flush=True)
    with urllib.request.urlopen(COIN_JSON_URL, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["database"]


def select_longaxis_pairs(database, min_span_seconds, min_steps, limit=None):
    """One (past_step, future_step) candidate per video: the first and last
    annotated step by segment start time -- the widest span COIN's own
    annotations support for that video, matching the "procedure planning"
    (start, goal) framing this data source is modeled on.
    """
    pairs = []
    for video_id, entry in database.items():
        ann = entry.get("annotation") or []
        if len(ann) < min_steps:
            continue
        ann_sorted = sorted(ann, key=lambda a: a["segment"][0])
        past_step, future_step = ann_sorted[0], ann_sorted[-1]
        span = future_step["segment"][1] - past_step["segment"][0]
        if span < min_span_seconds:
            continue
        pairs.append({
            "video_id": video_id,
            "video_url": entry["video_url"],
            "class": entry.get("class"),
            "span_seconds": span,
            "past_segment": past_step["segment"],
            "past_label": past_step["label"],
            "future_segment": future_step["segment"],
            "future_label": future_step["label"],
            "n_steps": len(ann_sorted),
        })
    pairs.sort(key=lambda p: p["video_id"])
    if limit:
        pairs = pairs[:limit]
    return pairs


YT_DLP_JS_RUNTIME = os.environ.get("YT_DLP_JS_RUNTIME", "node:/opt/conda/bin/node")


def _download_worker(conn, video_id, out_path_str):
    try:
        result = subprocess.run(
            [
                "yt-dlp", "--no-progress", "--quiet",
                # Confirmed 2026-09-11: without a JS runtime, yt-dlp warns
                # "No supported JavaScript runtime could be found. Only deno
                # is enabled by default" and then fails "Requested format is
                # not available" on EVERY format selector including a bare
                # "best" fallback -- YouTube extraction now depends on JS
                # execution to see most of the real format list. node
                # already exists in this project's base image at
                # /opt/conda/bin/node, just not on PATH or told to yt-dlp.
                "--js-runtimes", YT_DLP_JS_RUNTIME,
                # Confirmed 2026-09-11: "best[ext=mp4]" can still resolve to an
                # AV1-in-mp4 stream for videos where YouTube's default "best"
                # tier is AV1 -- decord's own bundled ffmpeg build has no AV1
                # decoder and fails with "cannot find video stream with wanted
                # index: -1" on an otherwise perfectly valid, fully-downloaded
                # file. avc1 (H.264) is what every other decord-based script in
                # this project already relies on successfully.
                "-f", "best[vcodec^=avc1][ext=mp4]/best[ext=mp4]/best",
                "-o", out_path_str,
                f"https://www.youtube.com/watch?v={video_id}",
            ],
            capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT_S,
        )
        if result.returncode != 0:
            conn.send(("err", RuntimeError(f"yt-dlp exit {result.returncode}: {result.stderr[-500:]}")))
        else:
            conn.send(("ok", None))
    except subprocess.TimeoutExpired:
        conn.send(("err", TimeoutError(f"yt-dlp exceeded {DOWNLOAD_TIMEOUT_S}s on {video_id}")))
    except Exception as e:
        conn.send(("err", e))
    finally:
        conn.close()


def download_video(video_id, out_path):
    """Runs yt-dlp in a throwaway subprocess with a hard timeout -- same
    "an external call can hang forever, isolate + timeout it" pattern this
    project has needed for CephFS reads and video decode all session.
    """
    parent_conn, child_conn = _MP_CTX.Pipe(duplex=False)
    proc = _MP_CTX.Process(target=_download_worker, args=(child_conn, video_id, str(out_path)))
    proc.start()
    child_conn.close()
    if parent_conn.poll(DOWNLOAD_TIMEOUT_S + 10):
        status, payload = parent_conn.recv()
    else:
        status, payload = "timeout", None
    parent_conn.close()
    proc.join(5)
    if proc.is_alive():
        proc.terminate()
        proc.join()
    if status == "timeout":
        raise TimeoutError(f"download_video wrapper exceeded timeout on {video_id}")
    if status == "err":
        raise payload
    if not out_path.exists():
        raise RuntimeError(f"yt-dlp reported success but {out_path} is missing")


def _extract_window_worker(conn, video_path_str, seg_start_s, seg_end_s, num_frames):
    try:
        from decord import VideoReader, cpu
        from cache_train.rebuild_causal_cache import uniform_indices
        reader = VideoReader(video_path_str, ctx=cpu(0), num_threads=1)
        fps = float(reader.get_avg_fps())
        total = int(len(reader))
        start_frame = max(0, int(seg_start_s * fps))
        end_frame = min(total, int(seg_end_s * fps))
        if end_frame <= start_frame:
            end_frame = min(total, start_frame + num_frames)
        # uniform_indices(start, end, num_frames) takes an ABSOLUTE end index,
        # not a length -- confirmed 2026-09-11 the hard way: every non-AV1
        # smoke-test failure was "empty source interval [start, length)"
        # because passing (end_frame - start_frame) here gets read as an end
        # index smaller than start whenever start_frame > 0. The one existing
        # usage this project had (extract_vlm_guidance_paired.py's
        # uniform_indices(0, total, n)) never exposed this: with start=0,
        # "end" and "length" are numerically identical, so that call worked
        # by coincidence regardless of which one it actually is.
        try:
            indices = uniform_indices(start_frame, max(end_frame, start_frame + 1), num_frames)
        except Exception as e:
            raise type(e)(
                f"{e} (seg=[{seg_start_s},{seg_end_s}]s fps={fps:.3f} "
                f"video_total_frames={total} start_frame={start_frame} end_frame={end_frame})"
            ) from e

        frames = np.asarray(reader.get_batch(indices).asnumpy())
        frames = np.ascontiguousarray(frames[..., :3].astype(np.uint8, copy=False))
        try:
            timestamps = np.asarray(reader.get_frame_timestamp(indices), dtype=np.float64)
            centers = (timestamps[:, 0] + timestamps[:, 1]) * 0.5
        except Exception:
            fps_safe = fps if fps > 0 else 30.0
            centers = indices.astype(np.float64) / fps_safe
        conn.send(("ok", (frames, centers)))
    except Exception as e:
        conn.send(("err", e))
    finally:
        conn.close()


def extract_window_frames(video_path, seg_start_s, seg_end_s, num_frames):
    """Same "isolate a native decode call in a spawned subprocess with a
    hard timeout" pattern extract_vlm_guidance_paired.py already uses --
    a decord VideoReader call on a bad/corrupted download can hang with no
    exception ever raised in-process, confirmed repeatedly elsewhere in
    this project (see that script's own DECODE_TIMEOUT_S).
    """
    parent_conn, child_conn = _MP_CTX.Pipe(duplex=False)
    proc = _MP_CTX.Process(
        target=_extract_window_worker,
        args=(child_conn, str(video_path), seg_start_s, seg_end_s, num_frames),
    )
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
        raise TimeoutError(f"window decode exceeded {DECODE_TIMEOUT_S}s on {video_path}")
    if status == "err":
        raise payload
    return payload


def tensor_or_array_to_dtype(value, save_dtype):
    array = np.asarray(value)
    if save_dtype == "fp16":
        return array.astype(np.float16, copy=False)
    return array.astype(np.float32, copy=False)


def is_valid_output(path):
    if not path.exists():
        return False
    try:
        with np.load(path) as data:
            required = {"video_id", "vjepa_input_feats", "vjepa_target_feats"}
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
    ap.add_argument("--coin_json", default=None, help="local cached COIN.json; downloads fresh if omitted/missing")
    ap.add_argument("--output_dir", default=os.path.join(WORK_BASE, "coin_longaxis_cache"))
    ap.add_argument("--video_cache_dir", default=os.path.join(WORK_BASE, "coin_raw_videos"),
                     help="downloaded whole videos are kept here (not deleted) so a re-run "
                          "or a --limit increase doesn't re-download an already-fetched video")
    ap.add_argument("--vjepa_checkpoint", default=os.environ.get("VJEPA2_CKPT", "/data/checkpoints/vjepa2/vitl.pt"))
    ap.add_argument("--min_span_seconds", type=float, default=30.0,
                     help="skip videos whose first-to-last-step span is under this -- avoids "
                          "near-degenerate 'long axis' pairs that are barely longer than VANS's own")
    ap.add_argument("--min_steps", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="cap number of pairs (smoke test)")
    ap.add_argument("--save_dtype", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()

    from cache_train.rebuild_causal_cache import (
        NUM_PAST_FRAMES, NUM_TARGET_FRAMES,
        load_vjepa_encoder, preprocess_vjepa_frames, encode_vjepa_clip,
        sha256_file,
    )
    import torch

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    video_cache_root = Path(args.video_cache_dir)
    video_cache_root.mkdir(parents=True, exist_ok=True)

    database = load_coin_annotations(args.coin_json)
    pairs = select_longaxis_pairs(database, args.min_span_seconds, args.min_steps, args.limit)
    print(f"[INFO] {len(pairs)} candidate long-axis pairs "
          f"(min_span_seconds={args.min_span_seconds}, min_steps={args.min_steps}) "
          f"out of {len(database)} COIN videos", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] loading V-JEPA2 checkpoint {args.vjepa_checkpoint} ...", flush=True)
    vjepa_checkpoint_sha = sha256_file(Path(args.vjepa_checkpoint))
    vjepa_model = load_vjepa_encoder(Path(args.vjepa_checkpoint), device)

    saved_count = skipped_count = failed_count = 0
    seen_exc_types = set()
    with torch.no_grad():
        for i, pair in enumerate(pairs, start=1):
            video_id = pair["video_id"]
            output_path = output_root / f"{video_id}.npz"
            if is_valid_output(output_path):
                skipped_count += 1
                print(f"[SKIP {i}/{len(pairs)}] valid {output_path}", flush=True)
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

                past_clip, past_imgs = preprocess_vjepa_frames(past_frames)
                target_clip, target_imgs = preprocess_vjepa_frames(target_frames)
                vjepa_input_feats = encode_vjepa_clip(vjepa_model, past_clip, device)
                vjepa_target_feats = encode_vjepa_clip(vjepa_model, target_clip, device)

                payload = {
                    "schema_name": np.asarray("coin_longaxis_v1_no_guidance"),
                    "video_id": np.asarray(video_id),
                    "video_url": np.asarray(pair["video_url"]),
                    "coin_class": np.asarray(pair["class"]),
                    "span_seconds": np.asarray(pair["span_seconds"], dtype=np.float64),
                    "n_steps": np.asarray(pair["n_steps"], dtype=np.int32),
                    "past_label": np.asarray(pair["past_label"]),
                    "future_label": np.asarray(pair["future_label"]),
                    "past_frame_times_seconds": past_times.astype(np.float64),
                    "target_frame_times_seconds": target_times.astype(np.float64),
                    "past_imgs": past_imgs,
                    "target_imgs": target_imgs,
                    "vjepa_input_feats": tensor_or_array_to_dtype(vjepa_input_feats, args.save_dtype),
                    "vjepa_target_feats": tensor_or_array_to_dtype(vjepa_target_feats, args.save_dtype),
                    "vjepa_model": np.asarray("vjepa2_vit_large_rope"),
                    "vjepa_checkpoint_sha256": np.asarray(vjepa_checkpoint_sha),
                }
                atomic_write_npz(output_path, payload)
                saved_count += 1
                print(f"[SAVED {i}/{len(pairs)}] {output_path} span={pair['span_seconds']:.0f}s", flush=True)
            except Exception as exc:
                failed_count += 1
                exc_name = type(exc).__name__
                print(f"[FAIL {i}/{len(pairs)}] video_id={video_id}: {exc_name}: {exc}", file=sys.stderr, flush=True)
                if exc_name not in seen_exc_types:
                    seen_exc_types.add(exc_name)
                    traceback.print_exc()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    print(f"[DONE] saved={saved_count} skipped_valid={skipped_count} failed={failed_count} "
          f"output={output_root}", flush=True)
    return 1 if failed_count else 0


if __name__ == "__main__":
    sys.exit(main())
