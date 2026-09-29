"""Extracts genuinely long-horizon (past, future) V-JEPA2 feature pairs from
Ego4D's GoalStep annotations, for the reverse direction's self-supervised
leg -- a third data source alongside VANS and COIN (see
extract_coin_longaxis.py's own docstring for why the reverse direction
wants a "long axis": VANS's own short-step annotations only support small
spans). Mirrors extract_coin_longaxis.py closely; see that script for the
fuller rationale behind this general approach (real V-JEPA2 self-supervised
pairs, no VLM guidance, output schema deliberately has no vlm_old/vlm_new).

Two annotation sources, chosen via --annotation_source:

  goalstep (default): GoalStep's schema (per-video list of `segments`, each
    with `start_time`/`end_time`/`step_description`) is structurally the
    same shape as COIN.json's per-video step annotations --
    select_longaxis_pairs_goalstep() is copied over almost unchanged from
    extract_coin_longaxis.py's own select_longaxis_pairs() (first/last
    top-level step segment as past/future, span = last.end - first.start).
    Only 583 videos total (job-50's full run), and mixed into a reverse
    pool already dominated by VANS (~8253 items) this would only ever
    supply a few percent of any given training step's draws -- too dilute
    to move the needle either way regardless of whether Ego4D data is
    genuinely useful.

  narration: narration.json covers 9,611 videos -- comparable in scale to
    VANS's own reverse pool, giving real statistical weight if job-50's
    small goalstep-only run comes back inconclusive. Its narrations are
    POINT timestamps (dense per-video "#C C walks into the kitchen"-style
    captions), not [start,end] segments, so there's no inherent window the
    way GoalStep/COIN's own step boundaries provide one directly.
    select_longaxis_pairs_narration() picks the first and last narrated
    timestamp per video as the past/future ANCHORS (mirroring GoalStep's
    own "first/last annotated point = widest span this video supports"
    choice, for consistency), then builds a fixed-width window
    (--window_seconds, centered on each anchor, clamped to video bounds)
    around each -- unlike GoalStep/COIN, this project has no prior
    precedent for a narration-point window width, so this parameter is a
    genuinely new judgment call, not a carried-over convention.

Why Ego4D CLI, not yt-dlp: Ego4D videos are NOT on YouTube -- they're
licensed footage served from a private S3 bucket
(ego4d-consortium-sharing), requiring the official `ego4d` pip package's
CLI (which itself wraps boto3 with the license-holder's AWS credentials)
rather than a public downloader. download_video() below shells out to
`python3 -m ego4d.cli.cli --datasets full_scale --video_uids <uid>`
per video instead of COIN's yt-dlp call.

AWS credentials: read from AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY env vars
(populated from a Kubernetes Secret in this project's job templates, NEVER
written into a job YAML literally -- those get committed to this repo's
public GitHub remote) and written to ~/.aws/credentials at startup, since
the `ego4d` CLI's boto3.session.Session(profile_name=...) call does NOT
fall back to plain env vars the way a profile-less Session would.

Only the download/annotation-loading pieces differ from
extract_coin_longaxis.py; window extraction (decord, isolated subprocess +
timeout), V-JEPA2 encoding, and the output npz schema all reuse the exact
same cache_train.rebuild_causal_cache pieces, unchanged.
"""
import argparse
import json
import os
import subprocess
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

DOWNLOAD_TIMEOUT_S = 600  # full_scale Ego4D videos can run tens of minutes long
                          # (a real goalstep example ran ~54 min); much bigger than
                          # COIN's 180s (COIN videos average ~142s total)
DECODE_TIMEOUT_S = 60    # matches extract_coin_longaxis.py's own constant

import multiprocessing as mp  # noqa: E402
_MP_CTX = mp.get_context("spawn")


def write_aws_credentials():
    key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not key or not secret:
        raise RuntimeError("AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY must be set "
                            "(from the ego4d-aws-credentials Secret)")
    aws_dir = Path.home() / ".aws"
    aws_dir.mkdir(parents=True, exist_ok=True)
    (aws_dir / "credentials").write_text(f"[default]\naws_access_key_id = {key}\naws_secret_access_key = {secret}\n")
    (aws_dir / "config").write_text(f"[default]\nregion = {os.environ.get('AWS_DEFAULT_REGION', 'us-west-1')}\n")


def load_goalstep_annotations(goalstep_json_path):
    with open(goalstep_json_path) as f:
        data = json.load(f)
    return data["videos"]


def select_longaxis_pairs_goalstep(videos, min_span_seconds, min_steps, limit=None):
    """One (past_step, future_step) candidate per video: the first and last
    TOP-LEVEL step segment by start time (not recursing into GoalStep's own
    nested sub-segments) -- the widest span this video's step annotations
    support, matching extract_coin_longaxis.py's own selection logic. Each
    pair already carries real [start,end] segments, so past_segment/
    future_segment are used as-is by extract_window_frames().
    """
    pairs = []
    for video in videos:
        segments = video.get("segments") or []
        if len(segments) < min_steps:
            continue
        segs_sorted = sorted(segments, key=lambda s: s["start_time"])
        past_step, future_step = segs_sorted[0], segs_sorted[-1]
        span = future_step["end_time"] - past_step["start_time"]
        if span < min_span_seconds:
            continue
        pairs.append({
            "video_uid": video["video_uid"],
            "goal_category": video.get("goal_category"),
            "span_seconds": span,
            "past_segment": [past_step["start_time"], past_step["end_time"]],
            "past_label": past_step.get("step_description") or past_step.get("step_category"),
            "future_segment": [future_step["start_time"], future_step["end_time"]],
            "future_label": future_step.get("step_description") or future_step.get("step_category"),
            "n_steps": len(segs_sorted),
        })
    pairs.sort(key=lambda p: p["video_uid"])
    if limit:
        pairs = pairs[:limit]
    return pairs


def load_narration_annotations(narration_json_path):
    with open(narration_json_path) as f:
        return json.load(f)


def select_longaxis_pairs_narration(narration_data, min_span_seconds, min_steps, window_seconds, limit=None):
    """One (past_narration, future_narration) candidate per video: the
    first and last narrated timestamp (across narration_pass_1, falling
    back to narration_pass_2 if pass_1 is missing/too sparse for this
    video) -- mirrors GoalStep/COIN's own "first/last annotated point =
    widest span this video supports" choice, for consistency, even though
    narrations are point events rather than pre-existing [start,end] spans.
    Builds a --window_seconds-wide window centered on each anchor
    (clamped to [0, span between the two anchors] so the past/future
    windows never overlap each other even when window_seconds is large
    relative to span_seconds) so past_segment/future_segment come out in
    the same [start,end] shape select_longaxis_pairs_goalstep() produces --
    everything downstream of pair selection is annotation-source-agnostic.
    """
    pairs = []
    for video_uid, entry in narration_data.items():
        narrations = ((entry.get("narration_pass_1") or {}).get("narrations")
                      or (entry.get("narration_pass_2") or {}).get("narrations") or [])
        if len(narrations) < min_steps:
            continue
        narr_sorted = sorted(narrations, key=lambda n: n["timestamp_sec"])
        past_n, future_n = narr_sorted[0], narr_sorted[-1]
        span = future_n["timestamp_sec"] - past_n["timestamp_sec"]
        if span < min_span_seconds:
            continue
        half_w = min(window_seconds / 2.0, span / 2.0)
        past_t, future_t = past_n["timestamp_sec"], future_n["timestamp_sec"]
        pairs.append({
            "video_uid": video_uid,
            "goal_category": None,
            "span_seconds": span,
            "past_segment": [max(0.0, past_t - half_w), past_t + half_w],
            "past_label": past_n["narration_text"],
            "future_segment": [future_t - half_w, future_t + half_w],
            "future_label": future_n["narration_text"],
            "n_steps": len(narr_sorted),
        })
    pairs.sort(key=lambda p: p["video_uid"])
    if limit:
        pairs = pairs[:limit]
    return pairs


def _download_worker(conn, video_uid, output_dir_str):
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "ego4d.cli.cli",
                "-o", output_dir_str,
                "--datasets", "full_scale",
                "--video_uids", video_uid,
                "--no-metadata",
                "-y",
            ],
            capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT_S,
        )
        if result.returncode != 0:
            conn.send(("err", RuntimeError(f"ego4d CLI exit {result.returncode}: {result.stderr[-800:]}")))
        else:
            conn.send(("ok", None))
    except subprocess.TimeoutExpired:
        conn.send(("err", TimeoutError(f"ego4d CLI exceeded {DOWNLOAD_TIMEOUT_S}s on {video_uid}")))
    except Exception as e:
        conn.send(("err", e))
    finally:
        conn.close()


def download_video(video_uid, output_dir):
    """Same "isolate an external call in a spawned subprocess with a hard
    timeout" pattern extract_coin_longaxis.py's download_video() uses for
    yt-dlp -- an S3 download that stalls (network hiccup, huge file) should
    not be able to hang this script forever.
    """
    parent_conn, child_conn = _MP_CTX.Pipe(duplex=False)
    proc = _MP_CTX.Process(target=_download_worker, args=(child_conn, video_uid, str(output_dir)))
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
        raise TimeoutError(f"download_video wrapper exceeded timeout on {video_uid}")
    if status == "err":
        raise payload


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
            required = {"video_uid", "vjepa_input_feats", "vjepa_target_feats"}
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
    ap.add_argument("--annotation_source", choices=["goalstep", "narration"], default="goalstep")
    ap.add_argument("--goalstep_json", default=None, help="path to goalstep_train.json (--annotation_source goalstep)")
    ap.add_argument("--narration_json", default=None, help="path to narration.json (--annotation_source narration)")
    ap.add_argument("--window_seconds", type=float, default=8.0,
                     help="narration source only: width of the extraction window centered on each "
                          "chosen narration timestamp (narrations are point events, unlike "
                          "GoalStep/COIN's own [start,end] step segments) -- no prior convention "
                          "in this project to match, chosen as a plausible short-clip width")
    ap.add_argument("--output_dir", default=os.path.join(WORK_BASE, "ego4d_longaxis_cache"))
    ap.add_argument("--video_cache_dir", default=os.path.join(WORK_BASE, "ego4d_raw_videos"),
                     help="downloaded whole videos are kept here (not deleted) so a re-run "
                          "or a --limit increase doesn't re-download an already-fetched video")
    ap.add_argument("--vjepa_checkpoint", default=os.environ.get("VJEPA2_CKPT", "/data/checkpoints/vjepa2/vitl.pt"))
    ap.add_argument("--min_span_seconds", type=float, default=30.0)
    ap.add_argument("--min_steps", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="cap number of pairs (smoke test)")
    ap.add_argument("--save_dtype", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()
    if args.annotation_source == "goalstep" and not args.goalstep_json:
        ap.error("--goalstep_json is required for --annotation_source goalstep")
    if args.annotation_source == "narration" and not args.narration_json:
        ap.error("--narration_json is required for --annotation_source narration")

    write_aws_credentials()

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

    if args.annotation_source == "goalstep":
        videos = load_goalstep_annotations(args.goalstep_json)
        pairs = select_longaxis_pairs_goalstep(videos, args.min_span_seconds, args.min_steps, args.limit)
        print(f"[INFO] {len(pairs)} candidate long-axis pairs "
              f"(min_span_seconds={args.min_span_seconds}, min_steps={args.min_steps}) "
              f"out of {len(videos)} GoalStep videos", flush=True)
    else:
        narration_data = load_narration_annotations(args.narration_json)
        pairs = select_longaxis_pairs_narration(
            narration_data, args.min_span_seconds, args.min_steps, args.window_seconds, args.limit,
        )
        print(f"[INFO] {len(pairs)} candidate long-axis pairs "
              f"(min_span_seconds={args.min_span_seconds}, min_steps={args.min_steps}, "
              f"window_seconds={args.window_seconds}) out of {len(narration_data)} narrated videos", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] loading V-JEPA2 checkpoint {args.vjepa_checkpoint} ...", flush=True)
    vjepa_checkpoint_sha = sha256_file(Path(args.vjepa_checkpoint))
    vjepa_model = load_vjepa_encoder(Path(args.vjepa_checkpoint), device)

    saved_count = skipped_count = failed_count = 0
    seen_exc_types = set()
    with torch.no_grad():
        for i, pair in enumerate(pairs, start=1):
            video_uid = pair["video_uid"]
            output_path = output_root / f"{video_uid}.npz"
            if is_valid_output(output_path):
                skipped_count += 1
                print(f"[SKIP {i}/{len(pairs)}] valid {output_path}", flush=True)
                continue
            video_path = video_cache_root / "full_scale" / f"{video_uid}.mp4"
            try:
                if not video_path.exists():
                    download_video(video_uid, video_cache_root)
                if not video_path.exists():
                    raise RuntimeError(f"ego4d CLI reported success but {video_path} is missing")

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
                    "schema_name": np.asarray("ego4d_longaxis_v1_no_guidance"),
                    "annotation_source": np.asarray(args.annotation_source),
                    "video_uid": np.asarray(video_uid),
                    "goal_category": np.asarray(pair["goal_category"] or ""),
                    "span_seconds": np.asarray(pair["span_seconds"], dtype=np.float64),
                    "n_steps": np.asarray(pair["n_steps"], dtype=np.int32),
                    "past_label": np.asarray(pair["past_label"] or ""),
                    "future_label": np.asarray(pair["future_label"] or ""),
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
                # Full-scale Ego4D videos can be large (tens of minutes) --
                # unlike COIN's short whole-video downloads, keeping every
                # downloaded video around indefinitely could exhaust disk at
                # 583-video scale. Delete after this video's pair is safely
                # saved (re-running would just re-download it, same as a
                # cache miss).
                try:
                    video_path.unlink()
                except FileNotFoundError:
                    pass
            except Exception as exc:
                failed_count += 1
                exc_name = type(exc).__name__
                print(f"[FAIL {i}/{len(pairs)}] video_uid={video_uid}: {exc_name}: {exc}", file=sys.stderr, flush=True)
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
