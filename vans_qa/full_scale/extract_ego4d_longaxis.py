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


def select_longaxis_pairs_goalstep(videos, min_span_seconds, min_steps, max_span_seconds=None, limit=None):
    """One (past_step, future_step) candidate per video: the first and last
    TOP-LEVEL step segment by start time (not recursing into GoalStep's own
    nested sub-segments) -- the widest span this video's step annotations
    support, matching extract_coin_longaxis.py's own selection logic. Each
    pair already carries real [start,end] segments, so past_segment/
    future_segment are used as-is by extract_window_frames().

    max_span_seconds: a real job-50 example (goal_category=
    COOKING:COOKING_GENERAL, span=1798.8s/~30min, n_steps=27 -- "Move cereal
    packs into box" -> "Clean kitchen") surfaced that first/last-step span
    can run far longer than COIN's own long-axis distribution (median 71s,
    p90 154s, max 733s) ever does -- at 30 minutes apart, past and future
    windows share almost no visual continuity for V-JEPA2's future-latent
    predictor to actually learn from, just two disconnected moments in the
    same kitchen. Capped (default 720s, just above COIN's own observed max)
    so GoalStep's long-axis difficulty stays in the same regime as the
    already-validated COIN source instead of injecting extreme outliers.
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
        if max_span_seconds is not None and span > max_span_seconds:
            continue
        pairs.append({
            "video_uid": video["video_uid"],
            "pair_idx": 0,
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


def select_shortaxis_pairs_goalstep(videos, min_gap_seconds, max_gap_seconds, pairs_per_video, limit=None):
    """Multiple (past_step, future_step) candidates per video: CONSECUTIVE
    top-level step segments (i, i+1), filtered to a sane gap range --
    unlike narration's point-based shortaxis, GoalStep segments already
    have real [start,end] spans, so past_segment/future_segment are the
    segments themselves, not a synthetic window around a point.

    gap = segs[i+1]['start_time'] - segs[i]['end_time']. Verified on the
    65 already-downloaded videos before writing this: gap has min=-556.2s
    (steps that overlap in time -- two annotators' segments crossing, or a
    genuinely concurrent sub-step -- these are not real "past then future"
    pairs and must be excluded, not just capped) and a long tail out to
    5014.4s (a GoalStep video with a huge annotation gap, same "30 minutes
    apart, no visual continuity" problem as the longaxis selector). But the
    bulk is excellent: median gap 0.7s, p90 10.6s -- tighter than COIN's
    own 71s long-axis median -- so a modest [min_gap_seconds,
    max_gap_seconds] window (defaults 0/60, matching narration shortaxis's
    own range for consistency) keeps the vast majority of real pairs while
    dropping both the overlapping and the extreme-outlier ends.

    pairs_per_video caps any single video's contribution (one GoalStep
    video had 242 valid consecutive pairs) so the pair pool isn't
    dominated by a handful of long, heavily-annotated videos; spread
    evenly across the video's own step sequence (not just the first K) for
    temporal diversity within one video, same selection method
    select_shortaxis_pairs_narration already uses.
    """
    pairs = []
    for video in videos:
        video_uid = video.get("video_uid")
        segments = video.get("segments") or []
        if len(segments) < 2:
            continue
        segs_sorted = sorted(segments, key=lambda s: s["start_time"])
        candidates = []
        for i in range(len(segs_sorted) - 1):
            cur, nxt = segs_sorted[i], segs_sorted[i + 1]
            gap = nxt["start_time"] - cur["end_time"]
            if min_gap_seconds <= gap <= max_gap_seconds:
                candidates.append((cur, nxt, gap))
        if not candidates:
            continue
        if len(candidates) > pairs_per_video:
            step = len(candidates) / pairs_per_video
            candidates = [candidates[int(i * step)] for i in range(pairs_per_video)]
        for pair_idx, (cur, nxt, gap) in enumerate(candidates):
            pairs.append({
                "video_uid": video_uid,
                "pair_idx": pair_idx,
                "goal_category": video.get("goal_category"),
                "span_seconds": gap,
                "past_segment": [cur["start_time"], cur["end_time"]],
                "past_label": cur.get("step_description") or cur.get("step_category"),
                "future_segment": [nxt["start_time"], nxt["end_time"]],
                "future_label": nxt.get("step_description") or nxt.get("step_category"),
                "n_steps": len(segs_sorted),
            })
    pairs.sort(key=lambda p: (p["video_uid"], p["pair_idx"]))
    if limit:
        pairs = pairs[:limit]
    return pairs


def load_narration_annotations(narration_json_path):
    with open(narration_json_path) as f:
        return json.load(f)


def select_longaxis_pairs_narration(narration_data, min_span_seconds, min_steps, window_seconds,
                                     max_span_seconds=None, limit=None):
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

    max_span_seconds: same cap and same rationale as
    select_longaxis_pairs_goalstep()'s own -- Ego4D videos can run over an
    hour, so first/last-narration span is just as prone to the "30 minutes
    apart, no real visual continuity" problem GoalStep's long-axis showed.
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
        if max_span_seconds is not None and span > max_span_seconds:
            continue
        half_w = min(window_seconds / 2.0, span / 2.0)
        past_t, future_t = past_n["timestamp_sec"], future_n["timestamp_sec"]
        pairs.append({
            "video_uid": video_uid,
            "pair_idx": 0,
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


def select_shortaxis_pairs_narration(narration_data, min_gap_seconds, max_gap_seconds,
                                      pairs_per_video, window_seconds, min_steps, limit=None):
    """Multiple (past_narration, future_narration) candidates per video:
    consecutive narration pairs (i, i+1) whose gap falls in
    [min_gap_seconds, max_gap_seconds], up to --pairs_per_video of them per
    video, spread evenly across that video's narration timeline (not just
    the first K) for temporal diversity within one video.

    Why: select_longaxis_pairs_narration's first/last-narration choice can
    span a whole long video (some over an hour), and decord has to seek
    from the nearest keyframe all the way to a window that far in --
    confirmed via job-53's own logs (many `window decode exceeded 60s`
    timeouts) that this seek, not the download, is the real bottleneck.
    Consecutive narrations sit close together in the SAME video, so both
    windows decode fast, and reusing one already-downloaded video for
    several pairs instead of just one raises yield-per-download too. This
    is a real return to something closer to VANS's own "short-horizon
    guided pairs" character (see extract_coin_longaxis.py's own docstring
    for why long-axis was pursued as a DIFFERENT, wider-horizon addition
    to that) -- the domain (egocentric Ego4D vs VANS's third-person video)
    is still the novel variable this data source adds, not the horizon
    length specifically.
    """
    pairs = []
    for video_uid, entry in narration_data.items():
        narrations = ((entry.get("narration_pass_1") or {}).get("narrations")
                      or (entry.get("narration_pass_2") or {}).get("narrations") or [])
        if len(narrations) < min_steps:
            continue
        narr_sorted = sorted(narrations, key=lambda n: n["timestamp_sec"])
        candidates = []
        for i in range(len(narr_sorted) - 1):
            past_n, future_n = narr_sorted[i], narr_sorted[i + 1]
            gap = future_n["timestamp_sec"] - past_n["timestamp_sec"]
            if min_gap_seconds <= gap <= max_gap_seconds:
                candidates.append((past_n, future_n, gap))
        if not candidates:
            continue
        if len(candidates) > pairs_per_video:
            step = len(candidates) / pairs_per_video
            candidates = [candidates[int(i * step)] for i in range(pairs_per_video)]
        for pair_idx, (past_n, future_n, gap) in enumerate(candidates):
            half_w = min(window_seconds / 2.0, gap / 2.0)
            past_t, future_t = past_n["timestamp_sec"], future_n["timestamp_sec"]
            pairs.append({
                "video_uid": video_uid,
                "pair_idx": pair_idx,
                "goal_category": None,
                "span_seconds": gap,
                "past_segment": [max(0.0, past_t - half_w), past_t + half_w],
                "past_label": past_n["narration_text"],
                "future_segment": [future_t - half_w, future_t + half_w],
                "future_label": future_n["narration_text"],
                "n_steps": len(narr_sorted),
            })
    pairs.sort(key=lambda p: (p["video_uid"], p["pair_idx"]))
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


def split_video_keys(video_keys, eval_frac, seed):
    """Video-level train/eval split, done BEFORE pair selection: shuffles the
    distinct video_uids with a fixed seed and carves off eval_frac as eval.
    Must be video-level, not pair-level -- pairs from the same video share
    scene/context/participants, so a pair-level split would leak."""
    import random
    keys = sorted(set(video_keys))
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_eval = int(round(len(keys) * eval_frac))
    eval_keys = set(keys[:n_eval])
    train_keys = set(keys[n_eval:])
    return train_keys, eval_keys


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
    ap.add_argument("--pair_mode", choices=["longaxis", "shortaxis"], default="longaxis",
                     help="'longaxis' (default) takes one (first, last) pair per video, spanning "
                          "as much of the video as its annotations allow -- confirmed on real data "
                          "(both GoalStep and narration) to often span minutes, too sparse for real "
                          "visual continuity. 'shortaxis' instead takes up to --pairs_per_video "
                          "CONSECUTIVE pairs per video (narration: narration points; goalstep: "
                          "top-level step segments), each within [--min_gap_seconds, "
                          "--max_gap_seconds] -- now implemented for BOTH annotation sources. "
                          "Confirmed via job-53's own logs that longaxis's far-apart windows in "
                          "long videos are what's actually slow (decord seeking from the nearest "
                          "keyframe all the way to a window near the end of an hour-long video), "
                          "not the download.")
    ap.add_argument("--min_gap_seconds", type=float, default=10.0, help="shortaxis mode only")
    ap.add_argument("--max_gap_seconds", type=float, default=60.0, help="shortaxis mode only")
    ap.add_argument("--pairs_per_video", type=int, default=3, help="shortaxis mode only")
    ap.add_argument("--eval_frac", type=float, default=0.0,
                     help="if > 0, holds out this fraction of VIDEOS (not pairs -- pairs from the "
                          "same video share scene/context, so splitting by pair would leak) as a "
                          "separate eval set, written to --eval_output_dir instead of --output_dir. "
                          "Split is seeded and video-level, done once before pair selection.")
    ap.add_argument("--eval_output_dir", default=None,
                     help="required if --eval_frac > 0 -- held-out videos' pairs are written here")
    ap.add_argument("--output_dir", default=os.path.join(WORK_BASE, "ego4d_longaxis_cache"))
    ap.add_argument("--video_cache_dir", default=os.path.join(WORK_BASE, "ego4d_raw_videos"),
                     help="downloaded whole videos are kept here (not deleted) so a re-run "
                          "or a --limit increase doesn't re-download an already-fetched video")
    ap.add_argument("--vjepa_checkpoint", default=os.environ.get("VJEPA2_CKPT", "/data/checkpoints/vjepa2/vitl.pt"))
    ap.add_argument("--min_span_seconds", type=float, default=30.0, help="longaxis mode only")
    ap.add_argument("--max_span_seconds", type=float, default=720.0,
                     help="longaxis mode only -- caps first/last span so it stays in the same "
                          "regime as COIN's own long-axis distribution (median 71s, p90 154s, "
                          "max 733s observed); without this, GoalStep/narration first-last spans "
                          "can run past 30 minutes with almost no visual continuity between past "
                          "and future windows for V-JEPA2's predictor to learn from. Pass a "
                          "negative value to disable the cap.")
    ap.add_argument("--min_steps", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="cap number of pairs (smoke test)")
    ap.add_argument("--save_dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--seed", type=int, default=42, help="for the --eval_frac video-level split")
    args = ap.parse_args()
    if args.annotation_source == "goalstep" and not args.goalstep_json:
        ap.error("--goalstep_json is required for --annotation_source goalstep")
    if args.annotation_source == "narration" and not args.narration_json:
        ap.error("--narration_json is required for --annotation_source narration")
    if args.eval_frac and not (0.0 < args.eval_frac < 1.0):
        ap.error("--eval_frac must be between 0 and 1 (exclusive)")
    if args.eval_frac and not args.eval_output_dir:
        ap.error("--eval_output_dir is required when --eval_frac > 0")

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

    max_span = args.max_span_seconds if args.max_span_seconds is not None and args.max_span_seconds >= 0 else None

    goalstep_videos = narration_data = None
    if args.annotation_source == "goalstep":
        goalstep_videos = load_goalstep_annotations(args.goalstep_json)
        all_video_keys = [v["video_uid"] for v in goalstep_videos]
    else:
        narration_data = load_narration_annotations(args.narration_json)
        all_video_keys = list(narration_data.keys())

    train_keys, eval_keys = None, None
    if args.eval_frac > 0:
        train_keys, eval_keys = split_video_keys(all_video_keys, args.eval_frac, args.seed)
        print(f"[INFO] video-level split: {len(train_keys)} train / {len(eval_keys)} eval videos "
              f"(eval_frac={args.eval_frac}, seed={args.seed}, {len(all_video_keys)} total)", flush=True)

    def subset_videos(keys):
        """Filters goalstep_videos/narration_data down to the given video_uid
        set, done BEFORE pair selection so the resulting pairs never
        straddle train/eval (pairs from the same video share scene/context/
        participants, so splitting by pair instead of by video would leak)."""
        if keys is None:
            return goalstep_videos, narration_data
        if args.annotation_source == "goalstep":
            return [v for v in goalstep_videos if v["video_uid"] in keys], None
        return None, {k: v for k, v in narration_data.items() if k in keys}

    def build_pairs(gv, nd, limit):
        if args.annotation_source == "goalstep" and args.pair_mode == "longaxis":
            return select_longaxis_pairs_goalstep(gv, args.min_span_seconds, args.min_steps, max_span, limit)
        if args.annotation_source == "goalstep" and args.pair_mode == "shortaxis":
            return select_shortaxis_pairs_goalstep(gv, args.min_gap_seconds, args.max_gap_seconds,
                                                     args.pairs_per_video, limit)
        if args.annotation_source == "narration" and args.pair_mode == "longaxis":
            return select_longaxis_pairs_narration(nd, args.min_span_seconds, args.min_steps,
                                                     args.window_seconds, max_span, limit)
        return select_shortaxis_pairs_narration(nd, args.min_gap_seconds, args.max_gap_seconds,
                                                  args.pairs_per_video, args.window_seconds,
                                                  args.min_steps, limit)

    train_gv, train_nd = subset_videos(train_keys)
    train_pairs = build_pairs(train_gv, train_nd, args.limit)
    print(f"[INFO] {len(train_pairs)} candidate {args.pair_mode} pairs for TRAIN output "
          f"(annotation_source={args.annotation_source})", flush=True)

    eval_pairs = []
    if eval_keys:
        eval_gv, eval_nd = subset_videos(eval_keys)
        eval_pairs = build_pairs(eval_gv, eval_nd, args.limit)
        print(f"[INFO] {len(eval_pairs)} candidate {args.pair_mode} pairs for EVAL output "
              f"(annotation_source={args.annotation_source})", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] loading V-JEPA2 checkpoint {args.vjepa_checkpoint} ...", flush=True)
    vjepa_checkpoint_sha = sha256_file(Path(args.vjepa_checkpoint))
    vjepa_model = load_vjepa_encoder(Path(args.vjepa_checkpoint), device)

    def run_pairs(pairs, output_root):
        output_root.mkdir(parents=True, exist_ok=True)
        saved_count = skipped_count = failed_count = 0
        seen_exc_types = set()
        n = len(pairs)
        with torch.no_grad():
            for i, pair in enumerate(pairs, start=1):
                video_uid = pair["video_uid"]
                pair_idx = pair.get("pair_idx", 0)
                # shortaxis mode can produce several pairs per video_uid (see
                # select_shortaxis_pairs_narration); longaxis/goalstep always
                # have pair_idx=0. Only delete the downloaded video once this
                # is the LAST pair for this video_uid in the (video_uid,
                # pair_idx)-sorted list -- pairs list production groups
                # same-video pairs together, so checking the next entry's
                # video_uid is enough.
                is_last_pair_for_video = (i == n) or (pairs[i]["video_uid"] != video_uid)
                output_path = output_root / f"{video_uid}_{pair_idx}.npz"
                if is_valid_output(output_path):
                    skipped_count += 1
                    print(f"[SKIP {i}/{n}] valid {output_path}", flush=True)
                    continue
                # Confirmed 2026-09-29 via a live single-video download: the ego4d
                # CLI nests an extra version directory ("v2" for the current
                # dataset version) between -o's output dir and the dataset name --
                # actual path is <output_dir>/v2/full_scale/<uid>.mp4, not
                # <output_dir>/full_scale/<uid>.mp4 as the CLI's own --help text
                # example ("output_dir/full_scale/") implies. job-50's first full
                # run got 0/546 saved from this exact mismatch (every download
                # "succeeded" per the CLI's exit code, but the file was never
                # found at the wrong assumed path).
                video_path = video_cache_root / "v2" / "full_scale" / f"{video_uid}.mp4"
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
                        "pair_mode": np.asarray(getattr(args, "pair_mode", "longaxis")),
                        "video_uid": np.asarray(video_uid),
                        "pair_idx": np.asarray(pair_idx, dtype=np.int32),
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
                    print(f"[SAVED {i}/{n}] {output_path} span={pair['span_seconds']:.0f}s", flush=True)
                except Exception as exc:
                    failed_count += 1
                    exc_name = type(exc).__name__
                    print(f"[FAIL {i}/{n}] video_uid={video_uid} pair_idx={pair_idx}: {exc_name}: {exc}",
                          file=sys.stderr, flush=True)
                    if exc_name not in seen_exc_types:
                        seen_exc_types.add(exc_name)
                        traceback.print_exc()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                # Full-scale Ego4D videos can be large (tens of minutes) --
                # unlike COIN's short whole-video downloads, keeping every
                # downloaded video around indefinitely could exhaust disk at
                # this scale. Delete only once every pair for this video_uid
                # has had its turn (shortaxis mode reuses one download across
                # several pairs) -- re-running would just re-download it, same
                # as a cache miss, whether that turn succeeded or failed.
                if is_last_pair_for_video:
                    try:
                        video_path.unlink()
                    except FileNotFoundError:
                        pass

        print(f"[DONE] saved={saved_count} skipped_valid={skipped_count} failed={failed_count} "
              f"output={output_root}", flush=True)
        return saved_count, skipped_count, failed_count

    _, _, train_failed = run_pairs(train_pairs, output_root)
    total_failed = train_failed
    if eval_pairs:
        _, _, eval_failed = run_pairs(eval_pairs, Path(args.eval_output_dir))
        total_failed += eval_failed

    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
