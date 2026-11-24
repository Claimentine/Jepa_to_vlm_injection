"""Restores live-decode access for the Ego4D reverse-guided pairs without
keeping their full-length source videos.

The live reverse loader (train_latent_world_model_full.load_pair_live) only
re-decodes each pair's PAST window: the frames at past_frame_times_seconds
(32 frames over ~8 s for narration, up to ~40 s for goalstep). The full
full_scale videos average ~3.6 GB each (126 still on disk = 450 GB), so the
~2,900 unique videos behind the 8,412 cached pairs would need ~10 TB --
more than /data has. Instead, per video: download it, cut each pair's past
window (+-pad) into a clip at the ORIGINAL resolution, delete the full
video. Resizing happens at load time (the Qwen processor's own qwen_res).

Cutting at an arbitrary frame needs a re-encode. ffmpeg does the decode as
well as the encode (decord's get_batch leaks every decoded frame: job-175
grew ~230 MB/s until OOMKilled at 32Gi): it seeks to half a frame before
source frame start_idx's own timestamp (from decord's index, no decode) and
takes the next end_idx - start_idx + 1 frames with timestamps passed
through, x264 CRF 18 (visually lossless), no B-frames -- so clip frame i is
source frame start_idx + i. Nothing here decodes with decord at all: even
3 check frames each decode from the previous keyframe, and that leak alone
took job-175 to 30 GB. Each clip is checked by timestamps instead: ffmpeg's
showinfo reports every output frame's time, which (plus the seek point) must lie
nearer its intended source frame's decord index timestamp (read from the
container index, no decode) than either neighbour's -- this catches a wrong
frame count or a one-frame shift, while tolerating the ~12 ms off-grid
timestamps some Ego4D videos have.

Output per pid, under --clips_root/<annotation_source>/:
  <pid>.mp4   the past-window clip at the source fps
  <pid>.json  {"offset_seconds", "start_idx", "source_fps", ...}; written last,
              so its presence means the clip is complete

The loader maps each cached time to its source frame round(t * source_fps)
(the full-video rule), subtracts start_idx, and decodes that exact clip
frame index -- the cached times sit on half-frame boundaries, so shifting
the float times instead could flip the rounding to a neighbouring frame.

Idempotent: videos whose pids all have a .json are skipped, so a rerun
retries only what failed.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_ego4d_longaxis import download_video, write_aws_credentials  # noqa: E402



def build_index(cache_dirs, index_path):
    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)
    by_video = {}
    paths = sorted(p for d in cache_dirs for p in glob.glob(os.path.join(d, "*.npz")))
    for i, p in enumerate(paths, start=1):
        with np.load(p) as d:
            uid = str(d["video_uid"])
            by_video.setdefault(uid, []).append({
                "pid": str(d["pid"]),
                "source": str(d["annotation_source"]),
                "past_times": [float(t) for t in d["past_frame_times_seconds"]],
            })
        if i % 1000 == 0:
            print(f"[index] {i}/{len(paths)} npz read", flush=True)
    tmp = index_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(by_video, f)
    os.replace(tmp, index_path)
    return by_video


def clip_paths(clips_root, item):
    base = os.path.join(clips_root, item["source"], item["pid"])
    return base + ".mp4", base + ".json"


def check_clip_times(ffmpeg_stderr, seek, index_starts, first, n):
    """index_starts: decord index start times for source frames lo..hi, which
    cover the clip's n frames plus one neighbour on each side where they exist;
    the clip's frame k is meant to be source frame lo + first + k. Every output
    frame's showinfo time + seek must lie closer to its intended source frame
    than to either neighbour -- some Ego4D videos' timestamps sit ~12 ms off
    the 1/fps grid, so an absolute tolerance either misfires or must be loose
    enough to hide a real one-frame shift. Returns (max distance to the
    intended frame in seconds, source size parsed from showinfo)."""
    import re

    times = [float(m) for m in re.findall(r"\bn:\s*\d+\s.*?\bpts_time:\s*(-?[\d.]+)", ffmpeg_stderr)]
    size = re.search(r"\bs:(\d+)x(\d+)", ffmpeg_stderr)
    return _match_frames(times, seek, np.asarray(index_starts, dtype=np.float64), first, n), \
        ([int(size.group(1)), int(size.group(2))] if size else None)


def _match_frames(times, seek, ts, first, n):
    # showinfo can see one frame past -frames:v (the filter runs ahead of the
    # encoder's stop), so only the first n are the clip's own frames.
    if len(times) < n:
        raise RuntimeError(f"clip has {len(times)} frames, expected {n}")
    worst = 0.0
    for k in range(n):
        t = times[k] + seek
        j = first + k
        d = abs(t - ts[j])
        for nb in (j - 1, j + 1):
            if 0 <= nb < len(ts) and abs(t - ts[nb]) <= d:
                raise RuntimeError(f"clip frame {k} (source time {t:.4f}s) is nearer source frame "
                                   f"offset {nb - first} than its own ({k}): {ts[nb]:.4f}s vs {ts[j]:.4f}s")
        worst = max(worst, d)
    return worst


def cut_clip(reader, src_path, item, out_mp4, out_json, pad_seconds, crf):
    import subprocess

    import imageio_ffmpeg

    total = len(reader)
    fps = float(reader.get_avg_fps())
    t0, t1 = item["past_times"][0], item["past_times"][-1]
    start_idx = max(0, int(np.floor((t0 - pad_seconds) * fps)))
    end_idx = min(total - 1, int(np.ceil((t1 + pad_seconds) * fps)))
    if end_idx <= start_idx:
        raise ValueError(f"empty window {start_idx}..{end_idx} (total={total}, fps={fps})")
    n_frames = end_idx - start_idx + 1
    # index timestamps only -- get_frame_timestamp reads the container index, no decode
    lo, hi = max(0, start_idx - 1), min(total - 1, end_idx + 1)
    index_starts = reader.get_frame_timestamp(list(range(lo, hi + 1)))[:, 0].astype(np.float64)
    first = start_idx - lo
    seek = max(0.0, float(index_starts[first]) - 0.5 / fps)

    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    tmp_mp4 = out_mp4 + f".tmp.{os.getpid()}.{threading.get_ident()}.mp4"
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-nostats", "-loglevel", "info",
        "-ss", f"{seek:.6f}", "-i", str(src_path),
        "-map", "0:v:0", "-an", "-frames:v", str(n_frames), "-fps_mode", "passthrough",
        "-vf", "showinfo,pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-bf", "0", "-threads", "2",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp_mp4,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg exit {result.returncode}: {result.stderr[-500:]}")
        time_err, size = check_clip_times(result.stderr, seek, index_starts, first, n_frames)
    except BaseException:
        if os.path.exists(tmp_mp4):
            os.remove(tmp_mp4)
        raise
    os.replace(tmp_mp4, out_mp4)
    meta = {
        "pid": item["pid"], "offset_seconds": start_idx / fps, "start_idx": start_idx,
        "end_idx": end_idx, "source_fps": fps, "source_size": size, "crf": crf,
        "check_time_err_ms": time_err * 1000,
    }
    with open(out_json + ".tmp", "w") as f:
        json.dump(meta, f)
    os.replace(out_json + ".tmp", out_json)


def process_video(uid, items, args):
    from decord import VideoReader, cpu

    todo = [it for it in items if not os.path.exists(clip_paths(args.clips_root, it)[1])]
    if not todo:
        return uid, "skip", None
    dl_dir = Path(args.download_root) / uid
    video_path = dl_dir / "v2" / "full_scale" / f"{uid}.mp4"
    try:
        if not video_path.exists():
            download_video(uid, dl_dir)
        if not video_path.exists():
            raise RuntimeError(f"ego4d CLI reported success but {video_path} is missing")
        reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        n_err = 0
        for it in todo:
            out_mp4, out_json = clip_paths(args.clips_root, it)
            try:
                cut_clip(reader, video_path, it, out_mp4, out_json, args.pad_seconds, args.crf)
            except Exception as e:
                n_err += 1
                print(f"[CLIP FAIL] {it['pid']}: {type(e).__name__}: {e}", flush=True)
        del reader
        return uid, ("ok" if n_err == 0 else "partial"), None
    except Exception as e:
        return uid, "fail", f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(dl_dir, ignore_errors=True)


def verify(by_video, uids, clips_root, n):
    """Decodes finished clips exactly the way load_pair_live will and reports
    how far the achieved frame times land from the cached past_frame_times."""
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, _repo)
    from common.live_vlm_guidance import decode_frames_at_timestamps
    from decord import VideoReader, cpu

    checked = 0
    for u in uids:
        for it in by_video[u]:
            mp4, js = clip_paths(clips_root, it)
            if not os.path.exists(js) or checked >= n:
                continue
            meta = json.load(open(js))
            target = np.asarray(it["past_times"])
            clip_idx = np.round(target * meta["source_fps"]) - meta["start_idx"]
            frames, _, fps = decode_frames_at_timestamps(mp4, clip_idx / meta["source_fps"])
            n_clip = len(VideoReader(mp4, ctx=cpu(0)))
            in_range = bool((clip_idx >= 0).all() and (clip_idx < n_clip).all())
            print(f"[VERIFY] {it['pid']}: frames={frames.shape} clip_fps={fps:.3f} src_fps={meta['source_fps']:.3f} "
                  f"clip_frames={n_clip} all_32_indices_inside_clip={in_range} "
                  f"cut_check={meta.get('check_time_err_ms', 'n/a')}ms size={os.path.getsize(mp4)/1e6:.1f}MB",
                  flush=True)
            checked += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    root = "/data/vans_work_ego4d_reverse_guided"
    ap.add_argument("--cache_dirs", nargs="+",
                    default=[f"{root}/cache_narration", f"{root}/cache_goalstep"])
    ap.add_argument("--clips_root", default=f"{root}/past_clips")
    ap.add_argument("--download_root", default=f"{root}/tmp_full_videos")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--pad_seconds", type=float, default=0.5)
    ap.add_argument("--crf", type=int, default=18, help="x264 CRF for the re-encoded clips (lower = closer to source)")
    ap.add_argument("--limit_videos", type=int, default=None, help="smoke test: only the first N videos")
    ap.add_argument("--passes", type=int, default=2, help="extra passes retry videos that failed")
    ap.add_argument("--verify", type=int, default=0, help="after fetching, decode-check this many clips")
    args = ap.parse_args()

    if not os.environ.get("LAB22_API_KEY"):
        write_aws_credentials()
    os.makedirs(args.clips_root, exist_ok=True)
    by_video = build_index(args.cache_dirs, os.path.join(args.clips_root, "index_by_video.json"))
    uids = sorted(by_video)
    if args.limit_videos:
        uids = uids[: args.limit_videos]
    n_pairs = sum(len(by_video[u]) for u in uids)
    print(f"[INFO] {len(uids)} videos, {n_pairs} pairs -> {args.clips_root}", flush=True)

    for p in range(1, args.passes + 1):
        t_start = time.time()
        counts = {"ok": 0, "skip": 0, "partial": 0, "fail": 0}
        with ThreadPoolExecutor(args.concurrency) as pool:
            futures = [pool.submit(process_video, u, by_video[u], args) for u in uids]
            for i, fut in enumerate(futures, start=1):
                try:
                    uid, status, err = fut.result()
                except Exception:
                    traceback.print_exc()
                    continue
                counts[status] += 1
                if err:
                    print(f"[FAIL {i}/{len(uids)}] {uid}: {err}", flush=True)
                if i % 20 == 0 or i == len(uids):
                    rate = (counts["ok"] + counts["partial"]) / max(1e-9, (time.time() - t_start) / 3600)
                    print(f"[pass {p}] {i}/{len(uids)} videos {counts} ({rate:.0f} videos/h)", flush=True)
        if counts["fail"] == 0 and counts["partial"] == 0:
            break

    done = sum(os.path.exists(clip_paths(args.clips_root, it)[1]) for u in uids for it in by_video[u])
    print(f"[DONE] clips with metadata: {done}/{n_pairs} pairs", flush=True)
    if args.verify:
        verify(by_video, uids, args.clips_root, args.verify)


if __name__ == "__main__":
    main()
