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

Cutting at an arbitrary frame needs a re-encode: every frame in the window
is decoded from the source and piped to x264 at CRF 18 (visually lossless),
same fps, no B-frames, so clip frame i is source frame start_idx + i.

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

CHUNK_FRAMES = 64


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


def cut_clip(reader, item, out_mp4, out_json, pad_seconds, crf):
    import subprocess

    import imageio_ffmpeg

    total = len(reader)
    fps = float(reader.get_avg_fps())
    t0, t1 = item["past_times"][0], item["past_times"][-1]
    start_idx = max(0, int(np.floor((t0 - pad_seconds) * fps)))
    end_idx = min(total - 1, int(np.ceil((t1 + pad_seconds) * fps)))
    if end_idx <= start_idx:
        raise ValueError(f"empty window {start_idx}..{end_idx} (total={total}, fps={fps})")

    h, w = reader[0].shape[:2]
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    tmp_mp4 = out_mp4 + f".tmp.{os.getpid()}.{threading.get_ident()}.mp4"
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-bf", "0",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp_mp4,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for s in range(start_idx, end_idx + 1, CHUNK_FRAMES):
            idx = list(range(s, min(s + CHUNK_FRAMES, end_idx + 1)))
            proc.stdin.write(np.ascontiguousarray(reader.get_batch(idx).asnumpy()[..., :3]).tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg exit {proc.returncode}: {err[-500:]}")
    except BaseException:
        proc.kill()
        if os.path.exists(tmp_mp4):
            os.remove(tmp_mp4)
        raise
    os.replace(tmp_mp4, out_mp4)
    meta = {
        "pid": item["pid"], "offset_seconds": start_idx / fps, "start_idx": start_idx,
        "end_idx": end_idx, "source_fps": fps, "source_size": [w, h], "crf": crf,
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
        reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
        n_err = 0
        for it in todo:
            out_mp4, out_json = clip_paths(args.clips_root, it)
            try:
                cut_clip(reader, it, out_mp4, out_json, args.pad_seconds, args.crf)
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
            frames, centers, fps = decode_frames_at_timestamps(mp4, clip_idx / meta["source_fps"])
            err = np.abs(centers + meta["offset_seconds"] - target)
            n_clip = len(VideoReader(mp4, ctx=cpu(0)))
            print(f"[VERIFY] {it['pid']}: frames={frames.shape} clip_fps={fps:.3f} src_fps={meta['source_fps']:.3f} "
                  f"clip_frames={n_clip} size={os.path.getsize(mp4)/1e6:.1f}MB "
                  f"max_time_err={err.max()*1000:.1f}ms (1 frame={1000/meta['source_fps']:.1f}ms)", flush=True)
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
