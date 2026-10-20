"""Builds a VANS-style 2-choice "predict what happens next" QA eval split
from Ego4D's GoalStep annotations -- a 5th/6th benchmark (egocentric,
daily-life domain, distinct from COIN/YouCook2's third-person instructional
cooking videos and from TempCompass/MVBench's general-purpose clips).

Unlike extract_ego4d_longaxis.py (self-supervised reverse-direction pairs,
one first/last-step pair per video, no captions), this builds REAL QA items
matching eval_forward_checkpoint.py's schema directly -- same role as
extract_crosstask_qa.py, just for Ego4D instead of CrossTask, and mirrors
that script closely. Two differences from extract_crosstask_qa.py:

  1. Every CONSECUTIVE step pair per video becomes a candidate (not just
     first/last), the same convention extract_crosstask_qa.py itself uses
     -- more QA items per video than extract_ego4d_longaxis.py's "widest
     span" selection, which is the right choice for *that* script's
     self-supervised objective but would waste most of a video's
     annotations here.
  2. No download step. Ego4D videos require the official `ego4d` CLI
     (private S3 bucket, license credentials) -- not re-usable the way
     extract_crosstask_qa.py's plain yt-dlp call is, and those credentials
     are currently expired (pending renewal at ego4d-data.org). This script
     only uses videos ALREADY downloaded by job-50's earlier GoalStep run
     (expected at --video_cache_dir/v2/full_scale/<video_uid>.mp4); any
     video not already present is skipped, not re-downloaded.

Mirrors extract_crosstask_qa.py's two-phase structure otherwise: Phase 1
trims per-step clips with ffmpeg, Phase 2 encodes V-JEPA2 features once per
unique clip. Output schema (pid/in_clip/out_clip/question/correct_caption/
distractor_caption/difficulty) is identical, so it plugs into
eval_forward_checkpoint.py unmodified.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

import numpy as np

BASE = os.environ.get("VANS_ROOT", "/data")
WORK_BASE = os.environ.get("VANS_WORK_ROOT", "/data/vans_work_ego4d_qa")
THINKJEPA_ROOT = os.environ.get("THINKJEPA_ROOT", "/home/jovyan/ThinkJEPA")
if THINKJEPA_ROOT not in sys.path:
    sys.path.insert(0, THINKJEPA_ROOT)

CLIP_TRIM_TIMEOUT_S = 60


def load_goalstep_annotations(goalstep_json_path):
    with open(goalstep_json_path) as f:
        data = json.load(f)
    return data["videos"]


def select_qa_pairs(videos, video_cache_dir, min_segments=2, limit_videos=None, seed=42):
    """Every consecutive (step i, step i+1) pair, for videos whose raw mp4
    is already downloaded locally -- no new download attempted."""
    raw_root = Path(video_cache_dir) / "v2" / "full_scale"
    candidates = []
    n_missing_video = 0
    for video in videos:
        video_uid = video.get("video_uid")
        if not video_uid:
            continue
        video_path = raw_root / f"{video_uid}.mp4"
        if not video_path.exists():
            n_missing_video += 1
            continue
        segments = video.get("segments") or []
        if len(segments) < min_segments:
            continue
        goal_category = video.get("goal_category") or "this activity"
        segs_sorted = sorted(segments, key=lambda s: s["start_time"])
        for i in range(len(segs_sorted) - 1):
            cur, nxt = segs_sorted[i], segs_sorted[i + 1]
            correct = nxt.get("step_description") or nxt.get("step_category")
            if not correct:
                continue
            candidates.append({
                "video_uid": video_uid,
                "video_path": str(video_path),
                "pair_idx": i,
                "goal_category": goal_category,
                "past_segment": [cur["start_time"], cur["end_time"]],
                "future_segment": [nxt["start_time"], nxt["end_time"]],
                "correct_caption": correct,
            })
    print(f"[INFO] {n_missing_video} videos skipped (not downloaded locally)", flush=True)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    if limit_videos:
        seen_videos = set()
        kept = []
        for c in candidates:
            if len(seen_videos) >= limit_videos and c["video_uid"] not in seen_videos:
                continue
            seen_videos.add(c["video_uid"])
            kept.append(c)
        candidates = kept
    all_captions_by_video = {}
    for c in candidates:
        all_captions_by_video.setdefault(c["video_uid"], []).append(c["correct_caption"])
    video_ids = list(all_captions_by_video.keys())
    for c in candidates:
        other_video = c["video_uid"]
        for _ in range(10):
            other_video = rng.choice(video_ids)
            if other_video != c["video_uid"]:
                break
        c["distractor_caption"] = rng.choice(all_captions_by_video[other_video])
    return candidates


def trim_clip(source_video, start_s, end_s, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.5, end_s - start_s)
    result = subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start_s), "-i", str(source_video), "-t", str(duration),
         "-vf", "scale=-2:480", "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-an",
         str(out_path)],
        capture_output=True, text=True, timeout=CLIP_TRIM_TIMEOUT_S,
    )
    if result.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"ffmpeg trim failed: {result.stderr[-500:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--goalstep_json", required=True, help="path to goalstep_train.json")
    ap.add_argument("--video_cache_dir", default="/data/vans_work/ego4d_raw_videos",
                     help="where job-50's earlier run already downloaded videos "
                          "(expects <this>/v2/full_scale/<uid>.mp4); never writes here, read-only")
    ap.add_argument("--output_qa_split", default=os.path.join(BASE, "raw_data/qa_split_ego4d.json"))
    ap.add_argument("--clips_root", default=os.path.join(WORK_BASE, "clips"))
    ap.add_argument("--vjepa_cache_dir", default=os.path.join(WORK_BASE, "vjepa_cache_full"))
    ap.add_argument("--vjepa_checkpoint", default=os.environ.get("VJEPA2_CKPT", "/data/checkpoints/vjepa2/vitl.pt"))
    ap.add_argument("--limit_videos", type=int, default=None, help="cap distinct videos (smoke test)")
    ap.add_argument("--num_frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.num_frames is None:
        from cache_train.rebuild_causal_cache import NUM_PAST_FRAMES
        args.num_frames = NUM_PAST_FRAMES

    videos = load_goalstep_annotations(args.goalstep_json)
    pairs = select_qa_pairs(videos, args.video_cache_dir, limit_videos=args.limit_videos, seed=args.seed)
    print(f"[INFO] {len(pairs)} candidate QA pairs across "
          f"{len({p['video_uid'] for p in pairs})} already-downloaded videos", flush=True)

    clips_root = Path(args.clips_root)
    vjepa_cache_root = Path(args.vjepa_cache_dir)

    # Phase 1: trim per-step clips (no download -- video_path already exists).
    items = []
    trimmed_count = failed_count = 0
    seen_exc = set()
    for pi, p in enumerate(pairs, start=1):
        in_clip = clips_root / p["video_uid"] / f"{p['pair_idx']}_in.mp4"
        out_clip = clips_root / p["video_uid"] / f"{p['pair_idx']}_out.mp4"
        try:
            if not in_clip.exists():
                trim_clip(p["video_path"], *p["past_segment"], in_clip)
            if not out_clip.exists():
                trim_clip(p["video_path"], *p["future_segment"], out_clip)
            p["in_clip"] = str(in_clip)
            p["out_clip"] = str(out_clip)
            items.append(p)
            trimmed_count += 1
        except Exception as e:
            failed_count += 1
            exc_name = type(e).__name__
            print(f"[FAIL trim {pi}/{len(pairs)}] {p['video_uid']} pair {p['pair_idx']}: {exc_name}: {e}",
                  file=sys.stderr, flush=True)
            if exc_name not in seen_exc:
                seen_exc.add(exc_name)
                traceback.print_exc()
        if pi % 200 == 0:
            print(f"[{pi}/{len(pairs)} pairs] trimmed={trimmed_count} failed={failed_count}", flush=True)

    print(f"[PHASE1 DONE] trimmed={trimmed_count} failed={failed_count}", flush=True)

    # Phase 2: V-JEPA2 feature extraction for every unique clip.
    from cache_train.rebuild_causal_cache import load_vjepa_encoder, preprocess_vjepa_frames, encode_vjepa_clip, uniform_indices
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] loading V-JEPA2 checkpoint {args.vjepa_checkpoint} ...", flush=True)
    vjepa_model = load_vjepa_encoder(Path(args.vjepa_checkpoint), device)

    unique_clips = sorted({c for p in items for c in (p["in_clip"], p["out_clip"])})
    encoded_count = enc_failed_count = 0
    for ci, clip_path_str in enumerate(unique_clips, start=1):
        safe_name = os.path.relpath(clip_path_str, str(clips_root)).replace("/", "__").replace(".mp4", "")
        out_npz = vjepa_cache_root / f"{safe_name}.npz"
        if out_npz.exists():
            encoded_count += 1
            continue
        try:
            from decord import VideoReader, cpu
            reader = VideoReader(clip_path_str, ctx=cpu(0), num_threads=1)
            total = int(len(reader))
            indices = uniform_indices(0, total, args.num_frames)
            frames = np.asarray(reader.get_batch(indices).asnumpy())
            frames = np.ascontiguousarray(frames[..., :3].astype(np.uint8, copy=False))
            clip_t, _ = preprocess_vjepa_frames(frames)
            with torch.no_grad():
                feats = encode_vjepa_clip(vjepa_model, clip_t, device)
            out_npz.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_npz.parent / f".{out_npz.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}.npz"
            with tmp.open("wb") as handle:
                np.savez_compressed(handle, feats=np.asarray(feats).astype(np.float32))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, out_npz)
            encoded_count += 1
        except Exception as e:
            enc_failed_count += 1
            exc_name = type(e).__name__
            print(f"[FAIL encode {ci}/{len(unique_clips)}] {clip_path_str}: {exc_name}: {e}",
                  file=sys.stderr, flush=True)
            if exc_name not in seen_exc:
                seen_exc.add(exc_name)
                traceback.print_exc()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if ci % 200 == 0:
            print(f"[{ci}/{len(unique_clips)} clips] encoded={encoded_count} failed={enc_failed_count}", flush=True)

    print(f"[PHASE2 DONE] encoded={encoded_count} failed={enc_failed_count}", flush=True)

    encoded_set = {p.stem for p in vjepa_cache_root.glob("*.npz")} if vjepa_cache_root.exists() else set()

    def clip_has_feats(clip_path_str):
        safe_name = os.path.relpath(clip_path_str, str(clips_root)).replace("/", "__").replace(".mp4", "")
        return safe_name in encoded_set

    final_items = []
    for p in items:
        if not (clip_has_feats(p["in_clip"]) and clip_has_feats(p["out_clip"])):
            continue
        pid = f"ego4d_{p['video_uid']}_{p['pair_idx']}"
        final_items.append({
            "pid": pid,
            "in_clip": p["in_clip"],
            "out_clip": p["out_clip"],
            "question": (
                f"This is first-person video of someone doing {p['goal_category'].lower()}. "
                f"Based on the video I just sent, what should I do next?"
            ),
            "correct_caption": p["correct_caption"],
            "distractor_caption": p["distractor_caption"],
            "difficulty": p["goal_category"],
        })

    qa_split = {"train": [], "val": [], "test": final_items}
    out_path = Path(args.output_qa_split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(qa_split, f, indent=2)
    print(f"[DONE] wrote {len(final_items)} QA items -> {out_path}", flush=True)
    return 1 if (failed_count and not final_items) else 0


if __name__ == "__main__":
    sys.exit(main())
