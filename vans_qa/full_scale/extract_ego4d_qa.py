"""Builds a VANS-style 2-choice "predict what happens next" QA eval split
from Ego4D's GoalStep OR narration annotations (--annotation_source) -- a
5th/6th benchmark (egocentric, daily-life domain, distinct from COIN/
YouCook2's third-person instructional cooking videos and from
TempCompass/MVBench's general-purpose clips).

Narration support added 2026-10-25 so job-109's own narration video
downloads (reverse-guided pipeline, ~266GB already fetched) aren't
goalstep-only-usable dead weight -- select_qa_pairs_narration mirrors
extract_ego4d_longaxis.py's own select_shortaxis_pairs_narration for the
windowing (half_w centered on each narration timestamp), combined with
select_qa_pairs's (goalstep) own walk/download/distractor-sampling
structure.

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
  2. Downloads on demand (UPDATED 2026-10-24: the original docstring here
     said AWS credentials were expired -- they are not; job-95's own
     full-scale shortaxis extraction has been downloading hundreds of
     Ego4D videos successfully all session via the same ego4d-aws-
     credentials Secret). Reuses extract_ego4d_longaxis.py's own
     download_video()/write_aws_credentials() rather than duplicating that
     logic. Each video is deleted once every pair for it has been trimmed
     (same reasoning as extract_ego4d_longaxis.py's own delete-after-use:
     full_scale Ego4D videos can run tens of minutes, keeping them all
     around at QA-benchmark scale would exhaust disk). --no_download
     restores the original read-only-cache behavior for a quick rerun
     against whatever's already present (e.g. job-50/job-95's own
     video_cache_dir), skipping any video not already there.

Mirrors extract_crosstask_qa.py's two-phase structure otherwise: Phase 1
trims per-step clips with ffmpeg, Phase 2 encodes V-JEPA2 features once per
unique clip. Output schema (pid/in_clip/out_clip/question/correct_caption/
distractor_caption/difficulty) is identical, so it plugs into
eval_forward_checkpoint.py unmodified.
"""
import argparse
import collections
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
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from extract_ego4d_longaxis import download_video, write_aws_credentials, load_narration_annotations  # noqa: E402

CLIP_TRIM_TIMEOUT_S = 60


def load_goalstep_annotations(goalstep_json_path):
    with open(goalstep_json_path) as f:
        data = json.load(f)
    return data["videos"]


def select_qa_pairs(videos, video_cache_dir, min_segments=2, limit_videos=None, seed=42, download=True):
    """Every consecutive (step i, step i+1) pair. Walks the FULL shuffled
    eligible-video pool (not just its first limit_videos entries) and stops
    once limit_videos have been SUCCESSFULLY downloaded -- a failed
    download doesn't eat into the target count, it just moves on to the
    next candidate. Videos with more segments (what min_segments selects
    for, since more steps = more QA pairs) skew longer in duration too, so
    a naive "take the first limit_videos, whatever happens to them" cap
    risks ending up with very few usable videos if several in a row hit
    download_video's own DOWNLOAD_TIMEOUT_S (confirmed in practice: job-104
    failed its first 5 draws in a row). download=False restores the old
    read-only-cache behavior (skip anything not already local, still
    stopping at limit_videos successes)."""
    eligible = [v for v in videos if v.get("video_uid") and len(v.get("segments") or []) >= min_segments]
    rng = random.Random(seed)
    rng.shuffle(eligible)

    raw_root = Path(video_cache_dir) / "v2" / "full_scale"
    candidates = []
    n_download_failed = n_skipped = n_succeeded = 0
    for vi, video in enumerate(eligible, start=1):
        if limit_videos and n_succeeded >= limit_videos:
            break
        video_uid = video["video_uid"]
        video_path = raw_root / f"{video_uid}.mp4"
        if not video_path.exists():
            if not download:
                n_skipped += 1
                continue
            try:
                download_video(video_uid, Path(video_cache_dir))
            except Exception as e:
                n_download_failed += 1
                print(f"[WARN] download failed for {video_uid}: {type(e).__name__}: {e}", flush=True)
                continue
        if not video_path.exists():
            n_download_failed += 1
            continue
        n_succeeded += 1
        goal_category = video.get("goal_category") or "this activity"
        segs_sorted = sorted(video["segments"], key=lambda s: s["start_time"])
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
        if vi % 10 == 0:
            print(f"[tried {vi} videos, succeeded={n_succeeded}/{limit_videos or '?'}] "
                  f"candidates={len(candidates)} download_failed={n_download_failed} "
                  f"skipped={n_skipped}", flush=True)
    print(f"[INFO] tried {vi} videos total: {n_succeeded} succeeded, {n_download_failed} failed to "
          f"download, {n_skipped} skipped (--no_download), {len(candidates)} candidate pairs across "
          f"{len({c['video_uid'] for c in candidates})} videos", flush=True)
    rng.shuffle(candidates)
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


def select_qa_pairs_narration(narration_data, video_cache_dir, window_seconds=8.0, min_gap_seconds=10.0,
                               max_gap_seconds=60.0, min_steps=2, pairs_per_video=5, limit_videos=None,
                               seed=42, download=True):
    """Narration analogue of select_qa_pairs (goalstep) -- same walk/
    download/distractor-sampling structure, but pairs are consecutive
    narration POINTS (not [start,end] step segments), windowed via the
    same half_w-centered-on-timestamp construction
    extract_ego4d_longaxis.py's own select_shortaxis_pairs_narration uses
    for the reverse direction, not duplicated-and-diverged from it.
    pairs_per_video caps a single video's contribution -- unlike GoalStep's
    high-level steps (few per video), narration is dense (can be dozens to
    hundreds of points per video), so capping keeps the candidate pool from
    being dominated by a handful of heavily-narrated videos."""
    eligible = [(uid, entry) for uid, entry in narration_data.items()
                if len(((entry.get("narration_pass_1") or {}).get("narrations")
                        or (entry.get("narration_pass_2") or {}).get("narrations") or [])) >= min_steps]
    rng = random.Random(seed)
    rng.shuffle(eligible)

    raw_root = Path(video_cache_dir) / "v2" / "full_scale"
    candidates = []
    n_download_failed = n_skipped = n_succeeded = 0
    for vi, (video_uid, entry) in enumerate(eligible, start=1):
        if limit_videos and n_succeeded >= limit_videos:
            break
        video_path = raw_root / f"{video_uid}.mp4"
        if not video_path.exists():
            if not download:
                n_skipped += 1
                continue
            try:
                download_video(video_uid, Path(video_cache_dir))
            except Exception as e:
                n_download_failed += 1
                print(f"[WARN] download failed for {video_uid}: {type(e).__name__}: {e}", flush=True)
                continue
        if not video_path.exists():
            n_download_failed += 1
            continue
        n_succeeded += 1
        narrations = ((entry.get("narration_pass_1") or {}).get("narrations")
                      or (entry.get("narration_pass_2") or {}).get("narrations") or [])
        narr_sorted = sorted(narrations, key=lambda n: n["timestamp_sec"])
        video_candidates = []
        for i in range(len(narr_sorted) - 1):
            past_n, future_n = narr_sorted[i], narr_sorted[i + 1]
            gap = future_n["timestamp_sec"] - past_n["timestamp_sec"]
            if not (min_gap_seconds <= gap <= max_gap_seconds):
                continue
            correct = future_n.get("narration_text")
            if not correct:
                continue
            half_w = min(window_seconds / 2.0, gap / 2.0)
            past_t, future_t = past_n["timestamp_sec"], future_n["timestamp_sec"]
            video_candidates.append({
                "video_uid": video_uid,
                "video_path": str(video_path),
                "pair_idx": i,
                "goal_category": None,
                "past_segment": [max(0.0, past_t - half_w), past_t + half_w],
                "future_segment": [future_t - half_w, future_t + half_w],
                "correct_caption": correct,
            })
        if len(video_candidates) > pairs_per_video:
            step = len(video_candidates) / pairs_per_video
            video_candidates = [video_candidates[int(i * step)] for i in range(pairs_per_video)]
        candidates.extend(video_candidates)
        if vi % 10 == 0:
            print(f"[tried {vi} videos, succeeded={n_succeeded}/{limit_videos or '?'}] "
                  f"candidates={len(candidates)} download_failed={n_download_failed} "
                  f"skipped={n_skipped}", flush=True)
    print(f"[INFO] tried {vi} videos total: {n_succeeded} succeeded, {n_download_failed} failed to "
          f"download, {n_skipped} skipped (--no_download), {len(candidates)} candidate pairs across "
          f"{len({c['video_uid'] for c in candidates})} videos", flush=True)
    rng.shuffle(candidates)
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
    ap.add_argument("--annotation_source", choices=["goalstep", "narration"], default="goalstep")
    ap.add_argument("--goalstep_json", default=None, help="path to goalstep_train.json")
    ap.add_argument("--narration_json", default=None, help="path to narration.json")
    ap.add_argument("--window_seconds", type=float, default=8.0, help="narration mode only")
    ap.add_argument("--min_gap_seconds", type=float, default=10.0, help="narration mode only")
    ap.add_argument("--max_gap_seconds", type=float, default=60.0, help="narration mode only")
    ap.add_argument("--min_steps", type=int, default=2, help="narration mode only")
    ap.add_argument("--pairs_per_video", type=int, default=5,
                     help="narration mode only -- caps one video's contribution (narration is dense, "
                          "unlike GoalStep's few-per-video high-level steps)")
    ap.add_argument("--video_cache_dir", default="/data/vans_work/ego4d_raw_videos",
                     help="videos are downloaded here on demand (see module docstring) unless "
                          "--no_download is passed, in which case this is read-only, matching the "
                          "old job-50-cache-only behavior")
    ap.add_argument("--no_download", action="store_true",
                     help="skip any video not already present at --video_cache_dir instead of "
                          "downloading it")
    ap.add_argument("--skip_phase2", action="store_true",
                     help="stop after Phase 1 (download+trim) -- no VJEPA2 load, no GPU touched or "
                          "requested at all. For a long CPU-only overnight download run: rerun the "
                          "same command without this flag tomorrow to do the VJEPA encoding, picking "
                          "up from the already-trimmed clips with no re-download needed.")
    ap.add_argument("--output_qa_split", default=os.path.join(BASE, "raw_data/qa_split_ego4d.json"))
    ap.add_argument("--clips_root", default=os.path.join(WORK_BASE, "clips"))
    ap.add_argument("--vjepa_cache_dir", default=os.path.join(WORK_BASE, "vjepa_cache_full"))
    ap.add_argument("--vjepa_checkpoint", default=os.environ.get("VJEPA2_CKPT", "/data/checkpoints/vjepa2/vitl.pt"))
    ap.add_argument("--limit_videos", type=int, default=None, help="cap distinct videos (smoke test)")
    ap.add_argument("--num_frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.annotation_source == "goalstep" and not args.goalstep_json:
        ap.error("--goalstep_json is required when --annotation_source=goalstep")
    if args.annotation_source == "narration" and not args.narration_json:
        ap.error("--narration_json is required when --annotation_source=narration")
    if args.num_frames is None:
        from cache_train.rebuild_causal_cache import NUM_PAST_FRAMES
        args.num_frames = NUM_PAST_FRAMES
    if not args.no_download:
        write_aws_credentials()

    if args.annotation_source == "goalstep":
        videos = load_goalstep_annotations(args.goalstep_json)
        pairs = select_qa_pairs(videos, args.video_cache_dir, limit_videos=args.limit_videos, seed=args.seed,
                                 download=not args.no_download)
    else:
        narration_data = load_narration_annotations(args.narration_json)
        pairs = select_qa_pairs_narration(
            narration_data, args.video_cache_dir, window_seconds=args.window_seconds,
            min_gap_seconds=args.min_gap_seconds, max_gap_seconds=args.max_gap_seconds,
            min_steps=args.min_steps, pairs_per_video=args.pairs_per_video,
            limit_videos=args.limit_videos, seed=args.seed, download=not args.no_download,
        )
    print(f"[INFO] {len(pairs)} candidate QA pairs across "
          f"{len({p['video_uid'] for p in pairs})} videos", flush=True)

    clips_root = Path(args.clips_root)
    vjepa_cache_root = Path(args.vjepa_cache_dir)

    # Phase 1: trim per-step clips. pairs is shuffled (see select_qa_pairs),
    # not grouped by video_uid, so "last pair for this video" is tracked via
    # a remaining-count map rather than extract_ego4d_longaxis.py's own
    # next-entry-lookahead trick -- delete each downloaded video once every
    # one of ITS pairs has had its turn, same disk-usage reasoning.
    remaining_for_video = collections.Counter(p["video_uid"] for p in pairs)
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
        remaining_for_video[p["video_uid"]] -= 1
        if not args.no_download and remaining_for_video[p["video_uid"]] <= 0:
            try:
                Path(p["video_path"]).unlink()
            except FileNotFoundError:
                pass
        if pi % 200 == 0:
            print(f"[{pi}/{len(pairs)} pairs] trimmed={trimmed_count} failed={failed_count}", flush=True)

    print(f"[PHASE1 DONE] trimmed={trimmed_count} failed={failed_count}", flush=True)

    if args.skip_phase2:
        # Download+trim only, no GPU touched at all -- not just lazy model
        # loading (which still leaves the pod HOLDING a GPU reservation the
        # whole time regardless of when the code gets around to using it,
        # exactly what trips NRP's idle-GPU policy), but genuinely never
        # requesting one. Re-running this same script tomorrow WITHOUT
        # --skip_phase2 resumes cleanly: already-trimmed in_clip/out_clip
        # files are skipped (see the `if not in_clip.exists()` checks
        # above), already-downloaded source videos were already deleted
        # right after their trim (by design, no longer needed), and
        # select_qa_pairs's own already-local-video short-circuit means a
        # full target count reached tonight costs ~nothing to re-derive.
        print("[DONE] skip_phase2 -- stopping here, rerun without --skip_phase2 to do VJEPA encoding", flush=True)
        return 0

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
        pid = f"ego4d_{args.annotation_source}_{p['video_uid']}_{p['pair_idx']}"
        activity = (p["goal_category"] or "an everyday activity").lower()
        final_items.append({
            "pid": pid,
            "in_clip": p["in_clip"],
            "out_clip": p["out_clip"],
            "question": (
                f"This is first-person video of someone doing {activity}. "
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
