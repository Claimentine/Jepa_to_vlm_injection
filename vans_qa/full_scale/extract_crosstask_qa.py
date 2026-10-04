"""Builds a VANS-style 2-choice QA eval split from CrossTask (Zhukov et al.,
CVPR 2019), as a domain-matched 4th out-of-domain benchmark alongside
TempCompass and MVBench.

Why CrossTask specifically: TempCompass/MVBench are general-purpose temporal-
reasoning benchmarks -- testing a 2B VLM's broad world knowledge more than
whether JEPA injection helps "predict what happens next in an instructional
video", which is VANS/COIN's own actual training objective. CrossTask's own
annotation schema is structurally identical to COIN's: 18 primary tasks
(cooking, furniture, car maintenance, etc.), each video is one ordered list
of manually-timed steps -- see extract_coin_longaxis.py's own docstring for
why that same structure made COIN a good long-axis source. Confirmed 2026-10
by downloading the real crosstask_release.zip (not just the paper): 2750
annotated videos across the 18 primary tasks, annotations/<task_id>_
<video_id>.csv rows are <step_number>,<start_sec>,<end_sec>, and
tasks_primary.txt gives each task's ordered step descriptions directly --
this script needs no new annotation-parsing design, it's the same
(task -> ordered steps -> per-video timed segments) shape COIN already has.

Unlike COIN's own long-axis extraction (self-supervised, no captions, no
VLM step), this produces real 2-choice QA pairs matching VANS's own schema
(in_clip/out_clip/question/correct_caption/distractor_caption/difficulty/
pid) so it plugs directly into eval_forward_checkpoint.py unmodified --
same role as build_mvbench_qa_split.py, just for a domain-matched source
instead of a general-purpose one.

Pair construction: for each video, every CONSECUTIVE annotated step pair
(step i, step i+1) becomes one QA item -- in_clip is step i's own segment,
out_clip is step i+1's (not used for scoring, kept for schema parity with
VANS/MVBench, which also carry an out_clip field even though eval_forward_
checkpoint.py only ever decodes in_clip). correct_caption is step i+1's own
text description (from tasks_primary.txt, by step number, 1-indexed);
distractor_caption is a different step's real description, sampled from a
DIFFERENT video after every candidate pair is built (same "real caption
from elsewhere, not a synthetic negative" convention VANS/MVBench both use).

Video handling mirrors extract_coin_longaxis.py exactly (whole-video yt-dlp
download once per video_id, deno JS-runtime requirement, isolated-subprocess
+ timeout for both the download and the decode) -- the only difference is
this script also writes a real per-step .mp4 CLIP file (via ffmpeg, frame-
accurate re-encode) at VANS_WORK_ROOT's own clips/<video_id>/<step>.mp4
convention, not just extracted frames, since train_qa_full.py's eval path
decodes the raw clip live (qwen-vl-utils/decord), unlike COIN long-axis's
self-supervised pairs which only ever needed frames for immediate V-JEPA2
encoding. V-JEPA2 features for that same clip are cached separately at
vjepa_cache_full/<video_id>__<step>.npz under the "feats" key, matching
train_qa_full.py's load_vjepa_in_feats()/has_features() convention exactly
(see that script: VJEPA_CACHE/{safe_name}.npz, key "feats") -- this is NOT
the same key/shape convention as COIN/Ego4D long-axis's own
vjepa_input_feats/vjepa_target_feats pair schema, intentionally: this is
the FORWARD-direction QA schema, not a reverse-direction self-supervised one.
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import traceback
import urllib.request
import uuid
import zipfile
from pathlib import Path

import numpy as np

BASE = os.environ.get("VANS_ROOT", "/data")
WORK_BASE = os.environ.get("VANS_WORK_ROOT", "/data/vans_work_crosstask")
THINKJEPA_ROOT = os.environ.get("THINKJEPA_ROOT", "/home/jovyan/ThinkJEPA")
if THINKJEPA_ROOT not in sys.path:
    sys.path.insert(0, THINKJEPA_ROOT)

CROSSTASK_ZIP_URL = "https://www.di.ens.fr/~dzhukov/crosstask/crosstask_release.zip"

DOWNLOAD_TIMEOUT_S = 180
DECODE_TIMEOUT_S = 60
CLIP_TRIM_TIMEOUT_S = 60

import multiprocessing as mp  # noqa: E402
_MP_CTX = mp.get_context("spawn")


def fetch_crosstask_annotations(cache_dir):
    """Downloads+extracts the small (~900KB) official annotations zip --
    not the 30GB feature archive or any video, just task lists / video ID
    list / per-video step timings. Idempotent: skips if already extracted.
    """
    out_dir = Path(cache_dir) / "crosstask_release"
    if (out_dir / "tasks_primary.txt").exists():
        return out_dir
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path = Path(cache_dir) / "crosstask_release.zip"
    print(f"[INFO] downloading {CROSSTASK_ZIP_URL} ...", flush=True)
    urllib.request.urlretrieve(CROSSTASK_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache_dir)
    return out_dir


def load_tasks_primary(tasks_path):
    """tasks_primary.txt: 5 lines per task (id, name, url, n_steps,
    comma-separated ordered step descriptions), blank line between tasks."""
    tasks = {}
    with open(tasks_path) as f:
        lines = [ln.rstrip("\n") for ln in f]
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        task_id = lines[i].strip()
        name = lines[i + 1].strip()
        n_steps = int(lines[i + 3].strip())
        steps = [s.strip() for s in lines[i + 4].split(",")]
        assert len(steps) == n_steps, f"task {task_id}: declared {n_steps} steps, got {len(steps)}"
        tasks[task_id] = {"name": name, "steps": steps}
        i += 5
    return tasks


def load_videos_csv(videos_path):
    """videos.csv: <task_id>,<youtube_id>,<url> -- returns [(task_id, video_id), ...]."""
    out = []
    with open(videos_path) as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                out.append((row[0].strip(), row[1].strip()))
    return out


def load_annotation_csv(path):
    """<task_id>_<video_id>.csv: <step_number>,<start_s>,<end_s>, NOT
    pre-sorted by time (confirmed on a real sample) -- sort by start here,
    same as extract_coin_longaxis.py sorts COIN's own step annotations."""
    rows = []
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) != 3:
                continue
            step_num, start_s, end_s = int(row[0]), float(row[1]), float(row[2])
            rows.append((step_num, start_s, end_s))
    rows.sort(key=lambda r: r[1])
    return rows


def select_qa_pairs(crosstask_dir, limit_videos=None, seed=42):
    tasks = load_tasks_primary(crosstask_dir / "tasks_primary.txt")
    videos = load_videos_csv(crosstask_dir / "videos.csv")
    videos = [(t, v) for t, v in videos if t in tasks]  # primary tasks only
    annot_dir = crosstask_dir / "annotations"

    candidates = []
    for task_id, video_id in videos:
        annot_path = annot_dir / f"{task_id}_{video_id}.csv"
        if not annot_path.exists():
            continue
        steps = load_annotation_csv(annot_path)
        if len(steps) < 2:
            continue
        task = tasks[task_id]
        for i in range(len(steps) - 1):
            cur_num, cur_start, cur_end = steps[i]
            nxt_num, nxt_start, nxt_end = steps[i + 1]
            if not (1 <= cur_num <= len(task["steps"])) or not (1 <= nxt_num <= len(task["steps"])):
                continue  # a handful of files have step numbers outside the declared range -- skip, don't guess
            candidates.append({
                "task_id": task_id, "task_name": task["name"], "video_id": video_id,
                "pair_idx": i,
                "past_segment": [cur_start, cur_end], "past_step_num": cur_num,
                "future_segment": [nxt_start, nxt_end], "future_step_num": nxt_num,
                "correct_caption": task["steps"][nxt_num - 1],
            })
    rng = random.Random(seed)
    rng.shuffle(candidates)
    if limit_videos:
        seen_videos = set()
        kept = []
        for c in candidates:
            if len(seen_videos) >= limit_videos and c["video_id"] not in seen_videos:
                continue
            seen_videos.add(c["video_id"])
            kept.append(c)
        candidates = kept
    # distractor_caption: a real step description from a DIFFERENT video,
    # sampled after the full candidate pool exists (same convention as
    # VANS/MVBench's own QA construction -- a real caption that's simply
    # wrong for this clip, not a synthetic negative)
    all_captions_by_video = {}
    for c in candidates:
        all_captions_by_video.setdefault(c["video_id"], []).append(c["correct_caption"])
    video_ids = list(all_captions_by_video.keys())
    for c in candidates:
        other_video = c["video_id"]
        for _ in range(10):
            other_video = rng.choice(video_ids)
            if other_video != c["video_id"]:
                break
        c["distractor_caption"] = rng.choice(all_captions_by_video[other_video])
    return candidates


YT_DLP_JS_RUNTIME = os.environ.get("YT_DLP_JS_RUNTIME", "deno:/tmp/deno")


def _download_worker(conn, video_id, out_path_str):
    try:
        result = subprocess.run(
            [
                "yt-dlp", "--no-progress", "--quiet",
                "--js-runtimes", YT_DLP_JS_RUNTIME,
                "-f", "bestvideo[height<=480][vcodec^=avc1]+bestaudio/"
                      "best[height<=480][vcodec^=avc1]/best[height<=480]/best",
                "--merge-output-format", "mp4",
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


def trim_clip(source_video, start_s, end_s, out_path):
    """ffmpeg frame-accurate trim into its own small mp4 -- re-encodes (not
    stream-copy) since -ss/-t on a copy can only cut at keyframes, same
    reasoning as this session's earlier sample-clip extraction for the
    reverse-training-data-samples artifact."""
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


def _vjepa_encode_worker(conn, clip_path_str, vjepa_checkpoint_str, num_frames):
    try:
        from decord import VideoReader, cpu
        from cache_train.rebuild_causal_cache import (
            uniform_indices, load_vjepa_encoder, preprocess_vjepa_frames, encode_vjepa_clip,
        )
        import torch
        reader = VideoReader(clip_path_str, ctx=cpu(0), num_threads=1)
        total = int(len(reader))
        indices = uniform_indices(0, total, num_frames)
        frames = np.asarray(reader.get_batch(indices).asnumpy())
        frames = np.ascontiguousarray(frames[..., :3].astype(np.uint8, copy=False))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_vjepa_encoder(Path(vjepa_checkpoint_str), device)
        clip_t, _ = preprocess_vjepa_frames(frames)
        with torch.no_grad():
            feats = encode_vjepa_clip(model, clip_t, device)
        conn.send(("ok", np.asarray(feats).astype(np.float32)))
    except Exception as e:
        conn.send(("err", e))
    finally:
        conn.close()


# a fresh V-JEPA2 model load per clip (inside the spawned worker) would be
# far too slow at thousands-of-clips scale -- unlike extract_coin_longaxis.py
# (one model load for the whole run, reused across pairs in-process), this
# script's clip-trim step already isolates ffmpeg/yt-dlp per-call, so V-JEPA
# encoding is done in-process instead, after all clips are trimmed, loading
# the encoder once. See main() for the two-phase structure.


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations_cache_dir", default=os.path.join(WORK_BASE, "crosstask_annotations"))
    ap.add_argument("--output_qa_split", default=os.path.join(BASE, "raw_data/qa_split_crosstask.json"))
    ap.add_argument("--video_cache_dir", default=os.path.join(WORK_BASE, "crosstask_raw_videos"))
    ap.add_argument("--clips_root", default=os.path.join(WORK_BASE, "clips"))
    ap.add_argument("--vjepa_cache_dir", default=os.path.join(WORK_BASE, "vjepa_cache_full"))
    ap.add_argument("--vjepa_checkpoint", default=os.environ.get("VJEPA2_CKPT", "/data/checkpoints/vjepa2/vitl.pt"))
    ap.add_argument("--limit_videos", type=int, default=None, help="cap distinct videos (smoke test)")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    crosstask_dir = fetch_crosstask_annotations(args.annotations_cache_dir)
    pairs = select_qa_pairs(crosstask_dir, limit_videos=args.limit_videos, seed=args.seed)
    print(f"[INFO] {len(pairs)} candidate QA pairs across "
          f"{len({p['video_id'] for p in pairs})} videos", flush=True)

    video_cache_root = Path(args.video_cache_dir)
    clips_root = Path(args.clips_root)
    vjepa_cache_root = Path(args.vjepa_cache_dir)
    video_cache_root.mkdir(parents=True, exist_ok=True)

    # Phase 1: download + trim. One whole-video download per video_id,
    # reused for every pair drawn from it (matches extract_coin_longaxis.py's
    # own download-once-per-video economy).
    items = []
    by_video = {}
    for p in pairs:
        by_video.setdefault(p["video_id"], []).append(p)

    n = len(by_video)
    trimmed_count = failed_count = 0
    seen_exc = set()
    for vi, (video_id, video_pairs) in enumerate(by_video.items(), start=1):
        video_path = video_cache_root / f"{video_id}.mp4"
        try:
            if not video_path.exists():
                download_video(video_id, video_path)
        except Exception as e:
            failed_count += len(video_pairs)
            exc_name = type(e).__name__
            print(f"[FAIL download {vi}/{n}] {video_id}: {exc_name}: {e}", file=sys.stderr, flush=True)
            if exc_name not in seen_exc:
                seen_exc.add(exc_name)
                traceback.print_exc()
            continue

        for p in video_pairs:
            in_clip = clips_root / video_id / f"{p['past_step_num']}.mp4"
            out_clip = clips_root / video_id / f"{p['future_step_num']}.mp4"
            try:
                if not in_clip.exists():
                    trim_clip(video_path, *p["past_segment"], in_clip)
                if not out_clip.exists():
                    trim_clip(video_path, *p["future_segment"], out_clip)
                p["in_clip"] = str(in_clip)
                p["out_clip"] = str(out_clip)
                items.append(p)
                trimmed_count += 1
            except Exception as e:
                failed_count += 1
                exc_name = type(e).__name__
                print(f"[FAIL trim {vi}/{n}] {video_id} pair {p['pair_idx']}: {exc_name}: {e}",
                      file=sys.stderr, flush=True)
                if exc_name not in seen_exc:
                    seen_exc.add(exc_name)
                    traceback.print_exc()

        try:
            video_path.unlink()
        except FileNotFoundError:
            pass

        if vi % 50 == 0:
            print(f"[{vi}/{n} videos] trimmed={trimmed_count} failed={failed_count}", flush=True)

    print(f"[PHASE1 DONE] trimmed={trimmed_count} failed={failed_count}", flush=True)

    # Phase 2: V-JEPA2 feature extraction for every unique clip (in_clip and
    # out_clip can repeat across pairs -- a step is both "future" for pair i
    # and "past" for pair i+1 -- so dedupe by path first).
    from cache_train.rebuild_causal_cache import load_vjepa_encoder, preprocess_vjepa_frames, encode_vjepa_clip
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
            from cache_train.rebuild_causal_cache import uniform_indices
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

    # Only keep items whose BOTH clips got real V-JEPA features -- a clip
    # that failed encoding would otherwise look present (mp4 exists) but
    # fail has_features() silently at eval time.
    encoded_set = {p.stem for p in vjepa_cache_root.glob("*.npz")} if vjepa_cache_root.exists() else set()

    def clip_has_feats(clip_path_str):
        safe_name = os.path.relpath(clip_path_str, str(clips_root)).replace("/", "__").replace(".mp4", "")
        return safe_name in encoded_set

    final_items = []
    for p in items:
        if not (clip_has_feats(p["in_clip"]) and clip_has_feats(p["out_clip"])):
            continue
        pid = f"crosstask_{p['task_id']}_{p['video_id']}_{p['pair_idx']}"
        final_items.append({
            "pid": pid,
            "in_clip": p["in_clip"],
            "out_clip": p["out_clip"],
            "question": (
                f"I'm in the middle of {p['task_name'].lower()}. "
                f"Based on the video I just sent, what should I do next?"
            ),
            "correct_caption": p["correct_caption"],
            "distractor_caption": p["distractor_caption"],
            "difficulty": p["task_name"],
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
