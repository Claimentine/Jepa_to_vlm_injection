"""Builds Ego4D shortaxis pairs DIRECTLY in the reverse-direction joint-
training cache format (pid/vlm_old/vlm_new/vjepa_input_feats/
vjepa_target_feats, see extract_vlm_guidance_paired.py's own
is_valid_output()) -- the format train_joint_coupled_normalized.py's
--cache_dir/--rev_qa_split actually consume, NOT job-95's raw-pair format
(extract_ego4d_longaxis.py), which has no vlm_old/vlm_new at all and so
can't be fed into joint training directly.

Why a new script instead of reusing job-95's already-completed pairs:
extract_vlm_guidance_paired.py's Qwen teacher-forcing call
(infer_qwen_teacher_forced) needs RAW uint8 decoded frames, but job-95
only ever saved the VJEPA-preprocessed version (resized/cropped/float-
normalized via preprocess_vjepa_frames) -- the raw frames were discarded
in-process, never written to its npz. So there is no way to recover them
from job-95's existing output; the video has to be (re-)downloaded and
(re-)decoded regardless. Given that, this script downloads once and does
BOTH the VJEPA encoding AND the Qwen guidance captioning in the same pass
-- unlike running job-95 then a separate guidance-pairing pass over its
output, which would need a second download per video for no benefit.

Pairing/question/caption convention: future_label (the next step's own
step_description/step_category, or the next narration's text) is used as
correct_caption for Qwen's teacher-forced generation -- the same real,
dataset-authored supervision extract_ego4d_qa.py uses for its
correct_caption field, just not packaged as a full QA item (no distractor,
no multi-choice framing) since the reverse/joint training side only needs
the guidance text, not a QA answer.

Output: one {pid}.npz per pair in --cache_dir (directly usable as
--cache_dir for train_joint_coupled_normalized.py), plus one
--split_out JSON with {"train": [{"pid": ...}, ...], "val": [...],
"test": [...]} (the minimal membership format load_qa_split_membership
needs) usable as --rev_qa_split.
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
WORK_BASE = os.environ.get("VANS_WORK_ROOT", "/data/vans_work_ego4d_reverse_guided")
THINKJEPA_ROOT = os.environ.get("THINKJEPA_ROOT", "/home/jovyan/ThinkJEPA")
if THINKJEPA_ROOT not in sys.path:
    sys.path.insert(0, THINKJEPA_ROOT)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from extract_ego4d_longaxis import (  # noqa: E402
    load_goalstep_annotations, load_narration_annotations,
    select_shortaxis_pairs_goalstep, select_shortaxis_pairs_narration,
    download_video, write_aws_credentials, extract_window_frames,
    split_video_keys,
)
from extract_vlm_guidance_paired import infer_qwen_teacher_forced  # noqa: E402


def build_question(goal_category):
    activity = (goal_category or "an everyday activity").lower()
    return (f"This is first-person video of someone doing {activity}. "
            f"Based on the video I just sent, what should I do next?")


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
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotation_source", choices=["goalstep", "narration"], default="goalstep")
    ap.add_argument("--goalstep_json", default=None)
    ap.add_argument("--narration_json", default=None)
    ap.add_argument("--min_gap_seconds", type=float, default=0.0)
    ap.add_argument("--max_gap_seconds", type=float, default=60.0)
    ap.add_argument("--pairs_per_video", type=int, default=3)
    ap.add_argument("--window_seconds", type=float, default=8.0, help="narration mode only")
    ap.add_argument("--min_steps", type=int, default=2, help="narration mode only")
    ap.add_argument("--limit", type=int, default=None, help="cap candidate pairs")
    ap.add_argument("--eval_frac", type=float, default=0.1, help="video-level held-out fraction -> val split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--download_only", action="store_true",
                     help="download each pair's video and stop -- no Qwen/VJEPA load, no GPU touched "
                          "or requested at all (unlike lazy loading, a pod that merely holds a GPU "
                          "reservation it never uses still trips NRP's idle-GPU policy, so this is for "
                          "running as a genuinely GPU-less pod). Rerun without this flag tomorrow to "
                          "do the extraction -- download_video() no-ops for videos already present.")
    ap.add_argument("--download_concurrency", type=int, default=16,
                     help="--download_only mode only: concurrent download_video() calls (thread pool "
                          "-- pure I/O, no GIL contention). Serial downloads at Ego4D's ~50% "
                          "per-attempt failure rate (confirmed in practice, job-109's first attempt) "
                          "are on the order of weeks for a several-thousand-video target; this is the "
                          "actual fix, not a tuning knob to leave at 1.")
    ap.add_argument("--video_cache_dir", default=os.path.join(WORK_BASE, "raw_videos"))
    ap.add_argument("--cache_dir", default=os.path.join(WORK_BASE, "cache"))
    ap.add_argument("--split_out", default=os.path.join(BASE, "raw_data/qa_split_ego4d_reverse_guided.json"))
    ap.add_argument("--pretrained", default="Qwen/Qwen3-VL-2B-Thinking")
    ap.add_argument("--qwen_revision", default="main")
    ap.add_argument("--qwen_checkpoint_sha", default="")
    ap.add_argument("--vjepa_checkpoint", default=os.environ.get("VJEPA2_CKPT", "/data/checkpoints/vjepa2/vitl.pt"))
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 12, 16, 20, 24, 27])
    ap.add_argument("--qwen_res", type=int, default=256)
    ap.add_argument("--save_dtype", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()

    write_aws_credentials()

    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    from cache_train.rebuild_causal_cache import (
        NUM_PAST_FRAMES, NUM_TARGET_FRAMES, load_qwen, load_vjepa_encoder,
        preprocess_vjepa_frames, encode_vjepa_clip, tensor_to_numpy, sha256_file,
    )

    if args.annotation_source == "goalstep":
        videos = load_goalstep_annotations(args.goalstep_json)
        all_video_keys = [v["video_uid"] for v in videos]
        pairs = select_shortaxis_pairs_goalstep(videos, args.min_gap_seconds, args.max_gap_seconds,
                                                 args.pairs_per_video, args.limit)
    else:
        narration_data = load_narration_annotations(args.narration_json)
        all_video_keys = list(narration_data.keys())
        pairs = select_shortaxis_pairs_narration(narration_data, args.min_gap_seconds, args.max_gap_seconds,
                                                  args.pairs_per_video, args.window_seconds, args.min_steps,
                                                  args.limit)
    print(f"[INFO] {len(pairs)} candidate {args.annotation_source} shortaxis pairs across "
          f"{len({p['video_uid'] for p in pairs})} videos", flush=True)

    train_keys, val_keys = split_video_keys(all_video_keys, args.eval_frac, args.seed)
    video_part = {}
    for k in train_keys:
        video_part[k] = "train"
    for k in val_keys:
        video_part[k] = "val"

    video_cache_root = Path(args.video_cache_dir)
    video_cache_root.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    if args.download_only:
        # Concurrent, not the serial per-pair loop below: downloading is
        # pure network I/O (no GPU, no shared model state to worry about),
        # and at Ego4D's observed ~50% per-attempt failure rate (up to 600s
        # each), serial downloads are catastrophically slow -- confirmed in
        # practice, job-109's first overnight attempt did only ~76 unique
        # videos in 11 hours (serial), on pace for ~2 WEEKS to reach the
        # full narration+goalstep target. A thread pool is fine here (no
        # GIL-bound work, each download_video() call just blocks on a
        # subprocess) and lets successful downloads finish without being
        # stuck behind a slow/timing-out one ahead of them in a queue.
        import concurrent.futures
        unique_video_uids = sorted({p["video_uid"] for p in pairs})
        print(f"[INFO] download_only: {len(unique_video_uids)} unique videos needed "
              f"({args.download_concurrency} concurrent)", flush=True)
        n_dl_ok = n_dl_fail = n_dl_skip = 0

        def _dl_one(video_uid):
            video_path = video_cache_root / "v2" / "full_scale" / f"{video_uid}.mp4"
            if video_path.exists():
                return video_uid, "skip", None
            try:
                download_video(video_uid, video_cache_root)
                return video_uid, "ok", None
            except Exception as e:
                return video_uid, "fail", f"{type(e).__name__}: {e}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.download_concurrency) as pool:
            for i, (video_uid, status, err) in enumerate(pool.map(_dl_one, unique_video_uids), start=1):
                if status == "skip":
                    n_dl_skip += 1
                elif status == "ok":
                    n_dl_ok += 1
                else:
                    n_dl_fail += 1
                    print(f"[FAIL download {i}/{len(unique_video_uids)}] {video_uid}: {err}", flush=True)
                if i % 20 == 0:
                    print(f"[{i}/{len(unique_video_uids)} videos] ok={n_dl_ok} skip={n_dl_skip} "
                          f"fail={n_dl_fail}", flush=True)
        print(f"[DONE] download_only -- ok={n_dl_ok} skip={n_dl_skip} fail={n_dl_fail} "
              f"video_cache_dir={video_cache_root} -- rerun without --download_only tomorrow to do "
              f"the VJEPA/Qwen extraction", flush=True)
        return 1 if (n_dl_fail and not n_dl_ok and not n_dl_skip) else 0

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Loaded lazily, on first pair whose download actually succeeds -- not
    # here. Ego4D downloads have a ~50% failure rate at up to 600s/attempt
    # (confirmed this session, job-104); loading both models eagerly would
    # hold the GPU reserved-but-idle through however many failed downloads
    # happen before the first success, exactly the pattern that tripped
    # job-95's own NRP idle-GPU warning (fixed there the same way, see
    # extract_ego4d_longaxis.py's get_model()).
    vjepa_checkpoint_sha = sha256_file(Path(args.vjepa_checkpoint))
    _model_holder = {}

    def get_models():
        if "qwen" not in _model_holder:
            print(f"[INFO] loading {args.pretrained} + hooks on layers {args.layers} ...", flush=True)
            _model_holder["qwen"] = load_qwen(args, device)
            print(f"[INFO] loading V-JEPA2 checkpoint {args.vjepa_checkpoint} ...", flush=True)
            _model_holder["vjepa"] = load_vjepa_encoder(Path(args.vjepa_checkpoint), device)
        return _model_holder["qwen"] + (_model_holder["vjepa"],)

    n = len(pairs)
    saved_count = skipped_count = failed_count = 0
    seen_exc_types = set()
    split_membership = {"train": [], "val": [], "test": []}
    with torch.no_grad():
        for i, pair in enumerate(pairs, start=1):
            video_uid = pair["video_uid"]
            pair_idx = pair["pair_idx"]
            is_last_pair_for_video = (i == n) or (pairs[i]["video_uid"] != video_uid)
            pid = f"ego4d_shortaxis_{args.annotation_source}_{video_uid}_{pair_idx}"
            output_path = cache_root / f"{pid}.npz"
            part = video_part.get(video_uid, "train")
            if is_valid_output(output_path):
                skipped_count += 1
                split_membership[part].append({"pid": pid})
                print(f"[SKIP {i}/{n}] valid {output_path}", flush=True)
                continue
            video_path = video_cache_root / "v2" / "full_scale" / f"{video_uid}.mp4"
            try:
                if not video_path.exists():
                    download_video(video_uid, video_cache_root)
                if not video_path.exists():
                    raise RuntimeError(f"ego4d CLI reported success but {video_path} is missing")
                qwen_model, qwen_processor, qwen_saved, qwen_sha, vjepa_model = get_models()

                past_frames, past_times = extract_window_frames(
                    video_path, pair["past_segment"][0], pair["past_segment"][1], NUM_PAST_FRAMES,
                )
                target_frames, target_times = extract_window_frames(
                    video_path, pair["future_segment"][0], pair["future_segment"][1], NUM_TARGET_FRAMES,
                )

                question = build_question(pair["goal_category"])
                correct_caption = pair["future_label"] or "the next step of this activity"
                # infer_qwen_teacher_forced needs a source fps for sample_fps
                # placement -- extract_window_frames doesn't return one (it
                # already resamples to a fixed frame count/timestamps), so
                # derive an effective rate from the actual frame timestamps
                # it DID return, same fallback extract_vlm_guidance_paired.py's
                # own decode_clip_with_timeout uses when a reader can't report fps.
                span = max(1e-6, float(past_times[-1] - past_times[0]))
                in_fps = max(1.0, (len(past_times) - 1) / span)
                qwen_result = infer_qwen_teacher_forced(
                    args, qwen_model, qwen_processor, qwen_saved, device,
                    past_frames, past_times, in_fps, question, correct_caption,
                )

                past_clip, _ = preprocess_vjepa_frames(past_frames)
                target_clip, _ = preprocess_vjepa_frames(target_frames)
                vjepa_input_feats = encode_vjepa_clip(vjepa_model, past_clip, device)
                vjepa_target_feats = encode_vjepa_clip(vjepa_model, target_clip, device)

                def to_dtype(value):
                    array = np.asarray(value)
                    return array.astype(np.float16 if args.save_dtype == "fp16" else np.float32, copy=False)

                payload = {
                    "schema_name": np.asarray("ego4d_reverse_guided_v1"),
                    "pid": np.asarray(pid),
                    "annotation_source": np.asarray(args.annotation_source),
                    "video_uid": np.asarray(video_uid),
                    "pair_idx": np.asarray(pair_idx, dtype=np.int32),
                    "goal_category": np.asarray(pair["goal_category"] or ""),
                    "question": np.asarray(question),
                    "correct_caption": np.asarray(correct_caption),
                    "past_frame_times_seconds": past_times.astype(np.float64),
                    "target_frame_times_seconds": target_times.astype(np.float64),
                    "vjepa_input_feats": to_dtype(vjepa_input_feats),
                    "vjepa_target_feats": to_dtype(vjepa_target_feats),
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
                split_membership[part].append({"pid": pid})
                print(f"[SAVED {i}/{n}] {output_path}", flush=True)
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
            if is_last_pair_for_video:
                try:
                    video_path.unlink()
                except FileNotFoundError:
                    pass
            if i % 20 == 0:
                print(f"[{i}/{n}] saved={saved_count} skipped={skipped_count} failed={failed_count}",
                      flush=True)

    out_path = Path(args.split_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(split_membership, f, indent=2)
    print(f"[DONE] saved={saved_count} skipped_valid={skipped_count} failed={failed_count} "
          f"cache_dir={cache_root} split_out={out_path} "
          f"(train={len(split_membership['train'])} val={len(split_membership['val'])})", flush=True)
    return 1 if (failed_count and not saved_count and not skipped_count) else 0


if __name__ == "__main__":
    sys.exit(main())
