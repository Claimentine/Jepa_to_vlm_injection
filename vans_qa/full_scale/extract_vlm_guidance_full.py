"""
Full-scale VLM guidance extraction: stages symlinks for all 12,032 unique
in_clips referenced by qa_split_full.json, keyed by the same safe_name
scheme as extract_vjepa_features_full.py, then shells out to
cache_train/rebuild_causal_cache.py (unmodified) -- same approach as the demo
version.

Note: what this project's own wrapper scripts (and an earlier version of
this docstring) called "qwen3_cache_extractor.py" never existed in the
public ThinkJEPA release -- that was just a wrong filename baked into these
wrappers. The real, complete, unmodified-since-the-original-commit
implementation is cache_train/rebuild_causal_cache.py: it does both the
Qwen3-VL guidance extraction (vlm_old/vlm_new) *and* its own V-JEPA2
past/target encoding in one pass, per its own causal-cache schema (distinct
from this project's separate, single-clip V-JEPA cache used for the forward
QA-injection direction). Confirmed 2026-08-31 by reading the live checkout.
"""
import argparse
import json
import os
import subprocess
import sys

BASE = os.environ.get("VANS_ROOT", "/projects/bhay/william/ruixin/vans_world_model")
WORK_BASE = os.environ.get("VANS_WORK_ROOT", "/work/nvme/bdqf/yli8/vans_raw_data")  # relocated off the near-full /projects/bhay allocation
STAGE_DIR = os.path.join(BASE, "raw_data/vlm_stage_full")  # just symlinks, tiny, fine to leave on /projects
OUT_DIR = os.path.join(WORK_BASE, "vlm_guidance_cache_full")
EXTRACTOR = os.path.join(
    os.environ.get("THINKJEPA_ROOT", "/projects/bhay/william/ruixin/ThinkJEPA"),
    "cache_train/rebuild_causal_cache.py",
)
# Same checkpoint job-03 already downloads for the forward-direction V-JEPA2
# extraction (facebook/vjepa2-vitl-fpc64-256, original/model.pth) -- this
# script's VJEPA_IMAGE_SIZE=256/VJEPA_PATCH_SIZE=16/VJEPA_EMBED_DIM=1024
# constants match it exactly, so there's no separate checkpoint to source.
VJEPA_CHECKPOINT = os.environ.get("VJEPA2_CKPT", "/data/checkpoints/vjepa2/vitl.pt")

# rebuild_causal_cache.py hangs deterministically and indefinitely partway
# through processing these specific clips, confirmed via a watchdog-
# restarted rerun (see job-10-extract-vlm-guidance-full.yaml) landing on the
# exact same clip every attempt (17-20 consecutive fresh-process attempts,
# each with models freshly reloaded) -- despite the file decoding and
# frame-selecting fine when reproduced in isolation outside the main loop.
# The actual cause inside rebuild_causal_cache.py's per-video loop is
# unknown, but retrying clearly does not help for these, so skip them
# outright rather than burn another ~20 doomed retry cycles (each costing a
# ~1h model-reload). Add to this set as new persistent hangs are found.
EXCLUDE_CLIPS = {
    "-9oqHVK5-5c/598.mp4",  # confirmed 2026-09-02: hangs every attempt, 0% GPU util, decodes fine standalone
}


def safe_name(clip_path):
    rel = os.path.relpath(clip_path, os.path.join(WORK_BASE, "clips"))
    return rel.replace("/", "__").replace(".mp4", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 12, 16, 20, 24, 27])
    ap.add_argument("--vjepa_checkpoint", default=VJEPA_CHECKPOINT)
    ap.add_argument("--max_videos", type=int, default=0, help="forwarded to rebuild_causal_cache.py; 0=all assigned to this rank")
    ap.add_argument("--self_test", action="store_true", help="run rebuild_causal_cache.py's own --self_test and exit (no staging, no model load)")
    args = ap.parse_args()

    if args.self_test:
        cmd = [sys.executable, EXTRACTOR, "--file_dir", ".", "--output_dir", ".",
               "--vjepa_checkpoint", args.vjepa_checkpoint, "--self_test"]
        print("[INFO] running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return

    os.makedirs(STAGE_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    split = json.load(open(os.path.join(BASE, "raw_data/qa_split_full.json")))
    in_clips = set()
    for part in split.values():
        for item in part:
            in_clips.add(item["in_clip"])
    in_clips = sorted(in_clips)
    if args.limit:
        in_clips = in_clips[: args.limit]

    n_new = 0
    n_excluded = 0
    for clip_path in in_clips:
        rel = os.path.relpath(clip_path, os.path.join(WORK_BASE, "clips"))
        name = safe_name(clip_path)
        link_path = os.path.join(STAGE_DIR, f"{name}.mp4")
        if rel in EXCLUDE_CLIPS:
            n_excluded += 1
            # rebuild_causal_cache.py discovers videos by walking STAGE_DIR
            # directly, not through this script's in-memory clip list -- a
            # symlink staged by an earlier run would still be picked up even
            # though we're excluding it here now, so remove it too.
            if os.path.lexists(link_path):
                os.remove(link_path)
            continue
        # os.path.exists() follows symlinks and returns False for a *broken*
        # one (e.g. stale links left over from before the clips/ relocation
        # to /work), which then made os.symlink() below crash with
        # FileExistsError since the link itself was still there. lexists()
        # checks the link entry itself, so stale links get replaced cleanly.
        if os.path.lexists(link_path):
            if not os.path.exists(link_path):  # stale/broken -> replace it
                os.remove(link_path)
            else:
                continue
        os.symlink(os.path.abspath(clip_path), link_path)
        n_new += 1
    print(f"[INFO] staged {len(in_clips) - n_excluded} clips ({n_new} new symlinks, "
          f"{n_excluded} excluded as known-bad) -> {STAGE_DIR}")

    cmd = [
        sys.executable, EXTRACTOR,
        "--file_dir", STAGE_DIR,
        "--output_dir", OUT_DIR,
        "--pretrained", "Qwen/Qwen3-VL-2B-Thinking",
        "--vjepa_checkpoint", args.vjepa_checkpoint,
        "--layers", *[str(x) for x in args.layers],
        # max_frames/res/force_video_backend dropped: rebuild_causal_cache.py
        # hardcodes NUM_PAST_FRAMES=32 (matches what we were passing) and
        # decord (its only supported decode backend) -- neither is a CLI flag.
        "--max_new_token_num", "16",
        "--save_dtype", "fp16",
        "--qwen_res", "256",
        "--prompt", "Describe this video.",
    ]
    if args.max_videos:
        cmd += ["--max_videos", str(args.max_videos)]
    print("[INFO] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
