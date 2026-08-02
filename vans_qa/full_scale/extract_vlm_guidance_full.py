"""
Full-scale VLM guidance extraction: stages symlinks for all 12,032 unique
in_clips referenced by qa_split_full.json, keyed by the same safe_name
scheme as extract_vjepa_features_full.py, then shells out to the existing
qwen3_cache_extractor.py (unmodified) -- same approach as the demo version.
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
    "cache_train/qwen3_cache_extractor.py",
)


def safe_name(clip_path):
    rel = os.path.relpath(clip_path, os.path.join(WORK_BASE, "clips"))
    return rel.replace("/", "__").replace(".mp4", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_frames", type=int, default=32)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 12, 16, 20, 24, 27])
    args = ap.parse_args()

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
    for clip_path in in_clips:
        name = safe_name(clip_path)
        link_path = os.path.join(STAGE_DIR, f"{name}.mp4")
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
    print(f"[INFO] staged {len(in_clips)} clips ({n_new} new symlinks) -> {STAGE_DIR}")

    cmd = [
        sys.executable, EXTRACTOR,
        "--file_dir", STAGE_DIR,
        "--output_dir", OUT_DIR,
        "--pretrained", "Qwen/Qwen3-VL-2B-Thinking",
        "--layers", *[str(x) for x in args.layers],
        "--max_frames", str(args.max_frames),
        "--max_new_token_num", "16",
        "--save_dtype", "fp16",
        "--res", "256",
        "--prompt", "Describe this video.",
        "--force_video_backend", "decord",
    ]
    print("[INFO] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
