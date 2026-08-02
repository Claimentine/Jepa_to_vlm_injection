"""
Extract Qwen3-VL guidance features (vlm_old/vlm_new) for the VANS demo
pairs' "_in" clips only -- this is the "past/context" video the VLM branch
summarizes to condition the JEPA predictor, matching ThinkJEPA's design.

Just stages symlinks to the *_in.mp4 files (so qwen3_cache_extractor.py's
recursive mp4 walk doesn't also pick up the *_out.mp4 files we don't need
guidance for) and shells out to the existing, unmodified extractor script --
zero new extraction logic, 100% reuse of the already-tested ThinkJEPA code.
"""
import argparse
import glob
import os
import subprocess
import sys

VANS_ROOT = os.environ.get("VANS_ROOT", "/projects/bhay/william/ruixin/vans_world_model")
DEMO_DIR = os.path.join(VANS_ROOT, "hf_data/demo_sample/VANS_DATA_EXAMPLES")
STAGE_DIR = os.path.join(VANS_ROOT, "vlm_stage_in_only")
OUT_DIR = os.path.join(VANS_ROOT, "vlm_guidance_cache")
EXTRACTOR = os.path.join(
    os.environ.get("THINKJEPA_ROOT", "/projects/bhay/william/ruixin/ThinkJEPA"),
    "cache_train/qwen3_cache_extractor.py",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="stage only first N pairs (smoke test)")
    ap.add_argument("--max_frames", type=int, default=32)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 12, 16, 20, 24, 27])
    args = ap.parse_args()

    os.makedirs(STAGE_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    in_paths = sorted(glob.glob(os.path.join(DEMO_DIR, "*_in.mp4")))
    if args.limit:
        in_paths = in_paths[: args.limit]

    for p in in_paths:
        pid = os.path.basename(p)[: -len("_in.mp4")]
        link_path = os.path.join(STAGE_DIR, f"{pid}.mp4")
        if not os.path.exists(link_path):
            os.symlink(os.path.abspath(p), link_path)
    print(f"[INFO] staged {len(in_paths)} clips -> {STAGE_DIR}")

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
