"""
Extract V-JEPA2 (ThinkJEPA's dense JEPA encoder) features for VANS demo
(in, out) clip pairs.

Reuses cache_train.thinker_train's load_dense_jepa_encoder /
encode_dense_jepa_video / build_dense_jepa_video_transform verbatim (same
checkpoint, same preprocessing) so features are directly comparable to what
ThinkJEPA computes for EgoDex. That import path required two vjepa2 modules
(`src/datasets/utils/video/{transforms,volume_transforms}.py`) that are
missing from this repo's trimmed vjepa2 subtree (dead code path in the
public release -- all real runs used --skip_vjepa + precomputed cache, so
this was never exercised); they were copied in from the untrimmed
Thinker_World/vjepa2 checkout (same upstream V-JEPA2 utility code, no
EgoDex-specific logic) before this script could work.

Each demo pair -> one npz: in_feats (64,128,1024), out_feats (64,128,1024).
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(os.environ.get("THINKJEPA_ROOT", "/projects/bhay/william/ruixin/ThinkJEPA"))
for _p in (REPO_ROOT, REPO_ROOT / "cache_train", REPO_ROOT / "vjepa2", REPO_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from cache_train.thinker_train import (
    load_dense_jepa_encoder,
    encode_dense_jepa_video,
    build_dense_jepa_video_transform,
)

VITL_PT = os.environ.get("VJEPA2_CKPT", "/work/nvme/bdqf/william/charles/pretrain/vjepa2/vitl.pt")
NUM_FRAMES = 64
IMG_SIZE = 256


def read_clip_frames(mp4_path, n_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(mp4_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"no frames read from {mp4_path}")
    idx_want = np.linspace(0, max(total - 1, 0), n_frames).astype(int)
    frames = []
    want_set = set(idx_want.tolist())
    cur = 0
    got = {}
    while len(got) < len(want_set):
        ok, frame = cap.read()
        if not ok:
            break
        if cur in want_set:
            got[cur] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cur += 1
    cap.release()
    if not got:
        raise ValueError(f"could not decode any wanted frames from {mp4_path}")
    last = next(iter(got.values()))
    for i in idx_want:
        if i not in got:
            got[i] = last
        last = got[i]
    frames = [got[i] for i in idx_want]
    return frames  # list of (H,W,3) uint8 RGB numpy arrays, len == n_frames


def clip_to_model_input(frames, transform):
    clip_tensor = transform(frames)  # (C, T, H, W), normalized
    video = clip_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # (1, T, C, H, W)
    return video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_dir", default=os.path.join(os.environ.get("VANS_ROOT", "/projects/bhay/william/ruixin/vans_world_model"), "hf_data/demo_sample/VANS_DATA_EXAMPLES"))
    ap.add_argument("--out_dir", default=os.path.join(os.environ.get("VANS_ROOT", "/projects/bhay/william/ruixin/vans_world_model"), "vjepa_cache"))
    ap.add_argument("--limit", type=int, default=None, help="process only first N pairs (smoke test)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    in_paths = sorted(glob.glob(os.path.join(args.demo_dir, "*_in.mp4")))
    pair_ids = [os.path.basename(p)[: -len("_in.mp4")] for p in in_paths]
    if args.limit:
        pair_ids = pair_ids[: args.limit]

    print(f"[INFO] loading V-JEPA2 encoder from {VITL_PT} ...")
    model_pt, transform = load_dense_jepa_encoder(pt_model_path=VITL_PT)

    n_ok, n_err = 0, 0
    for pid in pair_ids:
        out_path = os.path.join(args.out_dir, f"{pid}.npz")
        if os.path.exists(out_path):
            n_ok += 1
            continue
        try:
            in_mp4 = os.path.join(args.demo_dir, f"{pid}_in.mp4")
            out_mp4 = os.path.join(args.demo_dir, f"{pid}_out.mp4")

            in_frames = read_clip_frames(in_mp4)
            out_frames = read_clip_frames(out_mp4)

            in_video = clip_to_model_input(in_frames, transform).cuda()
            out_video = clip_to_model_input(out_frames, transform).cuda()

            in_feats = encode_dense_jepa_video(in_video, model_pt)  # (1,T,P,D)
            out_feats = encode_dense_jepa_video(out_video, model_pt)

            np.savez(
                out_path,
                in_feats=in_feats[0].half().cpu().numpy(),
                out_feats=out_feats[0].half().cpu().numpy(),
            )
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f"[ERR] {pid}: {type(e).__name__}: {e}")

        if (n_ok + n_err) % 25 == 0:
            print(f"[{n_ok + n_err}/{len(pair_ids)}] ok={n_ok} err={n_err}")

    print(f"[DONE] ok={n_ok} err={n_err} -> {args.out_dir}")


if __name__ == "__main__":
    main()
