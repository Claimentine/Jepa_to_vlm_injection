"""
Full-scale V-JEPA2 feature extraction for qa_split_full.json's 18,147
unique clips (both in_clip and out_clip sides -- a clip can be an "in" for
one QA row and an "out" for another, so we extract once per unique path).

Same encoder/preprocessing as extract_vjepa_features.py (demo version):
load_dense_jepa_encoder / encode_dense_jepa_video, 64 frames, 256x256.
Output: vjepa_cache_full/{safe_name}.npz with a single `feats` (64,128,1024)
array (unlike the demo version's paired in_feats/out_feats -- here in/out
are independent unique clips, not always co-located pairs).
"""
import argparse
import json
import os
import sys
import zipfile
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
)
from cache_train.video_observation_adapter import VideoObservationAdapter

BASE = os.environ.get("VANS_ROOT", "/projects/bhay/william/ruixin/vans_world_model")
WORK_BASE = os.environ.get("VANS_WORK_ROOT", "/work/nvme/bdqf/yli8/vans_raw_data")  # relocated off the near-full /projects/bhay allocation
VITL_PT = os.environ.get("VJEPA2_CKPT", "/work/nvme/bdqf/william/charles/pretrain/vjepa2/vitl.pt")
NUM_FRAMES = 64
OUT_DIR = os.path.join(WORK_BASE, "vjepa_cache_full")


def safe_name(clip_path):
    rel = os.path.relpath(clip_path, os.path.join(WORK_BASE, "clips"))
    return rel.replace("/", "__").replace(".mp4", "")


def read_clip_frames(mp4_path, n_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(mp4_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"no frames read from {mp4_path}")
    idx_want = np.linspace(0, max(total - 1, 0), n_frames).astype(int)
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
    frames = []
    for i in idx_want:
        if i not in got:
            got[i] = last
        last = got[i]
        frames.append(got[i])
    return frames


def is_valid_npz(path):
    # A prior run wrote directly to out_path (see the save call below, now
    # fixed) and got killed mid-write on this preemptible cluster often
    # enough to leave ~41% of the cache as 0-byte or truncated-zip stubs.
    # Those still pass a bare os.path.exists() check, so the "already
    # extracted" skip below silently perpetuated them forever. Actually open
    # the archive and touch the array data before trusting a file as done.
    try:
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:
                return False
            with np.load(path) as d:
                d["feats"]
        return True
    except Exception:
        return False


def clip_to_model_input(frames, adapter):
    # load_dense_jepa_encoder's second return value is a transform *spec*
    # (resize/crop/normalize params), not a callable -- VideoObservationAdapter
    # is the nn.Module that actually applies it (see thinker_train.py's own
    # `fast_tx = VideoObservationAdapter(pt_video_transform); img = fast_tx(img)`
    # usage). Feed it (T,H,W,C) uint8; it returns (T,C,H,W) normalized.
    frames_t = torch.from_numpy(np.stack(frames))
    clip_tensor = adapter(frames_t)
    return clip_tensor.unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    split = json.load(open(os.path.join(BASE, "raw_data/qa_split_full.json")))
    clips = set()
    for part in split.values():
        for item in part:
            clips.add(item["in_clip"])
            clips.add(item["out_clip"])
    clips = sorted(clips)
    if args.limit:
        clips = clips[: args.limit]

    print(f"[INFO] loading V-JEPA2 encoder from {VITL_PT} ...")
    model_pt, transform = load_dense_jepa_encoder(pt_model_path=VITL_PT)
    adapter = VideoObservationAdapter(transform, antialias=True)

    n_ok, n_err, n_skip = 0, 0, 0
    for clip_path in clips:
        name = safe_name(clip_path)
        out_path = os.path.join(OUT_DIR, f"{name}.npz")
        if os.path.exists(out_path) and is_valid_npz(out_path):
            n_skip += 1
            continue
        try:
            frames = read_clip_frames(clip_path)
            video = clip_to_model_input(frames, adapter).cuda()
            feats = encode_dense_jepa_video(video, model_pt)
            # Write to a same-directory temp file and atomically rename into
            # place, so a pod eviction/OOM kill mid-write can never leave a
            # truncated .npz sitting at out_path -- either the full file
            # lands, or nothing does and the next run's is_valid_npz() check
            # above naturally redoes it.
            tmp_path = out_path + f".tmp{os.getpid()}"
            np.savez(tmp_path, feats=feats[0].half().cpu().numpy())
            os.replace(tmp_path, out_path)
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f"[ERR] {clip_path}: {type(e).__name__}: {e}")
        if (n_ok + n_err + n_skip) % 200 == 0:
            print(f"[{n_ok + n_err + n_skip}/{len(clips)}] ok={n_ok} err={n_err} skip={n_skip}", flush=True)

    print(f"[DONE] ok={n_ok} err={n_err} skip={n_skip} total={len(clips)}")


if __name__ == "__main__":
    main()
