"""Builds one EgoDex-hdf5-shaped supervision sidecar per EgoExo4D take.

Why this exists: the reverse-eval probe (probe_egodex_reverse_adapter.py)
loads pose supervision through NpzCacheDataset._repair_from_egodex_supervision,
which ALWAYS resolves a real .hdf5 file on disk (via source_video_relpath,
with the .mp4 extension swapped for .hdf5 under --supervision_root) and reads
it with load_supervision_from_hdf5 -- confirmed by reading
egodex/trajectory_dataset.py directly. There is no code path that reads pose
data out of the feature-cache npz itself: NpzCacheDataset._load_cache_sample
unconditionally rejects any cache npz containing an "egoexo_"-prefixed key
(or any of CAUSAL_FEATURE_FORBIDDEN_KEYS, which includes xyz_cam/cam_ext/etc.
too) before any of that code can run, so ThinkJEPA's own camera_mode="egoexo"
ingestion path is dead/unreachable and was never exercised. Writing a real
.hdf5 that load_supervision_from_hdf5 can read is the only path that is
actually live and production-tested (this exact mechanism already serves
EgoDex's own real probe runs, e.g. job-76).

load_supervision_from_hdf5's required layout (read directly from
egodex/trajectory_dataset.py, lines ~702-780):
    /transforms/camera          (T,4,4) float  -- world_from_cam per frame
    /transforms/<tf_name>       (T,4,4) float  -- per query joint, world frame,
                                 one dataset per name in NPZ_TARGET_QUERY_TFS
    /camera/intrinsic           (3,3) float (or (1,3,3)/(T,3,3))
    /confidences/<tf_name>      (T,) float, optional (defaults to 1.0 if the
                                 whole group is absent)
    attrs["llm_description"]    str (language instruction)

EgoExo4D only gives xyz positions (no per-joint orientation) and only at a
sparse set of annotated frames (not every raw frame), so:
  - every joint transform is identity-rotation + translation=xyz (matches
    ThinkJEPA's own _build_identity_transforms_from_xyz, which is the right
    thing to do when there's no orientation to carry).
  - both camera pose and joint xyz are nearest-frame-held across every raw
    frame 0..T-1 (T = the REAL decoded frame count of the actual mp4, must
    match exactly what rebuild_causal_cache.py's own video decode finds, or
    _repair_from_egodex_supervision's source-frame-count cross-check fails).
  - world coordinates are EgoExo4D's annotation3D values AS-IS (already a
    consistent, multi-view-triangulated per-take scene frame); xyz_cam is
    then the CORRECT inverse projection (apply cam_from_world to the world
    point) -- this is the opposite of what ThinkJEPA's own dead
    _build_camera_xyz_from_egoexo_pose/_repair_from_egoexo_extraction code
    does (it copies annotation3D straight into something it calls "xyz_cam"
    with no transform, then projects it AGAIN through cam_ext as if it were
    camera-local -- a double/wrong transform that was never caught because
    that path never runs). We don't reuse that function; load_supervision_
    from_hdf5 does its own (correct) cam_ext^-1 @ tfs internally, so we only
    need to hand it correct WORLD-frame joint transforms and a correct
    world_from_cam camera track.
"""
import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np

THINKJEPA_ROOT = os.environ.get("THINKJEPA_ROOT", "/home/jovyan/ThinkJEPA")
if THINKJEPA_ROOT not in sys.path:
    sys.path.insert(0, THINKJEPA_ROOT)

from egodex.trajectory_dataset import NpzCacheDataset, NPZ_TARGET_QUERY_TFS  # noqa: E402

normalize_name = NpzCacheDataset._normalize_hand_joint_name
map_candidates = NpzCacheDataset._map_query_joint_to_egoexo_candidates


def _nearest_fill(values: np.ndarray, src_frames: np.ndarray, num_frames: int) -> np.ndarray:
    """values: (S, ...) at src_frames (S,) -> (num_frames, ...) nearest-held.

    Frames before the first / after the last annotated frame hold the
    nearest edge value (standard nearest-neighbour extrapolation), matching
    the spirit of ThinkJEPA's own _match_nearest_frame_ids.
    """
    if values.shape[0] == 0:
        return None
    order = np.argsort(src_frames)
    src_sorted = src_frames[order]
    values_sorted = values[order]
    all_frames = np.arange(num_frames, dtype=np.float64)
    right = np.searchsorted(src_sorted, all_frames, side="left")
    right = np.clip(right, 0, src_sorted.shape[0] - 1)
    left = np.clip(right - 1, 0, src_sorted.shape[0] - 1)
    choose_right = np.abs(src_sorted[right] - all_frames) <= np.abs(src_sorted[left] - all_frames)
    nn = np.where(choose_right, right, left)
    return values_sorted[nn]


def _load_pose_sources(hand_path, body_path):
    """Returns {normalized_joint_name: (frames (N,), xyz (N,3), nviews (N,))}."""
    sources: dict[str, list] = {}
    for path in (hand_path, body_path):
        if path is None or not os.path.isfile(path):
            continue
        data = json.load(open(path, "r", encoding="utf-8"))
        for frame_str, entries in data.items():
            try:
                frame_num = float(frame_str)
            except ValueError:
                continue
            if not entries:
                continue
            ann3d = entries[0].get("annotation3D", {}) or {}
            for joint_name, v in ann3d.items():
                name = normalize_name(joint_name)
                if not name:
                    continue
                xyz = (float(v["x"]), float(v["y"]), float(v["z"]))
                nviews = float(v.get("num_views_for_3d", 1.0))
                sources.setdefault(name, []).append((frame_num, xyz, nviews))
    out = {}
    for name, rows in sources.items():
        rows.sort(key=lambda r: r[0])
        frames = np.asarray([r[0] for r in rows], dtype=np.float64)
        xyz = np.asarray([r[1] for r in rows], dtype=np.float64)
        nviews = np.asarray([r[2] for r in rows], dtype=np.float64)
        out[name] = (frames, xyz, nviews)
    return out


def _load_aria_camera_track(camera_pose_path):
    """Returns (aria_cam_name, intrinsic (3,3), frames (F,), cam_from_world (F,3,4))."""
    data = json.load(open(camera_pose_path, "r", encoding="utf-8"))
    aria_keys = [k for k in data.keys() if k != "metadata" and k.lower().startswith("aria")]
    if not aria_keys:
        raise ValueError(f"no aria camera entry in {camera_pose_path}")
    aria_name = aria_keys[0]
    cam = data[aria_name]
    intrinsic = np.asarray(cam["camera_intrinsics"], dtype=np.float64)
    ext = cam["camera_extrinsics"]
    if isinstance(ext, dict):
        frames = np.asarray([float(k) for k in ext.keys()], dtype=np.float64)
        mats = np.asarray([ext[k] for k in ext.keys()], dtype=np.float64)
    else:
        frames = np.asarray([0.0], dtype=np.float64)
        mats = np.asarray(ext, dtype=np.float64)[None, ...]
    order = np.argsort(frames)
    return aria_name, intrinsic, frames[order], mats[order]


def _to_4x4(mat3x4: np.ndarray) -> np.ndarray:
    out = np.broadcast_to(np.eye(4, dtype=np.float64), mat3x4.shape[:-2] + (4, 4)).copy()
    out[..., :3, :4] = mat3x4
    return out


def build_supervision_hdf5(
    *,
    take_uid: str,
    discovery_root: str,
    split: str,
    out_hdf5_path: str,
    task_name: str,
    num_frames: int,
):
    hand_path = os.path.join(
        discovery_root, "annotations", "ego_pose", split, "hand", "annotation", f"{take_uid}.json"
    )
    body_path = os.path.join(
        discovery_root, "annotations", "ego_pose", split, "body", "annotation", f"{take_uid}.json"
    )
    cam_path = os.path.join(
        discovery_root, "annotations", "ego_pose", split, "camera_pose", f"{take_uid}.json"
    )
    if not os.path.isfile(cam_path):
        raise FileNotFoundError(f"missing camera_pose for {take_uid}: {cam_path}")

    aria_name, intrinsic, cam_frames, cam_from_world_3x4 = _load_aria_camera_track(cam_path)
    cam_from_world_4x4 = _to_4x4(cam_from_world_3x4)
    world_from_cam_4x4 = np.linalg.inv(cam_from_world_4x4)
    world_from_cam_dense = _nearest_fill(world_from_cam_4x4, cam_frames, num_frames)
    if world_from_cam_dense is None:
        raise ValueError(f"no aria camera_extrinsics samples for {take_uid}")

    pose_sources = _load_pose_sources(hand_path, body_path)

    right_wrist = pose_sources.get("right-wrist")
    left_wrist = pose_sources.get("left-wrist")
    right_elbow = pose_sources.get("right-elbow", right_wrist)
    left_elbow = pose_sources.get("left-elbow", left_wrist)

    def _fallback_source(side: str):
        if side == "right":
            return right_elbow if right_elbow is not None else right_wrist
        return left_elbow if left_elbow is not None else left_wrist

    joint_tfs = np.zeros((num_frames, len(NPZ_TARGET_QUERY_TFS), 4, 4), dtype=np.float32)
    joint_tfs[:, :, :, :] = np.eye(4, dtype=np.float32)
    joint_confs = np.zeros((num_frames, len(NPZ_TARGET_QUERY_TFS)), dtype=np.float32)
    matched = 0
    for j, q_name in enumerate(NPZ_TARGET_QUERY_TFS):
        qn = normalize_name(q_name)
        cands = map_candidates(qn)
        src = None
        for c in cands:
            if c in pose_sources:
                src = pose_sources[c]
                break
        if src is None:
            side = "right" if qn.startswith("right-") else "left"
            src = _fallback_source(side)
        if src is None:
            continue
        frames, xyz, nviews = src
        xyz_dense = _nearest_fill(xyz, frames, num_frames)
        nviews_dense = _nearest_fill(nviews[:, None], frames, num_frames)[:, 0]
        joint_tfs[:, j, :3, 3] = xyz_dense.astype(np.float32)
        joint_confs[:, j] = nviews_dense.astype(np.float32)
        matched += 1

    os.makedirs(os.path.dirname(out_hdf5_path), exist_ok=True)
    with h5py.File(out_hdf5_path, "w") as f:
        f.create_dataset("transforms/camera", data=world_from_cam_dense.astype(np.float32))
        for j, q_name in enumerate(NPZ_TARGET_QUERY_TFS):
            f.create_dataset(f"transforms/{q_name}", data=joint_tfs[:, j, :, :])
            f.create_dataset(f"confidences/{q_name}", data=joint_confs[:, j])
        f.create_dataset("camera/intrinsic", data=intrinsic.astype(np.float32))
        f.attrs["llm_description"] = task_name or ""
    return {"matched_joints": matched, "total_joints": len(NPZ_TARGET_QUERY_TFS), "aria_camera": aria_name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery_root", required=True)
    ap.add_argument("--takes_json", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--video_root", required=True, help="Root the Aria mp4s were downloaded under (takes/<take_name>/frame_aligned_videos/*.mp4)")
    ap.add_argument("--supervision_root", required=True)
    ap.add_argument("--take_uids", nargs="+", required=True)
    args = ap.parse_args()

    import decord

    takes = json.load(open(args.takes_json, "r", encoding="utf-8"))
    by_uid = {t["take_uid"]: t for t in takes}

    for take_uid in args.take_uids:
        t = by_uid.get(take_uid)
        if t is None:
            print(f"[SKIP] {take_uid}: not in takes.json")
            continue
        aria_cams = [k for k, v in t["frame_aligned_videos"].items() if k.lower().startswith("aria")]
        if not aria_cams:
            print(f"[SKIP] {take_uid}: no aria camera in frame_aligned_videos")
            continue
        rgb = t["frame_aligned_videos"][aria_cams[0]].get("rgb")
        if rgb is None:
            print(f"[SKIP] {take_uid}: no rgb stream for {aria_cams[0]}")
            continue
        rel_mp4 = rgb["relative_path"]  # e.g. frame_aligned_videos/aria03_214-1.mp4
        mp4_path = os.path.join(args.video_root, t["root_dir"], rel_mp4)
        if not os.path.isfile(mp4_path):
            print(f"[SKIP] {take_uid}: mp4 not found at {mp4_path}")
            continue
        vr = decord.VideoReader(mp4_path)
        num_frames = len(vr)
        out_path = os.path.join(args.supervision_root, t["root_dir"], os.path.splitext(rel_mp4)[0] + ".hdf5")
        info = build_supervision_hdf5(
            take_uid=take_uid,
            discovery_root=args.discovery_root,
            split=args.split,
            out_hdf5_path=out_path,
            task_name=t.get("task_name", ""),
            num_frames=num_frames,
        )
        print(f"[OK] {take_uid}: T={num_frames} matched={info['matched_joints']}/{info['total_joints']} aria={info['aria_camera']} -> {out_path}")


if __name__ == "__main__":
    main()
