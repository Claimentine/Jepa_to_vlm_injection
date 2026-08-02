"""
Prototype: derive grasp-state / inter-hand / contact-event signals directly
from EgoDex hdf5 hand skeleton transforms (no object pose needed).

v2: adds dynamic active-hand selection. EgoDex episodes are frequently
single-handed (or one hand rests out of the interaction for stretches), so a
hard-coded "always use the right hand" signal produces flat/degenerate labels
on a meaningful fraction of episodes (confirmed empirically: right-hand
std==0.0 on 2 of 13 sampled episodes while the left hand had real motion).

Signals computed per frame:
  - grasp_aperture (both hands): distance(thumbTip, indexTip), normalized by
    a per-episode hand-scale reference (thumbKnuckle <-> littleKnuckle span)
  - active_hand: per-frame argmax of a windowed activity score (aperture
    variance + hand speed) between left/right -> which hand's signal is used
    as the "primary" grasp signal for that frame
  - grasp_state (primary): {opening, closing, steady} via smoothed derivative
    + hysteresis, computed on the active hand's aperture
  - inter_hand_distance: distance(rightHand, leftHand), independent of which
    hand is "active"
  - inter_hand_state: {approaching, departing, steady}
  - contact_candidate: heuristic flag where the ACTIVE hand's speed drops
    sharply while its aperture is closing (approach + grasp coincidence)
"""
import json
import glob
import os

import h5py
import numpy as np

HDF5_ROOT = os.environ.get(
    "EGODEX_HDF5_ROOT",
    "/work/nvme/bdqf/william/charles/data/hf_staging/egodex_part2_video_cache_subset2000_ratio0.9_seed42/hdf5/egodex/part2",
)
OUT_DIR = os.environ.get("EGODEX_PROBE_OUT_DIR", os.path.dirname(os.path.abspath(__file__)))

SMOOTH_WIN = 5                # frames, odd, for derivative smoothing
ACTIVITY_WIN = 15             # frames, odd, rolling window for active-hand scoring
STEADY_EPS_APERTURE = 0.03    # normalized units/frame, dead-zone for open/close derivative
STEADY_EPS_HANDS = 0.02       # normalized units/frame, dead-zone for approach/depart derivative
CONTACT_SPEED_PCTL = 15       # local speed below this percentile (of the active hand) => "slow/stopped"
CONTACT_APERTURE_CLOSE_THR = -0.02  # aperture derivative below this => actively closing


def pos(tf):
    """tf: [T,4,4] rigid transforms -> [T,3] translation."""
    return tf[:, :3, 3]


def moving_average(x, win):
    if win <= 1:
        return x.copy()
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(win) / win
    return np.convolve(xp, kernel, mode="valid")


def rolling_std(x, win):
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = xp[i : i + win].std()
    return out


def classify_with_hysteresis(deriv, eps):
    """+1 increasing, -1 decreasing, 0 steady, with dead-zone eps."""
    state = np.zeros_like(deriv, dtype=int)
    state[deriv > eps] = 1
    state[deriv < -eps] = -1
    return state


def load_episode(path):
    with h5py.File(path, "r") as f:
        T = f["transforms/camera"].shape[0]

        r_thumb = pos(f["transforms/rightThumbTip"][()])
        r_index = pos(f["transforms/rightIndexFingerTip"][()])
        r_knuckle_thumb = pos(f["transforms/rightThumbKnuckle"][()])
        r_knuckle_little = pos(f["transforms/rightLittleFingerKnuckle"][()])
        r_hand = pos(f["transforms/rightHand"][()])

        l_thumb = pos(f["transforms/leftThumbTip"][()])
        l_index = pos(f["transforms/leftIndexFingerTip"][()])
        l_knuckle_thumb = pos(f["transforms/leftThumbKnuckle"][()])
        l_knuckle_little = pos(f["transforms/leftLittleFingerKnuckle"][()])
        l_hand = pos(f["transforms/leftHand"][()])

        meta = {
            "task": f.attrs.get("task", ""),
            "llm_description": f.attrs.get("llm_description", ""),
            "llm_objects": list(f.attrs.get("llm_objects", [])),
            "llm_verbs": list(f.attrs.get("llm_verbs", [])),
        }

    def hand_scale(knuckle_thumb, knuckle_little):
        span = np.linalg.norm(knuckle_thumb - knuckle_little, axis=-1)
        return max(np.median(span), 1e-6)

    r_scale = hand_scale(r_knuckle_thumb, r_knuckle_little)
    l_scale = hand_scale(l_knuckle_thumb, l_knuckle_little)

    r_aperture = np.linalg.norm(r_thumb - r_index, axis=-1) / r_scale
    l_aperture = np.linalg.norm(l_thumb - l_index, axis=-1) / l_scale

    r_aperture_s = moving_average(r_aperture, SMOOTH_WIN)
    l_aperture_s = moving_average(l_aperture, SMOOTH_WIN)

    r_aperture_deriv = np.gradient(r_aperture_s)
    l_aperture_deriv = np.gradient(l_aperture_s)

    r_speed = np.linalg.norm(np.gradient(r_hand, axis=0), axis=-1) / r_scale
    l_speed = np.linalg.norm(np.gradient(l_hand, axis=0), axis=-1) / l_scale
    r_speed_s = moving_average(r_speed, SMOOTH_WIN)
    l_speed_s = moving_average(l_speed, SMOOTH_WIN)

    # --- dynamic active-hand selection ---
    # windowed activity score = local aperture variability + local hand speed
    r_activity = rolling_std(r_aperture_s, ACTIVITY_WIN) + r_speed_s
    l_activity = rolling_std(l_aperture_s, ACTIVITY_WIN) + l_speed_s
    active_hand = np.where(r_activity >= l_activity, 1, -1)  # +1 = right, -1 = left

    primary_aperture = np.where(active_hand == 1, r_aperture_s, l_aperture_s)
    primary_deriv = np.where(active_hand == 1, r_aperture_deriv, l_aperture_deriv)
    primary_speed = np.where(active_hand == 1, r_speed_s, l_speed_s)
    primary_grasp_state = classify_with_hysteresis(primary_deriv, STEADY_EPS_APERTURE)

    inter_hand_scale = max(np.median(np.linalg.norm(r_hand - l_hand, axis=-1)), 1e-6)
    inter_hand_dist = np.linalg.norm(r_hand - l_hand, axis=-1) / inter_hand_scale
    inter_hand_dist_s = moving_average(inter_hand_dist, SMOOTH_WIN)
    inter_hand_deriv = np.gradient(inter_hand_dist_s)
    inter_hand_state = classify_with_hysteresis(inter_hand_deriv, STEADY_EPS_HANDS)

    speed_thr = np.percentile(primary_speed, CONTACT_SPEED_PCTL)
    contact_candidate = (
        (primary_speed <= speed_thr) & (primary_deriv <= CONTACT_APERTURE_CLOSE_THR)
    ).astype(int)

    return {
        "meta": meta,
        "T": int(T),
        "r_aperture": r_aperture_s.tolist(),
        "l_aperture": l_aperture_s.tolist(),
        "r_speed": r_speed_s.tolist(),
        "l_speed": l_speed_s.tolist(),
        "active_hand": active_hand.tolist(),
        "primary_aperture": primary_aperture.tolist(),
        "primary_grasp_state": primary_grasp_state.tolist(),
        "inter_hand_dist": inter_hand_dist_s.tolist(),
        "inter_hand_state": inter_hand_state.tolist(),
        "contact_candidate": contact_candidate.tolist(),
    }


def print_raw_sample(key, ep, n_frames=30, start=None):
    """Print raw per-frame numbers for manual inspection."""
    T = ep["T"]
    if start is None:
        # start somewhere with actual state changes, not the very beginning
        states = ep["primary_grasp_state"]
        start = next((i for i, s in enumerate(states) if s != 0), 0)
        start = max(0, start - 5)
    end = min(start + n_frames, T)

    print(f"\n=== RAW SAMPLE: {key}  frames [{start}:{end}] of {T} ===")
    print(f"task={ep['meta']['task']}  objects={ep['meta']['llm_objects']}  verbs={ep['meta']['llm_verbs']}")
    header = f"{'t':>4} {'active':>7} {'r_aper':>8} {'l_aper':>8} {'primary_aper':>13} {'state':>7} {'contact':>8} {'inter_hand':>11} {'ih_state':>9}"
    print(header)
    for t in range(start, end):
        active = "R" if ep["active_hand"][t] == 1 else "L"
        state_lbl = {1: "open+", -1: "close-", 0: "steady"}[ep["primary_grasp_state"][t]]
        ih_lbl = {1: "apart+", -1: "near-", 0: "steady"}[ep["inter_hand_state"][t]]
        print(
            f"{t:>4} {active:>7} {ep['r_aperture'][t]:>8.3f} {ep['l_aperture'][t]:>8.3f} "
            f"{ep['primary_aperture'][t]:>13.3f} {state_lbl:>7} {ep['contact_candidate'][t]:>8d} "
            f"{ep['inter_hand_dist'][t]:>11.3f} {ih_lbl:>9}"
        )


def main():
    out = {}
    for cat_dir in sorted(glob.glob(os.path.join(HDF5_ROOT, "*"))):
        cat = os.path.basename(cat_dir)
        files = sorted(glob.glob(os.path.join(cat_dir, "*.hdf5")))
        if not files:
            continue
        ep_path = files[0]
        ep_id = os.path.splitext(os.path.basename(ep_path))[0]
        try:
            result = load_episode(ep_path)
        except Exception as e:
            print(f"[SKIP] {cat}/{ep_id}: {e}")
            continue
        key = f"{cat}/{ep_id}"
        out[key] = result
        n_right = sum(1 for a in result["active_hand"] if a == 1)
        n_left = len(result["active_hand"]) - n_right
        print(
            f"[OK] {key:55s} T={result['T']:5d}  "
            f"active_hand(R/L)={n_right}/{n_left}  contact_frames={sum(result['contact_candidate'])}"
        )

    out_path = os.path.join(OUT_DIR, "kinematic_signals.json")
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nSaved {len(out)} episodes -> {out_path}")

    # print raw data for the two episodes that were flagged as degenerate
    # under the old right-hand-only logic, to show the fix in action
    for key in [
        "assemble_disassemble_furniture_bench_desk/1004",
        "basic_pick_place/10006",
        "assemble_disassemble_furniture_bench_lamp/104",
    ]:
        if key in out:
            print_raw_sample(key, out[key])


if __name__ == "__main__":
    main()
