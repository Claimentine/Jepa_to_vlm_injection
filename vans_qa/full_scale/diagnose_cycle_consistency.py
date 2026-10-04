"""Free, no-training diagnostic: does a checkpoint's bridge actually being
more cycle-consistent (lower ||vlm_to_jepa(jepa_to_vlm(x)) - x||) correlate
with it scoring better on the real downstream benchmarks (COIN/TempCompass/
MVBench)? If not, cycle-consistency per se isn't the mechanism behind
bridge-capacity's gains, whatever else it's doing.

Reuses the EXACT pooling convention train_joint_coupled_normalized.py uses
at training time (see its own pooled_jepa/pooled_vlm lines) so the numbers
here are the same quantity the cycle loss term actually optimized, not a
different metric that merely sounds similar:
  pooled_jepa = vjepa_in_feats.mean(dim=(1,2))      -- forward QA items' own
                                                        in_clip V-JEPA2 feats
  pooled_vlm  = vlm_old.mean(dim=(0,1), keepdim=True) -- reverse-direction
                                                        cache's own vlm_old

Forward and reverse samples are drawn independently (unpaired) from each
direction's own held-out TEST split, matching how training itself never
pairs a specific QA item with a specific reverse pair either -- the cycle
loss only ever saw independently-sampled pooled vectors from each side.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from common.cross_modal_bridge import CrossModalBridge  # noqa: E402
import train_qa_full as fwd  # noqa: E402
import train_latent_world_model_full as rev  # noqa: E402

BASE = os.environ.get("VANS_ROOT", "/data")

# known downstream numbers for each checkpoint, gathered this session --
# printed alongside the cycle-consistency numbers for a side-by-side read,
# not recomputed here (that's eval_forward_checkpoint.py's job).
KNOWN_RESULTS = {
    "job22_joint_vans_only": {"coin": 0.6829, "tempcompass": 0.5665, "mvbench": 0.5604},
    "job34_joint_normalized": {"coin": 0.6720, "tempcompass": 0.7031, "mvbench": 0.6072},
    "job36_joint_cycle0": {"coin": None, "tempcompass": 0.6396, "mvbench": 0.5517},
    "job42_bridge_mlp2048": {"coin": 0.6671, "tempcompass": 0.7571, "mvbench": 0.6309},
    "job49_coin_mix": {"coin": 0.6713, "tempcompass": 0.6952, "mvbench": 0.5882},
    "job56_norm_bridge": {"coin": 0.6146, "tempcompass": 0.6140, "mvbench": None},
}

CHECKPOINTS = {
    "job22_joint_vans_only": os.path.join(BASE, "raw_data/joint_coupled_runs/best_forward.pt"),
    "job34_joint_normalized": os.path.join(BASE, "raw_data/joint_coupled_normalized_runs/best_forward.pt"),
    "job36_joint_cycle0": os.path.join(BASE, "raw_data/joint_coupled_cycle0_runs/best_forward.pt"),
    "job42_bridge_mlp2048": os.path.join(BASE, "raw_data/joint_coupled_bridge_mlp_runs/best_forward.pt"),
    "job49_coin_mix": os.path.join(BASE, "raw_data/joint_coupled_mixed_normalized_nobridge_runs/best_forward.pt"),
    "job56_norm_bridge": os.path.join(BASE, "raw_data/joint_coupled_norm_bridge_runs/best_forward.pt"),
}


def detect_hidden_dim(bridge_state):
    if "jepa_to_vlm.weight" in bridge_state:
        return None
    return bridge_state["jepa_to_vlm.0.weight"].shape[0]


def load_bridge(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    bridge_state = ckpt["bridge_state"]
    hidden_dim = detect_hidden_dim(bridge_state)
    bridge = CrossModalBridge(jepa_dim=1024, vlm_dim=2048, hidden_dim=hidden_dim).to(device)
    bridge.load_state_dict(bridge_state)
    bridge.eval()
    return bridge, hidden_dim


def sample_pooled_jepa(qa_split_path, n, device):
    with open(qa_split_path) as f:
        split = json.load(f)
    items = [it for it in split["test"] if fwd.has_features(it)][:n]
    pooled = []
    for it in items:
        feats = torch.from_numpy(fwd.load_vjepa_in_feats(it["in_clip"])).unsqueeze(0).to(device)
        pooled.append(feats.mean(dim=(1, 2)))
    return torch.cat(pooled, dim=0)  # (n, 1024)


def sample_pooled_vlm(qa_split_full_path, cache_dir, n, device):
    membership = rev.load_qa_split_membership(qa_split_full_path)
    cached_pids = rev.list_cached_pids(cache_dir)
    test_pids = [pid for pid in cached_pids if membership.get(pid) == "test"][:n]
    pooled = []
    for pid in test_pids:
        _, _, extras = rev.load_pair(cache_dir, pid, device)
        pooled.append(extras["vlm_old"].mean(dim=(0, 1)))  # (2048,)
    return torch.stack(pooled, dim=0)  # (n, 2048)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qa_split", default=os.path.join(BASE, "raw_data/qa_split_temporal.json"))
    ap.add_argument("--qa_split_full", default=os.path.join(BASE, "raw_data/qa_split_full.json"))
    ap.add_argument("--rev_cache_dir", default=os.path.join(
        os.environ.get("VANS_WORK_ROOT", "/data/vans_work"), "vlm_guidance_cache_paired"))
    ap.add_argument("--n_samples", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] sampling {args.n_samples} held-out forward (pooled_jepa) items ...", flush=True)
    pooled_jepa = sample_pooled_jepa(args.qa_split, args.n_samples, device)
    print(f"[INFO] sampling {args.n_samples} held-out reverse (pooled_vlm) items ...", flush=True)
    pooled_vlm = sample_pooled_vlm(args.qa_split_full, args.rev_cache_dir, args.n_samples, device)
    print(f"[INFO] pooled_jepa={tuple(pooled_jepa.shape)} pooled_vlm={tuple(pooled_vlm.shape)}", flush=True)

    print("", flush=True)
    header = f"{'checkpoint':28s} {'hidden_dim':>10s} {'cycle_mse':>10s} {'coin':>8s} {'tempcmp':>8s} {'mvbench':>8s}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    rows = []
    for name, path in CHECKPOINTS.items():
        if not os.path.exists(path):
            print(f"{name:28s} -- checkpoint not found: {path}", flush=True)
            continue
        bridge, hidden_dim = load_bridge(path, device)
        with torch.no_grad():
            cyc = bridge.cycle_loss(pooled_jepa, pooled_vlm).item()
        known = KNOWN_RESULTS.get(name, {})
        fmt = lambda v: f"{v:.4f}" if v is not None else "n/a"
        hd = str(hidden_dim) if hidden_dim is not None else "linear"
        print(f"{name:28s} {hd:>10s} {cyc:>10.4f} {fmt(known.get('coin')):>8s} "
              f"{fmt(known.get('tempcompass')):>8s} {fmt(known.get('mvbench')):>8s}", flush=True)
        rows.append((name, hidden_dim, cyc, known))

    print("", flush=True)
    valid = [(n, c, k) for n, _, c, k in rows if k.get("tempcompass") is not None]
    if len(valid) >= 3:
        cycs = np.array([c for _, c, _ in valid])
        temps = np.array([k["tempcompass"] for _, _, k in valid])
        corr = float(np.corrcoef(cycs, temps)[0, 1])
        print(f"[CORRELATION] cycle_mse vs TempCompass acc across {len(valid)} checkpoints: r={corr:.3f}", flush=True)
        print("  (near 0 or positive r = lower cycle error does NOT predict better TempCompass score --", flush=True)
        print("   cycle-consistency isn't obviously the mechanism; strongly negative r = it plausibly is)", flush=True)


if __name__ == "__main__":
    main()
