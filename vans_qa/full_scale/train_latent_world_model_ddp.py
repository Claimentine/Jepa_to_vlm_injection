"""Multi-GPU (DDP) + real batching pilot for the reverse (VLM-guides-JEPA)
direction, on top of train_latent_world_model_full.py (job-15/16's tested
baseline, imported unmodified -- new file, not an edit, per this project's
established convention).

Why this direction first: it has no live frozen VLM backbone and no
forward-hook injection mechanism (vlm_old/vlm_new/vjepa features are all
precomputed .npz on disk) -- the simplest of this project's training loops
to batch and wrap in DDP without touching anything architecture-sensitive.
The forward (JEPA->VLM QA) direction's hook-based per-example condition
injection would need real batching work on DecoderLayerInjectionHook before
this same approach applies there -- deliberately out of scope here.

Batching: confirmed 2026-09-17 by reading cache_train/thinker_predictor.py
directly that CortexGuidedVideoPredictor's own _normalize_guidance_stream
already accepts a real [B,L,S,D] guidance tensor (it only auto-unsqueezes a
[L,S,D] single-sample tensor when batch_size==1, and raises a clear error on
any batch mismatch -- never silently broadcasts). forward_step() in the base
script is already fully parameterized by B = in_feats.shape[0]. So batching
here is just: stack B pids' vjepa_input_feats/vjepa_target_feats/vlm_old/
vlm_new along a new leading axis instead of unsqueeze(0)-ing one at a time --
no changes needed to forward_step, build_predictor, or the predictor itself.

DDP: single-node only (see this script's own smoke-test job's header comment
for why -- this cluster's node availability/eviction pattern makes
multi-node rendezvous not worth the added fragility right now). Launched via
`torchrun --nproc_per_node=N`; falls back to plain single-process/single-GPU
behavior when WORLD_SIZE is unset or 1, so it's safe to smoke-test with a
plain `python` invocation before scaling up. Rank sharding is a plain
`ids[rank::world_size]` slice (this project's loops are hand-rolled, not
torch.utils.data.Dataset/DataLoader, so there's no DistributedSampler to
reach for) -- all ranks shuffle with the SAME seed before slicing so shards
are disjoint and deterministic. Logging, validation, and checkpointing only
happen on rank 0.
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_latent_world_model_full as base  # noqa: E402


def is_ddp():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def load_batch(cache_dir, pids, device):
    """Same schema as base.load_pair, but stacks B pids along a real batch
    axis instead of unsqueeze(0)-ing a single sample."""
    in_list, out_list, old_list, new_list = [], [], [], []
    for pid in pids:
        d = np.load(os.path.join(cache_dir, f"{pid}.npz"))
        in_list.append(d["vjepa_input_feats"].astype(np.float32))
        out_list.append(d["vjepa_target_feats"].astype(np.float32))
        old_list.append(d["vlm_old"].astype(np.float32))
        new_list.append(d["vlm_new"].astype(np.float32))
    in_feats = torch.from_numpy(np.stack(in_list, axis=0)).to(device)   # (B,16,256,1024)
    out_feats = torch.from_numpy(np.stack(out_list, axis=0)).to(device)  # (B,16,256,1024)
    extras = {
        "vlm_old": torch.from_numpy(np.stack(old_list, axis=0)).to(device),  # (B,L,S,D)
        "vlm_new": torch.from_numpy(np.stack(new_list, axis=0)).to(device),  # (B,L,S,D)
    }
    return in_feats, out_feats, extras


def chunk(ids, batch_size):
    for i in range(0, len(ids), batch_size):
        chunk_ids = ids[i:i + batch_size]
        if len(chunk_ids) == batch_size:  # drop a ragged final batch -- keeps
            yield chunk_ids               # every rank's step count identical,
                                           # which DDP's gradient sync needs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache_dir", default=os.path.join(base.WORK_BASE, "vlm_guidance_cache_paired"))
    ap.add_argument("--qa_split", default=os.path.join(base.BASE, "raw_data/qa_split_full.json"))
    ap.add_argument("--guidance", choices=["crossattn", "film", "adaln", "none"], required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_every_steps", type=int, default=200)
    ap.add_argument("--save_every_steps", type=int, default=500)
    ap.add_argument("--max_val_items", type=int, default=200)
    ap.add_argument("--max_train_items", type=int, default=None, help="smoke-test cap, applied before rank sharding")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=os.path.join(base.BASE, "raw_data/latent_world_model_ddp_runs"))
    args = ap.parse_args()

    if is_ddp():
        rank, world_size, local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    run_dir = os.path.join(args.out_dir, args.guidance)
    if is_main:
        os.makedirs(run_dir, exist_ok=True)

    membership = base.load_qa_split_membership(args.qa_split)
    cached_pids = base.list_cached_pids(args.cache_dir)
    by_split = {"train": [], "val": [], "test": []}
    for pid in cached_pids:
        part = membership.get(pid)
        if part in by_split:
            by_split[part].append(pid)
    rng = random.Random(args.seed)  # same seed on every rank -- shuffle order
    for part in by_split.values():  # must match so ids[rank::world_size]
        rng.shuffle(part)           # gives disjoint, deterministic shards
    train_ids, val_ids, test_ids = by_split["train"], by_split["val"], by_split["test"]
    if args.max_train_items:
        train_ids = train_ids[: args.max_train_items]

    my_train_ids = train_ids[rank::world_size]
    if is_main:
        print(f"[INFO] world_size={world_size} batch_size={args.batch_size} "
              f"total_train={len(train_ids)} per_rank={len(my_train_ids)} "
              f"val={len(val_ids)} test={len(test_ids)}", flush=True)

    predictor = base.build_predictor(device, args.guidance)
    if is_ddp():
        predictor = DDP(predictor, device_ids=[local_rank])
    raw_predictor = predictor.module if is_ddp() else predictor

    if is_main:
        n_params = sum(p.numel() for p in raw_predictor.parameters())
        print(f"[INFO] guidance={args.guidance} predictor params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(predictor.parameters(), lr=args.lr)

    log_f = open(os.path.join(run_dir, "train_log.jsonl"), "a") if is_main else None
    best_val = float("inf")
    step = 0

    for epoch in range(args.epochs):
        rng2 = random.Random(args.seed + 1000 + epoch)
        rng2.shuffle(my_train_ids)
        for pid_batch in chunk(my_train_ids, args.batch_size):
            step += 1
            in_feats, out_feats, extras = load_batch(args.cache_dir, pid_batch, device)
            metrics = base.forward_step(predictor, in_feats, out_feats, extras, args.guidance, device)
            loss = metrics["pred_loss"]

            opt.zero_grad()
            loss.backward()
            opt.step()

            if is_main:
                log_f.write(json.dumps({"step": step, "epoch": epoch, "train_loss": float(loss.item())}) + "\n")
                log_f.flush()

            if is_main and step % args.val_every_steps == 0:
                val_metrics, n_val = base.evaluate(raw_predictor, args.cache_dir, val_ids, args.guidance, device, args.max_val_items)
                print(
                    f"[step {step} x{world_size}gpu bs{args.batch_size}] train_loss={loss.item():.4f}  "
                    f"val_mse={val_metrics['pred_loss']:.4f}  "
                    f"val_latent_dist={val_metrics['pred_latent_dist']:.4f}  "
                    f"val_cosine_dist={val_metrics['pred_latent_cosine_distance']:.4f}  (n={n_val})",
                    flush=True,
                )
                log_f.write(json.dumps({"step": step, "epoch": epoch, **{f"val_{k}": v for k, v in val_metrics.items()}}) + "\n")
                log_f.flush()
                if val_metrics["pred_loss"] < best_val:
                    best_val = val_metrics["pred_loss"]
                    torch.save({"step": step, "guidance": args.guidance, "best_val_mse": best_val,
                                "predictor_state": raw_predictor.state_dict()},
                               os.path.join(run_dir, "best.pt"))
            if is_main and step % args.save_every_steps == 0:
                torch.save({"step": step, "guidance": args.guidance,
                            "predictor_state": raw_predictor.state_dict()},
                           os.path.join(run_dir, f"step_{step}.pt"))

    if is_main:
        best_path = os.path.join(run_dir, "best.pt")
        if os.path.exists(best_path):
            raw_predictor.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["predictor_state"])
        test_metrics, n_test = base.evaluate(raw_predictor, args.cache_dir, test_ids, args.guidance, device)
        print(f"[TEST] guidance={args.guidance} n={n_test} {test_metrics}", flush=True)
        with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
            json.dump({"n_test": n_test, "world_size": world_size, "batch_size": args.batch_size, **test_metrics}, f, indent=2)
        log_f.write(json.dumps({"done": True, "steps_completed": step, "best_val_mse": best_val}) + "\n")
        log_f.flush()
        log_f.close()

    if is_ddp():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
