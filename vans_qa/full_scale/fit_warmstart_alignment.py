"""Fits a closed-form (ridge regression) linear alignment between pooled
V-JEPA2 features and pooled VLM (Qwen3-VL) hidden states, using the paired
data extract_vlm_guidance_paired.py already produced -- vjepa_input_feats,
vjepa_target_feats, vlm_old, vlm_new all live in the same npz per pid, so no
new extraction is needed for this.

Motivation (the "warm-start coupling" idea): two otherwise-unrelated models
each start a cross-modal adapter from pure random init:
  - forward direction (train_qa_full.py): LayerWiseJEPAInjector's
    conditioner.adapter.mlp[0], a Linear(JEPA embed_dim=1024 ->
    VLM hidden_size=2048) inside SoftPromptAdapterTemporal.
  - reverse direction (train_latent_world_model_full.py):
    CortexGuidedVideoPredictor's guidance_old_adapter / guidance_new_adapter,
    Linear(VLM hidden_size=2048 -> predictor_embed_dim=384, bias=False).

Instead of starting these from random noise the optimizer has to discover a
sane subspace for via gradient descent, a linear regression fit on real
(JEPA, VLM) feature pairs gives a starting point already in roughly the
right region of the target space.

Pooling: both target spaces are pooled globally (mean over every temporal/
spatial/token/layer position) before fitting. SoftPromptAdapterTemporal.mlp
and guidance_old_adapter/guidance_new_adapter both apply their Linear
per-position with weights SHARED across positions, so a single global
alignment is the natural per-position initialization -- there's no
token-by-token correspondence between JEPA's spatiotemporal patches and
VLM's text tokens to fit against directly anyway.

Reverse-direction composition: the fitted VLM->JEPA(1024) regression is
composed with CortexGuidedVideoPredictor's OWN (randomly initialized, but
consistent) context_adapter (Linear(1024, predictor_embed_dim), the same
layer the predictor uses to embed its real rollout tokens into its working
space) rather than fit directly against the 384-dim predictor_embed_dim
space. predictor_embed_dim has no independent pretrained meaning of its
own -- unlike VLM's hidden_size, a frozen pretrained space with real
semantic content, predictor_embed_dim is whatever this same training run
ends up learning end to end, so there's no stable target to regress against
before training starts. Composing through context_adapter instead puts the
guidance adapter's *initial* output in the same coordinate frame the
predictor will actually see real vjepa rollout tokens land in via that same
layer, at initialization time.

Caveat: this assumes vjepa_cache_full/{name}.npz's "feats" (what
train_qa_full.py actually feeds the forward-direction injector at train
time) lives in the same V-JEPA2 ViT-L feature space as this script's
vjepa_input_feats. Both extractions call the same shared
cache_train.rebuild_causal_cache.{preprocess_vjepa_frames, encode_vjepa_clip}
against the same checkpoint, so this should hold, but wasn't independently
re-verified against a live vjepa_cache_full file (that cache predates this
project's V-JEPA2 encoding work being consolidated through
rebuild_causal_cache.py).

Output: a single .pt file consumed by train_qa_full.py's
--warmstart_alignment and train_latent_world_model_full.py's
--warmstart_alignment flags.
"""
import argparse
import glob
import os

import numpy as np
import torch

BASE = os.environ.get("VANS_ROOT", "/data")
WORK_BASE = os.environ.get("VANS_WORK_ROOT", "/data/vans_work")


def pooled_stats(cache_dir, limit):
    files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
    if limit:
        files = files[:limit]
    X_in, X_tgt, Y_old, Y_new = [], [], [], []
    n_skipped = 0
    for i, f in enumerate(files, start=1):
        if i % 200 == 0:
            print(f"[PROGRESS] {i}/{len(files)} files read", flush=True)
        try:
            d = np.load(f)
            vj_in = d["vjepa_input_feats"].astype(np.float64)   # (T,N,1024)
            vj_tgt = d["vjepa_target_feats"].astype(np.float64)
            v_old = d["vlm_old"].astype(np.float64)             # (L,S,2048)
            v_new = d["vlm_new"].astype(np.float64)
            X_in.append(vj_in.mean(axis=(0, 1)))
            X_tgt.append(vj_tgt.mean(axis=(0, 1)))
            Y_old.append(v_old.mean(axis=(0, 1)))
            Y_new.append(v_new.mean(axis=(0, 1)))
        except Exception as e:
            n_skipped += 1
            if n_skipped <= 5:
                print(f"[WARN] skip {f}: {e}", flush=True)
    print(f"[INFO] pooled {len(X_in)} pairs ({n_skipped} skipped) from {cache_dir}", flush=True)
    return (np.stack(X_in), np.stack(X_tgt), np.stack(Y_old), np.stack(Y_new))


def fit_ridge(X, Y, lam):
    """X: (N, d_in), Y: (N, d_out). Closed-form ridge with an intercept
    (via an augmented ones-column) -- returns (W, b) in nn.Linear's own
    convention: W has shape (d_out, d_in), b has shape (d_out,)."""
    n = X.shape[0]
    Xc = np.concatenate([X, np.ones((n, 1), dtype=X.dtype)], axis=1)  # (N, d_in+1)
    d = Xc.shape[1]
    A = np.linalg.solve(Xc.T @ Xc + lam * np.eye(d), Xc.T @ Y)  # (d_in+1, d_out)
    W = A[:-1].T
    b = A[-1]
    return W, b


def r_squared(X, Y, W, b):
    pred = X @ W.T + b
    ss_res = ((Y - pred) ** 2).sum()
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum()
    return 1.0 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache_dir", default=os.path.join(WORK_BASE, "vlm_guidance_cache_paired"))
    ap.add_argument("--limit", type=int, default=1500,
                     help="cap #pairs used for the fit -- closed-form ridge doesn't need all ~11.8k, "
                          "and each file's vlm_old array is large enough that reading thousands of "
                          "them from CephFS dominates runtime (confirmed 2026-09-09: a --limit 4000 "
                          "run took 45+ min just reading, all I/O-bound)")
    ap.add_argument("--ridge_lambda", type=float, default=10.0)
    ap.add_argument("--predictor_embed_dim", type=int, default=384)
    ap.add_argument("--context_adapter_seed", type=int, default=0,
                     help="seeds a throwaway Linear(1024, predictor_embed_dim) used only to put the "
                          "composed guidance-adapter weights in a plausible scale/subspace for that "
                          "shape -- does NOT need to match the real predictor's own context_adapter "
                          "init (that one starts equally arbitrary/unlearned regardless of seed; any "
                          "reasonably-scaled random linear map serves the same purpose here)")
    ap.add_argument("--out", default=os.path.join(BASE, "raw_data/warmstart_alignment.pt"))
    args = ap.parse_args()

    X_in, X_tgt, Y_old, Y_new = pooled_stats(args.cache_dir, args.limit)

    W_fwd, b_fwd = fit_ridge(X_in, Y_old, args.ridge_lambda)
    W_rev_old, b_rev_old = fit_ridge(Y_old, X_in, args.ridge_lambda)
    W_rev_new, b_rev_new = fit_ridge(Y_new, X_tgt, args.ridge_lambda)

    print(f"[INFO] fwd  (jepa_in  -> vlm_old) R^2={r_squared(X_in, Y_old, W_fwd, b_fwd):.4f}", flush=True)
    print(f"[INFO] rev_old (vlm_old -> jepa_in ) R^2={r_squared(Y_old, X_in, W_rev_old, b_rev_old):.4f}", flush=True)
    print(f"[INFO] rev_new (vlm_new -> jepa_tgt) R^2={r_squared(Y_new, X_tgt, W_rev_new, b_rev_new):.4f}", flush=True)

    torch.manual_seed(args.context_adapter_seed)
    context_adapter = torch.nn.Linear(1024, args.predictor_embed_dim, bias=True)
    Wc = context_adapter.weight.detach().numpy().astype(np.float64)  # (predictor_embed_dim, 1024)
    bc = context_adapter.bias.detach().numpy().astype(np.float64)

    W_rev_old_composed = Wc @ W_rev_old            # (predictor_embed_dim, 2048)
    W_rev_new_composed = Wc @ W_rev_new
    # guidance_old_adapter/guidance_new_adapter are bias=False in
    # CortexGuidedVideoPredictor -- these intercepts can't be applied there,
    # kept only for inspection/debugging.
    b_rev_old_ref = Wc @ b_rev_old + bc
    b_rev_new_ref = Wc @ b_rev_new + bc

    torch.save({
        "n_pairs": int(X_in.shape[0]),
        "ridge_lambda": args.ridge_lambda,
        "context_adapter_seed": args.context_adapter_seed,
        "predictor_embed_dim": args.predictor_embed_dim,
        # forward direction: LayerWiseJEPAInjector.conditioner.adapter.mlp[0]
        "fwd_soft_prompt_mlp0_weight": torch.from_numpy(W_fwd).float(),
        "fwd_soft_prompt_mlp0_bias": torch.from_numpy(b_fwd).float(),
        # reverse direction: CortexGuidedVideoPredictor.guidance_old_adapter / guidance_new_adapter
        "rev_guidance_old_adapter_weight": torch.from_numpy(W_rev_old_composed).float(),
        "rev_guidance_new_adapter_weight": torch.from_numpy(W_rev_new_composed).float(),
        "rev_guidance_old_adapter_bias_ref": torch.from_numpy(b_rev_old_ref).float(),
        "rev_guidance_new_adapter_bias_ref": torch.from_numpy(b_rev_new_ref).float(),
    }, args.out)
    print(f"[DONE] saved alignment to {args.out}", flush=True)


if __name__ == "__main__":
    main()
