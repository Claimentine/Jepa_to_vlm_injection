"""Rebuilds an EXISTING ThinkJEPA-schema VLM-guidance cache bundle's own
vlm_old/vlm_new arrays using a trained --lora_vlm checkpoint's VLM-LoRA
weights, instead of the original frozen Qwen3-VL those arrays were cached
with. Written 2026-11-20 per explicit user request ("mark the issue, write
the script, don't run it yet") -- NOT executed anywhere yet; see the
VERIFY-BEFORE-RUNNING markers below.

Why this exists: train_joint_coupled_normalized.py's --lora_vlm --live_vlm
recipe trains CortexGuidedVideoPredictor against vlm_old/vlm_new produced
LIVE by a LoRA-adapted Qwen3-VL (the adapter drifts over training, even at
a small --vlm_lr). Every bundle this project's eval scripts read
(EGODEX_BUNDLE for probe_egodex_reverse_adapter.py, EgoExo4D's
egoexo4d_probe_bundle_full) was cached by ThinkJEPA's own
cache_train/rebuild_causal_cache.py against the FROZEN, un-adapted model --
correct for every PRIOR checkpoint this project has probed (job15/42/49/
60/72, all dense- or predictor-LoRA-only, never --lora_vlm), but a real
train/eval distribution mismatch for a --lora_vlm checkpoint: the
predictor learned to interpret guidance vectors from a model that no
longer exists in its original form by the time the probe runs. Rebuilding
each npz's vlm_old/vlm_new with the checkpoint's own LoRA weights removes
that confound -- the probe then measures only the predictor's own latent
quality, the thing it's actually meant to measure.

Schema (confirmed directly from a live EgoExo4D cache npz, job-171):
  imgs                        (64,256,256,3) uint8  -- ALREADY decoded frames,
                                                         no video re-decode needed
  vlm_observation_frame_indices (32,) int64         -- which of imgs' 64 frames
                                                         the ORIGINAL VLM call saw
  input_ids                   (1187,) int32          -- the exact tokenized
                                                         prompt+placeholder sequence
  vlm_new_token_ids           (15,) int32             -- the teacher-forced
                                                         answer/target token ids
  text                        <U64 string            -- ASSUMED to hold the
                                                         question text (see VERIFY #1)
  layers                      (8,) int32              -- which decoder layers
                                                         vlm_old/vlm_new were
                                                         read from (reuse as-is)
  vlm_old  (8,1187,2048) / vlm_new (8,15,2048)         -- the arrays this script
                                                         replaces

*** VERIFY BEFORE RUNNING ***
1. `text`'s exact contents were never printed (job-171 only printed shape/
   dtype, not the string value) -- confirm it's actually the literal
   question text infer_qwen_teacher_forced_live expects, not a template
   name/id. If it's not, ThinkJEPA's own cache_train/rebuild_causal_cache.py
   (prompt_base/prompt_overlay/dataset_prompt_mode fields in the same npz)
   needs reading to learn the real prompt-construction convention.
2. decoding vlm_new_token_ids via the Qwen tokenizer to recover a
   correct_caption STRING (then re-tokenizing it inside
   infer_qwen_teacher_forced_live) is a round-trip that should be a no-op
   for ordinary text, but hasn't been checked against a real sample --
   confirm token_ids -> decode -> re-encode reproduces the same ids before
   trusting this on the full bundle.
3. --layers here defaults to this project's own global convention
   ([0,4,8,12,16,20,24,27]); the npz's own `layers` field should be read
   and asserted equal instead of assumed, in case some bundle used a
   different set.
4. Not yet smoke-tested on even one npz end-to-end.
5. --layers here is the fallback: this script asserts it against the
   npz's own `layers` field in rebuild_one and raises if they differ,
   rather than silently trusting the default (see #3) -- but that assert
   itself has not been exercised against a real file yet.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

THINKJEPA_ROOT = os.environ.get("THINKJEPA_ROOT", "/home/jovyan/ThinkJEPA")
if THINKJEPA_ROOT not in sys.path:
    sys.path.insert(0, THINKJEPA_ROOT)


def rebuild_one(npz_path, out_path, model, processor, saved, qwen_args, device):
    from common.live_vlm_guidance import infer_qwen_teacher_forced_live

    d = dict(np.load(npz_path))
    if "layers" in d and list(d["layers"]) != list(qwen_args.layers):
        raise ValueError(f"{npz_path}: cached with layers={list(d['layers'])}, "
                          f"but this run's --qwen_guidance_layers={list(qwen_args.layers)} -- "
                          "rebuilding with mismatched layers would silently produce a vlm_old/"
                          "vlm_new shaped for the wrong layer set.")
    obs_idx = d["vlm_observation_frame_indices"]
    frames = d["imgs"][obs_idx]  # (32,H,W,3) uint8 -- the exact frames the original VLM call saw
    past_times = d["past_frame_times_seconds"] if "past_frame_times_seconds" in d else obs_idx.astype(np.float64)
    span = max(1e-6, float(past_times[-1] - past_times[0]))
    in_fps = max(1.0, (len(past_times) - 1) / span)

    # VERIFY #1 (see module docstring): assumes `text` is the literal question.
    question = str(d["text"])
    # VERIFY #2: round-trip decode of the teacher-forced answer's token ids
    # back to a string, so infer_qwen_teacher_forced_live can re-tokenize it
    # itself (that function's own signature takes a string, not pre-tokenized
    # ids -- a lower-level variant that accepts ids directly would avoid this
    # round-trip entirely, but doesn't exist yet).
    correct_caption = processor.tokenizer.decode(d["vlm_new_token_ids"], skip_special_tokens=True)

    vlm_old, vlm_new = infer_qwen_teacher_forced_live(
        qwen_args, model, processor, saved, device,
        frames, past_times, in_fps, question, correct_caption,
    )

    d["vlm_old"] = vlm_old.detach().float().cpu().numpy().astype(np.float16)
    d["vlm_new"] = vlm_new.detach().float().cpu().numpy().astype(np.float16)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **d)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="a train_joint_coupled_normalized.py "
                     "checkpoint with a vlm_lora_state field")
    ap.add_argument("--in_cache_dir", required=True, help="existing ThinkJEPA-schema bundle cache "
                     "dir, e.g. /data/raw_data/egoexo4d_probe_bundle_full/cache")
    ap.add_argument("--out_cache_dir", required=True, help="rebuilt bundle's own cache dir -- "
                     "never writes into --in_cache_dir, so the original (frozen-VLM) bundle "
                     "stays available for comparison")
    ap.add_argument("--pretrained", default="Qwen/Qwen3-VL-2B-Thinking")
    ap.add_argument("--qwen_revision", default="main")
    ap.add_argument("--lora_vlm_r", type=int, default=8)
    ap.add_argument("--lora_vlm_alpha", type=float, default=16.0)
    ap.add_argument("--qwen_res", type=int, default=224)
    ap.add_argument("--qwen_guidance_layers", type=int, nargs="+", default=[0, 4, 8, 12, 16, 20, 24, 27])
    ap.add_argument("--limit", type=int, default=None, help="rebuild only the first N npz, for a "
                     "smoke test before committing to the full bundle")
    args = ap.parse_args()

    from types import SimpleNamespace
    from common.lora import load_lora_state_dict
    from common.live_vlm_guidance import load_qwen_with_lora

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qwen_args = SimpleNamespace(pretrained=args.pretrained, qwen_revision=args.qwen_revision,
                                 layers=args.qwen_guidance_layers, qwen_res=args.qwen_res)
    # load_qwen_with_lora wraps with a FRESH (zero-init lora_B) LoRA adapter --
    # that's immediately overwritten below by load_lora_state_dict with the
    # checkpoint's own trained weights, so the r/alpha passed here only need
    # to match the checkpoint's SHAPE (same r/alpha it was trained with),
    # not produce a meaningful adapter on their own.
    model, processor, saved, _layers_path, _wrapped = load_qwen_with_lora(
        qwen_args, device, lora_r=args.lora_vlm_r, lora_alpha=args.lora_vlm_alpha,
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "vlm_lora_state" not in ckpt:
        raise ValueError(f"{args.checkpoint} has no vlm_lora_state -- it wasn't trained with --lora_vlm, "
                          "nothing to rebuild (its bundle's existing frozen-VLM cache is already correct)")
    load_lora_state_dict(model, ckpt["vlm_lora_state"])
    model.eval()

    npz_paths = sorted(glob.glob(os.path.join(args.in_cache_dir, "**", "*.npz"), recursive=True))
    if args.limit:
        npz_paths = npz_paths[: args.limit]
    print(f"[INFO] rebuilding {len(npz_paths)} npz from {args.in_cache_dir} -> {args.out_cache_dir} "
          f"using {args.checkpoint}'s own vlm_lora_state", flush=True)

    n_ok = n_err = 0
    with torch.no_grad():
        for i, npz_path in enumerate(npz_paths, start=1):
            rel = os.path.relpath(npz_path, args.in_cache_dir)
            out_path = os.path.join(args.out_cache_dir, rel)
            try:
                rebuild_one(npz_path, out_path, model, processor, saved, qwen_args, device)
                n_ok += 1
            except Exception as e:
                n_err += 1
                print(f"[ERR {i}/{len(npz_paths)}] {rel}: {type(e).__name__}: {e}", flush=True)
            if i % 20 == 0:
                print(f"[{i}/{len(npz_paths)}] ok={n_ok} err={n_err}", flush=True)
    print(f"[DONE] ok={n_ok} err={n_err} total={len(npz_paths)}", flush=True)


if __name__ == "__main__":
    main()
