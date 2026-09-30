"""Pre-builds evaluate_logprob's per-item model inputs (decode + processor
tensor-building) ahead of time, as a separate CPU-only pass -- splitting
that work out of the GPU eval job entirely, rather than trying to overlap
it with GPU compute inside the same process.

Why: job-54's own live nvidia-smi sampling showed ~0% GPU utilization over
17+ hours, with the MAIN process (not the decode workers) burning ~80% CPU
the whole time -- confirmed via `ps aux`'s accumulated CPU time. The
existing --prefetch_depth concurrent-decode queue (see train_qa_full.py's
evaluate_logprob) only parallelizes the raw video-decode step; the
processor() call that turns decoded frames into model input tensors
(resize/normalize/patchify) still runs synchronously in the main process,
one item at a time, with no overlap against anything -- and MVBench's 11
source datasets vary in resolution far more than TempCompass's did, making
that step heavy enough to become the new bottleneck on its own. A GPU job
whose only per-item CPU work is `torch.load` + `.to(device)` should stay
GPU-bound instead.

This script needs no GPU and no frozen-VLM weights loaded -- only the
processor (tokenizer + image/video preprocessor), which is lightweight.
Reuses train_qa_full.py's start_decode/finish_decode/build_inputs_from_decoded/
build_option_block/build_prompt_text unmodified, plus the new
save_precomputed_inputs() -- the actual decode+build logic is identical to
what evaluate_logprob does live, just run here instead and written to
disk. evaluate_logprob() then consumes this cache transparently via its
own precompute_cache_dir argument (falls back to live building on any
per-item cache miss, so a partially-built or still-running cache never
blocks an eval run, just leaves it exactly as slow as before for whichever
items aren't cached yet).

--num_shards/--shard_idx let this run as several independent CPU-only pods
in parallel (each takes items[shard_idx::num_shards]) -- trivial to scale
this way, unlike in-process threading which is still capped by how many
CPU cores one pod requests.
"""
import argparse
import collections
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import train_qa_full as fwd  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qa_split", required=True)
    ap.add_argument("--split_part", default="test", choices=["val", "test", "train"])
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--seed", type=int, default=42,
                     help="only affects which letter (A/B) ends up correct for each item -- "
                          "baked into the cache at build time, so this does NOT need to match "
                          "whatever --seed an eval run later passes (see _start_item_decode's own "
                          "docstring for why a cache hit doesn't consume the eval run's rng)")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="cap on items in this shard (smoke test)")
    ap.add_argument("--prefetch_depth", type=int, default=12,
                     help="no GPU step downstream here (unlike evaluate_logprob), so this can "
                          "run deeper than that function's own default of 4 -- limited only by "
                          "this pod's own CPU request")
    args = ap.parse_args()
    if not (0 <= args.shard_idx < args.num_shards):
        ap.error("--shard_idx must be in [0, --num_shards)")

    with open(args.qa_split) as f:
        split = json.load(f)
    items = [it for it in split[args.split_part] if fwd.has_features(it)]
    items = items[args.shard_idx::args.num_shards]
    if args.limit:
        items = items[: args.limit]
    print(f"[INFO] {args.split_part} split: {len(items)} items in shard {args.shard_idx}/{args.num_shards}",
          flush=True)

    print(f"[INFO] loading {fwd.MODEL_ID} processor only (no model, no GPU) ...", flush=True)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(fwd.MODEL_ID)

    import random
    rng = random.Random(args.seed)

    next_to_start = 0
    queue = collections.deque()

    def _top_up():
        nonlocal next_to_start
        while next_to_start < len(items) and len(queue) < args.prefetch_depth:
            queue.append(fwd._start_item_decode(items[next_to_start], rng))
            next_to_start += 1

    n_saved = n_skipped_existing = n_failed = 0
    seen_exc_types = set()
    _top_up()
    for i, item in enumerate(items, start=1):
        cur_ok, cur_payload = queue.popleft()
        _top_up()
        pid = item.get("pid")
        out_path = fwd.precomputed_input_path(args.cache_dir, pid)
        if os.path.exists(out_path):
            n_skipped_existing += 1
            if i % 200 == 0:
                print(f"[{i}/{len(items)}] saved={n_saved} skipped={n_skipped_existing} failed={n_failed}",
                      flush=True)
            continue
        try:
            if not cur_ok:
                raise cur_payload
            option_lines, correct_letter, prompt_text, handle = cur_payload
            decoded = fwd.finish_decode(handle)
            inputs = fwd.build_inputs_from_decoded(processor, handle.abs_path, prompt_text, decoded)
            fwd.save_precomputed_inputs(args.cache_dir, pid, inputs, correct_letter)
            n_saved += 1
        except Exception as e:
            n_failed += 1
            exc_name = type(e).__name__
            print(f"[WARN] prebuild skip {pid}: {exc_name}: {e}", flush=True)
            if exc_name not in seen_exc_types:
                seen_exc_types.add(exc_name)
                import traceback
                traceback.print_exc()
        if i % 200 == 0:
            print(f"[{i}/{len(items)}] saved={n_saved} skipped={n_skipped_existing} failed={n_failed}", flush=True)

    print(f"[DONE] saved={n_saved} skipped_existing={n_skipped_existing} failed={n_failed} "
          f"cache_dir={args.cache_dir}", flush=True)


if __name__ == "__main__":
    main()
