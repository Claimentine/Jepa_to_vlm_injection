"""
Lightweight availability probe: for a random sample of the 11,453 target
YouTube videos, checks whether yt-dlp can still resolve them (metadata-only,
no download) so we know the realistic yield *before* committing to a
multi-hour bulk download.
"""
import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp

BASE = os.environ.get("VANS_ROOT", "/projects/bhay/william/ruixin/vans_world_model")


def check_one(target):
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True, "simulate": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(target["url"], download=False)
        return target["video_id"], True, None
    except Exception as e:
        return target["video_id"], False, str(e)[:150]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_size", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(os.path.join(BASE, "raw_data/download_targets.json")) as f:
        targets = json.load(f)

    rng = random.Random(args.seed)
    coin = [t for t in targets if t["dataset"] == "coin"]
    yc = [t for t in targets if t["dataset"] == "youcook2"]
    n_coin = int(args.sample_size * len(coin) / len(targets))
    n_yc = args.sample_size - n_coin
    sample = rng.sample(coin, min(n_coin, len(coin))) + rng.sample(yc, min(n_yc, len(yc)))
    print(f"[INFO] probing {len(sample)} videos ({n_coin} coin, {n_yc} youcook2) with {args.workers} workers ...")

    ok, fail = 0, 0
    fail_examples = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_one, t): t for t in sample}
        for i, fut in enumerate(as_completed(futures)):
            vid, success, err = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                if len(fail_examples) < 8:
                    fail_examples.append((vid, err))
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(sample)}] ok={ok} fail={fail}")

    print(f"\n[RESULT] {ok}/{len(sample)} available ({100*ok/len(sample):.1f}%)")
    print(f"[RESULT] estimated available out of full {len(targets)}: ~{int(len(targets)*ok/len(sample))}")
    print("\n[SAMPLE FAILURES]")
    for vid, err in fail_examples:
        print(f"  {vid}: {err}")


if __name__ == "__main__":
    main()
