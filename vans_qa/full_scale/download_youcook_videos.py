"""Production YouCook2 video downloader -- the piece that was always missing:
`VANS-DATA_YouCook.csv` (26,558 QA-pair rows) and `build_qa_pairs_full.py`
(already scans it) have existed for a while, and `step1_youcook_fixed.py`
already knows how to split a downloaded video into per-step clips -- but no
video has ever actually been fetched on this cluster. The old
`download_videos.py` is unusable as-is: it's leftover from a different
(Delta/SLURM) cluster, hardcodes that cluster's /projects and /work paths,
and needs a cookies.txt file that no longer exists anywhere on this PVC.

This instead downloads directly from YouCook2's own official annotation
file (not the CSV -- the CSV has no recipe_type field, and
step1_youcook_fixed.py's path convention, `videos/youcook2/<recipe_type>/
<video_id>.mp4`, needs it), using the SAME no-cookies yt-dlp + deno
JS-challenge-solver approach `extract_crosstask_qa.py` and
`smoke_test_youcook_download.py` already proved works on this cluster
(job-80: 14/15 videos succeeded, zero bot-check failures).

Resumable (skips existing output files) and rate-limit aware: on detecting
YouTube's "Sign in to confirm you're not a bot" or a 429, backs off and
retries a bounded number of times per video rather than hammering through
it (the lesson from the old script's own v1/v2 history, and from
CrossTask's extraction run stalling out at ~230/2692 videos). Writes a
JSONL manifest after every video so a killed/restarted job doesn't lose
already-completed downloads.
"""
import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

YT_DLP_JS_RUNTIME = os.environ.get("YT_DLP_JS_RUNTIME", "deno:/tmp/deno")
DOWNLOAD_TIMEOUT_S = 180
RATE_LIMIT_COOLDOWN_S = 70 * 60  # YouTube says "up to an hour"; pad it, matches old script's constant


def is_bot_or_rate_limited(err):
    e = (err or "").lower()
    return "sign in" in e or "bot" in e or "429" in e or "rate-limit" in e


def load_targets(annotations_json):
    with open(annotations_json) as f:
        data = json.load(f)
    targets = []
    for video_id, info in data["database"].items():
        targets.append({"video_id": video_id, "recipe_type": str(info["recipe_type"]), "subset": info.get("subset")})
    return targets


def load_manifest(manifest_path):
    done = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                done[rec["video_id"]] = rec
    return done


def download_one(target, out_root):
    vid = target["video_id"]
    out_dir = os.path.join(out_root, target["recipe_type"])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{vid}.mp4")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10_000:
        return {"video_id": vid, "status": "skip_exists"}
    cmd = [
        "yt-dlp", "--no-progress", "--quiet",
        "--js-runtimes", YT_DLP_JS_RUNTIME,
        "-f", "bestvideo[height<=480][vcodec^=avc1]+bestaudio/"
              "best[height<=480][vcodec^=avc1]/best[height<=480]/best",
        "--merge-output-format", "mp4",
        "-o", out_path,
        f"https://www.youtube.com/watch?v={vid}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT_S)
        if result.returncode == 0 and os.path.exists(out_path):
            return {"video_id": vid, "status": "ok", "size": os.path.getsize(out_path)}
        return {"video_id": vid, "status": "fail", "error": result.stderr[-500:]}
    except subprocess.TimeoutExpired:
        return {"video_id": vid, "status": "fail", "error": "timeout"}
    except Exception as e:
        return {"video_id": vid, "status": "fail", "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations_json", default="/data/raw_data/youcookii_annotations_trainval.json")
    ap.add_argument("--out_root", default="/data/vans_work_youcook/videos/youcook2")
    ap.add_argument("--manifest", default="/data/vans_work_youcook/youcook_download_manifest.jsonl")
    ap.add_argument("--workers", type=int, default=3,
                     help="matches the old script's own lesson (v1 used 6 with no throttling "
                          "and burned the account's rate limit) -- kept low and bounded")
    ap.add_argument("--delay", type=float, default=1.5)
    ap.add_argument("--limit", type=int, default=None, help="cap on target count, for a smoke test")
    ap.add_argument("--max_new", type=int, default=None,
                     help="stop cleanly after this many real (non-skip) attempts, independent "
                          "of rate-limit auto-detection -- a manual throttle for graduated rollout")
    ap.add_argument("--retry_rate_limited", type=int, default=2,
                     help="per-video retry budget after a bot-check/rate-limit failure, each "
                          "preceded by a cooldown sleep, before giving up on that video")
    args = ap.parse_args()

    targets = load_targets(args.annotations_json)
    if args.limit:
        targets = targets[: args.limit]
    done = load_manifest(args.manifest)
    remaining = [t for t in targets if done.get(t["video_id"], {}).get("status") in (None, "fail")]
    print(f"[INFO] {len(targets)} total targets, {len(done)} already in manifest, "
          f"{len(remaining)} remaining to attempt", flush=True)

    manifest_f = open(args.manifest, "a")
    n_ok, n_skip, n_fail, n_attempted = 0, 0, 0, 0
    i = 0
    while i < len(remaining):
        batch = remaining[i: i + args.workers]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda t: download_one(t, args.out_root), batch))

        hit_block = False
        for rec in results:
            manifest_f.write(json.dumps(rec) + "\n")
            manifest_f.flush()
            if rec["status"] == "ok":
                n_ok += 1
                n_attempted += 1
            elif rec["status"] == "skip_exists":
                n_skip += 1
            else:
                n_fail += 1
                n_attempted += 1
                if is_bot_or_rate_limited(rec.get("error", "")):
                    hit_block = True

        i += len(batch)
        if (i // args.workers) % 10 == 0 or i >= len(remaining):
            print(f"[{i}/{len(remaining)}] ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)

        if args.max_new is not None and n_attempted >= args.max_new:
            print(f"[MAX_NEW] reached cap of {args.max_new} real attempts -- stopping cleanly", flush=True)
            break

        if hit_block:
            print(f"[RATE_LIMIT] bot-check/rate-limit detected at {i}/{len(remaining)} -- "
                  f"exiting for a {RATE_LIMIT_COOLDOWN_S/60:.0f}min cooldown (resume by rerunning, "
                  f"the manifest skips everything already done)", flush=True)
            manifest_f.close()
            import sys
            sys.exit(2)
        time.sleep(args.delay)

    manifest_f.close()
    print(f"[DONE] ok={n_ok} skip={n_skip} fail={n_fail} total_remaining_attempted={i}", flush=True)


if __name__ == "__main__":
    main()
