"""
Bulk-downloads the 11,453 target COIN/YouCook2 videos referenced by the
VANS CSVs, using yt-dlp + the cookies.txt the user exported (bypasses
YouTube's bot-check) + Node.js for the JS-challenge solver (yt-dlp-ejs).

Format: best[height<=480], single-file progressive mp4 -- deliberately
avoids formats that need separate video+audio muxing since ffmpeg isn't
available on this machine.

Output layout matches what COIN's own official download_videos.py and
VANS's data_pipeline/step1.py expect: videos/{recipe_type}/{video_id}.mp4
-- so the existing (untouched) step1.py can be reused unmodified to split
these into the per-step clips the VANS-DATA CSVs reference.

Resumable: skips videos whose output file already exists. Writes a
manifest (download_manifest.jsonl) recording success/failure per video so
a re-run only retries what's missing.
"""
import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = os.environ.get("VANS_ROOT", "/projects/bhay/william/ruixin/vans_world_model")
WORK_BASE = os.environ.get("VANS_WORK_ROOT", "/work/nvme/bdqf/yli8/vans_raw_data")  # relocated off the near-full /projects/bhay allocation
COOKIES = os.path.join(BASE, "raw_data/cookies_2.txt")  # fresh account -- old one's rate-limit never fully cleared
YTDLP = os.environ.get("YTDLP_BIN", "/u/yli8/.conda/envs/thinkjepa/bin/yt-dlp")
OUT_ROOT = os.path.join(WORK_BASE, "videos")  # new downloads go straight to /work, never touch /projects again
MANIFEST_PATH = os.path.join(BASE, "raw_data/download_manifest.jsonl")  # small text log, fine to keep on /projects

# v1 (6 workers, no throttling, no rate-limit handling) burned through the
# account's YouTube rate limit after ~10k requests: 6887/7082 failures were
# "Your account has been rate-limited by YouTube for up to an hour", not
# real unavailability (~95 genuinely gone). v2: much lower concurrency, a
# per-request delay, and -- critically -- back off for a full cooldown the
# moment a rate-limit error is seen instead of continuing to hammer it.
REQUEST_DELAY_SEC = 2.0
RATE_LIMIT_COOLDOWN_SEC = 70 * 60  # YouTube says "up to an hour"; pad it


def is_rate_limited(err: str) -> bool:
    return "rate-limited" in (err or "").lower()


def download_one(target):
    vid = target["video_id"]
    recipe_dir = os.path.join(OUT_ROOT, target["dataset"], str(target["recipe_type"]))
    os.makedirs(recipe_dir, exist_ok=True)
    out_path = os.path.join(recipe_dir, f"{vid}.mp4")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 10_000:
        return {"video_id": vid, "status": "skip_exists", "path": out_path}

    cmd = [
        YTDLP,
        "--cookies", COOKIES,
        "--js-runtimes", "node",
        "-f", "best[height<=480]",
        "--no-warnings",
        "--retries", "2",
        "--socket-timeout", "30",
        "-o", out_path,
        target["url"],
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode == 0 and os.path.exists(out_path):
            return {"video_id": vid, "status": "ok", "path": out_path, "size": os.path.getsize(out_path)}
        return {"video_id": vid, "status": "fail", "error": result.stderr[-300:]}
    except subprocess.TimeoutExpired:
        return {"video_id": vid, "status": "fail", "error": "timeout"}
    except Exception as e:
        return {"video_id": vid, "status": "fail", "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2,
                    help="v1 used 6 with no throttling and burned the account's rate limit; "
                         "default lowered to 2")
    ap.add_argument("--delay", type=float, default=REQUEST_DELAY_SEC)
    ap.add_argument("--limit", type=int, default=None, help="cap for smoke tests")
    ap.add_argument("--max_new", type=int, default=None,
                     help="stop cleanly (exit 0, not a rate-limit exit) after this many real "
                          "(non-skip) attempts -- a manual throttle for graduated rollout on a "
                          "fresh account, independent of rate-limit auto-detection")
    args = ap.parse_args()

    with open(os.path.join(BASE, "raw_data/download_targets.json")) as f:
        targets = json.load(f)
    if args.limit:
        targets = targets[: args.limit]

    print(f"[INFO] {len(targets)} targets, {args.workers} workers, {args.delay}s delay/request, "
          f"max_new={args.max_new}")
    os.makedirs(OUT_ROOT, exist_ok=True)

    n_ok, n_skip, n_fail = 0, 0, 0
    i = 0
    with open(MANIFEST_PATH, "a") as manifest:
        while i < len(targets):
            batch = targets[i : i + args.workers]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                results = list(pool.map(download_one, batch))

            hit_rate_limit = False
            for rec in results:
                manifest.write(json.dumps(rec) + "\n")
                manifest.flush()
                if rec["status"] == "ok":
                    n_ok += 1
                elif rec["status"] == "skip_exists":
                    n_skip += 1
                else:
                    n_fail += 1
                    if is_rate_limited(rec.get("error", "")):
                        hit_rate_limit = True

            i += len(batch)
            if (i // args.workers) % 25 == 0 or i >= len(targets):
                print(f"[{i}/{len(targets)}] ok={n_ok} skip={n_skip} fail={n_fail}")

            if args.max_new is not None and (n_ok + n_fail) >= args.max_new:
                print(f"[MAX_NEW] reached cap of {args.max_new} real attempts (ok={n_ok} fail={n_fail}) "
                      f"-- stopping cleanly for manual review, not a rate-limit exit", flush=True)
                return

            if hit_rate_limit:
                # v2 slept in-process for 70 min here -- fragile, got killed
                # mid-sleep when the session/background task was torn down.
                # v3: exit immediately (exit code 2 signals "rate limited,
                # resume later") and let an outer shell loop own the wait,
                # so no single process has to stay alive for an hour.
                print(f"[RATE_LIMIT] hit at {i}/{len(targets)} -- exiting for cooldown "
                      f"({RATE_LIMIT_COOLDOWN_SEC/60:.0f} min recommended before resuming)", flush=True)
                import sys
                sys.exit(2)
            time.sleep(args.delay)

    print(f"[DONE] ok={n_ok} skip={n_skip} fail={n_fail} total={len(targets)}")


if __name__ == "__main__":
    main()
