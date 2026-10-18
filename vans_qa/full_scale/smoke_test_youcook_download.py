"""Pure download-only smoke test: does yt-dlp (+ the deno JS-challenge
solver, no cookies) still get through YouTube's bot-check right now, or
does it hit "Sign in to confirm you're not a bot" immediately the way
CrossTask's extraction did after ~230/2692 videos (job-59-derived full run)?

Deliberately does NOT do any ffmpeg clip-trimming or V-JEPA encoding --
step1_youcook_fixed.py / extract_vjepa_features_full.py already cover that
once downloading itself is confirmed viable again. This only answers "is
the YouTube-side rate limit/bot-check still blocking us," using the exact
same no-cookies yt-dlp + deno command extract_crosstask_qa.py already
proved works (when it works) on this cluster.

Video ids come from VANS-DATA_YouCook.csv's input_video_path column
(format "/<video_id>/<step>.mp4"), deduped, in file order (NOT YouCook2's
own train/val split -- this is a pure connectivity probe, not a real
extraction run).
"""
import argparse
import csv
import os
import subprocess
import time

YT_DLP_JS_RUNTIME = os.environ.get("YT_DLP_JS_RUNTIME", "deno:/tmp/deno")
DOWNLOAD_TIMEOUT_S = 120


def extract_video_ids(csv_path, limit):
    seen = []
    seen_set = set()
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row.get("input_video_path") or ""
            parts = path.split("/")
            if len(parts) < 2 or not parts[1]:
                continue
            vid = parts[1]
            if vid not in seen_set:
                seen_set.add(vid)
                seen.append(vid)
            if len(seen) >= limit:
                break
    return seen


def try_download(video_id, out_dir):
    out_path = os.path.join(out_dir, f"{video_id}.mp4")
    cmd = [
        "yt-dlp", "--no-progress", "--quiet",
        "--js-runtimes", YT_DLP_JS_RUNTIME,
        "-f", "bestvideo[height<=480][vcodec^=avc1]+bestaudio/"
              "best[height<=480][vcodec^=avc1]/best[height<=480]/best",
        "--merge-output-format", "mp4",
        "-o", out_path,
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT_S)
        if result.returncode == 0 and os.path.exists(out_path):
            return "ok", None
        return "fail", result.stderr[-500:]
    except subprocess.TimeoutExpired:
        return "fail", "timeout"
    except Exception as e:
        return "fail", f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv_path", default="/data/hf_data/VANS-DATA_YouCook.csv")
    ap.add_argument("--out_dir", default="/data/vans_work_youcook/smoke_download")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--delay", type=float, default=2.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    video_ids = extract_video_ids(args.csv_path, args.limit)
    print(f"[INFO] probing {len(video_ids)} unique video ids from {args.csv_path}", flush=True)

    n_ok, n_fail, n_bot = 0, 0, 0
    for i, vid in enumerate(video_ids, 1):
        status, err = try_download(vid, args.out_dir)
        if status == "ok":
            n_ok += 1
            print(f"[{i}/{len(video_ids)}] {vid}: OK", flush=True)
        else:
            n_fail += 1
            is_bot = "sign in" in (err or "").lower() or "bot" in (err or "").lower()
            if is_bot:
                n_bot += 1
            print(f"[{i}/{len(video_ids)}] {vid}: FAIL ({'BOT-CHECK' if is_bot else 'other'}) {err[-200:] if err else ''}", flush=True)
        time.sleep(args.delay)

    print(f"[SUMMARY] ok={n_ok} fail={n_fail} bot_check_fails={n_bot} total={len(video_ids)}", flush=True)
    if n_ok == 0 and n_bot > 0:
        print("[CONCLUSION] still fully blocked by YouTube's bot-check -- rate limit has NOT cleared", flush=True)
    elif n_bot == 0 and n_fail < len(video_ids):
        print("[CONCLUSION] no bot-check failures seen -- looks clear for now, "
              "but CrossTask also started clean before failing at ~230/2692, so this is "
              "only evidence for THIS batch size, not a guarantee at full YouCook2 scale", flush=True)
    else:
        print("[CONCLUSION] mixed result -- some bot-check failures, some successes; "
              "inconclusive at this sample size", flush=True)


if __name__ == "__main__":
    main()
