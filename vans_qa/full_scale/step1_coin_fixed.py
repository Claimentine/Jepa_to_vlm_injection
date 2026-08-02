"""
Same logic as VANS/data_pipeline/step1.py, except it looks up the source
video under `class` (the string task name, e.g. "PractiseKarate") instead of
the numeric `recipe_type` id -- matching how download_videos.py actually
named the COIN output folders (it used coin_db[vid]["class"], not the
numeric id). YouCook2 didn't have this mismatch (build_target_list.py used
the numeric recipe_type for it directly), so only COIN needs this variant.
"""
import json
import math
import multiprocessing
import os
import subprocess
from multiprocessing import Pool

FFMPEG = os.environ.get("FFMPEG_BIN", "/u/yli8/.conda/envs/thinkjepa/bin/ffmpeg")


def extract_single_segment(input_path, start_time, end_time, output_path):
    if os.path.exists(output_path):
        return True
    duration = end_time - start_time
    cmd = [FFMPEG, "-ss", str(start_time), "-i", input_path, "-t", str(duration),
           "-c", "copy", "-y", output_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except Exception:
        return False


def process_task_chunk(task_chunk):
    n = 0
    for task in task_chunk:
        os.makedirs(os.path.dirname(task["output_path"]), exist_ok=True)
        if extract_single_segment(task["input_path"], task["start_time"], task["end_time"], task["output_path"]):
            n += 1
    return n


def main():
    json_path = os.path.join(os.environ.get("VANS_ROOT", "/projects/bhay/william/ruixin/vans_world_model"), "raw_data/COIN.json")
    video_dir = os.path.join(os.environ.get("VANS_WORK_ROOT", "/work/nvme/bdqf/yli8/vans_raw_data"), "videos/coin")
    output_dir = os.path.join(os.environ.get("VANS_WORK_ROOT", "/work/nvme/bdqf/yli8/vans_raw_data"), "clips")
    num_processes = 8

    data = json.load(open(json_path))
    tasks = []
    n_missing = 0
    for video_id, info in data["database"].items():
        cls = info["class"]  # <-- the fix: class string, not numeric recipe_type
        original_video_path = os.path.join(video_dir, cls, f"{video_id}.mp4")
        if not os.path.exists(original_video_path):
            n_missing += 1
            continue
        for ann in info["annotation"]:
            seg = ann["segment"]
            tasks.append({
                "input_path": original_video_path,
                "start_time": seg[0],
                "end_time": seg[1],
                "output_path": os.path.join(output_dir, video_id, f"{ann['id']}.mp4"),
            })

    print(f"[INFO] {len(tasks)} segments to extract, {n_missing} source videos not downloaded (skipped)", flush=True)
    if not tasks:
        return

    n_proc = min(num_processes, len(tasks))
    chunk_size = math.ceil(len(tasks) / n_proc)
    chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]
    with Pool(processes=n_proc) as pool:
        results = pool.map(process_task_chunk, chunks)
    print(f"[DONE] success={sum(results)} total={len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
