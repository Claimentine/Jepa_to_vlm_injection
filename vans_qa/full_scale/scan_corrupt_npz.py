"""
Scans vjepa_cache_full/ and vlm_guidance_cache_full/ for corrupted .npz files
(the training job crashed on a BadZipFile "Overlapped entries" error while
reading a cached feature file -- likely a handful of files left truncated
by an interrupted extraction run or concurrent-write race). Actually opens
and reads every array in each file (np.load is lazy, so a simple np.load()
call without touching the arrays would not catch this).
"""
import os
import sys
import zipfile
import numpy as np
from multiprocessing import Pool

ROOTS = [
    os.path.join(os.environ.get("VANS_WORK_ROOT", "/work/nvme/bdqf/yli8/vans_raw_data"), "vjepa_cache_full"),
    os.path.join(os.environ.get("VANS_WORK_ROOT", "/work/nvme/bdqf/yli8/vans_raw_data"), "vlm_guidance_cache_full"),
]


def check_one(path):
    try:
        with np.load(path) as d:
            for k in d.files:
                _ = d[k][()] if d[k].shape == () else d[k][0:1]
        return None
    except Exception as e:
        return (path, f"{type(e).__name__}: {e}")


def main():
    all_files = []
    for root in ROOTS:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(".npz"):
                    all_files.append(os.path.join(dirpath, fn))
    print(f"[INFO] scanning {len(all_files)} .npz files across {len(ROOTS)} roots", flush=True)

    bad = []
    with Pool(processes=16) as pool:
        for i, result in enumerate(pool.imap_unordered(check_one, all_files, chunksize=64)):
            if result is not None:
                bad.append(result)
                print(f"[BAD] {result[0]}: {result[1]}", flush=True)
            if (i + 1) % 2000 == 0:
                print(f"[PROGRESS] {i + 1}/{len(all_files)} scanned, {len(bad)} bad so far", flush=True)

    print(f"[DONE] scanned {len(all_files)} files, {len(bad)} corrupted", flush=True)
    if bad:
        with open(os.path.join(os.environ.get("VANS_ROOT", "/projects/bhay/william/ruixin/vans_world_model"), "raw_data/corrupt_npz_list.txt"), "w") as f:
            for path, err in bad:
                f.write(f"{path}\t{err}\n")
        print("[INFO] wrote list to corrupt_npz_list.txt", flush=True)


if __name__ == "__main__":
    main()
