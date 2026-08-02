# External data & checkpoints (not in this repo)

Nothing under `data/`, `runs/`, `*.npz`, `*.mp4`, model weights, etc. is
committed (see `.gitignore`) -- all of it needs to exist on whatever cluster
this runs on next, at the paths given by the env vars below (or the current
Delta defaults baked into each script, which won't exist on a new machine).

## Models

| what | where it comes from | env var |
|---|---|---|
| V-JEPA2 ViT-L checkpoint (`vitl.pt`) | Meta's V-JEPA2 release | `VJEPA2_CKPT` |
| Qwen3-VL-2B-Thinking | `Qwen/Qwen3-VL-2B-Thinking` on HF Hub, public, no token needed | n/a (HF model id, downloaded automatically) |
| ThinkJEPA checkout | `https://github.com/Hai-chao-Zhang/ThinkJEPA` -- clone it, then run `scripts/patch_thinkjepa.sh` (two local fixes never upstreamed, see that script) | `THINKJEPA_ROOT` |

## EgoDex probe data

| what | env var |
|---|---|
| EgoDex video cache (raw episodes) | `EGODEX_DATA_ROOT` / `EGODEX_VIDEO_ROOT` |
| Cached V-JEPA2 feature `.npz` per episode | `EGODEX_NPZ_ROOT` |
| Kinematic/HDF5 signals (deprioritized proxy task, kept for reference) | `EGODEX_HDF5_ROOT` |

Built via `egodex_probe/sample_probe_episodes.py` -> `build_classification_split.py`.
2000 episodes, EgoDex "part2" subset, seed 42, 0.9 train/holdout ratio (see
the directory name baked into the current defaults).

## VANS QA data

Demo scale (500 pairs) came bundled as `VANS-DATA_demo.zip` from the
`KlingTeam/VANS` HF dataset (public, no token needed).

Full scale (12,032 matched pairs) is built from:
1. `VANS-DATA_COIN.csv` + `VANS-DATA_YouCook.csv` (same HF dataset) -- QA
   text + which COIN/YouCook2 step-segments to use.
2. Raw COIN (`COIN.json`) / YouCook2 (`youcookii_annotations_trainval.json`)
   annotation files -- step-level timestamps, used to cut the downloaded
   long videos into per-step clips.
3. The actual COIN/YouCook2 long-form videos, bulk-downloaded from YouTube
   via `vans_qa/full_scale/download_videos.py` (yt-dlp). **This step is the
   real bottleneck** -- YouTube's bot-detection rate-limits sustained bulk
   downloading at roughly the IP level (switching accounts didn't help past
   a certain point). Current state on Delta: ~6,296/11,453 target videos
   downloaded (~55%), yielding 12,032/28,226 possible QA pairs. Expect the
   same wall on a new cluster; budget for it or find an alternate video
   source.

`VANS_ROOT` (small metadata/config, e.g. `qa_split_full.json`) and
`VANS_WORK_ROOT` (large binary data: raw videos, clips, feature caches) are
intentionally separate env vars -- on Delta the latter was moved onto a
different, larger filesystem mid-project after a quota crisis. On a fresh
setup they can point at the same place.

## Backing up processed data

A private HF dataset repo (`Claimentine/vans-injection-data`) has a partial
backup of `clips/` (complete) and the V-JEPA2 / VLM-guidance feature caches
(uploading is slow -- HF rate-limits to 128 commits/hour and there are
30k+ small files, so `huggingface_hub.upload_large_folder` takes many
hours). Not wired into any script here; was done ad hoc with
`huggingface_hub.HfApi`.
