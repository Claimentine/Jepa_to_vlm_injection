# Nautilus workflow: env + VANS data setup

Scope: conda environments, VANS-Data-100K demo/CSV data (public, no HF
token), and V-JEPA2 feature extraction for the full-scale VANS QA pipeline.
EgoDex is explicitly out of scope for now.

Namespace: `williamli-scvl`. Run once per shell so you don't have to pass
`-n` every time (still fine to pass it explicitly, per the lab's Rule 5):

```bash
kubectl config set-context nautilus --namespace=williamli-scvl
```

## Apply order

```bash
kubectl apply -f pvc-conda-rbd.yaml -n williamli-scvl
kubectl apply -f pvc-data-cephfs.yaml -n williamli-scvl
kubectl apply -f job-01-setup-env.yaml -n williamli-scvl
kubectl apply -f job-02-fetch-vans-data.yaml -n williamli-scvl
kubectl apply -f job-03-extract-vjepa-features.yaml -n williamli-scvl   # after 01+02 finish; needs a GPU
```

job-01/job-02 are independent (different PVCs) and can run concurrently,
CPU-only. job-03 needs the conda env from job-01 and the clip data from
the (manual, see below) data restore, and requests a GPU.

Check status / logs:

```bash
kubectl get jobs -n williamli-scvl
kubectl logs -f job/ruixin-setup-env -n williamli-scvl
kubectl logs -f job/ruixin-fetch-vans-data -n williamli-scvl
kubectl logs -f job/ruixin-extract-vjepa-features -n williamli-scvl
```

**Delete the Jobs once they've completed** -- Jobs don't clean up their own
pods (Golden Rule #4):

```bash
kubectl delete job ruixin-setup-env ruixin-fetch-vans-data ruixin-extract-vjepa-features -n williamli-scvl
```

## Known state as of 2026-08-11 (read before re-running job-01)

- **job-01 silently half-failed once already.** Both conda envs ended up
  with only base packaging tools (no torch/opencv/etc.) -- `conda env
  create` had created the prefix dir before failing partway through the
  `pip:` section, and the old `[ -d "$ENVS/thinkjepa" ]` idempotency check
  treated that half-built dir as "done" forever. Fixed: the check now looks
  for a `.setup_complete` marker file written only after a successful
  install, and removes any stale partial dir first. Also fixed two real bugs
  found while manually reinstalling: `environment-thinkjepa.yml`'s bare
  `ffmpeg` was resolving to an outdated pytorch-channel build mismatched
  against conda-forge's `openh264` (missing `.so.5` at runtime, not at
  install time) -- pinned to `conda-forge::ffmpeg`. `environment-qwen3vl.yml`
  was missing `torchvision` (`qwen_vl_utils` imports it), which made a bare
  `pip install torchvision` silently upgrade the pinned `torch==2.10.0` ->
  `2.13.0` -- added `torchvision==0.25.0` (the matching pin) explicitly.
- **`/data/hf_data/vans_injection_backup/` (94GB, not created by any script
  here) has already been restored** from the private HF backup mentioned in
  `docs/data_and_checkpoints.md` -- 6,277 clip dirs (20,386 `.mp4`s) + 4,649
  V-JEPA2 feature `.npz` caches + a `qa_split_full.json`, ad hoc, outside
  these Jobs. Its layout is flat (`<video_id>/<n>.mp4`,
  `<video_id>__<n>.npz`), one level off from what the full_scale scripts
  expect (`$VANS_WORK_ROOT/clips/<video_id>/<n>.mp4` and
  `$VANS_WORK_ROOT/vjepa_cache_full/<video_id>__<n>.npz`). Wired up with two
  symlinks (safe to re-run, job-03 also does this itself so it isn't
  load-bearing on manual state):
  ```bash
  mkdir -p /data/vans_work
  ln -sfn /data/hf_data/vans_injection_backup /data/vans_work/clips
  ln -sfn /data/hf_data/vans_injection_backup /data/vans_work/vjepa_cache_full
  ```
  Both symlinks point at the *same* directory -- the two filename schemes
  (`<id>/<n>.mp4` vs `<id>__<n>.npz`) don't collide, so this avoids
  duplicating 94GB. Re-ran `build_qa_pairs_full.py` against this and got
  12,032/28,226 matched pairs, 18,147 unique clips needed, 4,649 (25.6%)
  already have cached V-JEPA2 features -- matches the historical numbers in
  the main README, so the restore looks complete for what it claims to be.
  Ran `vans_qa/full_scale/scan_corrupt_npz.py`: 131/4,649 (2.8%) of those
  cached `.npz` were corrupt (`BadZipFile`/`EOFError`, likely truncated
  during the ad hoc restore) -- listed in `raw_data/corrupt_npz_list.txt`
  and deleted, so job-03's `os.path.exists()` skip check won't mistake them
  for done. job-03 will therefore (re)extract 13,629 clips (131 corrupted +
  13,498 never cached), not 13,498.

## What each file does

| file | creates | purpose |
|---|---|---|
| `pvc-conda-rbd.yaml` | `ruixin-conda-vol` (rook-ceph-block, 50Gi) | conda envs, ThinkJEPA checkout |
| `pvc-data-cephfs.yaml` | `jepa-vlm-injection-data-vol` (rook-cephfs, 200Gi) | VANS CSVs + demo clips |
| `job-01-setup-env.yaml` | Job `ruixin-setup-env` | builds `thinkjepa`/`qwen3vl` conda envs onto the conda PVC, clones+patches ThinkJEPA |
| `job-02-fetch-vans-data.yaml` | Job `ruixin-fetch-vans-data` | downloads VANS-Data-100K CSVs + demo bundle from `KlingTeam/VANS` on HF into the data PVC |
| `job-03-extract-vjepa-features.yaml` | Job `ruixin-extract-vjepa-features` | downloads the V-JEPA2 ViT-L checkpoint (CPU initContainer) + extracts features for whichever of the 18,147 needed clips aren't already cached (GPU) |

All three Jobs are idempotent -- re-applying/re-running skips work that's
already done (checks for existing conda env dirs / downloaded files /
output `.npz` before redoing them).

## Next steps (not yet built)

- **Full-scale VANS pipeline**: `VANS-DATA_COIN.csv` / `VANS-DATA_YouCook.csv`
  are on the data PVC, and most of the raw COIN/YouCook2 clips too (see
  "Known state" above) -- the remaining gap is ~10,109 target videos never
  downloaded (bulk download via `yt-dlp`,
  `vans_qa/full_scale/download_videos.py`) -- per the repo's own docs this
  is the real bottleneck (YouTube bot-detection rate-limiting), budget for
  it separately as a long-running CPU Job.
- **VLM guidance caching** (`extract_vlm_guidance_full.py`) is currently
  **blocked, independent of the checkpoint**: it shells out to
  `$THINKJEPA_ROOT/cache_train/qwen3_cache_extractor.py`, which does not
  exist in the public `Hai-chao-Zhang/ThinkJEPA` checkout (verified --
  `scripts/patch_thinkjepa.sh`'s second patch step assumes it exists and
  just no-ops if it's missing). Needs sourcing that file from somewhere
  before a job-04 can be written.
- **EgoDex** intentionally excluded from this pass (out of scope per current
  instructions; also note the full public EgoDex release is ~2TB, so it'll
  need its own PVC sizing conversation before pulling anything).
- The private HF backup dataset (`Claimentine/vans-injection-data`,
  mentioned in `docs/data_and_checkpoints.md`) needs an HF token -- skipped
  for now per your choice of public-data-only. If you want it later, it
  needs a `kubectl create secret generic hf-token --from-literal=token=...`
  (never commit the token itself to this repo) plus an env var wired into
  whichever Job needs it.

## Reminders from the lab onboarding guide

- **No protected data** on Nautilus, ever (HIPAA/FERPA/PID/etc.) -- not
  applicable to this project's public video data, but worth restating.
- **6-month purge**: both PVCs above are active-computation storage, not
  archival. Move final results/checkpoints to lab-approved long-term
  storage when a project phase wraps up.
- **Never `sleep infinity`** in a Job, and never idle a GPU pod while
  writing/debugging code -- both trigger fair-share violations. Everything
  here runs as `kind: Job` and terminates on its own.
- Check the **Violations** page on the Nautilus portal periodically,
  especially before/after `job-03` (the first GPU Job here) -- run it once
  with `--limit 50` first and check the GPU dashboard before committing to
  the full ~13.5k-clip run (Section 1.2, "Be Truthful").
