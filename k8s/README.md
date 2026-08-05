# Nautilus workflow: env + VANS data setup

Scope of this first pass: conda environments + the VANS-Data-100K demo/CSV
data (public, no HF token). EgoDex and the full-scale COIN/YouCook2 video
pipeline are explicitly out of scope for now.

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
```

The two Jobs are independent (different PVCs) and can run concurrently.
Both are CPU-only -- no GPU is requested anywhere in this pass, so there's
no fair-share GPU-utilization risk to worry about yet.

Check status / logs:

```bash
kubectl get jobs -n williamli-scvl
kubectl logs -f job/ruixin-setup-env -n williamli-scvl
kubectl logs -f job/ruixin-fetch-vans-data -n williamli-scvl
```

**Delete the Jobs once they've completed** -- Jobs don't clean up their own
pods (Golden Rule #4):

```bash
kubectl delete job ruixin-setup-env ruixin-fetch-vans-data -n williamli-scvl
```

## What each file does

| file | creates | purpose |
|---|---|---|
| `pvc-conda-rbd.yaml` | `ruixin-conda-vol` (rook-ceph-block, 50Gi) | conda envs, ThinkJEPA checkout |
| `pvc-data-cephfs.yaml` | `jepa-vlm-injection-data-vol` (rook-cephfs, 200Gi) | VANS CSVs + demo clips |
| `job-01-setup-env.yaml` | Job `ruixin-setup-env` | builds `thinkjepa`/`qwen3vl` conda envs onto the conda PVC, clones+patches ThinkJEPA |
| `job-02-fetch-vans-data.yaml` | Job `ruixin-fetch-vans-data` | downloads VANS-Data-100K CSVs + demo bundle from `KlingTeam/VANS` on HF into the data PVC |

Both Jobs are idempotent -- re-applying/re-running skips work that's
already done (checks for existing conda env dirs / downloaded files before
redoing them).

## Next steps (not yet built)

- **Full-scale VANS pipeline**: `VANS-DATA_COIN.csv` / `VANS-DATA_YouCook.csv`
  are now on the data PVC, but the raw COIN/YouCook2 long-form videos still
  need bulk download via `yt-dlp` (`vans_qa/full_scale/download_videos.py`)
  -- per the repo's own docs this is the real bottleneck (YouTube
  bot-detection rate-limiting), budget for it separately as a long-running
  CPU Job.
- **V-JEPA2 feature extraction / VLM guidance caching**
  (`extract_vjepa_features*.py`, `extract_vlm_guidance*.py`) needs a GPU
  Job using Template 3 from the onboarding guide -- not written yet, since
  there's no data to extract features from until the video pipeline above
  runs.
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
  especially before/after the first GPU Job you add later.
