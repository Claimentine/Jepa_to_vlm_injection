#!/usr/bin/env bash
# v2: temporal (spatial-only) pooling + class-balanced sampling, testing
# whether these fix the v1 finding (random beat jepa: macro 0.626 vs 0.440,
# see injection_probe_report_v2.txt). Only reruns jepa/random -- linear_probe
# doesn't use the pooling adapter at all, so its v1 number is still valid
# as a reference point and isn't rerun here.
#
# Usage:
#   sbatch --account=bdqf-dtai-gh run_jepa_injection_probe_v2.sbatch
#   sbatch --account=bdjg-dtai-gh run_jepa_injection_probe_v2.sbatch

#SBATCH --job-name=jepa_injection_v2
#SBATCH --account=bdqf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail
cd /projects/bhay/william/ruixin/semantic_probe

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen3vl

echo "=========================================="
echo "[SMOKE TEST] tiny run, jepa+random, temporal pooling, to catch integration bugs cheaply"
echo "=========================================="
for injection in jepa random; do
  echo "--- smoke test: ${injection} ---"
  python train_jepa_injection_probe.py \
    --injection "${injection}" \
    --pooling temporal \
    --max_train_episodes 8 \
    --max_val_episodes 8 \
    --val_every_steps 8 \
    --epochs 1 \
    --out_dir jepa_injection_runs_v2_smoke
done
echo "[SMOKE TEST] passed."

echo "=========================================="
echo "[FULL RUN] training jepa + random with temporal pooling + class-balanced sampling"
echo "=========================================="
for injection in jepa random; do
  echo "--- training: ${injection} ---"
  python train_jepa_injection_probe.py \
    --injection "${injection}" \
    --pooling temporal \
    --epochs "${EPOCHS:-3}" \
    --val_every_steps "${VAL_EVERY:-200}" \
    --out_dir jepa_injection_runs_v2
done

echo "=========================================="
echo "[FULL RUN] held-out test evaluation (generate()-based, same protocol as v1 for direct comparison)"
echo "=========================================="
python eval_jepa_injection_probe.py --injection jepa   --checkpoint jepa_injection_runs_v2/jepa/best.pt   --out injection_eval_jepa_v2.jsonl
python eval_jepa_injection_probe.py --injection random --checkpoint jepa_injection_runs_v2/random/best.pt --out injection_eval_random_v2.jsonl

echo "=========================================="
echo "[SCORING] v2 (temporal pooling) vs v1 (global pooling, from injection_probe_report_v2.txt)"
echo "=========================================="
for f in injection_eval_jepa_v2.jsonl injection_eval_random_v2.jsonl; do
  echo "--- ${f} ---"
  python score_task_recognition_probe.py --results "${f}"
done | tee injection_probe_report_v3_temporal.txt

echo "[INFO] done."
