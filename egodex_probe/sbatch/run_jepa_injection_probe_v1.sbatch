#!/usr/bin/env bash
# JEPA-injection classification experiment: smoke test first (cheap, catches
# integration bugs in the hook-based soft-prompt injection before spending
# real GPU time), then the full 3-condition train+eval if the smoke test
# passes.
#
# Usage:
#   sbatch --account=bdqf-dtai-gh run_jepa_injection_probe.sbatch
#   sbatch --account=bdjg-dtai-gh run_jepa_injection_probe.sbatch

#SBATCH --job-name=jepa_injection_probe
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

if [[ ! -f classification_split.json ]]; then
  echo "[INFO] building classification split ..."
  python build_classification_split.py
fi

echo "=========================================="
echo "[SMOKE TEST] tiny run, all 3 conditions, to catch integration bugs cheaply"
echo "=========================================="
for injection in jepa random linear_probe; do
  echo "--- smoke test: ${injection} ---"
  python train_jepa_injection_probe.py \
    --injection "${injection}" \
    --max_train_episodes 8 \
    --max_val_episodes 8 \
    --val_every_steps 8 \
    --epochs 1 \
    --out_dir jepa_injection_runs_smoke
done
echo "[SMOKE TEST] all 3 conditions completed without error."

echo "=========================================="
echo "[FULL RUN] training all 3 conditions"
echo "=========================================="
for injection in jepa random linear_probe; do
  echo "--- training: ${injection} ---"
  python train_jepa_injection_probe.py \
    --injection "${injection}" \
    --epochs "${EPOCHS:-3}" \
    --val_every_steps "${VAL_EVERY:-200}" \
    --out_dir jepa_injection_runs
done

echo "=========================================="
echo "[FULL RUN] held-out test evaluation (generate()-based, comparable to zero-shot baseline)"
echo "=========================================="
python eval_jepa_injection_probe.py --injection jepa         --checkpoint jepa_injection_runs/jepa/best.pt         --out injection_eval_jepa.jsonl
python eval_jepa_injection_probe.py --injection random       --checkpoint jepa_injection_runs/random/best.pt       --out injection_eval_random.jsonl
python eval_jepa_injection_probe.py --injection linear_probe --checkpoint jepa_injection_runs/linear_probe/best.pt --out injection_eval_linear_probe.jsonl

echo "=========================================="
echo "[SCORING]"
echo "=========================================="
for f in injection_eval_jepa.jsonl injection_eval_random.jsonl injection_eval_linear_probe.jsonl; do
  echo "--- ${f} ---"
  python score_task_recognition_probe.py --results "${f}"
done | tee injection_probe_report.txt

echo "[INFO] done."
