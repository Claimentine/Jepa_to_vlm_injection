#!/usr/bin/env bash
# VANS 2-choice "what happens next" QA pilot (N=499, directional signal
# only -- see README_pilot_caveats.md). Smoke test first, then full run:
# train jepa + random (temporal pooling, validated on EgoDex), eval jepa /
# random / none (frozen baseline) via generate(), paired McNemar scoring.
#
# Usage: sbatch --account=bdqf-dtai-gh run_vans_qa_pilot.sbatch

#SBATCH --job-name=vans_qa_pilot
#SBATCH --account=bdqf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail
cd /projects/bhay/william/ruixin/vans_world_model

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen3vl

if [[ ! -f qa_split.json ]]; then
  python build_qa_pairs.py
fi

echo "=========================================="
echo "[SMOKE TEST] jepa + random, tiny run"
echo "=========================================="
for injection in jepa random; do
  python train_vans_qa_injection.py --injection "${injection}" \
    --max_train_items 8 --val_every_steps 8 --epochs 1 \
    --out_dir qa_injection_runs_smoke
done
echo "[SMOKE TEST] passed."

echo "=========================================="
echo "[FULL RUN] training jepa + random"
echo "=========================================="
for injection in jepa random; do
  python train_vans_qa_injection.py --injection "${injection}" \
    --epochs "${EPOCHS:-6}" --val_every_steps "${VAL_EVERY:-50}" \
    --out_dir qa_injection_runs
done

echo "=========================================="
echo "[FULL RUN] held-out test evaluation (generate()-based)"
echo "=========================================="
python eval_vans_qa_injection.py --injection none   --out qa_eval_none.jsonl
python eval_vans_qa_injection.py --injection jepa   --checkpoint qa_injection_runs/jepa/best.pt   --out qa_eval_jepa.jsonl
python eval_vans_qa_injection.py --injection random --checkpoint qa_injection_runs/random/best.pt --out qa_eval_random.jsonl

echo "=========================================="
echo "[SCORING]"
echo "=========================================="
python score_qa_pilot.py --results qa_eval_none.jsonl qa_eval_jepa.jsonl qa_eval_random.jsonl | tee qa_pilot_report.txt

echo "[INFO] done."
