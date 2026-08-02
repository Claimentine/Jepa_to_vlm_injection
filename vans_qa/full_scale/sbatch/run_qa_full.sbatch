#!/usr/bin/env bash
# Full-scale (~12K matched QA pairs vs demo's 499) VANS QA pipeline:
# V-JEPA2 features (18,147 unique clips) + VLM guidance (12,032 in_clips)
# + jepa/random training + generate()-based eval, same protocol as the
# demo pilot for direct comparison.
#
# Usage: sbatch --account=bdqf-dtai-gh run_qa_full.sbatch

#SBATCH --job-name=vans_qa_full
#SBATCH --account=bdqf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail
cd /projects/bhay/william/ruixin/vans_world_model/raw_data

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen3vl

echo "=========================================="
echo "[SMOKE TEST] small caps on everything"
echo "=========================================="
python extract_vjepa_features_full.py --limit 16
python extract_vlm_guidance_full.py --limit 16
python train_qa_full.py --injection jepa --max_train_items 8 --val_every_steps 8 --epochs 1 --out_dir qa_full_runs_smoke
python train_qa_full.py --injection random --max_train_items 8 --val_every_steps 8 --epochs 1 --out_dir qa_full_runs_smoke
echo "[SMOKE TEST] passed."

echo "=========================================="
echo "[FULL RUN] feature extraction (this is the slow part: 18,147 + 12,032 clips)"
echo "=========================================="
python extract_vjepa_features_full.py
python extract_vlm_guidance_full.py

echo "=========================================="
echo "[FULL RUN] training jepa + random"
echo "=========================================="
python train_qa_full.py --injection jepa   --epochs "${EPOCHS:-2}" --val_every_steps "${VAL_EVERY:-500}" --out_dir qa_full_runs
python train_qa_full.py --injection random --epochs "${EPOCHS:-2}" --val_every_steps "${VAL_EVERY:-500}" --out_dir qa_full_runs

echo "=========================================="
echo "[FULL RUN] held-out test evaluation (generate()-based, 300-item cap, comparable to demo pilot)"
echo "=========================================="
python eval_qa_full.py --injection none   --max_test_items 300
python eval_qa_full.py --injection jepa   --checkpoint qa_full_runs/jepa/best.pt   --max_test_items 300
python eval_qa_full.py --injection random --checkpoint qa_full_runs/random/best.pt --max_test_items 300

echo "=========================================="
echo "[SCORING]"
echo "=========================================="
python /projects/bhay/william/ruixin/vans_world_model/score_qa_pilot.py \
  --results qa_eval_full_none.jsonl qa_eval_full_jepa.jsonl qa_eval_full_random.jsonl \
  | tee qa_full_report.txt

echo "[INFO] done."
