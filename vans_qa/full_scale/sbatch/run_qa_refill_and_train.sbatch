#!/usr/bin/env bash
# Job 2761920 crashed 5h40m in, during val evaluation, on a corrupted vjepa
# feature .npz (BadZipFile "Overlapped entries" -- 471 files truncated,
# apparently during the /work relocation which only verified file *counts*,
# not content). Those 471 were deleted; this job backfills just them
# (extract_vjepa_features_full.py skips files that already exist, so this
# is fast -- not a full 18K-clip re-extraction), then runs training + eval.
# evaluate_teacher_forced() in train_qa_full.py was also patched to skip a
# corrupt feature file instead of crashing the whole job, as defense in depth.
#
# Usage: sbatch --account=bdqf-dtai-gh run_qa_refill_and_train.sbatch

#SBATCH --job-name=vans_qa_refill
#SBATCH --account=bdqf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail
cd /projects/bhay/william/ruixin/vans_world_model/raw_data

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen3vl

echo "=========================================="
echo "[REFILL] backfilling 471 deleted/corrupted vjepa features (skips the ~17,666 already-good ones)"
echo "=========================================="
python extract_vjepa_features_full.py

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
