#!/usr/bin/env bash
# VANS demo (500 in/out pairs) -> VLM-guided JEPA latent-prediction pipeline.
# Smoke test (8 pairs, cheap) first to catch integration bugs before the
# full run -- same discipline as the earlier task-recognition-probe jobs,
# which caught a real bug this way.
#
# Usage: sbatch --account=bdqf-dtai-gh run_vans_world_model.sbatch

#SBATCH --job-name=vans_world_model
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

echo "=========================================="
echo "[SMOKE TEST] 8 pairs, both feature extractors + a 2-epoch training run"
echo "=========================================="
python extract_vjepa_features.py --limit 8
python extract_vlm_guidance.py --limit 8
python train_latent_world_model.py --limit 8 --epochs 2 --run_name smoke_test
echo "[SMOKE TEST] passed."

echo "=========================================="
echo "[FULL RUN] extracting features for all 500 demo pairs"
echo "=========================================="
python extract_vjepa_features.py
python extract_vlm_guidance.py

echo "=========================================="
echo "[FULL RUN] training"
echo "=========================================="
python train_latent_world_model.py --epochs "${EPOCHS:-15}" --run_name vans_demo_latent_world_model

echo "[INFO] done."
