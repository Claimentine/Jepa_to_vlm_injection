#!/usr/bin/env bash
# Option A: zero-shot transfer -- reuse the EgoDex-trained temporal-pooling
# adapter checkpoint as-is on the VANS QA test set, NO training. Just one
# generate()-based eval pass (~75 items), short walltime to queue fast.
#
# Usage: sbatch --account=bdqf-dtai-gh run_qa_transfer_eval.sbatch

#SBATCH --job-name=qa_transfer_eval
#SBATCH --account=bdqf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.log

set -euo pipefail
cd /projects/bhay/william/ruixin/vans_world_model

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen3vl

EGODEX_JEPA_CKPT=/projects/bhay/william/ruixin/semantic_probe/jepa_injection_runs_v2/jepa/best.pt

echo "=== zero-shot transfer: EgoDex-trained jepa adapter -> VANS QA test set (no training) ==="
python eval_vans_qa_injection.py --injection jepa --checkpoint "${EGODEX_JEPA_CKPT}" --out qa_eval_jepa_transfer.jsonl

echo "=== scoring (vs none / vans-trained jepa / random, already on disk) ==="
python score_qa_pilot.py --results qa_eval_none.jsonl qa_eval_jepa.jsonl qa_eval_random.jsonl qa_eval_jepa_transfer.jsonl | tee qa_transfer_report.txt

echo "[INFO] done."
