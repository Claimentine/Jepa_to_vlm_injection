#!/usr/bin/env bash
# Submits the "does the VLM already know the task" probe end to end:
#   sample episodes -> run Qwen3-VL-2B-Thinking zero-shot -> score.
#
# Usage:
#   sbatch --account=bdqf-dtai-gh run_task_recognition_probe.sbatch
#   sbatch --account=bdjg-dtai-gh run_task_recognition_probe.sbatch
#
# (account can also be left to the #SBATCH default below and overridden
#  on the command line as shown above -- convenient for trying whichever
#  of bdjg-dtai-gh / bdqf-dtai-gh still has hours left)

#SBATCH --job-name=task_recognition_probe
#SBATCH --account=bdqf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

cd /projects/bhay/william/ruixin/semantic_probe

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen3vl

K_PER_CATEGORY="${K_PER_CATEGORY:-20}"

echo "[INFO] sampling episodes (k_per_category=${K_PER_CATEGORY}) ..."
python sample_probe_episodes.py --k_per_category "${K_PER_CATEGORY}" --out probe_manifest.json

echo "[INFO] running zero-shot Qwen3-VL-2B-Thinking probe ..."
python run_task_recognition_probe.py --manifest probe_manifest.json --out probe_results.jsonl

echo "[INFO] scoring ..."
python score_task_recognition_probe.py --results probe_results.jsonl | tee probe_report.txt

echo "[INFO] done."
