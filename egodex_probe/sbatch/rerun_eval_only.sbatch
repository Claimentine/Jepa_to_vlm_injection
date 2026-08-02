#!/usr/bin/env bash
# Cheap re-run: no training (checkpoints from job 2618512 are reused as-is),
# just re-does the generate()-based final eval for jepa/random with the
# hook bug fixed, plus a new no-injection "direct answer" control to isolate
# the "skip chain-of-thought" methodology confound from the injection
# content itself.
#
# Usage: sbatch --account=bdqf-dtai-gh rerun_eval_only.sbatch

#SBATCH --job-name=jepa_injection_eval_rerun
#SBATCH --account=bdqf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail
cd /projects/bhay/william/ruixin/semantic_probe

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen3vl

echo "=== re-eval: jepa (hook bug fixed) ==="
python eval_jepa_injection_probe.py --injection jepa   --checkpoint jepa_injection_runs/jepa/best.pt   --out injection_eval_jepa.jsonl

echo "=== re-eval: random (hook bug fixed) ==="
python eval_jepa_injection_probe.py --injection random --checkpoint jepa_injection_runs/random/best.pt --out injection_eval_random.jsonl

echo "=== new control: no injection, direct answer (no chain-of-thought) ==="
python eval_direct_answer_baseline.py --out injection_eval_no_injection_direct_answer.jsonl

echo "=== scoring ==="
for f in injection_eval_jepa.jsonl injection_eval_random.jsonl injection_eval_linear_probe.jsonl injection_eval_no_injection_direct_answer.jsonl; do
  echo "--- ${f} ---"
  python score_task_recognition_probe.py --results "${f}"
done | tee injection_probe_report_v2.txt

echo "[INFO] done."
