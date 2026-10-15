#!/usr/bin/env bash
#SBATCH --job-name=rec_v16e50
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-/data/home/dengzhonglang/program/recombinase_model_train}"
PYTHON_BIN="${PYTHON_BIN:-/data/home/dengzhonglang/miniforge3/envs/rec_rank/bin/python}"
RUN_DIR="${RUN_DIR:-runs/20261012_051847_7379931bab_s43}"
TARGET_EPOCHS="${TARGET_EPOCHS:-50}"

cd "$PROJECT_DIR"
if [[ ! -f "$RUN_DIR/checkpoints/last.pt" ]]; then
  echo "找不到续训 checkpoint: $PROJECT_DIR/$RUN_DIR/checkpoints/last.pt" >&2
  exit 2
fi
export MPLCONFIGDIR="$PROJECT_DIR/.matplotlib"

# Multiple jobs must never write checkpoints/history into the same run.
exec 9>"$RUN_DIR/.training.lock"
if ! flock -n 9; then
  echo "已有另一个任务正在续训同一 run: $PROJECT_DIR/$RUN_DIR" >&2
  exit 3
fi

echo "从 $RUN_DIR/checkpoints/last.pt 续训至 epoch $TARGET_EPOCHS"

srun "$PYTHON_BIN" -m rec_rank train \
  --config configs/v1_6_triple_aware.yaml \
  --device cuda \
  --run-dir "$RUN_DIR" \
  --resume \
  --epochs "$TARGET_EPOCHS"
