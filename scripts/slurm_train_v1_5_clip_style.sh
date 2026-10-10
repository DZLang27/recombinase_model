#!/usr/bin/env bash
#SBATCH --job-name=rec_v15clip
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-/data/home/dengzhonglang/program/recombinase_model_train}"
PYTHON_BIN="${PYTHON_BIN:-/data/home/dengzhonglang/miniforge3/envs/rec_rank/bin/python}"
SEED="${SEED:-43}"

cd "$PROJECT_DIR"
if [[ ! -f configs/v1_5_clip_style.yaml ]]; then
  echo "项目目录错误，找不到 $PROJECT_DIR/configs/v1_5_clip_style.yaml" >&2
  exit 2
fi
export MPLCONFIGDIR="$PROJECT_DIR/.matplotlib"

srun "$PYTHON_BIN" -m rec_rank train \
  --config configs/v1_5_clip_style.yaml \
  --device cuda \
  --seed "$SEED"
