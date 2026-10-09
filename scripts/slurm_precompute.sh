#!/usr/bin/env bash
#SBATCH --job-name=rec_embed
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "用法: sbatch $0 <protein|dna> <input.tsv> <output_dir>" >&2
  exit 2
fi

# sbatch 会把脚本复制到 spool，不能用 $0 推导项目路径。
# 从项目根目录提交时，SLURM_SUBMIT_DIR 是稳定的真实路径。
PROJECT_DIR="${SLURM_SUBMIT_DIR:-/data/home/dengzhonglang/program/recombinase_model_train}"
PYTHON_BIN="${PYTHON_BIN:-/data/home/dengzhonglang/miniforge3/envs/rec_rank/bin/python}"
MODALITY="$1"
INPUT_PATH="$2"
OUTPUT_PATH="$3"

cd "$PROJECT_DIR"
if [[ ! -f configs/v1.yaml ]]; then
  echo "项目目录错误，找不到 $PROJECT_DIR/configs/v1.yaml" >&2
  exit 2
fi
export MPLCONFIGDIR="$PROJECT_DIR/.matplotlib"

srun "$PYTHON_BIN" -m rec_rank embed \
  --config configs/v1.yaml \
  --modality "$MODALITY" \
  --input "$INPUT_PATH" \
  --output "$OUTPUT_PATH" \
  --device cuda
