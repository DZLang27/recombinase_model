#!/usr/bin/env bash
#SBATCH --job-name=rec_full_eval
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-/data/home/dengzhonglang/program/recombinase_model_train}"
PYTHON_BIN="${PYTHON_BIN:-/data/home/dengzhonglang/miniforge3/envs/rec_rank/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations/full_library_existing_v1}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-64}"

if [[ "$#" -eq 0 ]]; then
  echo "用法: sbatch $0 CHECKPOINT [CHECKPOINT ...]" >&2
  exit 2
fi

cd "$PROJECT_DIR"
if [[ ! -f configs/v1.yaml ]]; then
  echo "项目目录错误，找不到 $PROJECT_DIR/configs/v1.yaml" >&2
  exit 2
fi

checkpoint_args=()
for checkpoint in "$@"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "找不到 checkpoint: $checkpoint" >&2
    exit 2
  fi
  checkpoint_args+=(--checkpoint "$checkpoint")
done

srun "$PYTHON_BIN" -m rec_rank evaluate-full-library \
  "${checkpoint_args[@]}" \
  --train-candidate-embeddings embeddings/train_proteins \
  --test-candidate-embeddings embeddings/test_candidates \
  --query-embeddings embeddings/test_dnas \
  --processed-dir data/processed \
  --output "$OUTPUT_DIR" \
  --query-batch-size "$QUERY_BATCH_SIZE" \
  --device cuda
