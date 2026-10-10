#!/usr/bin/env bash
#SBATCH --job-name=rec_v13ens
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-/data/home/dengzhonglang/program/recombinase_model_train}"
PYTHON_BIN="${PYTHON_BIN:-/data/home/dengzhonglang/miniforge3/envs/rec_rank/bin/python}"
CHECKPOINT_42="${CHECKPOINT_42:-runs/20261010_082603_6452501e35_s42/checkpoints/best.pt}"
CHECKPOINT_43="${CHECKPOINT_43:-runs/20261010_085133_a453c8854e_s43/checkpoints/best.pt}"
CHECKPOINT_44="${CHECKPOINT_44:-runs/20261010_090603_616b95f815_s44/checkpoints/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations/v1_3_seed_42_43_44_ensemble}"

cd "$PROJECT_DIR"
for checkpoint in "$CHECKPOINT_42" "$CHECKPOINT_43" "$CHECKPOINT_44"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "找不到 checkpoint: $checkpoint" >&2
    exit 2
  fi
done

srun "$PYTHON_BIN" -m rec_rank evaluate-full-library-ensemble \
  --checkpoint "$CHECKPOINT_42" \
  --checkpoint "$CHECKPOINT_43" \
  --checkpoint "$CHECKPOINT_44" \
  --train-candidate-embeddings embeddings/train_proteins \
  --test-candidate-embeddings embeddings/test_candidates \
  --query-embeddings embeddings/test_dnas \
  --query-rc-embeddings embeddings/test_queries_rc_only \
  --processed-dir data/processed \
  --output "$OUTPUT_DIR" \
  --device cuda \
  --query-batch-size 64 \
  --top-k 10
