#!/bin/bash
#SBATCH --job-name=arrousal
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/%x_%j.log

# Full run on Fox: all 11 models, evaluation and error analysis (bash run_all.sh).
# The account is given on the sbatch command line (see fox/remote_deploy.sh).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/z00_lmod.sh; fi
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source venv/bin/activate

export MODEL_NAME="$PWD/models/distilbert-base-multilingual-cased"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "node=$(hostname) start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
bash run_all.sh
echo "end=$(date)"
