#!/bin/bash
# Example Slurm job. Replace the placeholders for your cluster.
#SBATCH --job-name=valence_arousal
#SBATCH --account=<your-project>
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Load or activate an environment with the packages in requirements.txt, e.g.:
# module load <python-module> <pytorch-module> <transformers-module>
# source .venv/bin/activate

# Compute nodes often have no internet access: point MODEL_NAME to a local
# copy of distilbert-base-multilingual-cased and run transformers offline.
# export MODEL_NAME=/path/to/distilbert-base-multilingual-cased
# export TRANSFORMERS_OFFLINE=1

bash run_all.sh
