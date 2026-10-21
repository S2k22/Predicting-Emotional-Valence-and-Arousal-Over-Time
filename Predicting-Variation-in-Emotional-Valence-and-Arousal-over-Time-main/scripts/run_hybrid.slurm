#!/bin/bash
#SBATCH --job-name=hybrid_ensemble
#SBATCH --account=ec403
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x-%j.out

mkdir -p logs
cd $SLURM_SUBMIT_DIR/..

echo "===== M11: Hybrid ensemble ====="
python scripts/m11_hybrid.py
