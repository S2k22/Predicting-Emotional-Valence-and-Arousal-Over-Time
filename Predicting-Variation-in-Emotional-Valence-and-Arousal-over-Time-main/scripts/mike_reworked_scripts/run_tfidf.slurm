#!/bin/bash
#SBATCH --job-name=tfidf_ablation
#SBATCH --account=ec403
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x-%j.out

mkdir -p logs
cd $SLURM_SUBMIT_DIR/..

echo "===== M1: TF-IDF text-only ====="
python scripts/m01_tfidf_text.py

echo "===== M2: TF-IDF + structural/intensity ====="
python scripts/m02_tfidf_structural.py

echo "===== M3: TF-IDF + structural/intensity + temporal/study-design ====="
python scripts/m03_tfidf_temporal.py

echo "===== M4: TF-IDF + structural/intensity + temporal/study-design + user baselines ====="
python scripts/m04_tfidf_user.py

echo "===== M5: TF-IDF + all features + text-type split ====="
python scripts/m05_tfidf_split.py
