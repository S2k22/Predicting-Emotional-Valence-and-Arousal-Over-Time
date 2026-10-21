#!/bin/bash
#SBATCH --job-name=bert_ablation
#SBATCH --account=ec403
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

mkdir -p logs

cd /fp/homes01/u01/ec-andremg/work/exam/valence_arousal

module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/01-foss-2024a-Python-3.12.3
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3
export TRANSFORMERS_OFFLINE=1

echo "===== M6: DistilBERT text-only ====="
python scripts/m06_bert_text.py

echo "===== M7: DistilBERT + structural/intensity ====="
python scripts/m07_bert_structural.py

echo "===== M8: DistilBERT + structural/intensity + temporal/study-design ====="
python scripts/m08_bert_temporal.py

echo "===== M9: DistilBERT + structural/intensity + temporal/study-design + user baselines ====="
python scripts/m09_bert_user.py

echo "===== M10: DistilBERT + all features + text-type split ====="
python scripts/m10_bert_split.py
