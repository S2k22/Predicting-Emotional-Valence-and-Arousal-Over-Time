#!/usr/bin/env bash
# Run every model, the ensemble, the evaluation and the error analysis.
# The DistilBERT models (M6-M10) need a GPU to finish in reasonable time.
set -euo pipefail

for script in m01_tfidf_text m02_tfidf_structural m03_tfidf_temporal m04_tfidf_user m05_tfidf_split \
              m06_bert_text m07_bert_structural m08_bert_temporal m09_bert_user m10_bert_split \
              m11_hybrid; do
    echo "===== ${script} ====="
    python "src/${script}.py"
done

python src/evaluate.py
python src/error_analysis.py
