#!/bin/bash
# Runs ON FOX from ~/arrousal (called by fox/deploy_to_fox.sh):
# venv (first time only) -> download DistilBERT (first time only) -> M1 on the login node
# -> short GPU smoke test -> sbatch the full run.
set -euo pipefail
: "${ACCOUNT:?ACCOUNT must be set}"

if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/z00_lmod.sh; fi
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

if [ ! -d venv ]; then
    echo "=== creating venv (first time only) ==="
    python -m venv --system-site-packages venv
    source venv/bin/activate
    # numpy, pandas, scipy and torch come from the module; transformers 4.44 still supports torch 2.1
    pip install --no-cache-dir scikit-learn "transformers==4.44.2"
else
    source venv/bin/activate
fi
python -c "import torch, transformers, sklearn, pandas; print('torch', torch.__version__, '| transformers', transformers.__version__, '| sklearn', sklearn.__version__, '| pandas', pandas.__version__)"

# Compute nodes may have no internet access, so keep a local copy of the model
MODEL_DIR="$PWD/models/distilbert-base-multilingual-cased"
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "=== downloading distilbert-base-multilingual-cased (first time only) ==="
    python - "$MODEL_DIR" <<'EOF'
import sys
from transformers import AutoModel, AutoTokenizer
name, out = "distilbert-base-multilingual-cased", sys.argv[1]
AutoTokenizer.from_pretrained(name).save_pretrained(out)
AutoModel.from_pretrained(name).save_pretrained(out)
EOF
fi

echo "=== check 1: M1 (TF-IDF) on the login node ==="
OUTPUT_DIR=outputs_check python src/m01_tfidf_text.py

echo "=== check 2: GPU smoke test (CUDA + model forward pass) ==="
srun --account="$ACCOUNT" --partition=accel --gres=gpu:1 --cpus-per-task=2 --mem=8G --time=00:10:00 \
    python - "$MODEL_DIR" <<'EOF'
import sys
import torch
from transformers import AutoModel, AutoTokenizer
assert torch.cuda.is_available(), "no GPU visible"
print("GPU:", torch.cuda.get_device_name(0))
tok = AutoTokenizer.from_pretrained(sys.argv[1])
model = AutoModel.from_pretrained(sys.argv[1]).cuda()
out = model(**tok(["I feel fine today."], return_tensors="pt").to("cuda"))
print("forward pass OK, hidden shape", tuple(out.last_hidden_state.shape))
EOF
rm -rf outputs_check
echo "=== checks PASSED ==="

echo "=== submitting the full run (M1-M11, evaluation, error analysis) ==="
mkdir -p logs
rm -rf outputs
sbatch --account="$ACCOUNT" fox/run_all.slurm
squeue -u "$USER"
