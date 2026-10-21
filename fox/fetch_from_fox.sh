#!/bin/bash
# Fetch results from Fox in ONE ssh call: job status, Slurm logs and outputs -> fox_results/.
# Safe to run while the job is still going.
#
# Usage, from the repo root in your own Git Bash terminal:
#   bash fox/fetch_from_fox.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p fox_results
echo "Connecting to Fox (answer the 2FA prompt once)..."
ssh fox 'bash -s' < fox/remote_fetch.sh | tar xf - -C fox_results
echo; echo "=== INVENTORY ==="; cat fox_results/inventory.txt
for f in fox_results/outputs/test_metrics.csv fox_results/outputs/validation_metrics.csv; do
    [ -f "$f" ] && { echo; echo "=== $f ==="; cat "$f"; }
done
exit 0
