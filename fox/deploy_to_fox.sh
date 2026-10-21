#!/bin/bash
# Deploy the code and data to Fox, set up the environment, run checks and submit the full run,
# all in ONE ssh call (so the 2FA prompt comes only once). The remote steps are in fox/remote_deploy.sh.
#
# Usage, from the repo root in your own Git Bash terminal:
#   bash fox/deploy_to_fox.sh                 # default account ec403
#   ACCOUNT=ecXXX bash fox/deploy_to_fox.sh   # another Fox project account
# data/train.csv and data/test.csv must exist locally (see data/README.md); they are small, so they
# are uploaded every time.
set -euo pipefail
cd "$(dirname "$0")/.."
ACCOUNT=${ACCOUNT:-ec403}

for f in data/train.csv data/test.csv; do
    [ -f "$f" ] || { echo "Missing $f, see data/README.md"; exit 1; }
done

echo "Bundling code + data..."
tar czf fox_bundle.tar.gz --exclude=__pycache__ src fox run_all.sh requirements.txt data/train.csv data/test.csv
echo "bundle: $(du -h fox_bundle.tar.gz | cut -f1)"

echo "Connecting to Fox (answer the 2FA prompt once)..."
if ssh fox "mkdir -p arrousal && cd arrousal && tar xzf - && ACCOUNT=$ACCOUNT bash fox/remote_deploy.sh" < fox_bundle.tar.gz; then
    STATUS=0
else
    STATUS=$?
fi
rm -f fox_bundle.tar.gz
if [ "$STATUS" -eq 0 ]; then
    echo "Done: checks passed and the job is submitted. Fetch results later with: bash fox/fetch_from_fox.sh"
else
    echo "Stopped (exit $STATUS), see output above. Nothing is submitted if a check failed."
fi
exit "$STATUS"
