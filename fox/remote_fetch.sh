#!/bin/bash
# Runs ON FOX via fox/fetch_from_fox.sh. Writes a tar of inventory.txt, logs/ and outputs/ to stdout.
set -euo pipefail
cd ~/arrousal
{
    echo "## queue"; squeue -u "$USER" || true
    echo; echo "## outputs"; ls -la outputs 2>/dev/null || echo "(none yet)"
    echo; echo "## last 5 lines of each log"
    for f in logs/*.log; do [ -f "$f" ] && { echo "--- $f"; tail -n 5 "$f"; }; done
} > inventory.txt 2>&1
tar cf - inventory.txt logs outputs 2>/dev/null || true
