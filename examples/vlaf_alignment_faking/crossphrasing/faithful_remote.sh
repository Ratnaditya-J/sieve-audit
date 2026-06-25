#!/bin/bash
# Runs ON the pod, from /root/VLAF. The script self-gates (diag -> baseline ->
# steering) and ALWAYS writes results (incl. raw diag) so we can inspect.
set -uo pipefail
cd /root/VLAF
pip install -q -U transformers jinja2 2>/dev/null
rm -f /root/FB_DONE /root/FB_FAIL
echo "=== RUN faithful B (maxtok 4096, staged gate) ==="
python build_faithful_B.py --per-foundation 10 --n-samples 3 --maxtok 4096 --batch 2 \
  --vec-items 200 --out faithful_B_results.json && touch /root/FB_DONE || touch /root/FB_FAIL
echo "=== REMOTE DONE ==="
