#!/bin/bash
# Runs ON the pod under nohup. Install -> tiny pre-flight (catch runtime bugs
# cheaply) -> full run. Sentinels /root/B_DONE or /root/B_FAIL signal the driver.
set -uo pipefail
cd /root/B
echo "=== pip install transformers ==="
pip install -q -U transformers >/root/pip.log 2>&1 || { echo PIP_FAIL; touch /root/B_FAIL; exit 1; }
echo "=== PRE-FLIGHT (tiny) ==="
python build_and_run_B.py --cal-scenarios 2 --cal-samples 1 --n-scenarios 2 \
  --n-samples 1 --vec-items 4 --maxtok 32 --out preflight.json \
  || { echo PREFLIGHT_FAIL; touch /root/B_FAIL; exit 1; }
echo "=== PRE-FLIGHT OK -> FULL RUN ==="
python build_and_run_B.py --n-scenarios 200 --n-samples 5 --cal-scenarios 30 \
  --cal-samples 3 --vec-items 220 --maxtok 128 --out B_results.json \
  && touch /root/B_DONE || touch /root/B_FAIL
echo "=== REMOTE DONE ==="
