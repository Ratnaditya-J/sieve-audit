#!/usr/bin/env bash
# Provision a RunPod GPU pod, run the 7B triptych, pull the cards back.
#
# Requires RUNPOD_API_KEY in the environment (and `runpodctl` on PATH:
#   wget -qO- cli.runpod.net | bash). Everything the run needs is in this repo;
# the pod only needs a PyTorch image + this checkout.
#
# The 7B measurement loads two zephyr-7b bf16 models one at a time, so a single
# 24GB GPU is plenty; 40-80GB just runs faster. ~1 GPU-hour end to end.
set -euo pipefail

: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY (this session did not have it; the user holds it)}"
GPU_TYPE="${GPU_TYPE:-NVIDIA A100 80GB PCIe}"
IMAGE="${IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
VOLUME_GB="${VOLUME_GB:-60}"
REPO_URL="${REPO_URL:-https://github.com/Ratnaditya-J/sieve-audit}"
BRANCH="${BRANCH:-claude/gram-sieve-detection-fit-1vrh9a}"

echo "== creating pod ($GPU_TYPE) =="
POD_JSON=$(runpodctl create pod \
  --name sieve-unlearning-7b \
  --gpuType "$GPU_TYPE" --gpuCount 1 \
  --imageName "$IMAGE" \
  --volumeSize "$VOLUME_GB" --containerDiskSize 40 \
  --ports '22/tcp' \
  --args "bash -c 'sleep infinity'")
echo "$POD_JSON"
POD_ID=$(echo "$POD_JSON" | grep -oE 'pod "[^"]+"' | head -1 | sed -E 's/pod "([^"]+)"/\1/')
[ -n "$POD_ID" ] || { echo "could not parse pod id"; exit 1; }
echo "pod id: $POD_ID"

echo "== waiting for SSH (poll runpodctl get pod) =="
for i in $(seq 1 40); do
  sleep 15
  runpodctl get pod "$POD_ID" | grep -q RUNNING && break
done

# runpodctl exposes an ssh string; scp is simplest via the pod's public ip/port.
# The remainder assumes you can `ssh pod` (configure from `runpodctl get pod`):
cat <<'REMOTE' > /tmp/_sieve_remote.sh
set -euo pipefail
cd /workspace
git clone --branch "$BRANCH" "$REPO_URL" sieve-audit || (cd sieve-audit && git fetch && git checkout "$BRANCH")
cd sieve-audit/examples/unlearning_completeness
bash pod_run.sh
tar czf /workspace/unlearning_reports.tgz reports/
REMOTE

echo "== NEXT (manual, needs the pod's ssh target from 'runpodctl get pod $POD_ID'): =="
echo "  scp /tmp/_sieve_remote.sh pod:/workspace/ && ssh pod 'BRANCH=$BRANCH REPO_URL=$REPO_URL bash /workspace/_sieve_remote.sh'"
echo "  scp pod:/workspace/unlearning_reports.tgz .   # then: runpodctl remove pod $POD_ID"
echo "  (SIEVE cards + triptych.json land in reports/)"
