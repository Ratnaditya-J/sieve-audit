#!/usr/bin/env bash
# One-command 7B unlearning-completeness triptych on a RunPod GPU pod.
#
# The real headline measurement: does RMU-unlearned Zephyr still linearly decode
# WMDP-bio knowledge (suppression), or is it genuinely gone (removal)? Runs the
# preregistered protocol on the true public pair — no approximations, no
# checkpoints to train.
#
#   base      HuggingFaceH4/zephyr-7b-beta   (pre-unlearning positive control)
#   unlearned cais/Zephyr_RMU                (the claim under audit)
#   anchor    cais/Zephyr_RMU --shuffle-labels (pipeline floor; phase-1 approx —
#             the true never-trained anchor arrives with GRAM in phase 2)
#
# Provision (from a machine WITH the RunPod key; needs a >=24GB GPU, e.g. A5000/
# A100). Two 7B bf16 models load one at a time, so 24GB is comfortable:
#   runpodctl create pod --gpuType 'NVIDIA A100 80GB PCIe' --imageName \
#     'runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04' --volumeSize 60
# then scp this repo + this script onto the pod and: bash pod_run.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$HERE"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
LAYERS="${LAYERS:-7 13 20 27}"          # preregistered a-priori layer set (32-layer zephyr)
DTYPE="${DTYPE:-bfloat16}"
DOMAIN="${DOMAIN:-wmdp-bio}"
N_STMT="${N_STMT:-300}"                 # WMDP questions -> 600 matched statements
N_MCQ="${N_MCQ:-200}"

echo "== [0/6] install =="
pip install -q -e "$REPO[runner]" datasets

echo "== [1/6] build real WMDP data ($DOMAIN) =="
python prepare_data.py --domain "$DOMAIN" --seed 0 \
  --n-statements "$N_STMT" --n-mcq "$N_MCQ" --min-family-questions 15 --out-dir data
STMTS="data/decode_statements.$DOMAIN.jsonl"
MCQ="data/mcq_eval.$DOMAIN.jsonl"
NFAM=$(python -c "import json;print(len(json.load(open('data/manifest.$DOMAIN.json'))['families']))")

echo "== [2/6] freeze preregistration (strict config + layers + decision rule) =="
python make_prereg.py --model cais/Zephyr_RMU --revision main \
  --layers $LAYERS --domain "$DOMAIN" --n-families "$NFAM" --out "prereg.$DOMAIN.json"

echo "== [3/6] extract + LOFO probe -> decodability bundles (3 conditions) =="
python unlearning_audit.py build-bundle \
  --model HuggingFaceH4/zephyr-7b-beta --model-tag zephyr-base --revision main \
  --statements "$STMTS" --layers $LAYERS --probes mean_diff logistic_regression \
  --domain "$DOMAIN" --encoding raw --dtype "$DTYPE" --seed 0 --out-dir runs
python unlearning_audit.py build-bundle \
  --model cais/Zephyr_RMU --model-tag zephyr-rmu --revision main \
  --statements "$STMTS" --layers $LAYERS --probes mean_diff logistic_regression \
  --domain "$DOMAIN" --encoding raw --dtype "$DTYPE" --seed 0 --out-dir runs
# anchor reuses the unlearned model's activations (no second extraction pass)
python unlearning_audit.py build-bundle \
  --model cais/Zephyr_RMU --model-tag zephyr-anchor --revision main \
  --statements "$STMTS" --layers $LAYERS --probes mean_diff logistic_regression \
  --domain "$DOMAIN" --encoding raw --seed 0 --out-dir runs \
  --shuffle-labels --reuse-activations runs --reuse-tag zephyr-rmu

echo "== [4/6] audit triptych (GPU-free engine) =="
python run_triptych.py \
  --index runs/bundles.zephyr-base.index.json \
          runs/bundles.zephyr-rmu.index.json \
          runs/bundles.zephyr-anchor.index.json \
  --roles base unlearned anchor \
  --prereg "prereg.$DOMAIN.json" --seed 0 --out-dir reports

echo "== [5/6] causal re-elicitation on any RESIDUAL (layer,probe) on the unlearned model =="
python - "$DOMAIN" "$MCQ" <<'PY'
import json, subprocess, sys
from pathlib import Path
domain, mcq = sys.argv[1], sys.argv[2]
trip = json.load(open("reports/triptych.json"))
launched = 0
for cell in trip["cells"]:
    if cell["reading"] and cell["reading"].startswith("RESIDUAL"):
        L, probe = cell["layer"], cell["probe"]
        vec = f"runs/vectors.zephyr-rmu.L{L}.npz"
        if not Path(vec).exists():
            print(f"[causal] missing {vec}; skip L{L}"); continue
        out = f"runs/reelicit.zephyr-rmu.L{L}.json"
        print(f"[causal] RESIDUAL at L{L}/{probe} -> re-elicitation")
        subprocess.run([sys.executable, "reelicit.py", "--model", "cais/Zephyr_RMU",
            "--revision", "main", "--vectors", vec, "--layer", str(L),
            "--mcq", mcq, "--decode-bundle", f"runs/bundle.zephyr-rmu.L{L}.mean_diff.json",
            "--domain", domain, "--dtype", "bfloat16", "--out", out], check=True)
        subprocess.run(["sieve", "audit", "--bundle", out, "--name",
            f"reelicit_zephyr-rmu_L{L}", "--out", "reports"], check=True)
        launched += 1
if not launched:
    print("[causal] no RESIDUAL cells; decodability stage settled the question")
PY

echo "== [6/6] done. cards + triptych in reports/ =="
ls -1 reports/ | sed 's/^/   /'
