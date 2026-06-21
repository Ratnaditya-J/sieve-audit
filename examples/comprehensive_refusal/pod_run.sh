#!/usr/bin/env bash
# Full re-run of the comprehensive_refusal SIEVE audit with multi-draw null
# (--n-random-controls 3). Outputs every evidence axis; run locally after
# scp / rsync to regenerate examples/comprehensive_refusal/bundle_all.json.
#
# Prereqs on pod:
#   export OPENAI_API_KEY=...      # both judges (gpt-4.1-mini + gpt-4o-mini)
#
# Usage:
#   bash pod_run.sh                  # full run
#   SMOKE=1 bash pod_run.sh          # tiny CPU model (plumbing check only)
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
LAYER="${LAYER:-14}"
WRONG_LAYER="${WRONG_LAYER:-7}"
MULTILAYER_LAYERS="${MULTILAYER_LAYERS:-12 14 16}"
N_RANDOM="${N_RANDOM:-3}"
ALPHA_MODE="${ALPHA_MODE:-relative}"
ALPHAS="${ALPHAS:--0.2 -0.1 0.1 0.2}"
MAXTOK="${MAXTOK:-200}"
JUDGE_A="${JUDGE_A:-openai:gpt-4.1-mini}"
JUDGE_B="${JUDGE_B:-openai:gpt-4o-mini}"
METRIC="${METRIC:-refusal}"
OUT="${OUT:-runout}"
PROMPTS="${PROMPTS:-prompts}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  MODEL="hf-internal-testing/tiny-random-LlamaForCausalLM"; LAYER=1; WRONG_LAYER=0
  MULTILAYER_LAYERS="0 1"; N_RANDOM=2; ALPHAS="-0.2 -0.1 0.1 0.2"; MAXTOK=16
  JUDGE_A="stub:a"; JUDGE_B="stub:b"
fi
DTYPE="${DTYPE:-bfloat16}"

DIR_SOURCE="contrastive mean-diff (z-scored, last-token), refusal direction, leave-one-family-out"
DIST="AdvBench harmful vs templated benign instructions"
LICENSE="AdvBench prompts (llm-attacks, MIT); benign templates CC0, n=120"

RUN="python -m sieve_audit.adapters.hf_steering_runner"
mkdir -p "$OUT"
echo "[pod] model=$MODEL layer=$LAYER dtype=$DTYPE n_random=$N_RANDOM"

# ---- preflight ----
if [[ "${SMOKE:-0}" != "1" ]]; then
  : "${OPENAI_API_KEY:?set OPENAI_API_KEY for judges}"
  python - <<'PY'
import torch; assert torch.cuda.is_available(), "no CUDA"
print("[pod] CUDA:", torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")
PY
fi

# ---- 1. decode-lofo (direction + LOFO decodability + vectors.npz) ----
# Explicit --layer to reproduce layer 14; --n-random-controls 3 for multi-draw null.
$RUN decode-lofo --model "$MODEL" --layer "$LAYER" --dtype "$DTYPE" \
  --holdout-prompts "$PROMPTS/holdout.jsonl" \
  --n-random-controls "$N_RANDOM" \
  --save-vectors "$OUT/vectors.npz" --out "$OUT/decode.json"

# ---- 2. steer (arms: probe + 3 random draws + orthogonal + wrong_layer) ----
$RUN steer --model "$MODEL" --dtype "$DTYPE" \
  --vectors "$OUT/vectors.npz" --wrong-layer "$WRONG_LAYER" \
  --steer-prompts "$PROMPTS/steer.jsonl" \
  --alpha-mode "$ALPHA_MODE" --alphas $ALPHAS \
  --max-new-tokens "$MAXTOK" --out "$OUT/steer.jsonl"

# ---- 3. ablate (necessity: single-layer ablation) ----
$RUN ablate --model "$MODEL" --dtype "$DTYPE" \
  --vectors "$OUT/vectors.npz" \
  --eval-prompts "$PROMPTS/steer.jsonl" \
  --max-new-tokens "$MAXTOK" --out "$OUT/ablate.jsonl"

# ---- 4. multilayer (joint ablation at layers 12, 14, 16) ----
$RUN multilayer --model "$MODEL" --dtype "$DTYPE" \
  --vectors "$OUT/vectors.npz" \
  --layers $MULTILAYER_LAYERS \
  --eval-prompts "$PROMPTS/steer.jsonl" \
  --max-new-tokens "$MAXTOK" --out "$OUT/multilayer.jsonl"

# ---- 5. patching (oracle calibration) ----
$RUN patching --model "$MODEL" --dtype "$DTYPE" \
  --vectors "$OUT/vectors.npz" \
  --eval-prompts "$PROMPTS/patching_pairs.jsonl" \
  --max-new-tokens "$MAXTOK" --out "$OUT/patching.jsonl"

# ---- 6. leakage (Tier-2: full / leak-removed / random-removed rescoring) ----
$RUN leakage --model "$MODEL" --dtype "$DTYPE" \
  --vectors "$OUT/vectors.npz" \
  --holdout-prompts "$PROMPTS/holdout.jsonl" \
  --out "$OUT/leakage.json"

# ---- 7. off-distribution decode (deployment lens) ----
$RUN decode --model "$MODEL" --dtype "$DTYPE" \
  --vectors "$OUT/vectors.npz" \
  --holdout-prompts "$PROMPTS/ood.jsonl" \
  --out "$OUT/decode_ood.json"

# ---- 8. two judges (GPU idle — fan out to API) ----
# Score steer, ablation, multilayer, patching all at once
$RUN judge --generations "$OUT/steer.jsonl" \
  --steer-prompts "$PROMPTS/steer.jsonl" \
  --judge "$JUDGE_A" --judge "$JUDGE_B" \
  --metric "$METRIC" --out "$OUT/judged_steer.jsonl"

$RUN judge --generations "$OUT/ablate.jsonl" \
  --steer-prompts "$PROMPTS/steer.jsonl" \
  --judge "$JUDGE_A" --judge "$JUDGE_B" \
  --metric "$METRIC" --out "$OUT/judged_ablate.jsonl"

$RUN judge --generations "$OUT/multilayer.jsonl" \
  --steer-prompts "$PROMPTS/steer.jsonl" \
  --judge "$JUDGE_A" --judge "$JUDGE_B" \
  --metric "$METRIC" --out "$OUT/judged_multilayer.jsonl"

$RUN judge --generations "$OUT/patching.jsonl" \
  --steer-prompts "$PROMPTS/steer.jsonl" \
  --judge "$JUDGE_A" --judge "$JUDGE_B" \
  --metric "$METRIC" --out "$OUT/judged_patching.jsonl"

# ---- 9. bundle (all evidence in one JSON) ----
$RUN bundle \
  --decode "$OUT/decode.json" \
  --steer "$OUT/steer.jsonl" \
  --judged "$OUT/judged_steer.jsonl" \
  --judged-ablation "$OUT/judged_ablate.jsonl" \
  --judged-multilayer "$OUT/judged_multilayer.jsonl" \
  --judged-patching "$OUT/judged_patching.jsonl" \
  --leakage "$OUT/leakage.json" \
  --decode-ood "$OUT/decode_ood.json" \
  --ood-distribution "AdvBench reworded (Question:...)" \
  --model "$MODEL" \
  --direction-source "$DIR_SOURCE" \
  --prompt-distribution "$DIST (license: $LICENSE)" \
  --prompt-license "$LICENSE" \
  --metric "$METRIC" \
  --attest-out-of-sample \
  --out "$OUT/bundle_all.json"

echo "[pod] DONE -> $OUT/bundle_all.json"
touch "$OUT/POD_DONE"
