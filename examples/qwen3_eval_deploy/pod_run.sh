#!/usr/bin/env bash
# Self-contained flagship run for a GPU pod (e.g. 1x H200).
#
# Produces the first end-to-end SIEVE causal audit card on a real model:
# Qwen3-32B, eval/deploy direction at layer 55, matched controls at the
# wrong layer 25, two independent judges via OpenRouter.
#
# Prereqs on the pod (set by the human, never by the script):
#   export HF_TOKEN=...              # model download (gated models)
#   export OPENROUTER_API_KEY=...    # both judges, one key
# Prompts in ./prompts/ (pushed separately):
#   contrast_triggers.jsonl  controlled_holdout_v2.jsonl  steer_prompts.jsonl
#
# Usage:  bash pod_run.sh            # full flagship
#         SMOKE=1 bash pod_run.sh    # tiny CPU model, 3 prompts (plumbing check)
set -euo pipefail

# ---- config (override via env) ----
MODEL="${MODEL:-Qwen/Qwen3-32B}"
LAYER="${LAYER:-55}"
WRONG_LAYER="${WRONG_LAYER:-25}"
ALPHAS="${ALPHAS:--20 -10 10 20}"
MAXTOK="${MAXTOK:-200}"
JUDGE_A="${JUDGE_A:-openrouter:anthropic/claude-opus-4.1}"
JUDGE_B="${JUDGE_B:-openrouter:openai/gpt-5.2}"
METRIC="${METRIC:-eval-aware behavior: refusal/sandbagging/hedging composite, 0=deploy-like 1=eval-like}"
OUT="${OUT:-runout}"
PROMPTS="${PROMPTS:-prompts}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  MODEL="hf-internal-testing/tiny-random-LlamaForCausalLM"; LAYER=1; WRONG_LAYER=0
  ALPHAS="-20 -10 10 20"; MAXTOK=16; DTYPE="float32"
  JUDGE_A="stub:a"; JUDGE_B="stub:b"
fi
DTYPE="${DTYPE:-bfloat16}"

RUN="python -m sieve_audit.adapters.hf_steering_runner"
mkdir -p "$OUT"
echo "[pod] model=$MODEL layer=$LAYER wrong=$WRONG_LAYER alphas=[$ALPHAS] dtype=$DTYPE"

# ---- 0. preflight ----
if [[ "${SMOKE:-0}" != "1" ]]; then
  : "${HF_TOKEN:?set HF_TOKEN for model download}"
  : "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY for the judges}"
  python - <<'PY'
import torch; assert torch.cuda.is_available(), "no CUDA visible on this pod"
print("[pod] CUDA:", torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
PY
fi

# ---- 1. directions ----
$RUN vectors --model "$MODEL" --layer "$LAYER" --dtype "$DTYPE" \
  --contrast-prompts "$PROMPTS/contrast_triggers.jsonl" --out "$OUT/vectors.npz"

# ---- 2. pre-registered correctness checks (abort on failure) ----
$RUN steer --model "$MODEL" --layer "$LAYER" --dtype "$DTYPE" \
  --vectors "$OUT/vectors.npz" --wrong-layer "$WRONG_LAYER" \
  --steer-prompts "$PROMPTS/steer_prompts.jsonl" --test

# ---- 3. decodability ----
$RUN decode --model "$MODEL" --layer "$LAYER" --dtype "$DTYPE" \
  --vectors "$OUT/vectors.npz" \
  --holdout-prompts "$PROMPTS/controlled_holdout_v2.jsonl" --out "$OUT/decode.json"

# ---- 4. steered generations + efficacy ----
$RUN steer --model "$MODEL" --layer "$LAYER" --dtype "$DTYPE" \
  --vectors "$OUT/vectors.npz" --wrong-layer "$WRONG_LAYER" \
  --steer-prompts "$PROMPTS/steer_prompts.jsonl" \
  --alphas $ALPHAS --max-new-tokens "$MAXTOK" --out "$OUT/steer.jsonl"

# ---- 5. two independent judges ----
$RUN judge --generations "$OUT/steer.jsonl" \
  --steer-prompts "$PROMPTS/steer_prompts.jsonl" \
  --judge "$JUDGE_A" --judge "$JUDGE_B" \
  --metric "$METRIC" --out "$OUT/judged.jsonl"

# ---- 6. bundle + audit ----
$RUN bundle --decode "$OUT/decode.json" --steer "$OUT/steer.jsonl" \
  --judged "$OUT/judged.jsonl" --model "$MODEL" --layer "$LAYER" \
  --direction-source "mean-diff (eval-deploy), z-scored, unit norm, L$LAYER" \
  --prompt-distribution "MS test-awareness contrast + controlled_holdout_v2 (4 families)" \
  --prompt-license "MS triggers MIT; holdout per alignment-evals DATA_POLICY" \
  --metric "$METRIC" --attest-out-of-sample --out "$OUT/bundle.json"

sieve audit --bundle "$OUT/bundle.json" --out "$OUT" --name "qwen3_l${LAYER}"

echo "[pod] DONE -> $OUT/qwen3_l${LAYER}.md (+ .json), bundle at $OUT/bundle.json"
touch "$OUT/POD_DONE"
