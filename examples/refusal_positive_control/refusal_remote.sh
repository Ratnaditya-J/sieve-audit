#!/bin/bash
# Runs ON the pod from /root/sieve (which has the edited sieve_audit + this example).
# Real SIEVE pipeline on the Arditi refusal direction, judge-free metric, with two
# hard reproduce-gates (sufficiency in calibrate.py, necessity in check_necessity_gate.py).
set -uo pipefail
cd /root/sieve
export PYTHONPATH=/root/sieve/src
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
pip install -q transformers accelerate scikit-learn scipy numpy 2>/dev/null
rm -f /root/R_DONE /root/R_FAIL
RUN="python -m sieve_audit.adapters.hf_steering_runner"
EX=examples/refusal_positive_control
OUT=$EX/out; mkdir -p "$OUT" reports
MODEL="Qwen/Qwen2.5-7B-Instruct"
WRONG=3
die(){ echo "$1"; touch /root/R_FAIL; exit 1; }

echo "=== B. decode-lofo (build direction + probe AUROC vs surface; CHEAP GATE) ==="
$RUN decode-lofo --model "$MODEL" --layer-candidates 12 13 14 15 16 --n-random-controls 1 \
  --holdout-prompts "$EX/prompts/decode.jsonl" --pool last --surface-text-field scenario \
  --save-vectors "$OUT/vectors.npz" --out "$OUT/decode.json" || die DECODE_FAIL
python - "$OUT/decode.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print("[gate] decode keys:", {k:d[k] for k in d if any(t in k.lower() for t in
      ("auroc","baseline","beats","layer","violation"))})
PY

echo "=== C. steer candidate pool at +/-0.3 (for swing calibration) ==="
$RUN steer --model "$MODEL" --vectors "$OUT/vectors.npz" --wrong-layer $WRONG \
  --steer-prompts "$EX/prompts/steer_pool.jsonl" --alphas -0.3 0.3 --max-new-tokens 128 \
  --out "$OUT/pool_steer.jsonl" || die POOLSTEER_FAIL

echo "=== D. calibrate borderline steer set (sufficiency reproduce-GATE) ==="
python "$EX/calibrate.py" "$OUT/pool_steer.jsonl" "$EX/prompts/steer_pool.jsonl" \
  "$OUT/steer.jsonl" "$OUT/calib_report.json" 40 || die CALIB_GATE_FAIL

echo "=== E. full steer on selected borderline set (symmetric grid) ==="
$RUN steer --model "$MODEL" --vectors "$OUT/vectors.npz" --wrong-layer $WRONG \
  --steer-prompts "$OUT/steer.jsonl" --alphas -0.3 -0.15 0.15 0.3 --max-new-tokens 128 \
  --out "$OUT/steer_gen.jsonl" || die STEER_FAIL

echo "=== F. judge steer (deterministic, judge-free) ==="
$RUN judge --generations "$OUT/steer_gen.jsonl" --steer-prompts "$OUT/steer.jsonl" \
  --judge refusal:lexical --judge refusal:graded --metric refusal \
  --out "$OUT/judged_steer.jsonl" || die JUDGE_FAIL

echo "=== G. ablate harmful set (necessity) ==="
$RUN ablate --model "$MODEL" --vectors "$OUT/vectors.npz" \
  --eval-prompts "$EX/prompts/ablate.jsonl" --max-new-tokens 128 \
  --out "$OUT/ablate_gen.jsonl" || die ABLATE_FAIL

echo "=== H. judge ablate + necessity reproduce-GATE ==="
$RUN judge --generations "$OUT/ablate_gen.jsonl" --steer-prompts "$EX/prompts/ablate.jsonl" \
  --judge refusal:lexical --judge refusal:graded --metric refusal \
  --out "$OUT/judged_ablate.jsonl" || die JUDGEABL_FAIL
python "$EX/check_necessity_gate.py" "$OUT/judged_ablate.jsonl" || die NEC_GATE_FAIL

echo "=== I. bundle (decode + steer + judged + judged-ablation -> one card) ==="
$RUN bundle --decode "$OUT/decode.json" --steer "$OUT/steer_gen.jsonl" \
  --judged "$OUT/judged_steer.jsonl" --judged-ablation "$OUT/judged_ablate.jsonl" \
  --model "$MODEL" \
  --direction-source "Arditi refusal direction: mean-diff (z-scored, last-token, unit-norm) of harmful vs harmless activations; XSTest; leave-one-family-out" \
  --prompt-distribution "XSTest (Rottger et al. 2024): surface-matched safe/unsafe prompts" \
  --prompt-license "XSTest (CC-BY-4.0)" --metric refusal --attest-out-of-sample \
  --out "$OUT/bundle.json" || die BUNDLE_FAIL

echo "=== J. audit -> verdict card ==="
python -m sieve_audit.cli audit --bundle "$OUT/bundle.json" --out reports/ --name refusal_qwen7b \
  || die AUDIT_FAIL

echo "=== VERDICT ==="
python - reports/refusal_qwen7b.json <<'PY'
import json,sys,glob
p=sys.argv[1]
try: d=json.load(open(p))
except:
  cand=glob.glob("reports/*refusal*json"); d=json.load(open(cand[0])) if cand else {}
print("verdict:", d.get("verdict"), "| label:", d.get("label"), "| status:", d.get("status"))
PY
touch /root/R_DONE
echo "=== REMOTE DONE ==="
