# Comprehensive end-to-end SIEVE audit — Qwen2.5-1.5B refusal direction

One real GPU run exercising **every** SIEVE axis on a refusal probe, then audited
GPU-free from the recorded bundle. This is the worked example for the four
validity-bounding features (Tier-2 leakage, multi-layer ablation, oracle
patching, deployment lens) alongside the original decodability / efficacy /
sufficiency / necessity stages.

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`, layer 14 (of 28)
- **Behavior:** refusal of harmful requests
- **Direction:** contrastive mean-difference (z-scored, last-token), built
  leave-one-family-out so every decodability score is out-of-family
- **Prompts:** AdvBench harmful (llm-attacks, public) vs templated benign
- **Judges:** `gpt-4o-mini` + `gpt-4.1-mini` (two independent judges)
- **Cost:** ~$0.54 on a community/secure A40 (generation + scoring on the pod;
  judging + audit run locally after teardown).

## Verdict: `surface_confounded · necessary`

The headline is the whole point of SIEVE in one line: a probe that is **perfectly
decodable yet surface-confounded**, but whose direction is **causally necessary**.
Decodability alone would have over-claimed; the causal tests tell the real story;
and where an axis can't be earned, the audit abstains instead of guessing.

| Axis | Result | Evidence |
| --- | --- | --- |
| Decodability | **surface_confounded** | probe AUROC **1.00**, but a TF-IDF baseline hits 0.969 and length 0.984 on held-out families — the separation is explained by surface text statistics |
| Tier-2 leakage | **not leaky** | removing the giveaway harmful spans (vs a random-span control) does not collapse the probe |
| Efficacy (steering) | effective; dose-responsive | median rel. residual movement 0.20 at max \|α\|; dose–response ρ=0.72 (p=0.001) |
| Sufficiency | not established | steering does not significantly move refusal beyond matched controls; superseded anyway by the decodability verdict |
| **Necessity (ablation)** | **NECESSARY** | probe-ablation refusal drop 0.33 vs random control (95% CI 0.19–0.49) |
| **Multi-layer (joint)** | **NECESSARY** | joint ablation across layers {12,14,16} also exceeds the random control — not a single-layer artifact |
| **Oracle (patching)** | **inconclusive** (honest) | single-site last-token patching at layer 14 produced no measurable behavioral gap to calibrate against; SIEVE refuses to certify direction-faithfulness without an oracle effect |
| Deployment lens | quantified | ROC for in-distribution / off-distribution / leakage-removed; recall reported at 1/5/10% FPR (≈100% here — the harmful-vs-benign task is trivially separable, which is also *why* surface-confounding shows up) |

## Files

- `bundle_all.json` — the complete evidence bundle (all eight sections). Re-audit
  with `sieve audit --bundle bundle_all.json`.
- `reports/comprehensive.md` / `.json` — the canonical audit card.
- `reports/comprehensive.html` — printable deployment report (embedded ROC).
- `reports/comprehensive.pdf` — the same as a single PDF.
- `reports/comprehensive.roc.svg` — standalone ROC chart.

## Reproduce

Datasets and the exact runner invocations are in `docs/comprehensive_gpu_run.md`.
The pod runs only generation + scoring; judging and the audit are GPU-free and
run locally from the pulled artifacts.

## Honest caveats

- The harmful-vs-benign task is **easy** (AUROC 1.0, deployment recall ~100%).
  That makes the surface-confounding finding clean but the deployment numbers
  uninteresting; a subtler behavior would stress the lens harder.
- The oracle is inconclusive because single-site last-token activation patching
  does not transfer refusal in this small model — a genuine limitation of that
  intervention here, surfaced rather than papered over. A stronger
  (multi-position / multi-layer) patch would be needed to calibrate faithfulness.
