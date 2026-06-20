# SIEVE roadmap (post v0.1)

Shipped on `sieve-validity-bounding` (the validity-bounding work): method-scoped
verdicts; the **necessity (ablation) gate** as a second causal method; a
**cross-method causal summary**; a **calibration** command (`sieve calibrate`)
that reports the criterion's error rate vs ground truth; a human **headline
label** that surfaces necessity instead of burying it under
`insufficient_protocol`; and a real end-to-end demo (`examples/refusal_necessity`).

The items below came out of design review. Each is additive and keeps the
"never claim more than you tested; report which checks ran" discipline.

## 1. Tier-2 leakage check (Boxo-style) — *keep* TF-IDF, *add* span-removal
TF-IDF/length baselines are a **GPU-free, coarse** surface check that runs on a
recorded bundle (and is the *only* leakage screen possible when auditing a
published artifact with no model access — e.g. the Apollo audit). Keep it.

Add a **stronger, model-access tier** (Boxo et al. 2509.21344): re-run the model
with the leaky spans removed — the elicitation prompt **and** the model's
verbalized reasoning — and check whether probe AUROC collapses. Crucial detail:
include a **random-span-removal control** (removing spans pushes the model
off-distribution; the control separates "removed the *leaky* tokens" from
"removed tokens in general"). New optional bundle evidence + a `leaky` gate.
Same shape as the necessity gate (adapter produces evidence; core adjudicates).

## 2. Deployment lens — a practitioner-facing FP/FN view
The multi-axis verdict is researcher-facing. Add a **second derived view** (not a
replacement) that translates the measured axes into the two errors a product
decision turns on: **false negatives** (misses the real thing — safety-critical)
and **false positives** (over-flags benign — operational cost).

Quantify only what's measured, and label each statement by its source axis:
- **Decodability AUROC / per-family** → ROC + **recall at a fixed FPR** (e.g.
  "catches 78% at 1% false-alarm"). Quantifiable.
- **Cross-distribution calibration** (pooled vs within-family) → threshold drift
  → both FP and FN rise off-distribution. Quantifiable (report the degradation).
- **Leaky** → when giveaway text is absent, detection collapses → **false
  negatives**. Quantifiable from the Tier-2 leakage drop.
- **Necessary / sufficient (causal)** → reliability caution, **not** an invented
  FP/FN number. Optionally add a **specificity** measurement (direction's
  presence in non-target contexts) to quantify the FP side.

Honest constraints to bake in: FP/FN require an **operating point + a
representative distribution**, so report "at threshold T on the audited
distribution … plus off-distribution degradation," with the caveat that real
deployment rates depend on the user's threshold and data. A missing axis →
"**not assessed**," loudly — especially the false-negative (safety) number.

## 3. Multi-layer / distributed interventions
Single-layer steering and ablation under-see **distributed** mechanisms (a
direction re-derived across many layers only partly bites at one). Add
multi-layer steer/ablate to close this gap (the original refusal-ablation result
needed all-layer ablation for a near-total effect). This is the deeper fix behind
the "committee"/under-powered-null limitation that calibration (#below) measures.

## 4. Oracle calibration (stronger #3)
Label *real* directions with an independent oracle — activation patching / causal
mediation — and feed those `(bundle, expected)` cases into `sieve calibrate` to
report the criterion's **measured false-negative rate** on real signals, not just
synthetic ground truth. Needs a GPU run; the report machinery is oracle-agnostic.
