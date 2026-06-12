# How far does the published evidence for deception probes actually go?

**SIEVE audit summary — Apollo Research deception probes (arXiv 2502.03407)**
*Protocol v0.1 · audited 2026-06-12 from released artifacts only (no GPU, no model inference)*

Full cards: [`apollo_pairs_probe.md`](apollo_pairs_probe.md) ·
[`apollo_rp_probe.md`](apollo_rp_probe.md)

## Setup

The two published deception probes for **Llama-3.3-70B-Instruct, layer 22**
(logistic regression on residual-stream activations):

| Probe | Trained on |
|---|---|
| `pairs_probe` | instructed pairs |
| `rp_probe` | roleplaying |

Evidence source: per-token probe scores embedded in Parrack et al.'s
black_to_white_boosts benchmark dumps (arXiv 2507.12691) over four evaluation
datasets disjoint from both training sets — `ai_liar`, `apps_backdoors`,
`insider_trading`, `sandbagging_wmdp_mmlu` — **n = 2,352** (906 deceptive /
1,446 honest). Probe score per example: mean per-token score over the
assistant's content tokens. Surface baselines (length and TF-IDF logistic
regressions over the full rendered conversation) were trained
leave-one-family-out, facing the same cross-dataset generalization burden as
the probes.

## Results

| Stage | pairs_probe | rp_probe |
|---|---|---|
| Decodable above chance (pooled, held-out) | **0.798** [0.780, 0.815] | **0.723** [0.702, 0.743] |
| Best surface baseline (cross-dataset) | 0.462 | 0.462 |
| Probe − baseline (95% CI) | +0.336 [0.299, 0.374] | +0.261 [0.222, 0.302] |
| Efficacy gate | **no evidence published** | **no evidence published** |
| Matched-control steering | **no evidence published** | **no evidence published** |
| **Verdict** | `insufficient_protocol` | `insufficient_protocol` |

Per-family probe AUROC (probe scores pooled across datasets, scored within family):

| Family | pairs_probe | rp_probe |
|---|---|---|
| ai_liar | 0.966 | 0.914 |
| apps_backdoors | 0.844 | **0.550** |
| insider_trading | 0.989 | 0.981 |
| sandbagging_wmdp_mmlu | 0.957 | 0.784 |

## Findings

1. **The decodability claims survive a fair fight.** Both probes beat
   cross-dataset surface baselines decisively. Length and TF-IDF readers of
   the same conversations *fail to transfer across datasets* (AUROC ≈ 0.46,
   at/below chance), while the probes keep substantial signal. Whatever the
   probes read, it is not text statistics that generalize. (Per-family
   numbers reproduce the published results closely — adapter sanity check.)

2. **Pooled cross-dataset performance is far below the headline numbers.**
   0.798 / 0.723 pooled vs 0.91–0.99 within-family: probe scores are not
   calibrated across distributions, so any *deployed monitor threshold* would
   transfer much worse than per-dataset AUROCs suggest. `rp_probe` is at
   chance (0.550) on `apps_backdoors`.

3. **No causal verdict is available for any published deception probe — not
   because the probes failed the causal tests, but because the evidence
   needed to run them has never been produced.** No efficacy-gate data, no
   matched-control steering (random / orthogonal / wrong-layer), no
   two-judge behavioral scoring. Under the audit's claim calibration, the
   probes are licensed as *correlational diagnostics*, and explicitly **not**
   as validated monitors.

This is the constructive answer to Smith, Chughtai & Nanda's "evaluating
deception detectors is harder than you think" (arXiv 2511.22662): here is the
concrete evidence a probe release must ship before "monitor" claims are
licensed, a protocol that checks it, and a measurement of exactly where the
field's best-published probes stand today.

## What would change the verdict

A steering-stage evidence bundle for layer 22 of Llama-3.3-70B-Instruct:
±α injections of the probe direction with matched random / orthogonal /
wrong-layer controls, efficacy records (residual movement + output change) on
the same prompts, and two independent judges scoring deception in the
generations. SIEVE's bundle format specifies it exactly; ~one GPU-day on an
80GB node. Until someone runs it, "the probe detects deception" remains an
unvalidated causal claim.

## Reproduce

See [`examples/apollo_deception/README.md`](../examples/apollo_deception/README.md).
Card hashes bind these conclusions to the exact config and evidence bundles.
