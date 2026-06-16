# A standard validity audit for AI-safety probes — and how the field's headline deception probes hold up

*Draft writeup · 2026-06-16 · companion to [SIEVE](https://github.com/Ratnaditya-J/sieve-audit)*

> **Status note.** Everything below about the Apollo deception probes is
> reproduced from released artifacts and is in the repo. The flagship
> full-protocol *causal* verdicts (Qwen3-32B L55, gpt-oss-120b L15/L34) require
> a GPU run that has not been executed yet; they are marked as such and are not
> claimed as results.

## The problem: "decodable" keeps getting read as "trustworthy"

A fast-growing literature claims to detect deception, sandbagging, or
evaluation-awareness inside language models by reading their activations. The
recurring failure is not that the probes are useless — it is that *decodability
gets overclaimed into causal, monitor-grade trust*. A probe hits AUROC 0.97 on
a held-out set, and that number starts being cited as if the probe had been
shown to track an internal deception state that would survive deployment. Three
things usually go unchecked:

1. **Surface confounds.** Can a length or TF-IDF reader of the same text match
   the probe? If so, the "internal signal" may be the prompt's surface
   statistics.
2. **Causal load-bearingness.** Does steering along the probe direction change
   behavior *more than a random / orthogonal / wrong-layer direction of the
   same magnitude*? Decodable ≠ causal; even random steering moves behavior
   somewhat ([Rogue Scalpel, 2025](https://arxiv.org/abs/2509.22067)).
3. **Whether the intervention even took effect.** A null steering result at a
   layer where steering can't bite is not evidence of "no causal effect"; it's
   evidence of nothing.

[Smith, Chughtai & Nanda (Nov 2025)](https://arxiv.org/abs/2511.22662) put the
underlying difficulty plainly: the field lacks confidently-labeled
deceptive/honest ground truth, and the proposed workarounds "seem insufficient
alone." Leading labs concede the point — Apollo's own probes, at AUROC
0.96–0.999, are described by their authors as "insufficient as a robust defence
against deception." The critique is well-posed. What's missing is a *reusable
tool* that operationalizes it: a gate any probe claim must pass before it's
trusted.

## What SIEVE is

SIEVE is that gate. Given an activation-based safety signal — a probe direction,
SAE feature, or monitor score — it runs a fixed battery and returns one of five
verdicts (or refuses):

1. **not_decodable** — no better than chance on held-out examples.
2. **surface_confounded** — decodable, but a length/TF-IDF baseline matches it.
3. **intervention_ineffective** — steering never took effect; causality is
   *untested*, **not** a null.
4. **not_causally_sufficient** — steering bites, but no more than matched
   random / orthogonal / wrong-layer controls.
5. **causally_sufficient** — exceeds all controls, dose-responsive, judge-agreed.

…or **insufficient_protocol** when the evidence to decide simply wasn't
produced. The output is a scoped, caveat-bound, hash-identified **audit card**:
claims can't be detached from the conditions that earned them, and a system
card citing the hash re-runs to the same result.

Two design choices make it a *standard* rather than a script:

- **A frozen profile.** The defaults *are* `SIEVE-v0.1-strict`. You may tighten
  the bar and keep a positive verdict; loosening any threshold **voids** the
  strong verdict (a hard gate, not a footnote), and the card states the profile
  as a binary an outsider can read. "Passed SIEVE" means passed *that exact
  object*.
- **Pre-registration.** Commit the config and scope to a hash before results
  exist; the card reports whether the run matched. This closes the
  after-the-fact moves — tuning thresholds or swapping to a luckier
  layer/direction once you've peeked.

Every protocol gap resolves *against* the stronger claim. SIEVE's most common
output is expected to be "decodable, but not a validated monitor" — which is
exactly the output a probe vendor would never publish about their own probe,
and exactly why a neutral third-party tool is the point.

## The first real audit: the Apollo deception probes

Using the per-token probe scores released in [Parrack et al.'s
benchmark](https://arxiv.org/abs/2507.12691) (which embed the
[Apollo probes](https://arxiv.org/abs/2502.03407) for Llama-3.3-70B, layer 22),
SIEVE audited both published probes across four evaluation datasets
(**n = 2,352**), entirely from released artifacts — no GPU.

| | pairs_probe | rp_probe |
|---|---|---|
| Decodable above chance (pooled, held-out) | **0.798** [0.780, 0.815] | **0.723** [0.702, 0.743] |
| Best surface baseline (cross-dataset) | 0.462 | 0.462 |
| Probe − baseline (95% CI) | +0.336 [0.299, 0.374] | +0.261 [0.222, 0.302] |
| Efficacy / matched-control steering | *no evidence published* | *no evidence published* |
| **Verdict** | `insufficient_protocol` | `insufficient_protocol` |

Three findings:

1. **The decodability claims survive a fair fight.** Length and TF-IDF readers
   of the same conversations fail to transfer across datasets (AUROC ≈ 0.46),
   while the probes keep real signal. Whatever they read, it isn't
   text-statistics that generalize.
2. **Pooled cross-dataset performance is far below the headline numbers**
   (0.798 / 0.723 vs 0.91–0.99 within-dataset). A single deployed threshold
   would transfer much worse than per-dataset AUROCs suggest; `rp_probe` is at
   chance on one dataset.
3. **No causal verdict is available for any published deception probe** — not
   because the probes failed the causal tests, but because the evidence to run
   them (matched-control steering, efficacy data, two-judge scoring) has never
   been released. SIEVE licenses the probes as *correlational diagnostics* and
   explicitly **not** as validated monitors.

That third point is the constructive answer to "evaluating deception detectors
is harder than you think": here is the specific evidence a probe release must
ship before "monitor" claims are licensed, a protocol that checks it, and a
measurement of exactly where the field's best-published probes stand today.

## Why no one else fills this slot

A [landscape sweep](related_work.md) (deep research + a direct source audit of
UK AISI's Inspect codebase + product audits of the named players) found the
niche empty. The field is densely populated *around* it — probe **builders**
(Apollo), feature **benchmarks** (SAEBench), interpretability/steering
**platforms** (Transluce, Goodfire), **red-teamers** and **guardrails** (Gray
Swan, Haize, Lakera, Robust Intelligence), and a black-box **eval framework**
(Inspect) — but none ships a third-party-runnable probe-*validity* verdict
engine. Even the near-miss names resolve away on inspection: Haize's "Verdict"
judges text outputs; Robust Intelligence's "AI Validation" is black-box model
stress-testing. (Standing caveat: private lab tooling is unknowable, and the
demand-side critiques are all recent, so the gap is real but time-sensitive.)

## What's next

- **The flagship causal cards.** Reproduce the companion paper's Qwen3-32B L55
  result (decodable, *not* causally sufficient) and the two gpt-oss companions
  through SIEVE's full protocol — the first end-to-end `not_causally_sufficient`
  / `intervention_ineffective` verdicts on real models. Staged and runbook-ready
  ([examples/qwen3_eval_deploy](../examples/qwen3_eval_deploy/)); needs ~1
  GPU-day.
- **Adoption.** SIEVE runs as an [Inspect](inspect_integration.md) task/scorer,
  so a verdict drops into existing eval pipelines.

## Honest limitations

SIEVE tests causal **sufficiency**, not necessity (ablation is roadmap); single
layer additive steering only; results are scope-bound to the audited prompt
distribution; behavioral metrics depend on judge quality (agreement is reported,
not assumed); and on closed weights the auditee self-runs the adapter, so SIEVE
provides a standardized honest *format*, not independent verification. None of
these are hidden — they print on every card.

---

*SIEVE is MIT-licensed and validated against synthetic ground truth (`sieve
selftest`, 6/6) and an anti-gaming regression suite hardened across two
adversarial reviews. Repo: <https://github.com/Ratnaditya-J/sieve-audit>.*
