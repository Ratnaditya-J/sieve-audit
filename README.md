# SIEVE

[![CI](https://github.com/Ratnaditya-J/sieve-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/Ratnaditya-J/sieve-audit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)

**Safety Indicator Evidence Validation Engine**: validity checks for AI safety signals.

> Does your safety signal survive controls, or is it just decodable?

SIEVE audits whether an activation-based safety **signal** (a probe direction, SAE
feature, monitor score, …) is *merely decodable* or actually *causally
load-bearing*, and emits a scoped, caveat-bound **evidence card** you can cite.
It is a **validity layer upstream of AI control**: vet a signal *before* it is
trusted in a control protocol or a public safety claim.

Its job is to **prevent overclaiming from activation evidence**, not to prove
models safe.

## Verdict, in one glance

A SIEVE audit returns one of five verdicts:

1. **not decodable**: no better than surface baselines
2. **surface-confounded**: decodable, but a length/TF-IDF baseline matches it
3. **intervention ineffective → inconclusive**: the steer never took effect (not a null!)
4. **not causally sufficient**: effective, but no more than random / orthogonal / wrong-layer controls
5. **causally sufficient**: exceeds controls, dose-responsive, judge-agreed

## Status

**v0.1 engine implemented and self-tested.** The design is specified in
[`DESIGN.md`](DESIGN.md). The engine audits *evidence bundles* (recorded probe
scores, residual-stream movement, judged steering outputs) so the verdict
logic is GPU-free and reproducible from `(bundle, config, seed)`; adapters
produce bundles from models or published artifacts.

- `sieve selftest`: six rigged ground-truth scenarios (noise probe,
  length-confounded probe, dead steering hook, decodable-but-epiphenomenal
  direction, truly causal direction, incomplete protocol) must return exactly
  the rigged verdicts. 6/6.
- Anti-gaming: hardened against an adversarial review (family gerrymandering,
  degenerate controls, duplicate judges, sandbagged efficacy records,
  weakened configs, in-sample probe scores, ...) with regression tests for
  every exploit.
- First real-world audit: the published Apollo deception probes
  (arXiv 2502.03407) from released artifacts.
  See [`examples/apollo_deception/`](examples/apollo_deception/).
- End-to-end GPU run: Qwen2.5-1.5B-Instruct refusal direction, all eight
  axes, multi-draw null (3 random controls).
  See [`examples/comprehensive_refusal/`](examples/comprehensive_refusal/).
  Verdict `surface_confounded · necessary`: probe is causally load-bearing
  (ablation) but scores are explained by surface text statistics on this
  small model. Harder target (non-surface-confounded, subtler behavior) is
  the natural next run.
- Unlearning-completeness auditing (the inverse question — is a *removal* claim
  hollow?). A three-model triptych over matched WMDP correct-vs-distractor
  completion pairs asks whether "unlearned" knowledge is gone or merely
  suppressed while still linearly decodable. CPU-validated end-to-end on real
  `cais/wmdp` (floor holds; a weak model's WMDP signal is correctly called
  `surface_confounded`); the 7B RMU headline (and GRAM, ICML 2026, whose
  deletion claim is loss-verified only) is staged and runbook-ready.
  See [`examples/unlearning_completeness/`](examples/unlearning_completeness/).

v0.1 scope: contrastive residual-stream directions in open-weight decoder LMs
under single-layer additive steering.

## Install

```bash
pip install "git+https://github.com/Ratnaditya-J/sieve-audit"   # pre-PyPI
sieve selftest   # six rigged ground-truth scenarios; verdicts must match (6/6)
```

For development (editable, with the test suite): `pip install -e ".[dev]"`.
Bundle-producing adapters need extras: `".[runner]"` (HF models) and
`".[judges]"` (LLM judges); the audit engine itself needs neither.

## Quickstart: audit your own probe

You bring a probe's per-example **scores**, the ground-truth **labels**, the
**texts** they were scored on, and a **family** id per example (so baselines and
the probe face the same held-out generalization test). SIEVE tells you whether
the signal beats surface baselines, or is just reading the prompt.

```python
from sieve_audit import EvidenceBundle, DecodabilityEvidence, run_audit

bundle = EvidenceBundle(
    model="my-model", revision=None, layers=[12],
    direction_source="my probe", prompt_distribution="my eval set",
    prompt_license="mine", behavioral_metrics=["n/a"], adapter="quickstart:0.1",
    decodability=DecodabilityEvidence(
        texts=texts,                 # raw prompt text per example
        labels=labels,               # 0/1 ground truth per example
        probe_scores=probe_scores,   # YOUR probe's score per example
        families=families,           # prompt-family id per example
        probe_scores_out_of_sample=True,  # attest: probe wasn't trained on these
    ),
)
card = run_audit(bundle).card
print(card.verdict.value)
```

If your probe scores 0.97 AUROC but the *raw text* is just as separable, SIEVE
says so:

```
surface_confounded     # probe AUROC 0.975, but a TF-IDF baseline scores 1.00
```

That's the point: a high AUROC that a word-counter matches isn't evidence the
*model* represents anything. To go past decodability to a **causal** verdict,
add `efficacy` + `steering` records (matched random / orthogonal / wrong-layer
arms) to the bundle. See [`docs/bundle_format.md`](docs/bundle_format.md), or
have an adapter build the bundle from a model (`sieve_audit.adapters.hf_steering_runner`)
or from published artifacts ([`examples/apollo_deception/`](examples/apollo_deception/)).

Audit a saved bundle from the CLI:

```bash
sieve audit --bundle my_bundle.json --name my_probe   # writes an evidence card
```

## Data policy

This repository ships **only openly-licensed prompts**. Raw activations, judged
run outputs, and non-redistributable benchmark prompts are never committed
(see `.gitignore`).

## Writeup

The field-facing summary (the standard, the Apollo audit results, and the
whitespace) is in
[`docs/standard-validity-audit.md`](docs/standard-validity-audit.md).

## Landscape

Where SIEVE sits relative to interpretability platforms, evals/red-teaming
companies, and the probe-validity literature (and why the reusable
validity-harness niche is empty) is documented with sources in
[`docs/related_work.md`](docs/related_work.md).

## Provenance

SIEVE generalizes the controlled causal-sufficiency protocol from the companion
paper *"Probing Is Not Enough"*
([alignment-evals](https://github.com/Ratnaditya-J/alignment-evals/blob/main/docs/probing-is-not-enough.pdf)).

## License

MIT. See [`LICENSE`](LICENSE).
