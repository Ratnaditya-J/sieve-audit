# Changelog

All notable changes to SIEVE are recorded here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); SIEVE uses the **protocol
version** (on every audit card) as the audit-compatibility contract, separate
from the package version.

## [Unreleased] — verbalizer faithfulness (branch `verbalizer-faithfulness`)

### Added
- **Verbalizer-faithfulness auditing**: SIEVE now audits activation
  verbalizers (Patchscopes / LatentQA style) as first-class signals — the
  claims are scalarized to `P(claim asserts Y)` and face the existing gates
  (confabulation → `surface_confounded`, CoT-redundancy → Tier-2 leakage,
  faithfulness → causal sufficiency). See
  [`docs/verbalizer_faithfulness.md`](docs/verbalizer_faithfulness.md).
- `VerbalizationEvidence` bundle section; when present with decodability, the
  engine enforces claim-scores ≡ probe-scores (no bait-and-switch verdicts).
- Named **`cot` span category** in the Tier-2 leakage gate, with its own delta
  and matched random control: `cot_leaky` (CoT-parroting) vs `cot_survives`
  (above-chance signal without the CoT — survival is earned, never defaulted).
  Both surface in the card headline (`· cot-leaky` / `· survives-cot-removal`).
- **Verbalizer adapter** (`adapters/verbalizer.py`): recorded-output path
  (GPU-free) and a model path (Patchscopes-style identity readout →
  claim-direction recovery → the existing HF-runner steering controls), with
  two-scorer claim scalarization (`assert:lexical` / `assert:graded` or LLM
  judges).
- `sieve selftest --verbalizer`: four rigged scenarios (faithful /
  confabulating / CoT-parroting / epiphenomenal) must return the rigged
  verdict AND the rigged `cot` flags; regression-tested across seeds in
  `tests/test_verbalizer.py`.

## [0.1.0] — unreleased (protocol v0.1)

First working engine. Audits recorded **evidence bundles** (GPU-free,
reproducible from `(bundle, config, seed)`); adapters produce bundles from
models or published artifacts.

### Added
- Five-state verdict taxonomy + `insufficient_protocol` refusal, with
  claim-calibrated, hash-identified audit cards (Markdown + JSON).
- Audit stages: surface baselines (length / TF-IDF, leave-one-family-out),
  decodability vs chance + baselines, the efficacy gate, matched-control
  steering (random / orthogonal / wrong-layer), two-judge agreement.
- **Frozen strict profile** (`SIEVE-v0.1-strict`): the defaults *are* the bar;
  loosening any threshold voids positive verdicts (hard gate, not a banner),
  while tightening is allowed.
- **Pre-registration mode**: commit config + scope to a hash before results;
  the card states whether the run matched.
- **Inspect integration**: `sieve_task` / `sieve_scorer` / `verdict_distribution`,
  tested against `inspect-ai` 0.3.240.
- **HF steering runner** (`adapters.hf_steering_runner`): vectors → decode →
  steer → judge → bundle, with the four pre-registered correctness checks.
- **Parrack/Apollo adapter** and the first real audit: both published Apollo
  deception probes (arXiv 2502.03407), decodability stage, from released
  artifacts — verdict `insufficient_protocol` (no causal evidence published).
- Ground-truth self-test (`sieve selftest`, 6/6) and an anti-gaming regression
  suite hardened against two adversarial reviews.

### Known limitations
- Causal **sufficiency** only; necessity/ablation is roadmap.
- Single-layer additive steering; distributed mechanisms untested.
- The flagship full-protocol causal cards (Qwen3-32B L55, gpt-oss-120b
  L15/L34) require a GPU run and are not yet produced.
