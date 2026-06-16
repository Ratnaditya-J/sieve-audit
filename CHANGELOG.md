# Changelog

All notable changes to SIEVE are recorded here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); SIEVE uses the **protocol
version** (on every audit card) as the audit-compatibility contract, separate
from the package version.

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
