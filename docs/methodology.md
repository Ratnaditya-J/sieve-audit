# Methodology

The audit protocol (probe decodability -> surface baselines -> efficacy gate ->
matched-control steering -> two-judge behavioral scoring -> verdict) is
specified in [`../DESIGN.md`](../DESIGN.md) sections 3-5 and implemented in
[`src/sieve_audit/`](../src/sieve_audit/):

| Step | Module |
|---|---|
| evidence-bundle contract | [`bundle.py`](../src/sieve_audit/bundle.py) ([format spec](bundle_format.md)) |
| surface baselines (length / TF-IDF) | [`baselines.py`](../src/sieve_audit/baselines.py) |
| decodability vs chance + baselines | [`decodability.py`](../src/sieve_audit/decodability.py) |
| efficacy gate (per arm) | [`efficacy.py`](../src/sieve_audit/efficacy.py) |
| matched-control steering + judges | [`controls.py`](../src/sieve_audit/controls.py) |
| statistics (stratified bootstrap, clustered permutation, kappa) | [`stats.py`](../src/sieve_audit/stats.py) |
| verdict decision + claim calibration | [`verdict.py`](../src/sieve_audit/verdict.py) |
| cross-stage consistency + orchestration | [`engine.py`](../src/sieve_audit/engine.py) |
| audit cards | [`card.py`](../src/sieve_audit/card.py) |

The engine is validated by a ground-truth self-test (`sieve selftest`): six
rigged evidence bundles whose correct verdicts are known by construction must
audit to exactly those verdicts, across seeds. Anti-gaming behavior is pinned
by regression tests reconstructing the exploits found in adversarial review
([`tests/test_antigaming.py`](../tests/test_antigaming.py)).

Origin: the controlled causal-sufficiency protocol from *"Probing Is Not
Enough"* (companion paper).
