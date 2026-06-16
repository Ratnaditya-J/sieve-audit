# Contributing to SIEVE

SIEVE is a validity auditor: its credibility rests on the verdict logic being
correct and ungameable. Contributions are welcome, with one non-negotiable
rule — **every change to the verdict logic must keep the ground-truth self-test
and the anti-gaming regression tests green.**

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # core + inspect + pytest
pytest                      # full suite (~6 min; bootstrap-heavy)
sieve selftest              # the six rigged ground-truth scenarios
```

## The bar for a change to the engine

1. **`sieve selftest` stays 6/6.** The rigged scenarios in
   `src/sieve_audit/synth.py` encode known ground truth; the engine must
   recover it. If you change a gate, add a scenario that exercises it.
2. **The anti-gaming suite stays green** (`tests/test_antigaming.py`,
   `tests/test_profile.py`). Each test reconstructs a way to fool the auditor;
   none may regress. New gating thresholds **must** be added to the profile
   maps in `config.py` — `tests/test_profile.py::test_every_config_field_is_profile_classified`
   enforces this.
3. **Asymmetry is sacred** (DESIGN.md §7). Any protocol gap must resolve
   *against* the stronger claim. A change that could let a weaker probe earn a
   stronger verdict will be rejected.
4. **Verdicts are reproducible.** Audits must be deterministic from
   `(bundle, config, seed)`; route all randomness through the seeded
   `numpy.random.Generator`.

## Scope discipline

SIEVE deliberately **wraps** activation extraction / probe training / inference
(nnsight, TransformerLens, Inspect) and **owns** only the validity spine. New
features should respect that boundary — adapters produce evidence bundles; the
core audits them and never touches a model.

## Tests

- Engine/stats/gates: `tests/test_gates.py`, `tests/test_selftest.py`
- Anti-gaming + frozen profile: `tests/test_antigaming.py`, `tests/test_profile.py`
- Pre-registration: `tests/test_prereg.py`
- Inspect integration (skipped without `inspect-ai`): `tests/test_inspect.py`

## Data policy

Never commit raw activations, judged run outputs, or non-redistributable
benchmark prompts (see `.gitignore`). Audit cards and evidence bundles contain
scores and flags only — but bundles can embed prompt text, so check the prompt
license before committing one.
