# Inspect integration

SIEVE ships an [Inspect](https://inspect.aisi.org.uk/) task/scorer so verdicts
and audit cards slot into the Inspect ecosystem — `inspect eval`, the log
viewer, and later `inspect_evals`. This is the adoption highway (DESIGN.md
§10): the mechanism by which the protocol becomes a *cited, runnable* standard
rather than a one-off script. Implemented in
[`src/sieve_audit/inspect_task.py`](../src/sieve_audit/inspect_task.py).

Install the extra:

```bash
pip install "sieve-audit[inspect]"
```

## What it provides

SIEVE audits recorded **evidence bundles**, not live model generations, so the
integration is deliberately model-free: each Inspect `Sample` points at a
bundle JSON, a no-op solver passes it through, and the scorer runs the audit.

- **`sieve_scorer(seed=0)`** — an Inspect scorer that loads the bundle named by
  each sample, runs `run_audit`, and returns a `Score` whose value is the
  verdict (or `insufficient_protocol`), whose `answer` is the verdict string,
  whose `explanation` is the full rendered Markdown audit card, and whose
  `metadata` carries the config/bundle hashes and the strict-profile status.
- **`verdict_distribution`** — a metric tallying the fraction of audited
  bundles landing on each verdict, e.g. `{causally_sufficient: 0.2,
  surface_confounded: 0.5, insufficient_protocol: 0.3}`.
- **`sieve_task(bundles, seed=0)`** — wraps a list of bundle paths (or a
  directory of `*.json`, or a comma-separated string) into a runnable `Task`.

## Run it

```bash
# audit a directory of evidence bundles
inspect eval sieve_audit.inspect_task@sieve_task -T bundles=reports/bundles/

# or a specific set
inspect eval sieve_audit.inspect_task@sieve_task -T bundles=a.json,b.json
```

or from Python:

```python
from inspect_ai import eval as inspect_eval
from sieve_audit.inspect_task import sieve_task

log = inspect_eval(sieve_task(["bundle_a.json", "bundle_b.json"]))[0]
for sample in log.samples:
    score = list(sample.scores.values())[0]
    print(sample.id, "->", score.value)   # the verdict
    print(score.explanation)              # the full audit card
```

(`inspect eval` requires a `--model`; SIEVE never calls it, so any placeholder
such as `mockllm/model` works.)

## Verified

`tests/test_inspect.py` runs a real `inspect eval` over rigged bundles and
asserts the verdicts and the distribution metric — so the integration is tested
against the installed Inspect API, not merely imported. Pinned working against
`inspect-ai` 0.3.240.
