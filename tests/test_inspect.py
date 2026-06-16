"""The Inspect integration runs end-to-end and returns the right verdicts.

Skipped if inspect_ai isn't installed (it's an optional extra). When present,
this is a real `inspect eval` over rigged bundles — the adoption highway must
actually work against the installed Inspect API, not just import.
"""
import json

import pytest

from sieve_audit.synth import (
    scenario_causally_sufficient,
    scenario_not_causally_sufficient,
    scenario_surface_confounded,
)

inspect_ai = pytest.importorskip("inspect_ai")


def _write_bundles(tmp_path):
    cases = {
        "cs": scenario_causally_sufficient(),
        "ncs": scenario_not_causally_sufficient(),
        "sconf": scenario_surface_confounded(),
    }
    paths = {}
    for name, bundle in cases.items():
        p = tmp_path / f"{name}.json"
        bundle.save(p)
        paths[name] = str(p)
    return paths


def test_inspect_eval_runs_and_returns_correct_verdicts(tmp_path):
    from inspect_ai import eval as ia_eval

    from sieve_audit.inspect_task import sieve_task

    paths = _write_bundles(tmp_path)
    expected = {
        "cs": "causally_sufficient",
        "ncs": "not_causally_sufficient",
        "sconf": "surface_confounded",
    }

    log = ia_eval(
        sieve_task(list(paths.values())),
        model="mockllm/model",
        display="none",
        log_dir=str(tmp_path / "logs"),
    )[0]

    assert log.status == "success"
    by_id = {s.id: list(s.scores.values())[0] for s in log.samples}
    for name, exp in expected.items():
        assert by_id[name].value == exp
        # the full audit card travels as the explanation
        assert by_id[name].explanation and "SIEVE audit card" in by_id[name].explanation


def test_verdict_distribution_metric_tallies_verdicts(tmp_path):
    from inspect_ai import eval as ia_eval

    from sieve_audit.inspect_task import sieve_task

    paths = _write_bundles(tmp_path)
    log = ia_eval(
        sieve_task(list(paths.values())),
        model="mockllm/model",
        display="none",
        log_dir=str(tmp_path / "logs"),
    )[0]
    metrics = {m.name: m.value for m in log.results.scores[0].metrics.values()}
    # three distinct verdicts, one each
    assert metrics["causally_sufficient"] == pytest.approx(1 / 3)
    assert metrics["not_causally_sufficient"] == pytest.approx(1 / 3)
    assert metrics["surface_confounded"] == pytest.approx(1 / 3)
