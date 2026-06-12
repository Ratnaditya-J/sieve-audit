"""Unit tests for the individual gates and statistical primitives."""
import numpy as np
import pytest

from sieve_audit import AuditConfig, run_audit
from sieve_audit.bundle import EfficacyRecord, SteeringRecord
from sieve_audit.controls import run_controls
from sieve_audit.efficacy import run_efficacy
from sieve_audit.stats import cohen_kappa, dose_response_clustered
from sieve_audit.synth import scenario_causally_sufficient
from sieve_audit.verdict import INSUFFICIENT_PROTOCOL, Verdict

CFG = AuditConfig()


# --- efficacy gate -----------------------------------------------------------


def _eff(alpha, delta, base=100.0, expected=None, changed=True, pid="p0", arm="probe"):
    return EfficacyRecord(
        alpha=alpha,
        prompt_id=pid,
        resid_delta_norm=delta,
        resid_base_norm=base,
        expected_delta_norm=abs(alpha) if expected is None else expected,
        output_changed=changed,
        arm=arm,
    )


def test_efficacy_gate_cannot_pass_by_omission():
    with pytest.raises(ValueError):
        run_efficacy([], CFG)
    with pytest.raises(ValueError):  # records exist, but not for the probe arm
        run_efficacy([_eff(20.0, 20.0, arm="random")], CFG, arm="probe")


def test_efficacy_gate_rejects_alpha_zero_only():
    with pytest.raises(ValueError):
        run_efficacy([_eff(0.0, 0.0, changed=False)], CFG)


def test_dirty_noop_is_a_hook_bug():
    records = [
        _eff(0.0, 5.0, changed=False),  # alpha=0 moved the stream: bug
        _eff(20.0, 20.0, expected=20.0),
    ]
    res = run_efficacy(records, CFG)
    assert not res.noop_ok and not res.effective


def test_dead_layer_is_ineffective_not_null():
    records = [_eff(0.0, 0.0, changed=False)] + [
        _eff(20.0, 1e-4, expected=20.0, changed=False, pid=f"p{i}") for i in range(5)
    ]
    res = run_efficacy(records, CFG)
    assert not res.effective


# --- controls / judges -------------------------------------------------------


def _steer(arm, alpha, pid, a, b=None):
    scores = {"judge_a": a, "judge_b": a if b is None else b}
    return SteeringRecord(arm=arm, alpha=alpha, prompt_id=pid, judge_scores=scores)


def test_missing_probe_arm_raises():
    with pytest.raises(ValueError):
        run_controls([_steer("random", 10.0, "p0", 0.5)], CFG)


def test_single_judge_refuses_causal_verdict():
    bundle = scenario_causally_sufficient()
    for r in bundle.steering:
        r.judge_scores = {"judge_a": r.judge_scores["judge_a"]}
    card = run_audit(bundle).card
    assert card.status == INSUFFICIENT_PROTOCOL
    assert card.verdict is None


def test_disagreeing_judges_block_causally_sufficient_but_not_the_negative():
    """Unreliable judges must never upgrade a signal, and must never rescue it."""
    rng = np.random.default_rng(0)
    bundle = scenario_causally_sufficient()
    for r in bundle.steering:  # judge_b becomes pure noise
        r.judge_scores["judge_b"] = float(rng.uniform())
    card = run_audit(bundle).card
    assert card.verdict == Verdict.NOT_CAUSALLY_SUFFICIENT


def test_missing_control_arm_refuses_causal_verdict():
    bundle = scenario_causally_sufficient()
    bundle.steering = [r for r in bundle.steering if r.arm != "wrong_layer"]
    card = run_audit(bundle).card
    assert card.status == INSUFFICIENT_PROTOCOL
    assert card.verdict is None


def test_decodability_only_bundle_gets_no_causal_verdict():
    bundle = scenario_causally_sufficient()
    bundle.steering = []
    bundle.efficacy = []
    card = run_audit(bundle).card
    assert card.status == INSUFFICIENT_PROTOCOL


# --- stats -------------------------------------------------------------------


def test_cohen_kappa_alternating_and_constant():
    a = np.array([0, 1, 0, 1, 0, 1] * 10)
    assert cohen_kappa(a, a) == 1.0
    assert abs(cohen_kappa(a, 1 - a)) > 0.9  # systematic disagreement -> strongly negative
    const = np.ones(60, dtype=int)
    assert np.isnan(cohen_kappa(const, const))  # constant raters: no evidence


def test_dose_response_needs_three_alphas():
    deltas = {1.0: {"p0": 0.1, "p1": 0.2}, 2.0: {"p0": 0.2, "p1": 0.3}}
    rho, p = dose_response_clustered(deltas, np.random.default_rng(0))
    assert rho == 0.0 and p == 1.0


def test_dose_response_monotone():
    rng = np.random.default_rng(0)
    deltas = {
        float(a): {f"p{i}": 0.01 * a + rng.normal(0, 0.02) for i in range(20)}
        for a in (-20, -10, 10, 20)
    }
    rho, p = dose_response_clustered(deltas, rng, n_perm=500)
    assert abs(rho) > 0.8 and p < 0.01


def test_dose_response_null_is_not_significant():
    """Within-prompt permutation must not inflate significance on null data."""
    rng = np.random.default_rng(0)
    # per-prompt offsets shared across alphas (the pseudo-replication trap)
    offsets = {f"p{i}": rng.normal(0, 0.2) for i in range(20)}
    deltas = {
        float(a): {p: off + rng.normal(0, 0.02) for p, off in offsets.items()}
        for a in (-20, -10, 10, 20)
    }
    _, p = dose_response_clustered(deltas, rng, n_perm=500)
    assert p > 0.05
