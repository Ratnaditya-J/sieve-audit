"""Regression tests for the multi-draw null distribution (Change 5).

Four invariants:
1. An extra random_N arm that equals the probe blocks causally_sufficient.
2. Missing efficacy for random_N → INSUFFICIENT_PROTOCOL (engine gap).
3. min_random_controls > actual draws → protocol violation → INSUFFICIENT_PROTOCOL.
4. min_random_controls satisfied → verdict unchanged (regression guard).
"""
import numpy as np
import pytest

from sieve_audit import AuditConfig, run_audit
from sieve_audit.bundle import EfficacyRecord, SteeringRecord
from sieve_audit.synth import scenario_causally_sufficient
from sieve_audit.verdict import INSUFFICIENT_PROTOCOL, Verdict


def _clone_arm(records, src_arm: str, dst_arm: str):
    """Return copies of `src_arm` records relabelled as `dst_arm`."""
    return [
        type(r)(
            **{**r.__dict__, "arm": dst_arm}
        )
        for r in records
        if r.arm == src_arm
    ]


# ---------------------------------------------------------------------------
# 1. extra random arm that matches probe blocks causally_sufficient
# ---------------------------------------------------------------------------

def test_extra_random_matching_probe_blocks_causally_sufficient():
    """random_1 records identical to probe → |probe| - |random_1| ≈ 0 → not causal."""
    bundle = scenario_causally_sufficient()
    # clone probe steering records as random_1 — same effect size, so probe cannot beat it
    bundle.steering = bundle.steering + _clone_arm(bundle.steering, "probe", "random_1")
    # provide valid efficacy for random_1 (clone from 'random' so liveness check passes)
    bundle.efficacy = bundle.efficacy + _clone_arm(bundle.efficacy, "random", "random_1")
    card = run_audit(bundle).card
    assert card.verdict == Verdict.NOT_CAUSALLY_SUFFICIENT


# ---------------------------------------------------------------------------
# 2. missing efficacy for random_1 → INSUFFICIENT_PROTOCOL
# ---------------------------------------------------------------------------

def test_extra_random_arm_without_efficacy_is_refused():
    """Steering records for random_1 present but no efficacy → engine flags the gap."""
    bundle = scenario_causally_sufficient()
    # add random_1 steering (benign — small effects like canonical random)
    bundle.steering = bundle.steering + _clone_arm(bundle.steering, "random", "random_1")
    # intentionally omit random_1 efficacy records
    card = run_audit(bundle).card
    assert card.status == INSUFFICIENT_PROTOCOL
    reasons = card.diagnostics.get("decision_reasons", [])
    assert any("random_1" in r and "no efficacy" in r for r in reasons), reasons


# ---------------------------------------------------------------------------
# 3. min_random_controls > supplied draws → protocol violation
# ---------------------------------------------------------------------------

def test_min_random_controls_violation_blocks_verdict():
    """Requiring 3 random draws with only 1 present → controls protocol violation."""
    bundle = scenario_causally_sufficient()  # has only canonical 'random'
    cfg = AuditConfig(min_random_controls=3)
    card = run_audit(bundle, cfg).card
    assert card.status == INSUFFICIENT_PROTOCOL
    reasons = card.diagnostics.get("decision_reasons", [])
    assert any("min_random_controls" in r or "random control draws" in r for r in reasons), reasons


# ---------------------------------------------------------------------------
# 4. min_random_controls satisfied → verdict unchanged (regression)
# ---------------------------------------------------------------------------

def test_min_random_controls_satisfied_preserves_verdict():
    """Requiring 2 draws with 2 present (random + random_1 with small effects) → causal."""
    bundle = scenario_causally_sufficient()
    # add random_1 steering with small effects (same as canonical random → probe beats it)
    bundle.steering = bundle.steering + _clone_arm(bundle.steering, "random", "random_1")
    # provide valid efficacy for random_1
    bundle.efficacy = bundle.efficacy + _clone_arm(bundle.efficacy, "random", "random_1")
    cfg = AuditConfig(min_random_controls=2)
    card = run_audit(bundle, cfg).card
    assert card.verdict == Verdict.CAUSALLY_SUFFICIENT
