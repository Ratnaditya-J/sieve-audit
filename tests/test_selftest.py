"""The ground-truth self-test: rigged scenarios must yield the rigged verdicts.

This is the validity check for the validity checker. Every scenario constructs
an evidence bundle whose correct verdict is known by construction; the engine
must recover it. Runs across multiple seeds so a verdict cannot pass by luck.
"""
import pytest

from sieve_audit import AuditConfig, run_audit
from sieve_audit.synth import SCENARIOS
from sieve_audit.verdict import INSUFFICIENT_PROTOCOL

SEEDS = [0, 1, 2]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("expected", list(SCENARIOS))
def test_rigged_scenario_returns_rigged_verdict(expected: str, seed: int):
    bundle = SCENARIOS[expected](seed=seed)
    result = run_audit(bundle, AuditConfig(seed=seed))
    got = result.card.verdict.value if result.card.verdict else result.card.status
    assert got == expected, (
        f"scenario rigged as {expected!r} audited as {got!r} (seed={seed}); "
        f"reasons: {result.card.diagnostics['decision_reasons']}"
    )


def test_insufficient_protocol_card_licenses_no_causal_claims():
    bundle = SCENARIOS[INSUFFICIENT_PROTOCOL]()
    card = run_audit(bundle).card
    assert card.verdict is None
    assert card.status == INSUFFICIENT_PROTOCOL
    # decodability passed cleanly, so its correlational claim is licensed,
    # but every causal/monitor claim must be refused
    assert any("decodable" in c for c in card.allowed_claims)
    assert any("NO causal" in c for c in card.allowed_claims)
    assert not any("causally sufficient" in c.lower() for c in card.allowed_claims)
    assert "Any safety or causal claim." in card.disallowed_claims


def test_cards_are_reproducible():
    bundle = SCENARIOS["causally_sufficient"]()
    a = run_audit(bundle, AuditConfig(seed=0)).card
    b = run_audit(bundle, AuditConfig(seed=0)).card
    assert a.bundle_hash == b.bundle_hash
    assert a.config_hash == b.config_hash
    assert a.diagnostics == b.diagnostics
