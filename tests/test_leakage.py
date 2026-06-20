"""Tier-2 leakage gate tests (Boxo-style).

Leaky ⟺ AUROC collapses under leak-span removal but survives random-span removal.
"""
from sieve_audit.bundle import EvidenceBundle, LeakageEvidence
from sieve_audit.config import AuditConfig
from sieve_audit.leakage import run_leakage


def _leak_ev(n: int, leak_collapses: bool) -> LeakageEvidence:
    labels = [i % 2 for i in range(n)]                 # balanced 0/1
    perfect = [float(l) for l in labels]               # AUROC ~1.0
    return LeakageEvidence(
        labels=labels,
        probe_scores_full=perfect,
        probe_scores_leak_removed=([0.5] * n if leak_collapses else perfect),
        probe_scores_random_removed=perfect,           # control survives
    )


def _bundle(leak_ev):
    return EvidenceBundle(
        model="m", revision=None, layers=[1], direction_source="d",
        prompt_distribution="x", prompt_license="y", behavioral_metrics=["b"],
        adapter="t", leakage=leak_ev,
    )


def test_leaky_when_signal_collapses_under_leak_removal_only():
    res = run_leakage(_leak_ev(60, leak_collapses=True), AuditConfig(seed=0))
    assert not res.inconclusive and res.leaky
    assert res.drop_leak.lo >= 0.05 and res.drop_leak.lo > res.drop_random.hi


def test_not_leaky_when_probe_survives_leak_removal():
    res = run_leakage(_leak_ev(60, leak_collapses=False), AuditConfig(seed=0))
    assert not res.inconclusive and not res.leaky


def test_leakage_round_trips():
    b = _bundle(_leak_ev(20, leak_collapses=True))
    again = EvidenceBundle.from_dict(b.to_dict())
    assert again.leakage is not None
    assert again.leakage.labels == b.leakage.labels


def test_leakage_surfaces_in_card():
    from sieve_audit import run_audit

    card = run_audit(_bundle(_leak_ev(60, leak_collapses=True))).card
    assert card.diagnostics["leakage"]["leaky"] is True
    assert "leaky" in card.label
    assert any("LEAKY" in r for r in card.residual_risks)
