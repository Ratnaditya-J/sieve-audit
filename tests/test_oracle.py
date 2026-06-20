"""Oracle (activation-patching) calibration tests.

Faithful ⟺ patching the audited direction recovers a large fraction of the
full-site (oracle) effect AND beats the random-patch control. The key value: a
direction can be necessary/sufficient-ish yet UNFAITHFUL — the site is causal but
the mechanism doesn't run through the audited coordinate.
"""
import pytest

from sieve_audit.bundle import EvidenceBundle, PatchingRecord
from sieve_audit.config import AuditConfig
from sieve_audit.oracle import run_oracle


def _patch(n, corrupt, full, direction, random, clean=None, layers=(5,), judges=("j1", "j2")):
    arms = [("corrupt", corrupt), ("patch_full", full),
            ("patch_direction", direction), ("patch_random", random)]
    if clean is not None:
        arms.append(("clean", clean))
    recs = []
    for i in range(n):
        pid = f"p{i}"
        for arm, v in arms:
            recs.append(PatchingRecord(arm=arm, prompt_id=pid, layers=list(layers),
                                       judge_scores={j: v for j in judges}))
    return recs


def _bundle(**kw):
    return EvidenceBundle(
        model="m", revision=None, layers=[5], direction_source="contrastive",
        prompt_distribution="d", prompt_license="x", behavioral_metrics=["b"],
        adapter="test", **kw,
    )


def test_faithful_when_direction_recovers_most_of_oracle_effect():
    res = run_oracle(
        _patch(25, corrupt=0.1, full=0.9, direction=0.8, random=0.2, clean=0.95),
        AuditConfig(seed=0),
    )
    assert not res.inconclusive and res.faithful
    assert res.recovered_fraction.lo >= 0.5
    assert res.direction_vs_random.lo > 0
    assert res.patch_completeness is not None


def test_unfaithful_when_direction_recovers_little():
    res = run_oracle(
        _patch(25, corrupt=0.1, full=0.9, direction=0.25, random=0.2),
        AuditConfig(seed=0),
    )
    assert not res.inconclusive and not res.faithful
    assert res.recovered_fraction.point < 0.5


def test_inconclusive_when_full_patch_restores_nothing():
    # full-site patch restores nothing (corrupt == patch_full): no causal content
    # at the site to calibrate the direction against
    res = run_oracle(
        _patch(25, corrupt=0.10, full=0.10, direction=0.10, random=0.10),
        AuditConfig(seed=0),
    )
    assert res.inconclusive and not res.faithful
    assert any("no measurable causal content" in n for n in res.notes)


def test_inconclusive_without_random_control():
    recs = [r for r in _patch(25, 0.1, 0.9, 0.8, 0.2) if r.arm != "patch_random"]
    res = run_oracle(recs, AuditConfig(seed=0))
    assert res.inconclusive and not res.faithful


def test_mismatched_sites_rejected():
    recs = _patch(25, 0.1, 0.9, 0.8, 0.2)
    recs[0].layers = [9]
    with pytest.raises(ValueError, match="share one site"):
        run_oracle(recs, AuditConfig(seed=0))


def test_patching_round_trips_and_rejects_dupes():
    recs = _patch(2, 0.1, 0.9, 0.8, 0.2)
    b = _bundle(patching=recs)
    b.validate()
    assert "patch_full" in b.patching_arms
    assert EvidenceBundle.from_dict(b.to_dict()).patching == recs
    with pytest.raises(ValueError, match="duplicate patching"):
        _bundle(patching=recs + [PatchingRecord(arm="corrupt", prompt_id="p0",
                layers=[5], judge_scores={"j": 0.1})]).validate()


def test_unfaithful_surfaces_in_card_end_to_end():
    from sieve_audit import run_audit
    from sieve_audit.synth import SCENARIOS

    bundle = SCENARIOS["causally_sufficient"]()
    bundle.patching = _patch(25, corrupt=0.1, full=0.9, direction=0.25, random=0.2)
    card = run_audit(bundle).card
    assert card.diagnostics["oracle"]["faithful"] is False
    assert card.diagnostics["causal_summary"]["oracle"] == "unfaithful"
    assert any("DIRECTION UNFAITHFUL" in r for r in card.residual_risks)
    assert any(i.startswith("activation patching") for i in card.tested_interventions)


def test_faithful_surfaces_in_card_end_to_end():
    from sieve_audit import run_audit
    from sieve_audit.synth import SCENARIOS

    bundle = SCENARIOS["causally_sufficient"]()
    bundle.patching = _patch(25, corrupt=0.1, full=0.9, direction=0.82, random=0.2)
    card = run_audit(bundle).card
    assert card.diagnostics["oracle"]["faithful"] is True
    assert card.diagnostics["causal_summary"]["oracle"] == "faithful"
    assert any("FAITHFUL coordinate" in c for c in card.allowed_claims)
