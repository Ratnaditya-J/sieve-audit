"""Tests for the necessity (ablation) gate — #2.

2a (this file, for now): the evidence bundle carries optional ablation evidence,
round-trips through (de)serialization, and rejects duplicate records — without
disturbing any existing steering-only audit (the gate logic lands in 2b).
"""
import pytest

from sieve_audit.bundle import AblationRecord, EvidenceBundle


def _bundle(ablation):
    return EvidenceBundle(
        model="m",
        revision=None,
        layers=[1],
        direction_source="contrastive",
        prompt_distribution="d",
        prompt_license="x",
        behavioral_metrics=["b"],
        adapter="test",
        ablation=ablation,
    )


def test_ablation_round_trips_and_lists_arms():
    recs = [
        AblationRecord(arm="baseline", prompt_id="p1", judge_scores={"j": 0.9}),
        AblationRecord(arm="probe", prompt_id="p1", judge_scores={"j": 0.2}),
        AblationRecord(arm="ablate_random", prompt_id="p1", judge_scores={"j": 0.85}),
    ]
    b = _bundle(recs)
    b.validate()
    assert b.ablation_arms == ["ablate_random", "baseline", "probe"]
    again = EvidenceBundle.from_dict(b.to_dict())
    assert again.ablation == recs


def test_duplicate_ablation_record_rejected():
    recs = [
        AblationRecord(arm="probe", prompt_id="p1", judge_scores={"j": 0.2}),
        AblationRecord(arm="probe", prompt_id="p1", judge_scores={"j": 0.3}),
    ]
    with pytest.raises(ValueError, match="duplicate ablation"):
        _bundle(recs).validate()


def test_bundle_without_ablation_is_unchanged():
    b = _bundle([])
    b.validate()
    assert b.ablation == []
    assert b.ablation_arms == []
    assert EvidenceBundle.from_dict(b.to_dict()).ablation == []


# --- 2b: the necessity gate ---------------------------------------------------
from sieve_audit.config import AuditConfig  # noqa: E402
from sieve_audit.necessity import run_necessity  # noqa: E402


def _abl(n, base, probe, rand, judges=("j1", "j2")):
    recs = []
    for i in range(n):
        pid = f"p{i}"
        for arm, v in (("baseline", base), ("probe", probe), ("ablate_random", rand)):
            recs.append(
                AblationRecord(arm=arm, prompt_id=pid, judge_scores={j: v for j in judges})
            )
    return recs


def test_necessity_detected_when_probe_ablation_dominates_control():
    res = run_necessity(_abl(30, base=0.9, probe=0.2, rand=0.88), AuditConfig(seed=0))
    assert not res.inconclusive
    assert res.necessary
    assert res.probe_drop.lo > 0 and res.probe_vs_random_drop.lo > 0


def test_not_necessary_when_drop_matches_random_control():
    res = run_necessity(_abl(30, base=0.9, probe=0.85, rand=0.85), AuditConfig(seed=0))
    assert not res.inconclusive
    assert not res.necessary


def test_inconclusive_without_random_control():
    recs = [r for r in _abl(30, 0.9, 0.2, 0.88) if r.arm != "ablate_random"]
    res = run_necessity(recs, AuditConfig(seed=0))
    assert res.inconclusive
    assert not res.necessary
    assert not res.has_random_control


def test_necessity_enriches_not_causally_sufficient_card_end_to_end():
    """Wiring (#2b-ii): attaching 'necessary' ablation evidence to a
    not_causally_sufficient audit must (a) leave the sufficiency verdict intact,
    (b) add 'ablation' to tested_interventions, and (c) surface the
    'necessary -> not causally inert' claim and necessity diagnostics."""
    from sieve_audit import run_audit
    from sieve_audit.synth import SCENARIOS

    bundle = SCENARIOS["not_causally_sufficient"]()
    bundle.ablation = _abl(30, base=0.9, probe=0.2, rand=0.88)  # necessary
    result = run_audit(bundle)
    card = result.card
    assert card.verdict.value == "not_causally_sufficient"
    assert "ablation" in card.tested_interventions
    assert result.necessity is not None and result.necessity.necessary
    assert card.diagnostics["necessity"]["necessary"] is True
    assert any("IS necessary" in c for c in card.allowed_claims)
