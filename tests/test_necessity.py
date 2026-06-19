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
