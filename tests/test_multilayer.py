"""Tests for the joint multi-layer ablation gate (committee / distributed mechanism).

The headline value: a single-layer ablation null is NOT a true "no causal role"
when a joint multi-layer intervention isn't tested. The gate reuses the necessity
adjudication over a joint layer set and the card compares the two.
"""
import pytest

from sieve_audit.bundle import AblationRecord, EvidenceBundle, MultiLayerRecord
from sieve_audit.config import AuditConfig
from sieve_audit.multilayer import run_multilayer


def _ml(n, base, probe, rand, layers=(4, 5, 6), judges=("j1", "j2")):
    recs = []
    for i in range(n):
        pid = f"p{i}"
        for arm, v in (("baseline", base), ("probe", probe), ("ablate_random", rand)):
            recs.append(
                MultiLayerRecord(
                    arm=arm, prompt_id=pid, layers=list(layers),
                    judge_scores={j: v for j in judges},
                )
            )
    return recs


def _abl(n, base, probe, rand, judges=("j1", "j2")):
    recs = []
    for i in range(n):
        pid = f"p{i}"
        for arm, v in (("baseline", base), ("probe", probe), ("ablate_random", rand)):
            recs.append(
                AblationRecord(arm=arm, prompt_id=pid, judge_scores={j: v for j in judges})
            )
    return recs


def _bundle(**kw):
    return EvidenceBundle(
        model="m", revision=None, layers=[5], direction_source="contrastive",
        prompt_distribution="d", prompt_license="x", behavioral_metrics=["b"],
        adapter="test", **kw,
    )


def test_multilayer_round_trips_and_rejects_dupes():
    recs = _ml(2, 0.9, 0.2, 0.88)
    b = _bundle(multilayer=recs)
    b.validate()
    assert b.multilayer_arms == ["ablate_random", "baseline", "probe"]
    again = EvidenceBundle.from_dict(b.to_dict())
    assert again.multilayer == recs
    dup = recs + [MultiLayerRecord(arm="probe", prompt_id="p0", layers=[4, 5, 6],
                                   judge_scores={"j": 0.1})]
    with pytest.raises(ValueError, match="duplicate multilayer"):
        _bundle(multilayer=dup).validate()


def test_joint_necessity_detected():
    res = run_multilayer(_ml(30, base=0.9, probe=0.2, rand=0.88), AuditConfig(seed=0))
    assert not res.inconclusive and res.necessary
    assert res.layers == [4, 5, 6]


def test_joint_not_necessary_when_matches_control():
    res = run_multilayer(_ml(30, base=0.9, probe=0.85, rand=0.85), AuditConfig(seed=0))
    assert not res.inconclusive and not res.necessary


def test_mismatched_layer_sets_rejected():
    recs = _ml(2, 0.9, 0.2, 0.88, layers=(4, 5))
    recs[0].layers = [7, 8]
    with pytest.raises(ValueError, match="share one joint layer set"):
        run_multilayer(recs, AuditConfig(seed=0))


def test_distributed_mechanism_signature_end_to_end():
    """Single-layer ablation NOT necessary, but joint multi-layer IS: the card
    must headline 'necessary (multi-layer)', surface the distributed claim, and
    flag the distributed-mechanism summary."""
    from sieve_audit import run_audit
    from sieve_audit.synth import SCENARIOS

    bundle = SCENARIOS["not_causally_sufficient"]()
    bundle.ablation = _abl(30, base=0.9, probe=0.85, rand=0.85)      # NOT necessary
    bundle.multilayer = _ml(30, base=0.9, probe=0.2, rand=0.88)      # joint necessary
    card = run_audit(bundle).card

    assert card.verdict.value == "not_causally_sufficient"
    assert card.label == "not_causally_sufficient · necessary (multi-layer)"
    cs = card.diagnostics["causal_summary"]
    assert cs["multilayer"] == "necessary_joint"
    assert "DISTRIBUTED MECHANISM" in cs["combined"]
    assert any("DISTRIBUTED MECHANISM" in c for c in card.allowed_claims)
    assert any(i.startswith("joint multi-layer ablation") for i in card.tested_interventions)


def test_single_layer_null_without_multilayer_is_qualified():
    """A single-layer not-necessary null with no joint evidence must warn that a
    distributed mechanism was not ruled out (the over-read guard)."""
    from sieve_audit import run_audit

    card = run_audit(_bundle(ablation=_abl(30, base=0.9, probe=0.85, rand=0.85))).card
    assert any("cannot be ruled out" in r and "multi-layer" in r
               for r in card.residual_risks)
