"""Pre-registration: a committed config+scope is checked against the run.

A matching run is attested; a run that swapped the layer/direction/prompts or
tuned a threshold after the fact is flagged MISMATCH on the card — without
silently voiding the science.
"""
from sieve_audit import AuditConfig, build_prereg, run_audit
from sieve_audit.prereg import PreRegistration, verify_prereg
from sieve_audit.synth import scenario_causally_sufficient


def test_matching_run_is_attested():
    bundle = scenario_causally_sufficient()
    prereg = build_prereg(bundle, AuditConfig())
    card = run_audit(bundle, AuditConfig(), prereg=prereg).card
    assert card.preregistration["matches"] is True
    assert card.preregistration["diffs"] == []


def test_swapped_layer_after_prereg_is_flagged():
    """Pre-register layer 7, then run on a bundle that actually used layer 9."""
    planned = scenario_causally_sufficient()
    prereg = build_prereg(planned, AuditConfig())

    run = scenario_causally_sufficient()
    run.layers = [9]  # the analysis target changed after committing
    card = run_audit(run, AuditConfig(), prereg=prereg).card
    assert card.preregistration["matches"] is False
    assert any("layers" in d for d in card.preregistration["diffs"])
    # the science still produced a verdict; the prereg claim is what failed
    assert card.verdict is not None or card.status


def test_tuned_threshold_after_prereg_is_flagged():
    bundle = scenario_causally_sufficient()
    prereg = build_prereg(bundle, AuditConfig())  # committed strict thresholds
    # later: run with a different (here, stricter) threshold than committed
    card = run_audit(bundle, AuditConfig(min_judge_kappa=0.5), prereg=prereg).card
    assert card.preregistration["matches"] is False
    assert any("min_judge_kappa" in d for d in card.preregistration["diffs"])


def test_prereg_roundtrip_and_hash_stable(tmp_path):
    bundle = scenario_causally_sufficient()
    a = build_prereg(bundle, AuditConfig(), note="flagship plan")
    path = tmp_path / "prereg.json"
    a.save(path)
    b = PreRegistration.load(path)
    assert b.prereg_hash == a.prereg_hash
    assert b.note == "flagship plan"
    # hash is independent of object identity, derived from content
    assert verify_prereg(b, bundle, AuditConfig()).matches


def test_audit_without_prereg_has_none():
    card = run_audit(scenario_causally_sufficient()).card
    assert card.preregistration is None
