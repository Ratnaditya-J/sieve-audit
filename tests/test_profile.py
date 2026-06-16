"""The frozen strict profile: loosening is a hard gate, not a soft banner.

`SIEVE-v0.1-strict` (the defaults) is the bar. You may tighten it and keep a
positive verdict; loosening any threshold voids `causally_sufficient` and
positive decodability claims, but — by the same asymmetry as everywhere in
SIEVE — never voids a verdict the probe failed to earn.
"""
import pytest

from sieve_audit import AuditConfig, run_audit
from sieve_audit.config import STRICT_PROFILE_NAME
from sieve_audit.synth import (
    scenario_causally_sufficient,
    scenario_not_causally_sufficient,
    scenario_surface_confounded,
)
from sieve_audit.verdict import INSUFFICIENT_PROTOCOL, Verdict

# (field, loosened value) pairs — each makes the bar easier to clear
LOOSENINGS = [
    ("auroc_baseline_margin", 0.0),
    ("auroc_chance_margin", 0.0),
    ("min_eval_n", 1),
    ("min_family_class_n", 1),
    ("min_resid_rel_delta", 0.0),
    ("require_output_change", False),
    ("min_steered_prompts", 1),
    ("require_symmetric_grid", False),
    ("dose_response_min_rho", 0.0),
    ("dose_response_max_p", 1.0),
    ("min_judges", 1),
    ("min_judge_kappa", 0.0),
    ("min_judge_spearman", 0.0),
    ("max_judge_spearman", 1.0),
    ("min_informative_judged", 0),
    ("noop_tolerance", 1.0),
    ("ci_level", 0.5),
]

TIGHTENINGS = [
    ("auroc_baseline_margin", 0.05),
    ("dose_response_min_rho", 0.6),
    ("min_judge_kappa", 0.5),
    ("min_steered_prompts", 25),
]


def test_default_config_is_the_strict_profile():
    status = AuditConfig().profile_status()
    assert status["status"] == "strict"
    assert status["profile"] == STRICT_PROFILE_NAME
    assert not status["loosened"] and not status["tightened"]


@pytest.mark.parametrize("field,value", LOOSENINGS)
def test_each_loosening_voids_causally_sufficient(field, value):
    bundle = scenario_causally_sufficient()
    card = run_audit(bundle, AuditConfig(**{field: value})).card
    assert card.verdict != Verdict.CAUSALLY_SUFFICIENT, (
        f"loosening {field}={value} bought a causal verdict"
    )
    assert card.status == INSUFFICIENT_PROTOCOL
    assert field in card.diagnostics["profile"]["loosened"]


@pytest.mark.parametrize("field,value", TIGHTENINGS)
def test_tightening_keeps_the_verdict(field, value):
    bundle = scenario_causally_sufficient()
    card = run_audit(bundle, AuditConfig(**{field: value})).card
    assert card.verdict == Verdict.CAUSALLY_SUFFICIENT
    assert card.diagnostics["profile"]["status"] == "stricter"
    assert field in card.diagnostics["profile"]["tightened"]


def test_loosening_never_hides_a_negative_verdict():
    """The asymmetry: a loosened bar cannot turn a real failure into a refusal."""
    card = run_audit(
        scenario_not_causally_sufficient(),
        AuditConfig(dose_response_min_rho=0.0, min_judge_kappa=0.0),
    ).card
    assert card.verdict == Verdict.NOT_CAUSALLY_SUFFICIENT

    card = run_audit(
        scenario_surface_confounded(), AuditConfig(auroc_baseline_margin=0.0)
    ).card
    assert card.verdict == Verdict.SURFACE_CONFOUNDED


def test_ambiguous_knob_change_is_treated_as_loosening():
    """A knob with no clear 'stricter' direction is conservatively a loosening."""
    status = AuditConfig(judge_binarize_threshold=0.4).profile_status()
    assert status["status"] == "loosened"
    assert "judge_binarize_threshold" in status["loosened"]


def test_loosened_profile_renders_on_card():
    from sieve_audit.card import card_to_markdown

    card = run_audit(
        scenario_causally_sufficient(), AuditConfig(min_judge_kappa=0.0)
    ).card
    md = card_to_markdown(card)
    assert "LOOSENED" in md
    assert "min_judge_kappa" in md
