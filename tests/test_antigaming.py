"""Regression tests for the adversarial-review exploits.

Each test reconstructs a gaming strategy a probe vendor could use and asserts
the engine now refuses or downgrades it. These exploits previously yielded
``causally_sufficient`` (or a face-saving ``intervention_ineffective``) and
must never do so again.
"""
import numpy as np

from sieve_audit import AuditConfig, run_audit
from sieve_audit.bundle import SteeringRecord
from sieve_audit.synth import (
    _texts_and_labels,
    scenario_causally_sufficient,
    scenario_not_causally_sufficient,
)
from sieve_audit.bundle import DecodabilityEvidence
from sieve_audit.verdict import INSUFFICIENT_PROTOCOL, Verdict


def _verdict_or_status(card):
    return card.verdict.value if card.verdict else card.status


def test_family_gerrymandering_is_refused():
    """One giant family + one tiny single-class family silenced the baselines;
    a pure length-confound probe then audited as causally_sufficient."""
    rng = np.random.default_rng(0)
    bundle = scenario_causally_sufficient()
    texts, labels, _ = _texts_and_labels(rng, 240, length_confound=True)
    bundle.decodability = DecodabilityEvidence(
        texts=texts,
        labels=labels,
        probe_scores=[float(len(t) / 50.0 + rng.normal(0, 0.6)) for t in texts],
        families=["giant"] * 236 + ["tiny"] * 4,  # tiny is nearly single-class
        probe_scores_out_of_sample=True,
    )
    card = run_audit(bundle).card
    assert card.verdict != Verdict.CAUSALLY_SUFFICIENT
    # either the confound is caught (surface_confounded) or the bundle refused
    assert _verdict_or_status(card) in ("surface_confounded", INSUFFICIENT_PROTOCOL)


def test_bimodal_control_effects_cannot_cancel():
    """A control moving half the prompts +0.4 and half -0.4 has |mean| ~ 0;
    a weak probe (+0.15) must not 'exceed' it."""
    bundle = scenario_causally_sufficient()
    rng = np.random.default_rng(0)
    new_steering = []
    for r in bundle.steering:
        if r.arm == "probe":
            # weak but real probe effect
            latent = 0.5 + 0.15 * r.alpha / 20.0
        elif r.alpha == 0.0:
            latent = 0.5
        else:
            # large mixed-sign control effect, canceling in the mean
            sign = 1.0 if int(r.prompt_id[1:]) % 2 == 0 else -1.0
            latent = 0.5 + sign * 0.4 * abs(r.alpha) / 20.0
        latent = float(np.clip(latent, 0.0, 1.0))
        new_steering.append(
            SteeringRecord(
                arm=r.arm,
                alpha=r.alpha,
                prompt_id=r.prompt_id,
                judge_scores={
                    j: float(np.clip(latent + rng.normal(0, 0.03), 0, 1))
                    for j in r.judge_scores
                },
            )
        )
    bundle.steering = new_steering
    card = run_audit(bundle).card
    assert card.verdict != Verdict.CAUSALLY_SUFFICIENT


def test_degenerate_control_arm_is_refused():
    """A 'control' that never moved the residual stream flatters any probe."""
    bundle = scenario_causally_sufficient()
    for r in bundle.efficacy:
        if r.arm == "random" and r.alpha != 0.0:
            r.resid_delta_norm = 1e-6  # the random arm never actually steered
            r.output_changed = False
    card = run_audit(bundle).card
    assert card.status == INSUFFICIENT_PROTOCOL


def test_control_arm_without_efficacy_records_is_refused():
    bundle = scenario_causally_sufficient()
    bundle.efficacy = [r for r in bundle.efficacy if r.arm != "orthogonal"]
    card = run_audit(bundle).card
    assert card.status == INSUFFICIENT_PROTOCOL


def test_duplicate_judges_are_refused():
    """judge_b = judge_a defeats the two-judge requirement with 'perfect' kappa."""
    bundle = scenario_causally_sufficient()
    for r in bundle.steering:
        r.judge_scores["judge_b"] = r.judge_scores["judge_a"]
    card = run_audit(bundle).card
    assert card.status == INSUFFICIENT_PROTOCOL
    assert card.verdict is None


def test_sandbagged_efficacy_is_inconsistent_not_inconclusive():
    """Dead-hook efficacy records + significant behavioral deltas cannot both
    describe the same intervention. Previously this downgraded a damning
    not_causally_sufficient to a face-saving intervention_ineffective."""
    bundle = scenario_not_causally_sufficient()
    for r in bundle.efficacy:
        if r.alpha != 0.0:
            r.resid_delta_norm = 1e-6
            r.output_changed = False
    card = run_audit(bundle).card
    assert card.verdict != Verdict.INTERVENTION_INEFFECTIVE
    assert card.status == INSUFFICIENT_PROTOCOL


def test_weakened_control_suite_cannot_certify():
    """required_controls=() made exceeds_all_controls vacuously true."""
    bundle = scenario_causally_sufficient()
    card = run_audit(bundle, AuditConfig(required_controls=())).card
    assert card.verdict != Verdict.CAUSALLY_SUFFICIENT
    assert card.status == INSUFFICIENT_PROTOCOL


def test_weakened_config_is_visible_on_card():
    bundle = scenario_causally_sufficient()
    card = run_audit(bundle, AuditConfig(auroc_baseline_margin=0.0)).card
    assert card.diagnostics["config_nondefault"] == {"auroc_baseline_margin": 0.0}


def test_in_sample_probe_scores_are_refused():
    """In-sample probe scores vs cross-validated baselines is an unfair fight."""
    bundle = scenario_causally_sufficient()
    bundle.decodability.probe_scores_out_of_sample = False
    card = run_audit(bundle).card
    assert card.status == INSUFFICIENT_PROTOCOL


def test_efficacy_for_a_different_intervention_is_refused():
    """Efficacy at alpha=+/-100 does not certify steering judged at +/-20."""
    bundle = scenario_causally_sufficient()
    for r in bundle.efficacy:
        r.alpha *= 5.0
        r.expected_delta_norm *= 5.0
    card = run_audit(bundle).card
    assert card.status == INSUFFICIENT_PROTOCOL


def test_efficacy_on_disjoint_prompts_is_refused():
    bundle = scenario_causally_sufficient()
    for r in bundle.efficacy:
        r.prompt_id = "q" + r.prompt_id[1:]
    card = run_audit(bundle).card
    assert card.status == INSUFFICIENT_PROTOCOL


def test_duplicate_steering_records_are_refused():
    """Retakes with last-wins overwrites let a vendor keep the lucky roll."""
    bundle = scenario_causally_sufficient()
    bundle.steering.append(bundle.steering[0])
    card = run_audit(bundle).card
    assert card.status == INSUFFICIENT_PROTOCOL


def test_one_sided_alpha_grid_is_refused():
    bundle = scenario_causally_sufficient()
    bundle.steering = [r for r in bundle.steering if r.alpha >= 0.0]
    bundle.efficacy = [r for r in bundle.efficacy if r.alpha >= 0.0]
    card = run_audit(bundle).card
    assert card.verdict != Verdict.CAUSALLY_SUFFICIENT
    assert card.status == INSUFFICIENT_PROTOCOL
