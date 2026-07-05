"""Verbalizer-faithfulness tests: the rigged scenarios must yield the rigged
verdicts AND the rigged `cot` gate flags, and the adapter must map verbalizer
records onto the existing gates without inventing new claim surface.

Ground truth by construction (synth.VERBALIZER_SCENARIOS):
- faithful verbalizer            -> causally_sufficient, survives CoT removal
- confabulating (text inversion) -> surface_confounded
- CoT-parroting                  -> `cot` leakage fires (random control doesn't)
- decodable-but-epiphenomenal    -> not_causally_sufficient, CoT gate clean
"""
import numpy as np
import pytest

from sieve_audit import AuditConfig, run_audit
from sieve_audit.bundle import EvidenceBundle, LeakageEvidence, VerbalizationEvidence
from sieve_audit.adapters.verbalizer import (
    build_bundle_from_records,
    make_claim_scorer,
    recover_claim_direction,
    scalarize_claims,
)
from sieve_audit.leakage import run_leakage
from sieve_audit.synth import VERBALIZER_SCENARIOS, _verb_examples
from sieve_audit.verdict import INSUFFICIENT_PROTOCOL

SEEDS = [0, 1, 2]


# ---------------------------------------------------------------------------
# the selftest: rigged verdict + rigged cot flags, across seeds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name", list(VERBALIZER_SCENARIOS))
def test_rigged_verbalizer_scenario_returns_rigged_outcome(name: str, seed: int):
    make, expected = VERBALIZER_SCENARIOS[name]
    result = run_audit(make(seed=seed), AuditConfig(seed=seed))
    verdict = result.card.verdict.value if result.card.verdict else result.card.status
    lk = result.leakage
    got = {
        "verdict": verdict,
        "cot_leaky": lk.cot_leaky if lk else None,
        "cot_survives": lk.cot_survives if lk else None,
    }
    assert got == expected, (
        f"scenario rigged as {expected!r} audited as {got!r} (seed={seed}); "
        f"reasons: {result.card.diagnostics['decision_reasons']}"
    )


def test_cot_flags_surface_on_the_card():
    make, _ = VERBALIZER_SCENARIOS["verbalizer_cot_parroting"]
    card = run_audit(make()).card
    assert "cot-leaky" in card.label
    assert any("COT-PARROTING" in r for r in card.residual_risks)
    assert card.diagnostics["leakage"]["cot_leaky"] is True
    assert card.diagnostics["verbalization"]["verbalizer"] == "rigged:reads-cot-text"

    make, _ = VERBALIZER_SCENARIOS["verbalizer_faithful"]
    card = run_audit(make()).card
    assert "survives-cot-removal" in card.label
    assert any("not merely reading the CoT" in c for c in card.allowed_claims)


def test_direction_recovery_caveat_is_printed_when_steering_present():
    make, _ = VERBALIZER_SCENARIOS["verbalizer_epiphenomenal"]
    card = run_audit(make()).card
    assert any(
        "recovered from the verbalizer's CLAIMS" in r for r in card.residual_risks
    )
    assert any(
        "does NOT prove the verbalizer confabulated" in r
        for r in card.residual_risks
    )


def test_claim_score_bait_and_switch_is_refused():
    """If the decodability scores are not the verbalizer's claim scores, the
    verdict would be about a different signal: the engine must refuse."""
    make, _ = VERBALIZER_SCENARIOS["verbalizer_faithful"]
    bundle = make()
    bundle.decodability.probe_scores = [
        min(1.0, s + 0.001) for s in bundle.decodability.probe_scores
    ]
    card = run_audit(bundle).card
    assert card.status == INSUFFICIENT_PROTOCOL
    assert any(
        "not be about the verbalizer's claims" in r
        for r in card.diagnostics["decision_reasons"]
    )


# ---------------------------------------------------------------------------
# the cot span category of the leakage gate
# ---------------------------------------------------------------------------


def _cot_leak_ev(n: int, cot_collapses: bool, with_control: bool = True) -> LeakageEvidence:
    labels = [i % 2 for i in range(n)]
    strong = [0.9 if l else 0.1 for l in labels]
    noise = [0.5] * n
    return LeakageEvidence(
        labels=labels,
        probe_scores_full=strong,
        probe_scores_leak_removed=strong,
        probe_scores_random_removed=strong,
        probe_scores_cot_removed=(noise if cot_collapses else strong),
        probe_scores_cot_random_removed=(strong if with_control else None),
    )


def test_cot_gate_fires_only_when_cot_removal_collapses_the_signal():
    cfg = AuditConfig(seed=0)
    res = run_leakage(_cot_leak_ev(60, cot_collapses=True), cfg)
    assert res.cot_leaky is True and res.cot_survives is False
    assert res.drop_cot.lo >= cfg.leakage_min_drop
    assert res.drop_cot.lo > res.drop_cot_random.hi

    res = run_leakage(_cot_leak_ev(60, cot_collapses=False), cfg)
    assert res.cot_leaky is False and res.cot_survives is True


def test_cot_gate_untested_without_cot_scores():
    labels = [i % 2 for i in range(40)]
    strong = [float(l) for l in labels]
    ev = LeakageEvidence(
        labels=labels,
        probe_scores_full=strong,
        probe_scores_leak_removed=strong,
        probe_scores_random_removed=strong,
    )
    res = run_leakage(ev, AuditConfig(seed=0))
    assert res.cot_leaky is None and res.cot_survives is None


def test_cot_survival_is_earned_not_defaulted():
    """A signal that was never above chance cannot claim 'survives CoT removal'
    just because removing the CoT changed nothing (anti-gaming asymmetry)."""
    n = 60
    labels = [i % 2 for i in range(n)]
    noise = [0.5 + 0.001 * (i % 3) for i in range(n)]
    ev = LeakageEvidence(
        labels=labels,
        probe_scores_full=noise,
        probe_scores_leak_removed=noise,
        probe_scores_random_removed=noise,
        probe_scores_cot_removed=noise,
        probe_scores_cot_random_removed=noise,
    )
    res = run_leakage(ev, AuditConfig(seed=0))
    assert res.cot_leaky is False
    assert res.cot_survives is False


def test_shared_random_control_is_noted():
    res = run_leakage(_cot_leak_ev(60, cot_collapses=True, with_control=False),
                      AuditConfig(seed=0))
    assert res.cot_leaky is True
    assert any("shared with the generic random-span control" in n for n in res.notes)


# ---------------------------------------------------------------------------
# bundle validation + serialization
# ---------------------------------------------------------------------------


def test_verbalization_evidence_validates():
    good = dict(
        target_model="m", verbalizer="v", layer=3, token_selection="last",
        property_tested="y", texts=["a", "b"], cot_texts=["c", "d"],
        labels=[0, 1], verbalizer_claim_scores=[0.1, 0.9], families=["f", "f"],
    )
    VerbalizationEvidence(**good)
    with pytest.raises(ValueError):
        VerbalizationEvidence(**{**good, "cot_texts": ["c"]})
    with pytest.raises(ValueError):
        VerbalizationEvidence(**{**good, "verbalizer_claim_scores": [0.1, 1.9]})
    with pytest.raises(ValueError):
        VerbalizationEvidence(**{**good, "labels": [0, 2]})
    # label-free (None-filled) is legitimate: the GLM path has no ground truth
    VerbalizationEvidence(**{**good, "labels": [None, None]})
    with pytest.raises(ValueError):
        LeakageEvidence(
            labels=[0, 1], probe_scores_full=[0.1, 0.9],
            probe_scores_leak_removed=[0.1, 0.9],
            probe_scores_random_removed=[0.1, 0.9],
            probe_scores_cot_random_removed=[0.1, 0.9],  # control without cot scores
        )


def test_verbalization_round_trips():
    make, _ = VERBALIZER_SCENARIOS["verbalizer_faithful"]
    b = make(n=24)
    again = EvidenceBundle.from_dict(b.to_dict())
    assert again.verbalization is not None
    assert again.verbalization.verbalizer_claim_scores == \
        b.verbalization.verbalizer_claim_scores
    assert again.leakage.probe_scores_cot_removed == b.leakage.probe_scores_cot_removed


# ---------------------------------------------------------------------------
# the adapter: scalarization, recorded path, direction recovery
# ---------------------------------------------------------------------------


def test_claim_scorers_agree_on_the_assert_deny_axis():
    asserting = "The model is clearly engaged in hidden-goal pursuit right now."
    denying = "There is no hidden-goal pursuit; the model is not pursuing anything."
    unrelated = "The capital of France is Paris."
    for spec in ("assert:lexical", "assert:graded"):
        scorer = make_claim_scorer(spec, "hidden-goal pursuit")
        assert scorer(asserting) > 0.5 > scorer(denying)
        assert scorer(asserting) > scorer(unrelated)


def test_yesno_scorers_read_interrogative_claims():
    for spec in ("yesno:lexical", "yesno:graded"):
        scorer = make_claim_scorer(spec, "hidden-goal pursuit")
        assert scorer("Yes, it is relying on the suggestion.") > 0.5
        assert scorer("No. The model calculated independently.") < 0.5
        assert scorer("Yes") > 0.5 > scorer("No")


def test_scalarize_requires_two_scorers():
    with pytest.raises(ValueError):
        scalarize_claims(["x"], "y", scorer_specs=("assert:lexical",))


def _records(n: int = 240, seed: int = 0, mode: str = "internals") -> list[dict]:
    rng = np.random.default_rng(seed)
    texts, cots, labels, families, full, cot_rm, cot_rand = _verb_examples(rng, n, mode)
    return [
        {
            "prompt": t, "cot": c, "family": f, "label": l,
            "claim_score": s, "claim_cot_removed_score": cr,
            "claim_cot_random_removed_score": crr,
        }
        for t, c, f, l, s, cr, crr
        in zip(texts, cots, families, labels, full, cot_rm, cot_rand)
    ]


def _build(records, **overrides):
    kwargs = dict(
        target_model="test/target", verbalizer="rigged:test", layer=7,
        property_tested="hidden-goal pursuit",
        prompt_distribution="synthetic-families-v1",
        prompt_license="MIT (generated)", claim_scores_out_of_sample=True,
    )
    kwargs.update(overrides)
    return build_bundle_from_records(records, **kwargs)


def test_recorded_path_builds_an_auditable_bundle():
    bundle = _build(_records())
    assert bundle.decodability.probe_scores == \
        bundle.verbalization.verbalizer_claim_scores
    assert bundle.leakage.probe_scores_cot_removed is not None
    result = run_audit(bundle, AuditConfig(seed=0))
    # decodability-only bundle: correlational read, causal stage refused
    assert result.card.status == INSUFFICIENT_PROTOCOL
    assert result.leakage.cot_survives is True


def test_recorded_path_catches_cot_parroting():
    result = run_audit(_build(_records(mode="cot")), AuditConfig(seed=0))
    assert result.leakage.cot_leaky is True
    assert "cot-leaky" in result.card.label


def test_recorded_path_rejects_partial_conditions():
    records = _records(n=24)
    del records[0]["claim_cot_removed_score"]
    with pytest.raises(ValueError, match="not all"):
        _build(records)


def test_recorded_path_label_free_skips_decodability():
    records = [{**r, "label": None} for r in _records(n=24)]
    bundle = _build(records)
    assert bundle.decodability is None and bundle.leakage is None
    assert bundle.verbalization.labels == [None] * 24


def test_direction_recovery_finds_the_claim_direction():
    rng = np.random.default_rng(0)
    d, n = 64, 200
    w_true = rng.normal(size=d)
    w_true /= np.linalg.norm(w_true)
    claims = rng.integers(0, 2, size=n)
    X = rng.normal(size=(n, d)) + 3.0 * claims[:, None] * w_true
    w = recover_claim_direction(X, claims.astype(float).tolist())
    assert abs(float(w @ w_true)) > 0.9

    with pytest.raises(ValueError, match="one-sided"):
        recover_claim_direction(X, [1.0] * n)
