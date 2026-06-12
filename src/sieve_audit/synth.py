"""Synthetic ground-truth scenarios: SIEVE auditing itself.

Each generator rigs an evidence bundle whose correct verdict is known *by
construction* — a noise probe, a length-confounded probe, a dead layer, a
decodable-but-epiphenomenal direction, and a truly causal direction. The
self-test (tests/test_selftest.py, `sieve selftest`) asserts that the engine
returns exactly the rigged verdict for each. A validity auditor whose own
verdicts haven't been validated against known ground truth would be an
embarrassment; this module is the antidote.
"""
from __future__ import annotations

import numpy as np

from .bundle import (
    DecodabilityEvidence,
    EfficacyRecord,
    EvidenceBundle,
    SteeringRecord,
)

_FAMILIES = ("math_eval", "code_eval", "chat_deploy", "agentic_deploy")
_ALPHAS = (-20.0, -10.0, 0.0, 10.0, 20.0)
_CONTROLS = ("random", "orthogonal", "wrong_layer")


def _scope(name: str) -> dict:
    return dict(
        model=f"synthetic/ground-truth-{name}",
        revision="v1",
        layers=[7],
        direction_source="rigged contrastive direction (synthetic)",
        prompt_distribution="synthetic-families-v1",
        prompt_license="MIT (generated)",
        behavioral_metrics=["eval_behavior_score"],
        adapter=f"sieve_audit.synth:{name}",
    )


def _texts_and_labels(
    rng: np.random.Generator, n: int, length_confound: bool
) -> tuple[list[str], list[int], list[str]]:
    """Synthetic prompts over families; optionally make label predictable from length."""
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    texts, labels, families = [], [], []
    for i in range(n):
        label = int(i % 2)
        # family must vary independently of label, or every family is
        # single-class and the audit (rightly) refuses the bundle
        family = _FAMILIES[(i // 2) % len(_FAMILIES)]
        if length_confound:
            # label leaks into raw text length: the surface confound
            n_words = rng.integers(20, 30) if label else rng.integers(5, 12)
        else:
            n_words = rng.integers(8, 25)
        texts.append(" ".join(rng.choice(words, n_words)))
        labels.append(label)
        families.append(family)
    return texts, labels, families


def _decod(
    rng: np.random.Generator,
    n: int,
    signal: float,
    length_confound: bool = False,
) -> DecodabilityEvidence:
    """Probe scores = signal * label + noise; signal=0 means an uninformative probe."""
    texts, labels, families = _texts_and_labels(rng, n, length_confound)
    scores = [
        float(signal * lab + rng.normal(0, 1.0))
        for lab in labels
    ]
    return DecodabilityEvidence(
        texts=texts,
        labels=labels,
        probe_scores=scores,
        families=families,
        probe_scores_out_of_sample=True,
    )


def _efficacy(rng: np.random.Generator, n: int, dead: bool) -> list[EfficacyRecord]:
    """Residual-movement records, one set per steering arm.

    `dead=True` rigs the gpt-oss L34 failure mode (for every arm: the layer
    itself is dead). Control arms are otherwise live interventions too — the
    engine requires that, so a degenerate control cannot flatter the probe.
    """
    records = []
    base_norm = 100.0
    w_norm = 1.0
    for arm in ("probe", *_CONTROLS):
        for alpha in _ALPHAS:
            for p in range(n):
                expected = abs(alpha) * w_norm
                if dead and alpha != 0.0:
                    # quantization swallows the injection: nothing moves
                    delta = float(abs(rng.normal(0, 1e-4)))
                    changed = False
                elif alpha == 0.0:
                    delta = 0.0
                    changed = False
                else:
                    delta = float(expected * (1 + rng.normal(0, 0.05)))
                    changed = bool(abs(alpha) >= 10.0)
                records.append(
                    EfficacyRecord(
                        alpha=alpha,
                        prompt_id=f"p{p}",
                        resid_delta_norm=delta,
                        resid_base_norm=base_norm,
                        expected_delta_norm=expected,
                        output_changed=changed,
                        arm=arm,
                    )
                )
    return records


def _steering(
    rng: np.random.Generator,
    n: int,
    probe_gain: float,
    control_gain: float,
    judge_noise: float = 0.03,
) -> list[SteeringRecord]:
    """Judged behavioral scores per arm.

    The latent behavioral score moves linearly with alpha at `probe_gain` for
    the probe arm and `control_gain` for every control arm; two judges read it
    with independent noise.
    """
    records = []
    max_alpha = max(abs(a) for a in _ALPHAS)
    for arm in ("probe", *_CONTROLS):
        gain = probe_gain if arm == "probe" else control_gain
        for alpha in _ALPHAS:
            for p in range(n):
                base = 0.5 + rng.normal(0, 0.02)
                latent = float(np.clip(base + gain * alpha / max_alpha, 0.0, 1.0))
                records.append(
                    SteeringRecord(
                        arm=arm,
                        alpha=alpha,
                        prompt_id=f"p{p}",
                        judge_scores={
                            "judge_a": float(np.clip(latent + rng.normal(0, judge_noise), 0, 1)),
                            "judge_b": float(np.clip(latent + rng.normal(0, judge_noise), 0, 1)),
                        },
                    )
                )
    return records


# ---------------------------------------------------------------------------
# the five rigged scenarios (+ one protocol-refusal case)
# ---------------------------------------------------------------------------


def scenario_not_decodable(seed: int = 0, n: int = 240) -> EvidenceBundle:
    """Probe scores are pure noise: nothing to decode."""
    rng = np.random.default_rng(seed)
    return EvidenceBundle(
        **_scope("not-decodable"),
        decodability=_decod(rng, n, signal=0.0),
        efficacy=_efficacy(rng, 30, dead=False),
        steering=_steering(rng, 30, probe_gain=0.0, control_gain=0.0),
    )


def scenario_surface_confounded(seed: int = 0, n: int = 240) -> EvidenceBundle:
    """Label leaks into prompt length; the probe simply reads the leak.

    The probe decodes well — and so does a logistic regression on length
    features, because the 'signal' is a property of the raw text.
    """
    rng = np.random.default_rng(seed)
    texts, labels, families = _texts_and_labels(rng, n, length_confound=True)
    # probe score ~ text length (the confound), not any internal state
    scores = [float(len(t) / 50.0 + rng.normal(0, 0.6)) for t in texts]
    return EvidenceBundle(
        **_scope("surface-confounded"),
        decodability=DecodabilityEvidence(
            texts=texts,
            labels=labels,
            probe_scores=scores,
            families=families,
            probe_scores_out_of_sample=True,
        ),
        efficacy=_efficacy(rng, 30, dead=False),
        steering=_steering(rng, 30, probe_gain=0.0, control_gain=0.0),
    )


def scenario_intervention_ineffective(seed: int = 0, n: int = 240) -> EvidenceBundle:
    """Decodable probe, but the steering hook is dead (the gpt-oss L34 lesson)."""
    rng = np.random.default_rng(seed)
    return EvidenceBundle(
        **_scope("intervention-ineffective"),
        decodability=_decod(rng, n, signal=2.5),
        efficacy=_efficacy(rng, 30, dead=True),
        steering=_steering(rng, 30, probe_gain=0.0, control_gain=0.0),
    )


def scenario_not_causally_sufficient(seed: int = 0, n: int = 240) -> EvidenceBundle:
    """Decodable, intervention bites, but the probe arm moves behavior no more
    than random/orthogonal/wrong-layer controls: correlate, not cause."""
    rng = np.random.default_rng(seed)
    return EvidenceBundle(
        **_scope("not-causally-sufficient"),
        decodability=_decod(rng, n, signal=2.5),
        efficacy=_efficacy(rng, 30, dead=False),
        steering=_steering(rng, 30, probe_gain=0.04, control_gain=0.04),
    )


def scenario_causally_sufficient(seed: int = 0, n: int = 240) -> EvidenceBundle:
    """The real thing: decodable, effective, dose-responsive, beats all controls."""
    rng = np.random.default_rng(seed)
    return EvidenceBundle(
        **_scope("causally-sufficient"),
        decodability=_decod(rng, n, signal=2.5),
        efficacy=_efficacy(rng, 30, dead=False),
        steering=_steering(rng, 30, probe_gain=0.35, control_gain=0.02),
    )


def scenario_insufficient_protocol(seed: int = 0, n: int = 240) -> EvidenceBundle:
    """Looks causal, but a control arm and a judge are missing: SIEVE must refuse."""
    rng = np.random.default_rng(seed)
    steering = [
        SteeringRecord(
            arm=r.arm,
            alpha=r.alpha,
            prompt_id=r.prompt_id,
            judge_scores={"judge_a": r.judge_scores["judge_a"]},
        )
        for r in _steering(rng, 30, probe_gain=0.35, control_gain=0.02)
        if r.arm in ("probe", "random")  # orthogonal + wrong_layer never run
    ]
    return EvidenceBundle(
        **_scope("insufficient-protocol"),
        decodability=_decod(rng, n, signal=2.5),
        efficacy=_efficacy(rng, 30, dead=False),
        steering=steering,
    )


SCENARIOS = {
    "not_decodable": scenario_not_decodable,
    "surface_confounded": scenario_surface_confounded,
    "intervention_ineffective": scenario_intervention_ineffective,
    "not_causally_sufficient": scenario_not_causally_sufficient,
    "causally_sufficient": scenario_causally_sufficient,
    "insufficient_protocol": scenario_insufficient_protocol,
}
