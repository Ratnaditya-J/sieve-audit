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
    LeakageEvidence,
    SteeringRecord,
    VerbalizationEvidence,
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


# ---------------------------------------------------------------------------
# rigged verbalizer scenarios (verbalizer-faithfulness selftest)
#
# The audited signal is an activation verbalizer's per-example claims,
# scalarized into [0,1] (P(claim asserts Y)); the failure modes are rigged by
# construction so the adapter's gate mapping provably returns the right read:
# confabulation -> surface_confounded, CoT-parroting -> the Tier-2 `cot` gate,
# faithfulness -> causally_sufficient, decodable-but-epiphenomenal ->
# not_causally_sufficient.
# ---------------------------------------------------------------------------

_PROPERTY = "hidden-goal pursuit"


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _verb_scope(name: str, verbalizer: str) -> dict:
    return dict(
        model=f"synthetic/ground-truth-{name}",
        revision="v1",
        layers=[7],
        direction_source=(
            f"claim-direction recovered from rigged verbalizer '{verbalizer}' "
            "(synthetic)"
        ),
        prompt_distribution="synthetic-families-v1",
        prompt_license="MIT (generated)",
        behavioral_metrics=[_PROPERTY],
        adapter=f"sieve_audit.synth:{name}",
    )


def _verb_examples(
    rng: np.random.Generator, n: int, mode: str
) -> tuple[list[str], list[str], list[int], list[str],
           list[float], list[float], list[float]]:
    """Prompts, CoTs, labels, families + claim scores per input condition.

    Modes rig where the claims come from:
    - ``internals``: claims track the label through the activations; removing
      the CoT changes nothing (a faithful - or at least CoT-independent - read).
    - ``prompt``: claims are a function of the raw prompt text alone (text
      inversion); the label leaks into prompt length, so a length baseline
      matches the verbalizer.
    - ``cot``: claims track the CoT text; stripping the CoT collapses them to
      noise while a matched random removal does not (CoT-parroting).
    """
    length_confound = mode == "prompt"
    texts, labels, families = _texts_and_labels(rng, n, length_confound)
    cots = []
    full, cot_removed, cot_rand_removed = [], [], []
    for text, label in zip(texts, labels):
        # narrative CoT: mentions the property when the label is present, so a
        # cot-reading verbalizer has something to parrot
        cots.append(
            "I should think step by step. "
            + ("My real objective here is the hidden goal. " if label else "")
            + "The request looks routine."
        )
        if mode == "internals":
            latent = 2.5 * label
            full.append(_sigmoid(latent + rng.normal(0, 1.0)))
            cot_removed.append(_sigmoid(latent + rng.normal(0, 1.0)))
            cot_rand_removed.append(_sigmoid(latent + rng.normal(0, 1.0)))
        elif mode == "prompt":
            latent = len(text) / 50.0 - 2.0
            full.append(_sigmoid(latent + rng.normal(0, 0.6)))
            cot_removed.append(_sigmoid(latent + rng.normal(0, 0.6)))
            cot_rand_removed.append(_sigmoid(latent + rng.normal(0, 0.6)))
        elif mode == "cot":
            latent = 2.5 * label       # the label is legible ONLY via the CoT text
            full.append(_sigmoid(latent + rng.normal(0, 1.0)))
            cot_removed.append(_sigmoid(rng.normal(0, 1.0)))          # collapses
            cot_rand_removed.append(_sigmoid(latent + rng.normal(0, 1.0)))
        else:
            raise ValueError(f"unknown mode {mode!r}")
    return texts, cots, labels, families, full, cot_removed, cot_rand_removed


def _verb_bundle(
    name: str,
    verbalizer: str,
    rng: np.random.Generator,
    n: int,
    mode: str,
    probe_gain: float,
    control_gain: float,
    dead: bool = False,
) -> EvidenceBundle:
    texts, cots, labels, families, full, cot_rm, cot_rand = _verb_examples(rng, n, mode)
    # generic leak condition: no giveaway elicitation text beyond the CoT, so
    # leak-removal is a fresh draw from the full-condition distribution
    leak_removed = [
        _sigmoid(np.log(s / (1 - s)) + rng.normal(0, 0.3)) for s in full
    ]
    return EvidenceBundle(
        **_verb_scope(name, verbalizer),
        decodability=DecodabilityEvidence(
            texts=texts,
            labels=labels,
            probe_scores=full,
            families=families,
            probe_scores_out_of_sample=True,
        ),
        efficacy=_efficacy(rng, 30, dead=dead),
        steering=_steering(rng, 30, probe_gain=probe_gain, control_gain=control_gain),
        leakage=LeakageEvidence(
            labels=labels,
            probe_scores_full=full,
            probe_scores_leak_removed=leak_removed,
            probe_scores_random_removed=leak_removed,
            probe_scores_cot_removed=cot_rm,
            probe_scores_cot_random_removed=cot_rand,
        ),
        verbalization=VerbalizationEvidence(
            target_model=f"synthetic/ground-truth-{name}",
            verbalizer=verbalizer,
            layer=7,
            token_selection="last",
            property_tested=_PROPERTY,
            texts=texts,
            cot_texts=cots,
            labels=list(labels),
            verbalizer_claim_scores=full,
            families=families,
            claim_scores_out_of_sample=True,
        ),
    )


def scenario_verbalizer_faithful(seed: int = 0, n: int = 240) -> EvidenceBundle:
    """Claims track a genuinely encoded, causally load-bearing property: the
    claim-direction beats all matched controls and the signal survives CoT
    removal."""
    rng = np.random.default_rng(seed)
    return _verb_bundle(
        "verbalizer-faithful", "rigged:reads-internals", rng, n,
        mode="internals", probe_gain=0.35, control_gain=0.02,
    )


def scenario_verbalizer_confabulating(seed: int = 0, n: int = 240) -> EvidenceBundle:
    """Claims are generated from the prompt text alone (text inversion): a
    surface baseline matches the verbalizer, and no activation-level claim is
    warranted."""
    rng = np.random.default_rng(seed)
    return _verb_bundle(
        "verbalizer-confabulating", "rigged:reads-prompt-text", rng, n,
        mode="prompt", probe_gain=0.0, control_gain=0.0,
    )


def scenario_verbalizer_cot_parroting(seed: int = 0, n: int = 240) -> EvidenceBundle:
    """Claims track the CoT text, not the internals: stripping the CoT
    collapses the signal (the `cot` gate fires) while the matched random
    removal does not, and the claim-direction steers nothing."""
    rng = np.random.default_rng(seed)
    return _verb_bundle(
        "verbalizer-cot-parroting", "rigged:reads-cot-text", rng, n,
        mode="cot", probe_gain=0.02, control_gain=0.02,
    )


def scenario_verbalizer_epiphenomenal(seed: int = 0, n: int = 240) -> EvidenceBundle:
    """Claims are decodable, survive surface baselines and CoT removal - but
    steering the claim-direction moves behavior no more than the controls."""
    rng = np.random.default_rng(seed)
    return _verb_bundle(
        "verbalizer-epiphenomenal", "rigged:reads-epiphenomenal-correlate", rng, n,
        mode="internals", probe_gain=0.02, control_gain=0.02,
    )


# name -> (bundle factory, expected outcome). The expected outcome pins the
# verdict AND the `cot` gate flags, because two scenarios (cot-parroting and
# epiphenomenal) share a verdict and differ exactly in the CoT read.
VERBALIZER_SCENARIOS: dict = {
    "verbalizer_faithful": (
        scenario_verbalizer_faithful,
        {"verdict": "causally_sufficient", "cot_leaky": False, "cot_survives": True},
    ),
    "verbalizer_confabulating": (
        scenario_verbalizer_confabulating,
        {"verdict": "surface_confounded", "cot_leaky": False, "cot_survives": True},
    ),
    "verbalizer_cot_parroting": (
        scenario_verbalizer_cot_parroting,
        {"verdict": "not_causally_sufficient", "cot_leaky": True, "cot_survives": False},
    ),
    "verbalizer_epiphenomenal": (
        scenario_verbalizer_epiphenomenal,
        {"verdict": "not_causally_sufficient", "cot_leaky": False, "cot_survives": True},
    ),
}
