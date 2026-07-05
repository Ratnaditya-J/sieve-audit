"""The evidence bundle: the serialized record a SIEVE audit consumes.

SIEVE's core never touches a model. An adapter (nnsight / TransformerLens /
Inspect / vendor script) runs the probe and the steering arms, records what
happened, and serializes it here. The core then audits the *evidence*. This
keeps the verdict logic reproducible, GPU-free, and signal-agnostic — and it
means a closed-model vendor can self-run the adapter while anyone can re-run
the audit from the bundle (DESIGN.md sections 4, 7).

A bundle has three sections, one per audit stage:

- ``decodability``: per-example probe scores, labels, raw texts, and a family
  id used for held-out-family generalization splits.
- ``efficacy``: per-(alpha, prompt) residual-stream movement and
  output-changed flags, used by the efficacy gate.
- ``steering``: per-(arm, alpha, prompt) behavioral scores from every judge.

Serialization is a single JSON file (small: scores and flags, never raw
activations), so bundles can be committed, diffed, and hashed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DecodabilityEvidence:
    """Per-example evidence that the probe can (or cannot) read the signal."""

    texts: list[str]            # raw prompt texts (surface baselines train on these)
    labels: list[int]           # ground-truth condition per example (0/1)
    probe_scores: list[float]   # the audited signal's score per example
    families: list[str]         # prompt-family id per example (held-out splits)
    # Explicit attestation that no listed example was used to TRAIN the probe.
    # Without it, in-sample probe scores would face cross-validated baselines:
    # an unfair fight the probe always wins. Defaults to False (untrusted).
    probe_scores_out_of_sample: bool = False

    def __post_init__(self) -> None:
        n = len(self.texts)
        if not (len(self.labels) == len(self.probe_scores) == len(self.families) == n):
            raise ValueError("decodability fields must have equal length")
        if n == 0:
            raise ValueError("decodability evidence is empty")
        if set(self.labels) - {0, 1}:
            raise ValueError("labels must be 0/1")


@dataclass
class EfficacyRecord:
    """One steered forward pass: did the intervention move anything?

    Recorded per steering arm: the efficacy gate applies to the probe arm,
    and every control arm must also demonstrably move the stream — otherwise
    a degenerate (e.g. near-zero-norm) "control" makes any probe look
    superior.
    """

    alpha: float
    prompt_id: str
    resid_delta_norm: float     # ||h_steered - h_base|| at the intervened layer
    resid_base_norm: float      # ||h_base|| (for relative movement)
    expected_delta_norm: float  # |alpha| * ||w|| (hook-correctness reference)
    output_changed: bool        # did any generated token differ from alpha=0?
    arm: str = "probe"          # which steering arm this pass belongs to


@dataclass
class SteeringRecord:
    """One judged steered generation in one arm of the control suite."""

    arm: str                    # "probe" | "random" | "orthogonal" | "wrong_layer"
    alpha: float
    prompt_id: str
    judge_scores: dict[str, float]  # judge name -> behavioral score in [0, 1]


@dataclass
class AblationRecord:
    """One judged generation with a direction projected OUT of the residual stream.

    The *necessity* counterpart to ``SteeringRecord``: instead of adding a
    direction (sufficiency), the adapter removes it and judges the result. Arms:

    - ``baseline``: no ablation (reference behavior),
    - ``probe``: the audited direction projected out,
    - ``ablate_random``: a random direction of equal norm projected out — the
      matched control, without which "behavior changed after ablation" is
      confounded by the generic effect of perturbing the forward pass.

    Necessity = the probe-ablation behavioral DROP (baseline − probe) exceeds the
    ablate-random drop. It is the complement to steering's sufficiency: a
    direction can be necessary (ablating it removes the behavior) without being
    sufficient (adding it does not induce the behavior), which is exactly the
    distributed-mechanism case a steering-only verdict cannot see.
    """

    arm: str                        # "baseline" | "probe" | "ablate_random"
    prompt_id: str
    judge_scores: dict[str, float]  # judge name -> target-behavior score in [0, 1]


@dataclass
class PatchingRecord:
    """One judged generation in an activation-patching (oracle) run.

    The strongest localization test SIEVE supports. Steering/ablation add or
    remove a *learned* direction; patching transplants the *actual* activation
    from a clean run into a corrupted run, so the full-site patch is a
    ground-truth ("oracle") measure of how much causal content lives at that
    site. The audit then asks the calibration question: does patching ONLY the
    audited direction's component recover that effect, or is the direction a
    correlate that the site's real mechanism doesn't run through? Arms:

    - ``corrupt``: behavior on the corrupted prompt (the counterfactual floor),
    - ``patch_full``: corrupt run with the FULL residual at the site patched from
      clean — the oracle (100%) restoration,
    - ``patch_direction``: corrupt run with ONLY the audited direction's component
      patched from clean — the direction's restoration,
    - ``patch_random``: corrupt run patching a random direction's component — the
      matched control, so "the direction restored behavior" is not confounded by
      the generic effect of patching some coordinate,
    - ``clean`` (optional): clean-prompt behavior, to report how complete the
      full-site patch is relative to the full clean↔corrupt gap.

    Faithful ⟺ direction-patch recovers a large fraction of the full-site
    (oracle) effect AND more than the random-patch control.
    """

    arm: str                        # clean | corrupt | patch_full | patch_direction | patch_random
    prompt_id: str
    layers: list[int]               # the patched site(s); identical across all records
    judge_scores: dict[str, float]  # judge name -> target-behavior score in [0, 1]


@dataclass
class DeploymentEvidence:
    """Optional off-distribution probe scores for the deployment lens.

    The decodability eval characterizes the probe on its own prompt mix; a
    deployer cares about the *different* mix it will see in the wild. This holds
    the probe scored on that other distribution so the lens can report the
    in-the-wild miss rate, not just the in-distribution one. Without it, the lens
    reports off-distribution as "not assessed".
    """

    distribution: str            # name of the off-distribution eval set
    labels: list[int]
    probe_scores: list[float]

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.probe_scores):
            raise ValueError("deployment evidence fields must have equal length")
        if len(self.labels) == 0:
            raise ValueError("deployment evidence is empty")
        if set(self.labels) - {0, 1}:
            raise ValueError("labels must be 0/1")


@dataclass
class MultiLayerRecord:
    """One judged generation with a *joint multi-layer* intervention.

    The committee gap: a probe reads one layer, so a single-layer steering or
    ablation can show no effect even when the direction is causally load-bearing
    — because the mechanism is distributed and the other layers compensate. This
    record carries an intervention applied at SEVERAL layers at once, so a
    single-layer null is not mistaken for "no causal role".

    Same arms as ``AblationRecord`` (``baseline`` / ``probe`` / ``ablate_random``)
    — necessity adjudicated exactly as for single-layer ablation, but over the
    joint layer set. ``layers`` is the set of layers intervened together and must
    be identical across every record in the section (one joint intervention).
    """

    arm: str                        # "baseline" | "probe" | "ablate_random"
    prompt_id: str
    layers: list[int]               # the layers jointly intervened (same for all records)
    judge_scores: dict[str, float]  # judge name -> target-behavior score in [0, 1]


@dataclass
class LeakageEvidence:
    """Probe scores under three input conditions, for the leakage (Boxo) check.

    Same held-out examples/labels, re-scored after removing spans:
    - ``probe_scores_full``: the unmodified input,
    - ``probe_scores_leak_removed``: the giveaway spans removed (elicitation
      prompt / the model's verbalized reasoning),
    - ``probe_scores_random_removed``: an equal amount of *random* spans removed
      — the matched control, so a drop can be attributed to removing the leaky
      content, not to removing tokens in general (Boxo et al. 2509.21344).

    Leaky ⟺ AUROC collapses under leak-removal but *not* under random-removal:
    the probe was reading the giveaway text, not an internal state.

    Optional named ``cot`` span category (the verbalizer-vs-CoT question): the
    same examples re-scored with ONLY the model's chain-of-thought stripped,
    reported as its own delta. If the signal collapses under CoT removal but not
    under the matched random removal, it was reading the CoT text — it adds
    nothing over reading the transcript. If it survives, it is reading
    something the CoT does not expose.
    - ``probe_scores_cot_removed``: the model's CoT/verbalized reasoning removed,
    - ``probe_scores_cot_random_removed``: a random span of matched length
      removed — the control for the CoT removal specifically. If absent, the
      generic ``probe_scores_random_removed`` control is used (and noted).
    """

    labels: list[int]
    probe_scores_full: list[float]
    probe_scores_leak_removed: list[float]
    probe_scores_random_removed: list[float]
    probe_scores_cot_removed: list[float] | None = None
    probe_scores_cot_random_removed: list[float] | None = None

    def __post_init__(self) -> None:
        n = len(self.labels)
        if not (len(self.probe_scores_full) == len(self.probe_scores_leak_removed)
                == len(self.probe_scores_random_removed) == n):
            raise ValueError("leakage evidence fields must have equal length")
        if n == 0:
            raise ValueError("leakage evidence is empty")
        if set(self.labels) - {0, 1}:
            raise ValueError("labels must be 0/1")
        if self.probe_scores_cot_removed is not None and \
                len(self.probe_scores_cot_removed) != n:
            raise ValueError("leakage evidence fields must have equal length")
        if self.probe_scores_cot_random_removed is not None:
            if self.probe_scores_cot_removed is None:
                raise ValueError(
                    "cot_random_removed scores supplied without cot_removed scores"
                )
            if len(self.probe_scores_cot_random_removed) != n:
                raise ValueError("leakage evidence fields must have equal length")


@dataclass
class VerbalizationEvidence:
    """Per-example claims from an activation verbalizer (Patchscopes / LatentQA
    style): the audited signal when SIEVE audits a verbalizer instead of a probe.

    The verbalizer reads a hidden state at (layer, token position) of the target
    model and emits a natural-language claim about a behavioral property Y. Its
    validity threat is confabulation: reproducing its own priors or paraphrasing
    the input/CoT text rather than reading the internals (arXiv 2509.13316).
    SIEVE catches that with the machinery it already has — the adapter
    scalarizes each claim into P(claim asserts Y) and feeds those scores through
    the decodability + surface-baseline + Tier-2 leakage gates unchanged:

    - claim predicted by a TF-IDF/length baseline on ``texts``
      -> surface_confounded (text inversion caught mechanically);
    - claim collapses when ``cot_texts`` are stripped (vs random removal)
      -> the ``cot`` leakage category fires (the verbalizer was reading the CoT);
    - causal stage: a direction recovered from the CLAIMS faces the full
      matched-control steering suite.

    This record documents what the verbalizer read and claimed — scores and
    strings only, never raw activations. ``verbalizer_claim_scores`` must equal
    the bundle's decodability ``probe_scores`` (the engine enforces it), so the
    verdict provably concerns the verbalizer's claims.
    """

    target_model: str            # the model whose activations were verbalized
    verbalizer: str              # verbalizer model/method + version (e.g. "patchscopes:llama3-8b")
    layer: int                   # layer the activation was read from
    token_selection: str         # which token position(s) were read (documented, not raw data)
    property_tested: str         # the behavioral property Y the claims are about
    texts: list[str]             # raw input prompts (surface baselines / span removal operate on these)
    cot_texts: list[str]         # target model's CoT per example (the Tier-2 `cot` removal span)
    labels: list[int | None]     # ground-truth Y per example (0/1); None-filled for label-free use
    verbalizer_claim_scores: list[float]  # P(claim asserts Y | activation), each in [0, 1]
    families: list[str]          # prompt-family id per example (held-out splits)
    # Explicit attestation that no listed example tuned the verbalizer or its
    # claim scorer — mirrors probe_scores_out_of_sample. Defaults to False.
    claim_scores_out_of_sample: bool = False

    def __post_init__(self) -> None:
        n = len(self.texts)
        if not (len(self.cot_texts) == len(self.labels)
                == len(self.verbalizer_claim_scores) == len(self.families) == n):
            raise ValueError("verbalization fields must have equal length")
        if n == 0:
            raise ValueError("verbalization evidence is empty")
        if set(l for l in self.labels if l is not None) - {0, 1}:
            raise ValueError("labels must be 0/1 (or None for label-free use)")
        if any(not (0.0 <= s <= 1.0) for s in self.verbalizer_claim_scores):
            raise ValueError("verbalizer_claim_scores must lie in [0, 1]")


@dataclass
class EvidenceBundle:
    """Everything a SIEVE audit needs, recorded by an adapter."""

    # --- scope (copied into the audit card verbatim) ---
    model: str
    revision: str | None
    layers: list[int]
    direction_source: str
    prompt_distribution: str
    prompt_license: str
    behavioral_metrics: list[str]
    adapter: str                 # what produced this bundle (name + version)

    # --- evidence ---
    decodability: DecodabilityEvidence | None = None
    efficacy: list[EfficacyRecord] = field(default_factory=list)
    steering: list[SteeringRecord] = field(default_factory=list)
    # optional necessity evidence (#2): empty by default, so a bundle without
    # ablation audits exactly as before — the necessity gate is purely additive.
    ablation: list[AblationRecord] = field(default_factory=list)
    # optional multi-layer (joint) ablation evidence: detects a distributed
    # ("committee") mechanism a single-layer intervention misses. Empty by
    # default, so the gate is purely additive.
    multilayer: list[MultiLayerRecord] = field(default_factory=list)
    # optional leakage evidence (Tier-2): probe scores under full / leak-removed
    # / random-removed inputs; absent by default, so the gate is purely additive.
    leakage: "LeakageEvidence | None" = None
    # optional off-distribution probe scores for the deployment lens; absent by
    # default, in which case the lens reports off-distribution as "not assessed".
    deployment: "DeploymentEvidence | None" = None
    # optional activation-patching (oracle) evidence: calibrates how faithfully
    # the audited direction captures the site's causal content. Empty by default,
    # so the gate is purely additive.
    patching: list[PatchingRecord] = field(default_factory=list)
    # optional verbalizer evidence: documents the audited signal when it is an
    # activation verbalizer's per-example claims. Absent by default; when
    # present alongside decodability, the engine enforces that the claim scores
    # ARE the audited probe_scores (no bait-and-switch).
    verbalization: "VerbalizationEvidence | None" = None

    bundle_version: str = "0.1"

    def validate(self) -> None:
        """Reject bundles with duplicate records (silent last-wins overwrites
        would let a vendor submit retakes and keep the favorable one)."""
        seen: set[tuple] = set()
        for r in self.steering:
            key = ("steer", r.arm, r.alpha, r.prompt_id)
            if key in seen:
                raise ValueError(f"duplicate steering record: {key}")
            seen.add(key)
        for r in self.efficacy:
            key = ("eff", r.arm, r.alpha, r.prompt_id)
            if key in seen:
                raise ValueError(f"duplicate efficacy record: {key}")
            seen.add(key)
        for r in self.ablation:
            key = ("ablate", r.arm, r.prompt_id)
            if key in seen:
                raise ValueError(f"duplicate ablation record: {key}")
            seen.add(key)
        for r in self.multilayer:
            key = ("ml", r.arm, r.prompt_id, tuple(sorted(r.layers)))
            if key in seen:
                raise ValueError(f"duplicate multilayer record: {key}")
            seen.add(key)
        for r in self.patching:
            key = ("patch", r.arm, r.prompt_id, tuple(sorted(r.layers)))
            if key in seen:
                raise ValueError(f"duplicate patching record: {key}")
            seen.add(key)

    # ---- (de)serialization ----

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceBundle":
        dec = d.get("decodability")
        lk = d.get("leakage")
        dep = d.get("deployment")
        verb = d.get("verbalization")
        return cls(
            model=d["model"],
            revision=d.get("revision"),
            layers=list(d["layers"]),
            direction_source=d["direction_source"],
            prompt_distribution=d["prompt_distribution"],
            prompt_license=d["prompt_license"],
            behavioral_metrics=list(d["behavioral_metrics"]),
            adapter=d["adapter"],
            decodability=DecodabilityEvidence(**dec) if dec else None,
            efficacy=[EfficacyRecord(**r) for r in d.get("efficacy", [])],
            steering=[SteeringRecord(**r) for r in d.get("steering", [])],
            ablation=[AblationRecord(**r) for r in d.get("ablation", [])],
            multilayer=[MultiLayerRecord(**r) for r in d.get("multilayer", [])],
            patching=[PatchingRecord(**r) for r in d.get("patching", [])],
            leakage=LeakageEvidence(**lk) if lk else None,
            deployment=DeploymentEvidence(**dep) if dep else None,
            verbalization=VerbalizationEvidence(**verb) if verb else None,
            bundle_version=d.get("bundle_version", "0.1"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceBundle":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ---- convenience views ----

    @property
    def steering_arms(self) -> list[str]:
        return sorted({r.arm for r in self.steering})

    @property
    def ablation_arms(self) -> list[str]:
        return sorted({r.arm for r in self.ablation})

    @property
    def multilayer_arms(self) -> list[str]:
        return sorted({r.arm for r in self.multilayer})

    @property
    def patching_arms(self) -> list[str]:
        return sorted({r.arm for r in self.patching})

    @property
    def judge_names(self) -> list[str]:
        names: set[str] = set()
        for r in self.steering:
            names.update(r.judge_scores)
        return sorted(names)

    @property
    def alpha_grid(self) -> list[float]:
        return sorted({r.alpha for r in self.steering})
