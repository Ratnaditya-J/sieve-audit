"""The audit engine: evidence bundle in, audit card out.

Beyond orchestration, the engine owns the cross-stage consistency checks that
no single stage can see (added after adversarial review):

- the efficacy evidence must describe the *same* intervention the steering
  evidence judged (same max |alpha|, overlapping prompts) — otherwise a
  vendor can demonstrate efficacy on one setup and judge behavior on another;
- a failed efficacy gate alongside *significant* behavioral steering deltas
  is internally inconsistent evidence (sandbagged efficacy records), and is
  refused rather than rewarded with the face-saving ``intervention_ineffective``;
- every required control arm must itself have passed a movement check, so a
  degenerate control (zero-norm vector, dead hook) cannot make the probe look
  superior;
- an audit configured with fewer than the canonical control arms can never
  issue a causal verdict;
- any stage crash on malformed evidence becomes an ``insufficient_protocol``
  refusal (DESIGN.md section 7.2), never a silent pass.
"""
from __future__ import annotations

from dataclasses import dataclass

from .bundle import EvidenceBundle
from .card import build_card
from .config import CANONICAL_CONTROLS, AuditConfig
from .controls import PROBE_ARM, ControlsResult, run_controls
from .decodability import DecodabilityResult, run_decodability
from .efficacy import EfficacyResult, run_efficacy_all_arms
from .leakage import LeakageResult, run_leakage
from .necessity import NecessityResult, run_necessity
from .verdict import AuditCard, Decision, decide


@dataclass
class AuditResult:
    card: AuditCard
    decodability: DecodabilityResult | None
    efficacy: dict[str, EfficacyResult] | None   # per steering arm
    controls: ControlsResult | None
    necessity: NecessityResult | None = None     # optional ablation gate (#2)
    leakage: LeakageResult | None = None          # optional Tier-2 leakage gate


def _cross_stage_gaps(
    bundle: EvidenceBundle,
    efficacy: dict[str, EfficacyResult] | None,
    controls: ControlsResult | None,
    cfg: AuditConfig,
) -> list[str]:
    gaps: list[str] = []

    if set(CANONICAL_CONTROLS) - set(cfg.required_controls):
        gaps.append(
            f"audit configured with a non-canonical control suite "
            f"{list(cfg.required_controls)}; causal verdicts require "
            f"{list(CANONICAL_CONTROLS)}"
        )

    if efficacy is None or controls is None:
        return gaps

    probe_eff = efficacy.get(PROBE_ARM)
    if probe_eff is None:
        gaps.append("no efficacy records for the probe arm")
        return gaps

    # efficacy must describe the intervention that was judged
    primary = max((abs(a) for a in controls.primary_alphas), default=0.0)
    if primary and probe_eff.max_alpha != primary:
        gaps.append(
            f"efficacy was demonstrated at |alpha|={probe_eff.max_alpha:g} but "
            f"steering was judged at |alpha|={primary:g}: not the same intervention"
        )

    eff_prompts = {
        r.prompt_id
        for r in bundle.efficacy
        if r.arm == PROBE_ARM and abs(r.alpha) == probe_eff.max_alpha
    }
    steer_prompts = {
        r.prompt_id
        for r in bundle.steering
        if r.arm == PROBE_ARM and abs(r.alpha) == primary
    }
    shared = eff_prompts & steer_prompts
    need = min(cfg.min_shared_efficacy_prompts, len(steer_prompts))
    if len(shared) < need:
        gaps.append(
            f"efficacy and steering evidence share only {len(shared)} prompts "
            f"at the primary alpha (>= {need} required): efficacy may describe "
            "a different run"
        )

    # sandbagging check: dead-hook efficacy + significant behavioral deltas
    # cannot both be true of the same intervention
    if not probe_eff.effective and controls.probe_effect_significant:
        gaps.append(
            "efficacy gate failed but steering shows significant behavioral "
            "deltas: internally inconsistent evidence (suspected sandbagged "
            "efficacy records)"
        )

    # every control arm must itself be a live intervention — but only once the
    # probe arm is live: with a dead probe hook the verdict is
    # intervention_ineffective regardless, and a dead layer is dead for every
    # arm (degenerate controls can only flatter an *effective* probe)
    if not probe_eff.effective:
        return gaps
    for control in cfg.required_controls:
        if control in controls.missing_controls:
            continue  # already a gap via missing_controls
        ctrl_eff = efficacy.get(control)
        if ctrl_eff is None:
            gaps.append(
                f"control arm {control!r} has no efficacy records: a degenerate "
                "control would make any probe look superior"
            )
        elif not ctrl_eff.injection_verified:
            # liveness, not behavioral strength: catches a zero-norm/dead-hook
            # control, but does not penalise a wrong-layer arm whose injection
            # is real yet small relative to that layer's residual norm
            gaps.append(
                f"control arm {control!r} applied no real perturbation "
                "(zero-norm direction or dead hook): degenerate control"
            )
    return gaps


def run_audit(
    bundle: EvidenceBundle,
    cfg: AuditConfig | None = None,
    bundle_path: str | None = None,
    prereg=None,
) -> AuditResult:
    """Run every stage the evidence supports, then decide and emit the card.

    All stages with available evidence are run (their diagnostics always land
    on the card), but the verdict logic consumes them in protocol order:
    decodability -> efficacy gate -> matched controls -> cross-stage checks.
    """
    cfg = cfg or AuditConfig()
    hard_gaps: list[str] = []

    try:
        bundle.validate()
    except ValueError as exc:
        hard_gaps.append(f"bundle validation failed: {exc}")

    decod = efficacy = controls = None
    if bundle.decodability is not None:
        try:
            decod = run_decodability(bundle.decodability, cfg)
        except ValueError as exc:
            hard_gaps.append(f"decodability stage failed: {exc}")
    if bundle.efficacy:
        try:
            efficacy = run_efficacy_all_arms(bundle.efficacy, cfg)
        except ValueError as exc:
            hard_gaps.append(f"efficacy stage failed: {exc}")
    if bundle.steering:
        try:
            controls = run_controls(bundle.steering, cfg)
        except ValueError as exc:
            hard_gaps.append(f"steering stage failed: {exc}")
    necessity = None
    if bundle.ablation:
        try:
            necessity = run_necessity(bundle.ablation, cfg)
        except ValueError:
            # necessity is additive; a failure must not block the sufficiency
            # verdict — it simply omits the necessity finding from the card.
            necessity = None
    leakage = None
    if bundle.leakage is not None:
        try:
            leakage = run_leakage(bundle.leakage, cfg)
        except ValueError:
            leakage = None  # additive; never blocks the sufficiency verdict

    hard_gaps.extend(_cross_stage_gaps(bundle, efficacy, controls, cfg))

    sufficiency_blockers: list[str] = []
    if decod is not None and not decod.adequate_n:
        sufficiency_blockers.append(
            f"decodability evidence has < {cfg.min_eval_n} examples in the "
            "smaller class"
        )

    profile = cfg.profile_status()
    decision: Decision = decide(
        decod,
        efficacy.get(PROBE_ARM) if efficacy else None,
        controls,
        hard_gaps=hard_gaps,
        sufficiency_blockers=sufficiency_blockers,
        min_judges=cfg.min_judges,
        loosened_fields=profile["loosened"],
    )

    prereg_check = None
    if prereg is not None:
        from .prereg import verify_prereg

        prereg_check = verify_prereg(prereg, bundle, cfg)

    card = build_card(
        bundle, cfg, decision, decod, efficacy, controls, bundle_path, prereg_check,
        necessity, leakage,
    )
    return AuditResult(
        card=card,
        decodability=decod,
        efficacy=efficacy,
        controls=controls,
        necessity=necessity,
        leakage=leakage,
    )
