"""The audit engine: evidence bundle in, audit card out."""
from __future__ import annotations

from dataclasses import dataclass

from .bundle import EvidenceBundle
from .card import build_card
from .config import AuditConfig
from .controls import ControlsResult, run_controls
from .decodability import DecodabilityResult, run_decodability
from .efficacy import EfficacyResult, run_efficacy
from .verdict import AuditCard, Decision, decide


@dataclass
class AuditResult:
    card: AuditCard
    decodability: DecodabilityResult | None
    efficacy: EfficacyResult | None
    controls: ControlsResult | None


def run_audit(
    bundle: EvidenceBundle,
    cfg: AuditConfig | None = None,
    bundle_path: str | None = None,
) -> AuditResult:
    """Run every stage the evidence supports, then decide and emit the card.

    All stages with available evidence are run (their diagnostics always land
    on the card), but the verdict logic consumes them in protocol order:
    decodability -> efficacy gate -> matched controls.
    """
    cfg = cfg or AuditConfig()

    decod = run_decodability(bundle.decodability, cfg) if bundle.decodability else None
    efficacy = run_efficacy(bundle.efficacy, cfg) if bundle.efficacy else None
    controls = run_controls(bundle.steering, cfg) if bundle.steering else None

    decision: Decision = decide(
        decod,
        efficacy,
        controls,
        required_controls=cfg.required_controls,
        min_judges=cfg.min_judges,
    )

    card = build_card(bundle, cfg, decision, decod, efficacy, controls, bundle_path)
    return AuditResult(card=card, decodability=decod, efficacy=efficacy, controls=controls)
