"""Verdict taxonomy and audit-card schema - the intellectual core of SIEVE.

The diagnostics, steering controls, and judges are not implemented yet; this
module fixes the *contract*: the five-state verdict (DESIGN.md section 3) and the
scoped, caveat-bound record that gets emitted (DESIGN.md section 6). The verdict
and its scope/caveats are deliberately one object so a claim cannot be quoted
without its caveats.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    """The five possible outcomes of a SIEVE audit (DESIGN.md section 3)."""

    NOT_DECODABLE = "not_decodable"
    SURFACE_CONFOUNDED = "surface_confounded"
    # Steering never took effect (magnitude/quantization). Inconclusive, NOT a null.
    INTERVENTION_INEFFECTIVE = "intervention_ineffective"
    NOT_CAUSALLY_SUFFICIENT = "not_causally_sufficient"
    CAUSALLY_SUFFICIENT = "causally_sufficient"


@dataclass
class AuditCard:
    """A scoped, caveat-bound, reproducible record of one audit (DESIGN.md section 6)."""

    # --- scope: what was actually tested ---
    model: str
    revision: str | None
    layers: list[int]
    direction_source: str          # how the contrastive direction was derived
    prompt_distribution: str       # dataset name
    prompt_license: str
    n_prompts: int
    alpha_grid: list[float]
    behavioral_metrics: list[str]
    judges: list[str]
    controls: list[str]            # e.g. ["random", "orthogonal", "wrong_layer"]
    seed: int

    # --- results ---
    diagnostics: dict = field(default_factory=dict)
    verdict: Verdict | None = None

    # --- claim calibration (DESIGN.md section 6) ---
    allowed_claims: list[str] = field(default_factory=list)
    disallowed_claims: list[str] = field(default_factory=list)
    residual_risks: list[str] = field(default_factory=list)

    # --- reproducibility ---
    protocol_version: str = "0.1"
    config_hash: str | None = None
