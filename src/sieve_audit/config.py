"""Audit configuration: every threshold the verdict depends on, in one place.

The config is part of the audit card's hashed scope (DESIGN.md section 6), so two
audits with different thresholds can never be confused for the same protocol.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class AuditConfig:
    """Thresholds and protocol requirements for one SIEVE audit."""

    # --- statistics ---
    n_boot: int = 2000              # bootstrap resamples for every CI
    ci_level: float = 0.95          # two-sided confidence level
    seed: int = 0

    # --- decodability gates ---
    auroc_chance_margin: float = 0.03   # probe must beat 0.5 by this (CI lower bound)
    auroc_baseline_margin: float = 0.02 # probe must beat best surface baseline by this
    min_eval_n: int = 50                # min held-out examples per class for AUROC

    # --- efficacy gate ---
    noop_tolerance: float = 1e-3        # alpha=0 must move the residual stream less than this (relative)
    min_resid_rel_delta: float = 0.05   # at max |alpha|, median relative residual movement must exceed this
    require_output_change: bool = True  # at max |alpha|, at least one output must change

    # --- causal-sufficiency gates ---
    required_controls: tuple[str, ...] = ("random", "orthogonal", "wrong_layer")
    min_steered_prompts: int = 20       # per arm, per alpha
    dose_response_min_rho: float = 0.5  # |Spearman rho| of effect vs alpha
    dose_response_max_p: float = 0.05

    # --- judges ---
    min_judges: int = 2
    judge_binarize_threshold: float = 0.5
    # kappa is only computed where judges' mean score is at least this far from
    # the binarization threshold; at the threshold, binarized agreement
    # measures noise, not judge reliability
    judge_deadband: float = 0.1
    min_judge_kappa: float = 0.4
    min_judge_spearman: float = 0.6   # continuous agreement over all records

    def to_dict(self) -> dict:
        d = asdict(self)
        d["required_controls"] = list(self.required_controls)
        return d
