"""Audit configuration: every threshold the verdict depends on, in one place.

The full config is printed on the audit card and folded into the config hash
(DESIGN.md section 6), and any deviation from the published defaults is
flagged on the card — so a weakened protocol can never masquerade as the
standard one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

# The canonical control suite. A causal verdict additionally requires the
# audit's required_controls to include ALL of these; configuring fewer
# downgrades the audit to insufficient_protocol (anti-weakening guarantee).
CANONICAL_CONTROLS: tuple[str, ...] = ("random", "orthogonal", "wrong_layer")


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
    min_family_class_n: int = 5         # min examples per class per family (anti-gerrymandering)

    # --- efficacy gate ---
    noop_tolerance: float = 1e-3        # alpha=0 must move the residual stream less than this (relative)
    min_resid_rel_delta: float = 0.05   # at max |alpha|, median relative residual movement must exceed this
    require_output_change: bool = True  # at max |alpha|, at least one output must change

    # --- causal-sufficiency gates ---
    required_controls: tuple[str, ...] = CANONICAL_CONTROLS
    min_steered_prompts: int = 20       # per arm, per alpha
    require_symmetric_grid: bool = True # causal verdicts need both +max and -max alpha
    min_shared_efficacy_prompts: int = 10  # efficacy and steering must cover shared prompts
    dose_response_min_rho: float = 0.5  # |Spearman rho| of effect vs alpha
    dose_response_max_p: float = 0.05
    n_perm: int = 1000                  # within-prompt permutations for the dose-response p

    # --- judges ---
    min_judges: int = 2
    judge_binarize_threshold: float = 0.5
    # kappa is only computed on records where EVERY judge's score is at least
    # this far from the binarization threshold; at the threshold, binarized
    # agreement measures noise, not judge reliability
    judge_deadband: float = 0.1
    min_judge_kappa: float = 0.4
    min_judge_spearman: float = 0.6     # continuous agreement over all records
    # near-perfect agreement over many records means duplicated judges, not
    # excellent ones; flagged as a protocol violation
    max_judge_spearman: float = 0.995
    duplicate_judge_min_n: int = 200
    min_informative_judged: int = 30    # records outside the deadband needed for kappa

    def to_dict(self) -> dict:
        d = asdict(self)
        d["required_controls"] = list(self.required_controls)
        return d

    def nondefault_fields(self) -> dict:
        """Fields that deviate from the published protocol defaults."""
        default = AuditConfig(seed=self.seed).to_dict()
        mine = self.to_dict()
        return {k: v for k, v in mine.items() if default[k] != v and k != "seed"}
