"""Stage 3: matched-control steering and two-judge behavioral scoring.

The causal-sufficiency question is never "did steering along the probe change
behavior?" but "did it change behavior *more than matched controls*" — a
random direction, an orthogonalized random direction, and the same direction
injected at a wrong layer (DESIGN.md section 4). The primary test points are
the largest-|alpha| values in the grid (fixed by the adapter's config before
results exist, so they cannot be chosen post hoc), and a monotone
dose-response across the grid is required so an isolated blip at one alpha
cannot carry a causal verdict.

Judges: every steered generation is scored by >=2 independent judges; SIEVE
reports inter-rater agreement (Cohen's kappa on binarized scores) and requires
every judge to agree on the *direction* of the probe-arm effect.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .bundle import SteeringRecord
from .config import AuditConfig
from .stats import CI, bootstrap_abs_mean_diff, bootstrap_mean, cohen_kappa, dose_response

PROBE_ARM = "probe"


@dataclass
class JudgeResult:
    judges: list[str]
    min_pairwise_spearman: float    # continuous agreement, all records
    min_pairwise_kappa: float       # binarized agreement, informative records only
    n_informative: int              # records outside the binarization deadband
    agreement_ok: bool
    judges_agree_on_direction: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "judges": self.judges,
            "min_pairwise_spearman": self.min_pairwise_spearman,
            "min_pairwise_kappa": self.min_pairwise_kappa,
            "n_informative": self.n_informative,
            "agreement_ok": self.agreement_ok,
            "judges_agree_on_direction": self.judges_agree_on_direction,
            "notes": self.notes,
        }


@dataclass
class ControlsResult:
    arms: list[str]
    missing_controls: list[str]
    primary_alphas: list[float]                      # largest-|alpha| points
    # arm -> {alpha: CI of mean per-prompt behavioral delta vs alpha=0}
    arm_effects: dict[str, dict[float, CI]]
    significant_probe_alphas: list[float]
    # alpha -> control arm -> CI of |probe effect| - |control effect|
    probe_vs_controls: dict[float, dict[str, CI]]
    dose_rho: float
    dose_p: float
    judge: JudgeResult
    probe_effect_significant: bool
    exceeds_all_controls: bool
    dose_response_ok: bool
    causally_sufficient: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "arms": self.arms,
            "missing_controls": self.missing_controls,
            "primary_alphas": self.primary_alphas,
            "arm_effects": {
                a: {str(al): ci.to_dict() for al, ci in d.items()}
                for a, d in self.arm_effects.items()
            },
            "significant_probe_alphas": self.significant_probe_alphas,
            "probe_vs_controls": {
                str(al): {c: ci.to_dict() for c, ci in d.items()}
                for al, d in self.probe_vs_controls.items()
            },
            "dose_rho": self.dose_rho,
            "dose_p": self.dose_p,
            "judge": self.judge.to_dict(),
            "probe_effect_significant": self.probe_effect_significant,
            "exceeds_all_controls": self.exceeds_all_controls,
            "dose_response_ok": self.dose_response_ok,
            "causally_sufficient": self.causally_sufficient,
            "notes": self.notes,
        }


def _mean_judge_score(r: SteeringRecord, judges: list[str]) -> float:
    return float(np.mean([r.judge_scores[j] for j in judges]))


def _paired_deltas(
    records: list[SteeringRecord], judges: list[str] | None = None
) -> dict[str, dict[float, np.ndarray]]:
    """Per-arm, per-alpha arrays of behavioral deltas vs the same prompt at alpha=0.

    ``judges=None`` averages all judges; passing a single judge name computes
    that judge's view (used for the direction-agreement check).
    """
    judge_list = judges
    by_arm: dict[str, dict[float, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    all_judges = sorted({j for r in records for j in r.judge_scores})
    use = judge_list if judge_list is not None else all_judges
    for r in records:
        by_arm[r.arm][r.alpha][r.prompt_id] = _mean_judge_score(r, use)

    out: dict[str, dict[float, np.ndarray]] = {}
    for arm, by_alpha in by_arm.items():
        base = by_alpha.get(0.0)
        if base is None:
            continue
        out[arm] = {}
        for alpha, scores in by_alpha.items():
            if alpha == 0.0:
                continue
            shared = sorted(set(scores) & set(base))
            out[arm][alpha] = np.array([scores[p] - base[p] for p in shared])
    return out


def _judge_agreement(
    records: list[SteeringRecord],
    deltas_by_judge: dict[str, dict[str, dict[float, np.ndarray]]],
    primary_alphas: list[float],
    cfg: AuditConfig,
) -> JudgeResult:
    judges = sorted({j for r in records for j in r.judge_scores})
    notes: list[str] = []
    if len(judges) < cfg.min_judges:
        return JudgeResult(
            judges=judges,
            min_pairwise_spearman=float("nan"),
            min_pairwise_kappa=float("nan"),
            n_informative=0,
            agreement_ok=False,
            judges_agree_on_direction=False,
            notes=[f"only {len(judges)} judge(s); protocol requires >= {cfg.min_judges}"],
        )

    # continuous agreement over every judged generation (all arms)
    from scipy.stats import spearmanr

    spearmans = []
    for a, b in combinations(judges, 2):
        sa = np.array([r.judge_scores[a] for r in records])
        sb = np.array([r.judge_scores[b] for r in records])
        rho = spearmanr(sa, sb).statistic
        spearmans.append(0.0 if np.isnan(rho) else float(rho))
    min_spearman = float(min(spearmans))

    # binarized agreement, but only where the judges' mean score is away from
    # the threshold — at the threshold, binarized (dis)agreement is pure noise
    thr, band = cfg.judge_binarize_threshold, cfg.judge_deadband
    informative = [
        r
        for r in records
        if abs(np.mean(list(r.judge_scores.values())) - thr) > band
    ]
    if informative:
        kappas = []
        for a, b in combinations(judges, 2):
            ra = np.array([r.judge_scores[a] > thr for r in informative])
            rb = np.array([r.judge_scores[b] > thr for r in informative])
            kappas.append(cohen_kappa(ra, rb))
        min_kappa = float(min(kappas))
        kappa_ok = min_kappa >= cfg.min_judge_kappa
    else:
        # no record left the deadband: nothing to binarize, kappa is moot
        # (and the probe effect cannot be significant anyway)
        min_kappa = float("nan")
        kappa_ok = True
        notes.append("no judged record outside the deadband; kappa not computable")

    agreement_ok = (min_spearman >= cfg.min_judge_spearman) and kappa_ok
    if not agreement_ok:
        notes.append(
            f"judge agreement failed (min spearman {min_spearman:.2f} "
            f"vs >= {cfg.min_judge_spearman}; min kappa {min_kappa:.2f} on "
            f"{len(informative)} informative records vs >= {cfg.min_judge_kappa})"
        )

    # every judge must see the probe effect point the same way at primary alphas
    direction_ok = True
    for alpha in primary_alphas:
        signs = []
        for j in judges:
            d = deltas_by_judge[j].get(PROBE_ARM, {}).get(alpha)
            if d is not None and len(d) and abs(d.mean()) > 1e-12:
                signs.append(np.sign(d.mean()))
        if len(set(signs)) > 1:
            direction_ok = False
            notes.append(f"judges disagree on probe effect direction at alpha={alpha:g}")

    return JudgeResult(
        judges=judges,
        min_pairwise_spearman=min_spearman,
        min_pairwise_kappa=min_kappa,
        n_informative=len(informative),
        agreement_ok=agreement_ok,
        judges_agree_on_direction=direction_ok,
        notes=notes,
    )


def run_controls(records: list[SteeringRecord], cfg: AuditConfig) -> ControlsResult:
    if not records:
        raise ValueError("no steering records")
    rng = np.random.default_rng(cfg.seed)
    notes: list[str] = []

    arms = sorted({r.arm for r in records})
    missing = [c for c in cfg.required_controls if c not in arms]
    if PROBE_ARM not in arms:
        raise ValueError("steering records contain no 'probe' arm")

    deltas = _paired_deltas(records)
    judges = sorted({j for r in records for j in r.judge_scores})
    deltas_by_judge = {j: _paired_deltas(records, judges=[j]) for j in judges}

    probe_alphas = sorted(deltas.get(PROBE_ARM, {}), key=abs)
    if not probe_alphas:
        raise ValueError("probe arm has no nonzero-alpha records paired with alpha=0")
    max_abs = abs(probe_alphas[-1])
    primary = sorted(a for a in deltas[PROBE_ARM] if abs(a) == max_abs)

    # per-arm effects with CIs
    arm_effects: dict[str, dict[float, CI]] = {}
    for arm, by_alpha in deltas.items():
        arm_effects[arm] = {}
        for alpha, d in sorted(by_alpha.items()):
            if len(d) < cfg.min_steered_prompts:
                notes.append(
                    f"{arm}@alpha={alpha:g}: only {len(d)} paired prompts "
                    f"(< {cfg.min_steered_prompts})"
                )
            arm_effects[arm][alpha] = bootstrap_mean(d, rng, cfg.n_boot, cfg.ci_level)

    underpowered = any("paired prompts" in n for n in notes)

    # probe effect significance at primary points
    significant = [a for a in primary if arm_effects[PROBE_ARM][a].excludes(0.0)]
    probe_sig = bool(significant)

    # probe vs each control, matched alpha, at the significant primary points
    probe_vs_controls: dict[float, dict[str, CI]] = {}
    exceeds_all = probe_sig and not missing
    for alpha in significant:
        probe_vs_controls[alpha] = {}
        for control in cfg.required_controls:
            d_control = deltas.get(control, {}).get(alpha)
            if d_control is None or len(d_control) == 0:
                notes.append(f"control '{control}' missing at alpha={alpha:g}")
                exceeds_all = False
                continue
            diff = bootstrap_abs_mean_diff(
                deltas[PROBE_ARM][alpha], d_control, rng, cfg.n_boot, cfg.ci_level
            )
            probe_vs_controls[alpha][control] = diff
            if not diff.lo > 0:
                exceeds_all = False

    # dose-response over the full grid (per-record points for power)
    pairs = [
        (alpha, v)
        for alpha, d in deltas[PROBE_ARM].items()
        for v in d
    ]
    dose_rho, dose_p = dose_response(
        np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    )
    dose_ok = abs(dose_rho) >= cfg.dose_response_min_rho and dose_p <= cfg.dose_response_max_p
    if len(deltas[PROBE_ARM]) < 2:
        dose_ok = False
        notes.append("fewer than 2 nonzero alphas: dose-response untestable")

    judge = _judge_agreement(records, deltas_by_judge, primary, cfg)

    causal = (
        probe_sig
        and exceeds_all
        and dose_ok
        and judge.agreement_ok
        and judge.judges_agree_on_direction
        and not missing
        and not underpowered
    )

    return ControlsResult(
        arms=arms,
        missing_controls=missing,
        primary_alphas=primary,
        arm_effects=arm_effects,
        significant_probe_alphas=significant,
        probe_vs_controls=probe_vs_controls,
        dose_rho=dose_rho,
        dose_p=dose_p,
        judge=judge,
        probe_effect_significant=probe_sig,
        exceeds_all_controls=exceeds_all,
        dose_response_ok=dose_ok,
        causally_sufficient=causal,
        notes=notes,
    )
