"""Oracle (activation-patching) calibration: is the audited direction the right
coordinate of the mechanism, or just a correlate of it?

Steering and ablation manipulate a *learned* direction. Activation patching
transplants the *actual* residual activation from a clean run into a corrupted
run, so the full-site patch is a ground-truth ("oracle") measure of how much
causal content lives at that site. The calibration question is then:

    of the effect the full-site patch restores, how much does patching ONLY the
    audited direction's component recover?

A high recovered fraction means the direction is a faithful coordinate of the
site's mechanism. A low fraction means the site is causal but the mechanism does
not run through the audited direction — a probe can be necessary/sufficient-ish
and still be an unfaithful readout. The denominator (full-site effect) is what
makes this stronger than the single-direction tests: it is the real causal
ceiling, not a learned proxy.

Faithful ⟺ recovered fraction (CI lower bound) >= ``oracle_min_recovered`` AND
the direction-patch beats the random-patch control. Anti-gaming asymmetry: a
missing arm, too few judges/prompts, or a full-site patch that itself restores
nothing (no causal content to attribute) yields ``inconclusive`` — never a free
"faithful".
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .bundle import PatchingRecord
from .config import AuditConfig
from .stats import CI, _percentile_ci, bootstrap_mean

CLEAN = "clean"
CORRUPT = "corrupt"
PATCH_FULL = "patch_full"
PATCH_DIRECTION = "patch_direction"
PATCH_RANDOM = "patch_random"


@dataclass
class OracleResult:
    layers: list[int]
    arms: list[str]
    n_paired: int
    judges: list[str]
    full_effect: CI | None           # mean(patch_full - corrupt): the oracle restoration
    direction_effect: CI | None      # mean(patch_direction - corrupt)
    recovered_fraction: CI | None     # direction_effect / full_effect (paired bootstrap)
    direction_vs_random: CI | None    # mean(patch_direction - patch_random)
    patch_completeness: CI | None     # mean((full-corrupt)/(clean-corrupt)) if clean present
    faithful: bool
    inconclusive: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        c = lambda v: v.to_dict() if v is not None else None  # noqa: E731
        return {
            "layers": self.layers,
            "arms": self.arms,
            "n_paired": self.n_paired,
            "judges": self.judges,
            "full_effect": c(self.full_effect),
            "direction_effect": c(self.direction_effect),
            "recovered_fraction": c(self.recovered_fraction),
            "direction_vs_random": c(self.direction_vs_random),
            "patch_completeness": c(self.patch_completeness),
            "faithful": self.faithful,
            "inconclusive": self.inconclusive,
            "notes": self.notes,
        }


def _by_arm_prompt(records, judges) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for r in records:
        out[r.arm][r.prompt_id] = float(np.mean([r.judge_scores[j] for j in judges]))
    return out


def run_oracle(records: list[PatchingRecord], cfg: AuditConfig) -> OracleResult:
    if not records:
        raise ValueError("no patching records")
    layer_sets = {tuple(sorted(r.layers)) for r in records}
    if len(layer_sets) != 1:
        raise ValueError(
            f"patching records must share one site (got {sorted(layer_sets)})"
        )
    layers = sorted(layer_sets.pop())
    rng = np.random.default_rng(cfg.seed)
    arms = sorted({r.arm for r in records})
    judges = sorted({j for r in records for j in r.judge_scores})
    notes: list[str] = []

    def _incon(reason: str) -> OracleResult:
        notes.append(reason)
        return OracleResult(
            layers=layers, arms=arms, n_paired=0, judges=judges,
            full_effect=None, direction_effect=None, recovered_fraction=None,
            direction_vs_random=None, patch_completeness=None,
            faithful=False, inconclusive=True, notes=notes,
        )

    required = {CORRUPT, PATCH_FULL, PATCH_DIRECTION, PATCH_RANDOM}
    if not required.issubset(arms):
        return _incon(
            "oracle calibration needs corrupt, patch_full, patch_direction and "
            f"patch_random arms (missing {sorted(required - set(arms))})"
        )
    if len(judges) < cfg.min_judges:
        return _incon(f"only {len(judges)} judge(s); oracle requires >= {cfg.min_judges}")

    per = _by_arm_prompt(records, judges)
    x, f, d, r = per[CORRUPT], per[PATCH_FULL], per[PATCH_DIRECTION], per[PATCH_RANDOM]
    shared = sorted(set(x) & set(f) & set(d) & set(r))
    if len(shared) < cfg.min_steered_prompts:
        return _incon(
            f"only {len(shared)} prompts shared across the patch arms "
            f"(< {cfg.min_steered_prompts}): oracle underpowered"
        )

    full = np.array([f[p] - x[p] for p in shared])
    direction = np.array([d[p] - x[p] for p in shared])
    dir_minus_rand = np.array([d[p] - r[p] for p in shared])

    full_effect = bootstrap_mean(full, rng, cfg.n_boot, cfg.ci_level)
    direction_effect = bootstrap_mean(direction, rng, cfg.n_boot, cfg.ci_level)
    direction_vs_random = bootstrap_mean(dir_minus_rand, rng, cfg.n_boot, cfg.ci_level)

    # the oracle must actually restore behavior, else there is nothing to attribute
    if full_effect.lo <= 0:
        return _incon(
            "full-site patch does not significantly restore behavior: the site "
            "carries no measurable causal content, so the direction cannot be "
            "calibrated against it"
        )

    # recovered fraction: ratio of means, with a paired-prompt bootstrap
    point = float(direction.mean() / full.mean())
    reps = np.empty(cfg.n_boot)
    n = len(shared)
    for b in range(cfg.n_boot):
        idx = rng.integers(0, n, n)
        fm = full[idx].mean()
        reps[b] = direction[idx].mean() / fm if fm != 0 else np.nan
    lo, hi = _percentile_ci(reps[~np.isnan(reps)], cfg.ci_level)
    recovered = CI(point, lo, hi, cfg.ci_level)

    patch_completeness = None
    if CLEAN in arms:
        c = per[CLEAN]
        sc = [p for p in shared if p in c and (c[p] - x[p]) != 0]
        if sc:
            comp = np.array([(f[p] - x[p]) / (c[p] - x[p]) for p in sc])
            patch_completeness = bootstrap_mean(comp, rng, cfg.n_boot, cfg.ci_level)

    faithful = (
        recovered.lo >= cfg.oracle_min_recovered
        and direction_vs_random.lo > 0
    )
    if not faithful and not notes:
        notes.append(
            "direction-patch recovers less of the full-site effect than the bar "
            "(or does not beat the random-patch control): the audited direction is "
            "not a faithful coordinate of the site's causal mechanism"
        )

    return OracleResult(
        layers=layers, arms=arms, n_paired=len(shared), judges=judges,
        full_effect=full_effect, direction_effect=direction_effect,
        recovered_fraction=recovered, direction_vs_random=direction_vs_random,
        patch_completeness=patch_completeness,
        faithful=faithful, inconclusive=False, notes=notes,
    )
