"""Stage 1: is the signal decodable at all, and does it beat surface baselines?

Verdict contributions (DESIGN.md section 3):
- probe no better than chance on held-out examples  -> not_decodable
- probe beaten/matched by a surface baseline        -> surface_confounded

Baselines are evaluated leave-one-family-out so they face the same
generalization burden the probe claims to pass. With a single family, SIEVE
falls back to stratified k-fold and records "family generalization untested"
as a residual risk.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .baselines import SURFACE_BASELINES, fit_baseline_scores
from .bundle import DecodabilityEvidence
from .config import AuditConfig
from .stats import CI, bootstrap_auroc, bootstrap_auroc_diff


@dataclass
class DecodabilityResult:
    probe_auroc: CI
    baseline_aurocs: dict[str, float]
    # AUROC(probe) - AUROC(baseline), paired bootstrap, per baseline
    probe_vs_baseline: dict[str, CI]
    beats_chance: bool
    beats_baselines: bool
    held_out_scheme: str            # "leave-one-family-out" | "stratified-kfold"
    n_examples: int
    n_families: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "probe_auroc": self.probe_auroc.to_dict(),
            "baseline_aurocs": self.baseline_aurocs,
            "probe_vs_baseline": {k: v.to_dict() for k, v in self.probe_vs_baseline.items()},
            "beats_chance": self.beats_chance,
            "beats_baselines": self.beats_baselines,
            "held_out_scheme": self.held_out_scheme,
            "n_examples": self.n_examples,
            "n_families": self.n_families,
            "notes": self.notes,
        }


def _held_out_baseline_scores(
    ev: DecodabilityEvidence, cfg: AuditConfig
) -> tuple[dict[str, np.ndarray], str, list[str]]:
    """Score every example with each baseline while it is held out of training."""
    labels = np.asarray(ev.labels)
    families = np.asarray(ev.families)
    unique_families = np.unique(families)
    notes: list[str] = []
    scores = {name: np.full(len(labels), np.nan) for name in SURFACE_BASELINES}

    if len(unique_families) >= 2:
        scheme = "leave-one-family-out"
        splits = [(families != f, families == f) for f in unique_families]
    else:
        scheme = "stratified-kfold"
        notes.append(
            "single prompt family: family generalization untested; "
            "baselines evaluated via stratified 5-fold instead"
        )
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
        splits = []
        for tr, te in skf.split(np.zeros(len(labels)), labels):
            tr_mask = np.zeros(len(labels), bool)
            te_mask = np.zeros(len(labels), bool)
            tr_mask[tr] = True
            te_mask[te] = True
            splits.append((tr_mask, te_mask))

    for tr_mask, te_mask in splits:
        tr_labels = labels[tr_mask]
        if len(np.unique(tr_labels)) < 2:
            notes.append("a training split had one class; baseline scores 0.5 there")
            for name in SURFACE_BASELINES:
                scores[name][te_mask] = 0.5
            continue
        tr_texts = [t for t, m in zip(ev.texts, tr_mask) if m]
        te_texts = [t for t, m in zip(ev.texts, te_mask) if m]
        for name in SURFACE_BASELINES:
            scores[name][te_mask] = fit_baseline_scores(
                name, tr_texts, tr_labels, te_texts, seed=cfg.seed
            )
    return scores, scheme, notes


def run_decodability(ev: DecodabilityEvidence, cfg: AuditConfig) -> DecodabilityResult:
    labels = np.asarray(ev.labels)
    probe_scores = np.asarray(ev.probe_scores, dtype=float)
    rng = np.random.default_rng(cfg.seed)

    baseline_scores, scheme, notes = _held_out_baseline_scores(ev, cfg)

    probe_auroc = bootstrap_auroc(labels, probe_scores, rng, cfg.n_boot, cfg.ci_level)
    # If the probe anti-predicts (AUROC < 0.5), flipping its sign is information
    # the auditee did not claim; we audit the direction as shipped.
    beats_chance = probe_auroc.lo > 0.5 + cfg.auroc_chance_margin

    baseline_aurocs: dict[str, float] = {}
    probe_vs_baseline: dict[str, CI] = {}
    beats_all = True
    for name, b_scores in baseline_scores.items():
        diff = bootstrap_auroc_diff(
            labels, probe_scores, b_scores, rng, cfg.n_boot, cfg.ci_level
        )
        from .stats import auroc as _auroc

        baseline_aurocs[name] = _auroc(labels, b_scores)
        probe_vs_baseline[name] = diff
        if not diff.lo > cfg.auroc_baseline_margin:
            beats_all = False

    n_per_class = min(int((labels == 0).sum()), int((labels == 1).sum()))
    if n_per_class < cfg.min_eval_n:
        notes.append(
            f"only {n_per_class} examples in the smaller class "
            f"(< {cfg.min_eval_n}); decodability estimates are low-powered"
        )

    return DecodabilityResult(
        probe_auroc=probe_auroc,
        baseline_aurocs=baseline_aurocs,
        probe_vs_baseline=probe_vs_baseline,
        beats_chance=beats_chance,
        beats_baselines=beats_all,
        held_out_scheme=scheme,
        n_examples=len(labels),
        n_families=len(set(ev.families)),
        notes=notes,
    )
