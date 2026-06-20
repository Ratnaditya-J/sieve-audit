"""Tier-2 leakage gate (Boxo et al. 2509.21344): does the probe survive removing
the giveaway text?

A probe can beat a TF-IDF surface baseline yet still read *textual evidence* in
the transcript — the elicitation prompt or the model's verbalized reasoning —
rather than an internal state. This gate re-scores the held-out examples with
those spans removed and checks whether AUROC collapses, but only relative to a
*random-span-removal* control, so the drop is attributable to the leaky content,
not to perturbing the input in general.

Leaky ⟺ leak-removal drops AUROC by at least ``leakage_min_drop`` (CI lower
bound) AND by clearly more than random-removal (leak-drop CI above random-drop
CI). Anti-gaming asymmetry: a degenerate/one-class case yields ``inconclusive``,
never a free "clean" verdict.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bundle import LeakageEvidence
from .config import AuditConfig
from .stats import CI, bootstrap_auroc, bootstrap_auroc_diff


@dataclass
class LeakageResult:
    auroc_full: CI | None
    auroc_leak_removed: CI | None
    auroc_random_removed: CI | None
    drop_leak: CI | None            # auroc_full - auroc_leak_removed (paired)
    drop_random: CI | None          # auroc_full - auroc_random_removed (control)
    leaky: bool
    inconclusive: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        ci = lambda c: c.to_dict() if c is not None else None  # noqa: E731
        return {
            "auroc_full": ci(self.auroc_full),
            "auroc_leak_removed": ci(self.auroc_leak_removed),
            "auroc_random_removed": ci(self.auroc_random_removed),
            "drop_leak": ci(self.drop_leak),
            "drop_random": ci(self.drop_random),
            "leaky": self.leaky,
            "inconclusive": self.inconclusive,
            "notes": self.notes,
        }


def run_leakage(ev: LeakageEvidence, cfg: AuditConfig) -> LeakageResult:
    rng = np.random.default_rng(cfg.seed)
    y = np.asarray(ev.labels)
    if len(np.unique(y)) < 2:
        return LeakageResult(
            None, None, None, None, None, leaky=False, inconclusive=True,
            notes=["only one label class present; leakage not assessable"],
        )
    full = np.asarray(ev.probe_scores_full, dtype=float)
    leak = np.asarray(ev.probe_scores_leak_removed, dtype=float)
    rand = np.asarray(ev.probe_scores_random_removed, dtype=float)

    a_full = bootstrap_auroc(y, full, rng, cfg.n_boot, cfg.ci_level)
    a_leak = bootstrap_auroc(y, leak, rng, cfg.n_boot, cfg.ci_level)
    a_rand = bootstrap_auroc(y, rand, rng, cfg.n_boot, cfg.ci_level)
    drop_leak = bootstrap_auroc_diff(y, full, leak, rng, cfg.n_boot, cfg.ci_level)
    drop_random = bootstrap_auroc_diff(y, full, rand, rng, cfg.n_boot, cfg.ci_level)

    leaky = (drop_leak.lo >= cfg.leakage_min_drop) and (drop_leak.lo > drop_random.hi)
    notes: list[str] = []
    if not leaky:
        notes.append(
            "probe survives leak-span removal, or its drop is not separable from "
            "the random-removal control: no leakage detected at this bar"
        )
    return LeakageResult(
        auroc_full=a_full,
        auroc_leak_removed=a_leak,
        auroc_random_removed=a_rand,
        drop_leak=drop_leak,
        drop_random=drop_random,
        leaky=leaky,
        inconclusive=False,
        notes=notes,
    )
