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

Named ``cot`` span category (optional, adjudicated with the same rule): the
same examples re-scored with only the model's chain-of-thought stripped, vs a
matched random-removal control. This is the verbalizer-vs-CoT question made
mechanical: ``cot_leaky`` means the signal was reading the CoT text (a
CoT-parroting verbalizer adds nothing over the transcript); ``cot_survives``
means the signal retains above-chance discrimination WITHOUT the CoT - the only
regime in which a verbalizer tells you something the CoT does not. The survival
claim is held to the same asymmetry: it requires the post-removal AUROC's CI
lower bound to clear chance, never just "the drop was small".
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
    # --- named `cot` span category (None throughout when not supplied) ---
    auroc_cot_removed: CI | None = None
    drop_cot: CI | None = None          # auroc_full - auroc_cot_removed (paired)
    drop_cot_random: CI | None = None   # matched random control for the cot removal
    cot_leaky: bool | None = None       # signal collapses under CoT removal only
    cot_survives: bool | None = None    # signal still above chance WITHOUT the CoT
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
            "auroc_cot_removed": ci(self.auroc_cot_removed),
            "drop_cot": ci(self.drop_cot),
            "drop_cot_random": ci(self.drop_cot_random),
            "cot_leaky": self.cot_leaky,
            "cot_survives": self.cot_survives,
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

    # --- named `cot` span category (the verbalizer-vs-CoT read) ---
    a_cot = drop_cot = drop_cot_random = None
    cot_leaky = cot_survives = None
    if ev.probe_scores_cot_removed is not None:
        cot = np.asarray(ev.probe_scores_cot_removed, dtype=float)
        if ev.probe_scores_cot_random_removed is not None:
            cot_rand = np.asarray(ev.probe_scores_cot_random_removed, dtype=float)
        else:
            cot_rand = rand
            notes.append(
                "cot removal control shared with the generic random-span control "
                "(no matched cot_random_removed scores supplied)"
            )
        a_cot = bootstrap_auroc(y, cot, rng, cfg.n_boot, cfg.ci_level)
        drop_cot = bootstrap_auroc_diff(y, full, cot, rng, cfg.n_boot, cfg.ci_level)
        drop_cot_random = bootstrap_auroc_diff(
            y, full, cot_rand, rng, cfg.n_boot, cfg.ci_level
        )
        cot_leaky = (drop_cot.lo >= cfg.leakage_min_drop) and (drop_cot.lo > drop_cot_random.hi)
        # the survival claim is positive evidence and must be earned, not
        # defaulted: post-removal AUROC's CI lower bound must itself beat chance
        cot_survives = (not cot_leaky) and (a_cot.lo > 0.5 + cfg.auroc_chance_margin)
        if cot_leaky:
            notes.append(
                "signal collapses when the model's CoT is removed but not under "
                "matched random removal: it was reading the CoT text, not an "
                "internal state (CoT-parroting)"
            )
        elif cot_survives:
            notes.append(
                "signal retains above-chance discrimination with the CoT removed: "
                "it reads something the CoT does not expose"
            )
        else:
            notes.append(
                "cot removal neither fires the leak rule nor leaves an "
                "above-chance signal: the CoT comparison is uninformative here"
            )

    return LeakageResult(
        auroc_full=a_full,
        auroc_leak_removed=a_leak,
        auroc_random_removed=a_rand,
        drop_leak=drop_leak,
        drop_random=drop_random,
        leaky=leaky,
        inconclusive=False,
        auroc_cot_removed=a_cot,
        drop_cot=drop_cot,
        drop_cot_random=drop_cot_random,
        cot_leaky=cot_leaky,
        cot_survives=cot_survives,
        notes=notes,
    )
