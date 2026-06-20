"""Multi-layer (joint) ablation gate: the committee / distributed-mechanism test.

A probe reads one layer. A single-layer ablation can show *no* necessity even
when the direction is causally load-bearing, because the mechanism is spread
across layers and the others compensate (the "committee" objection: removing one
member's vote changes nothing). This gate runs the necessity adjudication over a
*joint* intervention applied at several layers at once.

It deliberately reuses ``run_necessity``: the arms (baseline/probe/ablate_random)
and the anti-gaming asymmetry are identical — only the intervention is joint
rather than single-layer. The card then compares the two:

- joint-necessary while single-layer was NOT  → distributed-mechanism signature;
- both not necessary                          → a stronger, layer-robust null;
- single-layer null with NO joint evidence    → the null is qualified (a
  distributed mechanism was never tested and cannot be ruled out).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .bundle import AblationRecord, MultiLayerRecord
from .config import AuditConfig
from .necessity import NecessityResult, run_necessity


@dataclass
class MultiLayerResult:
    layers: list[int]               # the joint layer set intervened together
    necessary: bool                 # joint ablation removes behavior beyond control
    inconclusive: bool
    necessity: NecessityResult      # full joint-ablation necessity detail
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "layers": self.layers,
            "necessary": self.necessary,
            "inconclusive": self.inconclusive,
            "necessity": self.necessity.to_dict(),
            "notes": self.notes,
        }


def run_multilayer(records: list[MultiLayerRecord], cfg: AuditConfig) -> MultiLayerResult:
    if not records:
        raise ValueError("no multilayer records")
    layer_sets = {tuple(sorted(r.layers)) for r in records}
    if len(layer_sets) != 1:
        raise ValueError(
            "multilayer records must all share one joint layer set "
            f"(got {sorted(layer_sets)})"
        )
    layers = sorted(layer_sets.pop())
    notes: list[str] = []
    if len(layers) < 2:
        notes.append(
            f"joint layer set has only {len(layers)} layer(s); this is not a "
            "genuine multi-layer test of a distributed mechanism"
        )

    abl = [
        AblationRecord(arm=r.arm, prompt_id=r.prompt_id, judge_scores=r.judge_scores)
        for r in records
    ]
    nec = run_necessity(abl, cfg)
    return MultiLayerResult(
        layers=layers,
        necessary=nec.necessary,
        inconclusive=nec.inconclusive,
        necessity=nec,
        notes=notes + list(nec.notes),
    )
