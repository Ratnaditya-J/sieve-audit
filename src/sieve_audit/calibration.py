"""#3 scaffold: calibrate the verdict against ground-truth-labeled cases.

SIEVE's causal criterion (matched-control steering + ablation necessity) is one
test with known blind spots (see verdict.py / DESIGN.md). Rather than assert the
criterion is "correct," this measures its ERROR RATE against cases whose true
status is known by construction — turning an unsettled criterion into a bounded,
reported one. The cell of interest is the **false negative**: a truly-causal
direction the criterion returns as ``not_causally_sufficient``.

The default case set is the rigged self-test scenarios (synth.py). The stronger
version labels *real* directions with an independent oracle — activation
patching / causal mediation — and slots those ``(name, bundle, expected)`` cases
into the same report; that labeling needs a GPU run, but the report machinery
here is oracle-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import AuditConfig
from .engine import run_audit
from .synth import SCENARIOS


@dataclass
class CalibrationReport:
    n: int
    n_correct: int
    accuracy: float
    false_negatives: int          # expected causally_sufficient, got not_causally_sufficient
    confusion: dict = field(default_factory=dict)   # expected -> {got: count}
    mismatches: list = field(default_factory=list)   # [{name, expected, got}]

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "n_correct": self.n_correct,
            "accuracy": self.accuracy,
            "false_negatives": self.false_negatives,
            "confusion": self.confusion,
            "mismatches": self.mismatches,
        }


def default_cases() -> list[tuple[str, object, str]]:
    """(name, bundle, expected_verdict) from the rigged scenarios; the SCENARIOS
    key IS the verdict each case is constructed to produce."""
    return [(name, SCENARIOS[name](), name) for name in SCENARIOS]


def run_calibration(cases=None, cfg: AuditConfig | None = None) -> CalibrationReport:
    """Audit each labeled case and tally verdict-vs-truth.

    ``cases`` is an iterable of ``(name, EvidenceBundle, expected_verdict_str)``;
    defaults to the rigged self-test set. Returns a CalibrationReport with the
    overall accuracy and the false-negative count (truly-causal scored
    not_causally_sufficient — the criterion's most consequential error mode)."""
    cases = list(cases) if cases is not None else default_cases()
    cfg = cfg or AuditConfig()
    confusion: dict = {}
    mismatches: list = []
    n_correct = 0
    false_negatives = 0
    for name, bundle, expected in cases:
        card = run_audit(bundle, cfg).card
        got = card.verdict.value if card.verdict else card.status
        confusion.setdefault(expected, {})
        confusion[expected][got] = confusion[expected].get(got, 0) + 1
        if got == expected:
            n_correct += 1
        else:
            mismatches.append({"name": name, "expected": expected, "got": got})
        if expected == "causally_sufficient" and got == "not_causally_sufficient":
            false_negatives += 1
    n = len(cases)
    return CalibrationReport(
        n=n,
        n_correct=n_correct,
        accuracy=(n_correct / n if n else 0.0),
        false_negatives=false_negatives,
        confusion=confusion,
        mismatches=mismatches,
    )
