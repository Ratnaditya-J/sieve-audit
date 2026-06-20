"""Deployment lens: the practitioner's FP/FN view of the probe.

The audit verdict answers "is this signal causally load-bearing?". A deployer
asks a different question: "if I wire this probe into a monitor, how many real
cases will it miss, and how many false alarms will I eat?" This module turns the
recorded probe scores into that view — ROC curves and recall at a fixed
false-alarm (FPR) budget — across whatever conditions the bundle supports:

- ``in_distribution``  — the decodability eval set (always available);
- ``off_distribution`` — an optional different prompt mix, closer to the wild;
- ``leakage_removed``  — the Boxo leak-spans removed, the realistic floor when
  the behavior isn't spelled out in the transcript.

Everything is quantified only where it was measured. Conditions with no evidence
are reported as "not assessed", never silently assumed fine. Numbers stay
numpy/sklearn-only (no new dependency); the ROC chart and PDF report are
optional renderers layered on top.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_curve

from .config import AuditConfig
from .stats import CI, _percentile_ci, _stratified_indices, auroc


@dataclass
class RocCurve:
    name: str
    fpr: list[float]
    tpr: list[float]
    auroc: float
    n_pos: int
    n_neg: int

    def to_dict(self) -> dict:
        return {
            "name": self.name, "fpr": self.fpr, "tpr": self.tpr,
            "auroc": self.auroc, "n_pos": self.n_pos, "n_neg": self.n_neg,
        }


@dataclass
class OperatingPoint:
    fpr_target: float
    recall: CI            # TPR at the threshold that holds FPR <= target
    threshold: float

    def to_dict(self) -> dict:
        return {
            "fpr_target": self.fpr_target,
            "recall": self.recall.to_dict(),
            "threshold": self.threshold,
        }


@dataclass
class DeploymentLensResult:
    curves: list[RocCurve]
    operating_points: dict[str, list[OperatingPoint]]   # condition -> points
    plain_language: list[str]
    out_of_sample: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "curves": [c.to_dict() for c in self.curves],
            "operating_points": {
                k: [p.to_dict() for p in v] for k, v in self.operating_points.items()
            },
            "plain_language": self.plain_language,
            "out_of_sample": self.out_of_sample,
            "notes": self.notes,
        }


def _recall_at_fpr(y: np.ndarray, s: np.ndarray, target: float) -> tuple[float, float]:
    """Max recall (TPR) achievable while holding FPR <= target, and the score
    threshold that achieves it. NaN if a class is absent."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y, np.asarray(s, dtype=float))
    ok = fpr <= target
    if not ok.any():
        return 0.0, float("inf")
    i = int(np.max(np.flatnonzero(ok)))
    return float(tpr[i]), float(thr[i])


def _operating_point(
    y: np.ndarray, s: np.ndarray, target: float, cfg: AuditConfig
) -> OperatingPoint:
    rng = np.random.default_rng(cfg.seed)
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    point, thr = _recall_at_fpr(y, s, target)
    reps = np.empty(cfg.n_boot)
    for b in range(cfg.n_boot):
        idx = _stratified_indices(y, rng)
        reps[b], _ = _recall_at_fpr(y[idx], s[idx], target)
    lo, hi = _percentile_ci(reps[~np.isnan(reps)], cfg.ci_level)
    return OperatingPoint(
        fpr_target=target, recall=CI(point, lo, hi, cfg.ci_level), threshold=thr
    )


def _curve(name: str, y, s) -> RocCurve:
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if len(np.unique(y)) < 2:
        return RocCurve(name, [0.0, 1.0], [0.0, 1.0], 0.5, n_pos, n_neg)
    fpr, tpr, _ = roc_curve(y, s)
    return RocCurve(name, fpr.tolist(), tpr.tolist(), auroc(y, s), n_pos, n_neg)


def _pct(x: float) -> str:
    return "n/a" if (x is None or np.isnan(x)) else f"{x * 100:.0f}%"


def run_deployment(bundle, cfg: AuditConfig) -> DeploymentLensResult | None:
    """Build the deployment lens from whatever conditions the bundle supports.
    Returns None when there is no labelled probe-score evidence at all."""
    conditions: list[tuple[str, list, list]] = []
    out_of_sample = False
    if bundle.decodability is not None:
        d = bundle.decodability
        conditions.append(("in_distribution", d.labels, d.probe_scores))
        out_of_sample = d.probe_scores_out_of_sample
    dep = getattr(bundle, "deployment", None)
    if dep is not None:
        conditions.append(("off_distribution", dep.labels, dep.probe_scores))
    if bundle.leakage is not None:
        lk = bundle.leakage
        conditions.append(("leakage_removed", lk.labels, lk.probe_scores_leak_removed))
    if not conditions:
        return None

    targets = list(cfg.deployment_fpr_targets)
    curves = [_curve(name, y, s) for name, y, s in conditions]
    operating_points = {
        name: [_operating_point(np.asarray(y), np.asarray(s), t, cfg) for t in targets]
        for name, y, s in conditions
    }
    headline_t = 0.05 if 0.05 in targets else targets[len(targets) // 2]

    def recall_at(name: str) -> tuple[CI | None, float]:
        for p in operating_points.get(name, []):
            if p.fpr_target == headline_t:
                return p.recall, headline_t
        return None, headline_t

    plain: list[str] = []
    notes: list[str] = []
    by_name = {c.name: c for c in curves}

    ci_in, t = recall_at("in_distribution")
    if ci_in is not None:
        tail = "" if out_of_sample else " (in-sample scores — treat as an optimistic ceiling)"
        plain.append(
            f"In-distribution, at a {t * 100:.0f}% false-alarm rate this probe flags "
            f"about {_pct(ci_in.point)} of the cases it should "
            f"(95% CI {_pct(ci_in.lo)}–{_pct(ci_in.hi)}){tail}."
        )
        if not out_of_sample:
            notes.append(
                "in-distribution recall is computed on in-sample probe scores; the "
                "real held-out number can only be lower"
            )

    if "off_distribution" in by_name:
        ci_off, t = recall_at("off_distribution")
        plain.append(
            f"Off-distribution (a different prompt mix, closer to the wild), the same "
            f"{t * 100:.0f}% false-alarm setting catches about {_pct(ci_off.point)} "
            f"(95% CI {_pct(ci_off.lo)}–{_pct(ci_off.hi)})."
        )
    else:
        plain.append(
            "Off-distribution (in-the-wild) performance: NOT ASSESSED — no "
            "off-distribution evidence was supplied, so the real-world miss rate "
            "is unknown."
        )

    if "leakage_removed" in by_name:
        ci_lk, t = recall_at("leakage_removed")
        plain.append(
            f"With the giveaway text removed, recall at {t * 100:.0f}% FPR falls to "
            f"about {_pct(ci_lk.point)}: the realistic floor when the behavior isn't "
            "spelled out in the transcript."
        )

    plain.append(
        "Bottom line: choose your threshold from the ROC curve — lower it to miss "
        "fewer cases (more false alarms), raise it to cut false alarms (more misses)."
    )

    return DeploymentLensResult(
        curves=curves,
        operating_points=operating_points,
        plain_language=plain,
        out_of_sample=out_of_sample,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# self-contained ROC chart (pure SVG string — no plotting dependency)
# ---------------------------------------------------------------------------

_COLORS = {
    "in_distribution": "#2563eb",
    "off_distribution": "#16a34a",
    "leakage_removed": "#dc2626",
}


def roc_svg(curves: list[dict], width: int = 440, height: int = 440) -> str:
    """Render the ROC curves as a standalone SVG string (embeddable in HTML/MD).

    Takes a list of curve dicts (``{name, fpr, tpr, auroc}``) — the
    ``RocCurve.to_dict()`` shape — so it renders straight from a card's
    diagnostics without needing the live result object."""
    pad = 50
    w, h = width - 2 * pad, height - 2 * pad

    def X(fpr: float) -> float:
        return pad + fpr * w

    def Y(tpr: float) -> float:
        return pad + (1 - tpr) * h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="sans-serif" font-size="12">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        # axes
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad + h}" stroke="#222"/>',
        f'<line x1="{pad}" y1="{pad + h}" x2="{pad + w}" y2="{pad + h}" stroke="#222"/>',
        # chance diagonal
        f'<line x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}" '
        f'stroke="#bbb" stroke-dasharray="4 4"/>',
        f'<text x="{pad + w / 2}" y="{height - 12}" text-anchor="middle">'
        f'False-alarm rate (FPR)</text>',
        f'<text x="14" y="{pad + h / 2}" text-anchor="middle" '
        f'transform="rotate(-90 14 {pad + h / 2})">Recall (TPR)</text>',
    ]
    # gridline labels at 0, .5, 1
    for v in (0.0, 0.5, 1.0):
        parts.append(
            f'<text x="{X(v)}" y="{pad + h + 16}" text-anchor="middle" '
            f'fill="#555">{v:.1f}</text>'
        )
        parts.append(
            f'<text x="{pad - 8}" y="{Y(v) + 4}" text-anchor="end" '
            f'fill="#555">{v:.1f}</text>'
        )
    legend_y = pad + 6
    for c in curves:
        color = _COLORS.get(c["name"], "#7c3aed")
        pts = " ".join(f"{X(f):.1f},{Y(t):.1f}" for f, t in zip(c["fpr"], c["tpr"]))
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        label = f"{c['name']} (AUROC {c['auroc']:.2f})"
        parts.append(
            f'<rect x="{pad + 10}" y="{legend_y - 9}" width="12" height="3" fill="{color}"/>'
        )
        parts.append(f'<text x="{pad + 28}" y="{legend_y - 4}">{label}</text>')
        legend_y += 18
    parts.append("</svg>")
    return "\n".join(parts)
