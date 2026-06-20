"""Optional PDF deployment report (lazy matplotlib import).

The Markdown + JSON card and the HTML report are the canonical, dependency-free
artifacts. This adds a true ``.pdf`` for practitioners who want a single file:
the ROC curves (page 1) plus the plain-language summary and operating-point
table (page 2). matplotlib is imported lazily so the audit core has no plotting
dependency — only `sieve audit --pdf` pulls it in.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

from .verdict import AuditCard


def write_pdf(card: AuditCard, path: str | Path) -> Path:
    dep = card.diagnostics.get("deployment")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception as exc:  # pragma: no cover - exercised only without matplotlib
        raise RuntimeError(
            "PDF export needs matplotlib (pip install matplotlib); the Markdown, "
            "JSON and HTML reports need no extra dependency"
        ) from exc

    path = Path(path)
    verdict = card.label or (card.verdict.value if card.verdict else card.status)

    with PdfPages(path) as pdf:
        # page 1 — ROC curves
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        if dep:
            for c in dep["curves"]:
                ax.plot(c["fpr"], c["tpr"],
                        label=f"{c['name']} (AUROC {c['auroc']:.2f})", linewidth=2)
        ax.plot([0, 1], [0, 1], "--", color="#bbbbbb", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False-alarm rate (FPR)")
        ax.set_ylabel("Recall (TPR)")
        ax.set_title(f"SIEVE deployment ROC\n{verdict}", fontsize=11)
        if dep:
            ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # page 2 — plain-language summary + operating-point table
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        y = 0.97
        ax.text(0.05, y, "SIEVE deployment report", fontsize=16, weight="bold")
        y -= 0.035
        ax.text(0.05, y, f"Verdict: {verdict}", fontsize=11)
        y -= 0.035
        ax.text(0.05, y, "What this means for a deployer:", fontsize=11, weight="bold")
        y -= 0.03
        for p in (dep["plain_language"] if dep else []):
            for line in textwrap.wrap(p, 92):
                ax.text(0.06, y, line, fontsize=9.5)
                y -= 0.022
            y -= 0.008
        if dep:
            rows = []
            for name, pts in dep["operating_points"].items():
                for p in pts:
                    r = p["recall"]
                    rows.append([
                        name, f"{p['fpr_target'] * 100:.0f}%",
                        f"{r['point'] * 100:.0f}% "
                        f"[{r['lo'] * 100:.0f}, {r['hi'] * 100:.0f}]",
                    ])
            y -= 0.02
            tbl = ax.table(
                cellText=rows,
                colLabels=["Condition", "FPR budget", "Recall (95% CI)"],
                loc="upper left", cellLoc="left", bbox=[0.05, max(0.05, y - 0.4), 0.9, 0.4],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
        pdf.savefig(fig)
        plt.close(fig)
    return path
