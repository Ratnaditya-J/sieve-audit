"""Inspect integration: run SIEVE audits as an Inspect task/scorer.

This is the adoption highway (DESIGN.md section 10). It lets a SIEVE verdict
slot into the Inspect ecosystem — `inspect eval`, the log viewer, and later
`inspect_evals` — so the audit is a *cited, runnable* protocol rather than a
one-off script.

SIEVE audits recorded **evidence bundles**, not live model generations, so the
integration is unusual for Inspect: there is no model call. Each Inspect
``Sample`` points at a bundle JSON; a no-op solver passes it through; and the
``sieve_scorer`` loads the bundle, runs ``run_audit``, and returns the verdict
as the Score (with the full rendered audit card as the explanation). The
``verdict_distribution`` metric tallies how a set of probes fared.

Usage::

    from sieve_audit.inspect_task import sieve_task
    # inspect eval sieve_audit.inspect_task@sieve_task -T bundles=path/to/bundles
    task = sieve_task(["reports/bundle_a.json", "reports/bundle_b.json"])

Requires the ``inspect`` extra: ``pip install "sieve-audit[inspect]"``.
"""
from __future__ import annotations

from pathlib import Path

from .bundle import EvidenceBundle
from .card import card_to_markdown
from .config import AuditConfig
from .engine import run_audit


def _require_inspect():
    try:
        import inspect_ai  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "the Inspect integration needs the inspect extra: "
            "pip install 'sieve-audit[inspect]'"
        ) from exc


# ---------------------------------------------------------------------------
# metric: the distribution of verdicts across the audited probes
# ---------------------------------------------------------------------------


def _make_verdict_distribution():
    from inspect_ai.scorer import metric

    @metric
    def verdict_distribution():
        """Fraction of audited bundles landing on each verdict / refusal."""

        # No parameter annotations: under `from __future__ import annotations`
        # they become strings that Inspect resolves in this module's globals,
        # where the Inspect types are not present — so we leave them off.
        def compute(scores):
            # Inspect numericizes Score.value for metrics (a categorical verdict
            # collapses to 0.0), so read the verdict from the string-preserving
            # `answer` field the scorer sets, falling back to metadata.
            counts: dict[str, float] = {}
            for s in scores:
                score = s if hasattr(s, "answer") else s.score
                verdict = score.answer or (score.metadata or {}).get("verdict") or "unknown"
                counts[str(verdict)] = counts.get(str(verdict), 0.0) + 1.0
            total = max(len(scores), 1)
            return {k: v / total for k, v in sorted(counts.items())}

        return compute

    return verdict_distribution


# ---------------------------------------------------------------------------
# scorer: run one audit, return the verdict + card
# ---------------------------------------------------------------------------


def sieve_scorer(seed: int = 0):
    """An Inspect scorer that audits the evidence bundle named by each sample.

    The bundle path is read from ``sample.metadata['bundle_path']`` (falling
    back to the sample input). The Score value is the verdict string (or
    ``insufficient_protocol``); the explanation is the full Markdown card; the
    metadata carries the config/bundle hashes and the profile status so a log
    reader sees whether the strict bar was used.
    """
    _require_inspect()
    from inspect_ai.scorer import Score, Target, scorer
    from inspect_ai.solver import TaskState

    verdict_distribution = _make_verdict_distribution()

    @scorer(metrics=[verdict_distribution()])
    def _scorer():
        async def score(state: TaskState, target: Target) -> Score:
            meta = state.metadata or {}
            bundle_path = meta.get("bundle_path") or state.input_text
            if not bundle_path:
                return Score(
                    value="insufficient_protocol",
                    answer="insufficient_protocol",
                    explanation="no bundle_path on the sample",
                )
            bundle = EvidenceBundle.load(bundle_path)
            result = run_audit(bundle, AuditConfig(seed=seed), bundle_path=str(bundle_path))
            card = result.card
            verdict = card.verdict.value if card.verdict else card.status
            return Score(
                value=verdict,
                answer=verdict,
                explanation=card_to_markdown(card),
                metadata={
                    "verdict": verdict,
                    "config_hash": card.config_hash,
                    "bundle_hash": card.bundle_hash,
                    "profile": card.diagnostics.get("profile"),
                    "decision_reasons": card.diagnostics.get("decision_reasons"),
                },
            )

        return score

    return _scorer()


# ---------------------------------------------------------------------------
# task: wrap a set of bundles into a runnable Inspect task
# ---------------------------------------------------------------------------


def sieve_task(bundles: list[str] | str, seed: int = 0):
    """Build an Inspect ``Task`` that audits each evidence bundle in ``bundles``.

    ``bundles`` is a list of bundle-JSON paths, or a single path, or a path to
    a directory of ``*.json`` bundles. No model is used — the audit is the work.
    """
    _require_inspect()
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.solver import Generate, TaskState, solver

    paths = _resolve_bundle_paths(bundles)
    if not paths:
        raise SystemExit(f"no bundle JSONs found in {bundles!r}")

    samples = [
        Sample(input=str(p), id=Path(p).stem, metadata={"bundle_path": str(p)})
        for p in paths
    ]

    @solver
    def _passthrough():
        # SIEVE audits recorded evidence; there is no model generation step.
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            return state

        return solve

    return Task(
        dataset=MemoryDataset(samples),
        solver=_passthrough(),
        scorer=sieve_scorer(seed=seed),
        name="sieve_audit",
    )


def _resolve_bundle_paths(bundles: list[str] | str) -> list[Path]:
    if isinstance(bundles, str):
        p = Path(bundles)
        if p.is_dir():
            return sorted(p.glob("*.json"))
        # comma-separated convenience for `-T bundles=a.json,b.json`
        return [Path(x) for x in bundles.split(",") if x.strip()]
    return [Path(b) for b in bundles]
