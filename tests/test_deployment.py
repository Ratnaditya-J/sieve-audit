"""Deployment lens tests: ROC + recall@FPR + plain-language, quantified only
where measured ('not assessed' otherwise), plus the chart/HTML/PDF artifacts."""
import numpy as np

from sieve_audit.bundle import (
    DecodabilityEvidence,
    DeploymentEvidence,
    EvidenceBundle,
    LeakageEvidence,
)
from sieve_audit.config import AuditConfig
from sieve_audit.deployment import roc_svg, run_deployment


def _scores(labels):
    out = []
    for i, l in enumerate(labels):
        if l == 1:
            out.append(0.9 if i % 7 else 0.4)    # ~1/7 positives missed
        else:
            out.append(0.1 if i % 11 else 0.8)   # ~1/11 negatives false-alarm
    return out


def _decod(n=70, oos=True):
    labels = [i % 2 for i in range(n)]
    return DecodabilityEvidence(
        texts=[f"t{i}" for i in range(n)],
        labels=labels,
        probe_scores=_scores(labels),
        families=["f0"] * n,
        probe_scores_out_of_sample=oos,
    )


def _bundle(**kw):
    return EvidenceBundle(
        model="m", revision=None, layers=[5], direction_source="contrastive",
        prompt_distribution="d", prompt_license="x", behavioral_metrics=["b"],
        adapter="test", **kw,
    )


def test_in_distribution_lens_computed():
    res = run_deployment(_bundle(decodability=_decod()), AuditConfig(seed=0))
    assert res is not None
    names = [c.name for c in res.curves]
    assert names == ["in_distribution"]
    pts = res.operating_points["in_distribution"]
    assert {p.fpr_target for p in pts} == {0.01, 0.05, 0.10}
    for p in pts:
        assert 0.0 <= p.recall.point <= 1.0
        assert p.recall.lo <= p.recall.point <= p.recall.hi
    assert any("false-alarm rate" in s for s in res.plain_language)


def test_off_distribution_not_assessed_when_absent():
    res = run_deployment(_bundle(decodability=_decod()), AuditConfig(seed=0))
    assert any("NOT ASSESSED" in s for s in res.plain_language)


def test_off_distribution_and_leakage_quantified_when_present():
    labels = [i % 2 for i in range(60)]
    dep = DeploymentEvidence(distribution="wild", labels=labels, probe_scores=_scores(labels))
    lk = LeakageEvidence(
        labels=labels,
        probe_scores_full=_scores(labels),
        probe_scores_leak_removed=[0.5] * 60,
        probe_scores_random_removed=_scores(labels),
    )
    res = run_deployment(
        _bundle(decodability=_decod(), deployment=dep, leakage=lk), AuditConfig(seed=0)
    )
    names = {c.name for c in res.curves}
    assert names == {"in_distribution", "off_distribution", "leakage_removed"}
    assert any("Off-distribution" in s and "catches about" in s for s in res.plain_language)
    assert any("giveaway text removed" in s for s in res.plain_language)


def test_out_of_sample_flag_warns_when_in_sample():
    res = run_deployment(_bundle(decodability=_decod(oos=False)), AuditConfig(seed=0))
    assert not res.out_of_sample
    assert any("optimistic ceiling" in s for s in res.plain_language)
    assert any("in-sample" in n for n in res.notes)


def test_returns_none_without_any_evidence():
    assert run_deployment(_bundle(), AuditConfig(seed=0)) is None


def test_roc_svg_renders_curves():
    res = run_deployment(_bundle(decodability=_decod()), AuditConfig(seed=0))
    svg = roc_svg([c.to_dict() for c in res.curves])
    assert svg.startswith("<svg") and "polyline" in svg and "AUROC" in svg


def test_deployment_surfaces_in_card_and_artifacts(tmp_path):
    from sieve_audit import run_audit
    from sieve_audit.card import card_to_html, write_card

    card = run_audit(_bundle(decodability=_decod())).card
    assert "deployment" in card.diagnostics
    md = card_to_html(card)
    assert "<svg" in md and "deployment report" in md
    write_card(card, tmp_path, "c")
    assert (tmp_path / "c.roc.svg").exists()
    assert (tmp_path / "c.html").exists()


def test_pdf_export_when_matplotlib_available(tmp_path):
    import importlib.util

    if importlib.util.find_spec("matplotlib") is None:
        return  # optional dependency; skip silently
    from sieve_audit import run_audit
    from sieve_audit.report import write_pdf

    card = run_audit(_bundle(decodability=_decod())).card
    out = write_pdf(card, tmp_path / "c.pdf")
    assert out.exists() and out.stat().st_size > 0
