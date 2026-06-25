"""Audit-card construction and rendering (DESIGN.md section 6).

The card is the only output of an audit: scope, diagnostics, verdict, allowed
and disallowed claims, residual risks, and a config hash, in one inseparable
record. A system card cites the hash; re-running the hash reproduces the card.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .bundle import EvidenceBundle
from .config import AuditConfig
from .controls import ControlsResult
from .deployment import roc_svg
from .decodability import DecodabilityResult
from .efficacy import EfficacyResult
from .verdict import (
    ALLOWED_CLAIMS,
    DISALLOWED_CLAIMS,
    DISALLOWED_CLAIMS_ALWAYS,
    AuditCard,
    Decision,
    Verdict,
)


def _ml_state(multilayer) -> str:
    """'untested' | 'inconclusive' | 'necessary_joint' | 'not_necessary_joint'."""
    if multilayer is None:
        return "untested"
    if multilayer.inconclusive:
        return "inconclusive"
    return "necessary_joint" if multilayer.necessary else "not_necessary_joint"


def _oracle_state(oracle) -> str:
    """'untested' | 'inconclusive' | 'faithful' | 'unfaithful'."""
    if oracle is None:
        return "untested"
    if oracle.inconclusive:
        return "inconclusive"
    return "faithful" if oracle.faithful else "unfaithful"


def _causal_summary(verdict, necessity, multilayer=None, oracle=None) -> dict:
    """Cross-method agreement (#4): integrate sufficiency (steering), necessity
    (single-layer ablation), the joint multi-layer test, and oracle (patching)
    calibration into one read, flagging — never hiding — disagreement or
    single-method coverage. No single method's result stands in for the whole
    causal picture."""
    if verdict == Verdict.CAUSALLY_SUFFICIENT:
        suff = "sufficient"
    elif verdict == Verdict.NOT_CAUSALLY_SUFFICIENT:
        suff = "not_sufficient"
    else:  # intervention_ineffective / not_decodable / surface_confounded / None
        suff = "untested"
    if necessity is None:
        nec = "untested"
    elif necessity.inconclusive:
        nec = "inconclusive"
    elif necessity.necessary:
        nec = "necessary"
    else:
        nec = "not_necessary"
    ml = _ml_state(multilayer)

    # The distributed-mechanism signature dominates the read: joint multi-layer
    # ablation is necessary while the single-layer test was not. This is the
    # exact false-negative a single-layer (steering- or ablation-only) verdict
    # cannot see, so it is surfaced above the single-layer combination.
    if ml == "necessary_joint" and nec in ("not_necessary", "inconclusive", "untested"):
        combined = ("DISTRIBUTED MECHANISM: joint multi-layer ablation is necessary "
                    "while single-layer ablation is not — the signal is causally "
                    "load-bearing across layers, invisible to single-layer "
                    "interventions")
    elif suff == "sufficient" and nec == "necessary":
        combined = ("sufficient AND necessary — strong causal evidence under the "
                    "tested interventions")
    elif suff == "untested" and nec == "necessary":
        combined = (
            "DIRECTION IS CAUSALLY NECESSARY (ablation) but the probe is "
            "surface-confounded at the decodability stage: the direction is a "
            "load-bearing intervention coordinate — ablating it degrades the "
            "behavior — but its scores are explained by surface text statistics "
            "and it is NOT licensed as a deployment signal detector. "
            "Causal involvement ≠ valid monitor."
        )
    elif suff == "not_sufficient" and nec == "necessary":
        combined = ("necessary but NOT sufficient — distributed/partial-mechanism "
                    "signature; the signal is causally involved, not inert")
    elif suff == "sufficient" and nec == "not_necessary":
        combined = ("sufficient but NOT necessary — likely one of several redundant "
                    "pathways")
    elif suff == "not_sufficient" and nec == "not_necessary":
        extra = (
            " (and joint multi-layer ablation also not necessary — layer-robust null)"
            if ml == "not_necessary_joint" else ""
        )
        combined = ("neither sufficient nor necessary under the tested interventions "
                    "— limited causal role" + extra)
    else:
        combined = (f"single-method evidence only (sufficiency={suff}, "
                    f"necessity={nec}); cross-method agreement not established")
    orc = _oracle_state(oracle)
    if orc == "unfaithful":
        combined += (
            " — but oracle (patching) calibration shows the site is causal while "
            "the audited DIRECTION is not its faithful coordinate (low recovered "
            "fraction): treat the probe as a correlate, not the mechanism"
        )
    elif orc == "faithful":
        combined += (
            " — oracle (patching) calibration confirms the audited direction is a "
            "faithful coordinate of the site's mechanism"
        )
    return {
        "sufficiency": suff, "necessity": nec, "multilayer": ml,
        "oracle": orc, "combined": combined,
    }


def _necessity_phrase(necessity, multilayer) -> str | None:
    """Human phrase rolling single-layer and joint multi-layer necessity.

    Returns None when neither test reached a conclusion (so the headline falls
    back to the bare verdict/status)."""
    ml_nec = multilayer is not None and not multilayer.inconclusive and multilayer.necessary
    sl_known = necessity is not None and not necessity.inconclusive
    sl_nec = sl_known and necessity.necessary
    if ml_nec and not sl_nec:
        return "necessary (multi-layer)"
    if sl_nec:
        return "necessary"
    if sl_known:   # single-layer concluded not-necessary, joint did not rescue it
        return "not necessary"
    return None


def _headline_label(verdict, status: str, necessity, leakage=None, multilayer=None) -> str:
    """Human-facing headline. Rolls the sufficiency-pipeline verdict together
    with the necessity finding — single-layer or joint multi-layer (so a real
    necessity result is surfaced, not buried under a bare 'insufficient_protocol')
    — and flags leakage. Falls back to the formal verdict/status when no
    necessity test reached a conclusion."""
    base = verdict.value if verdict is not None else status
    leak = " · leaky" if (leakage is not None and leakage.leaky) else ""
    nec = _necessity_phrase(necessity, multilayer)
    if nec is None:
        return base + leak
    if verdict is None:
        return f"{nec} · sufficiency not established{leak}"
    return f"{base} · {nec}{leak}"


def _base_residual_risks(efficacy, controls, necessity, multilayer) -> list[str]:
    """Residual risks that match the evidence actually supplied."""
    risks: list[str] = []

    if efficacy is not None or controls is not None:
        risks.append(
            "Sufficiency evidence uses single-layer additive steering; other "
            "intervention forms may still reveal different causal behavior."
        )
    else:
        risks.append("Causal sufficiency via steering was not tested.")

    if necessity is None:
        risks.append("Necessity (ablation) untested.")
    elif not necessity.inconclusive:
        risks.append(
            "Necessity was tested by directional ablation under the audited "
            "scope; other ablation methods and prompt distributions may differ."
        )

    if multilayer is None:
        risks.append("Distributed/multi-layer mechanisms untested.")
    elif not multilayer.inconclusive:
        risks.append(
            f"Joint multi-layer ablation tested layer set {multilayer.layers}; "
            "other layer sets and distributed mechanisms remain outside scope."
        )

    risks.extend([
        "Results are specific to the audited prompt distribution and may not transfer.",
        "Behavioral metrics depend on judge quality; judge agreement is reported, not guaranteed.",
    ])
    return risks


def _strip_license_suffix(distribution: str) -> str:
    """Remove a trailing ' (license: ...)' that was inadvertently baked into
    the distribution string.  The license belongs in the separate prompt_license
    field; the renderer appends it explicitly, so embedding it in distribution
    produces duplicate text."""
    idx = distribution.find(" (license:")
    return distribution[:idx].strip() if idx != -1 else distribution


def _canonical_hash(obj: object) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def scope_sentence(bundle: EvidenceBundle) -> str:
    layers = ",".join(str(l) for l in bundle.layers)
    methods = []
    if bundle.steering:
        methods.append("single-layer additive steering")
    if bundle.ablation:
        methods.append("ablation")
    if bundle.multilayer:
        methods.append("multilayer ablation")
    if bundle.patching:
        methods.append("activation patching")
    method_str = "/".join(methods) if methods else "decodability only"
    return (
        f"[model={bundle.model}"
        + (f"@{bundle.revision}" if bundle.revision else "")
        + f", layer(s)={layers}, direction={bundle.direction_source}, "
        f"prompts={bundle.prompt_distribution}, "
        f"metrics={'/'.join(bundle.behavioral_metrics)}, "
        f"{method_str}]"
    )


def build_card(
    bundle: EvidenceBundle,
    cfg: AuditConfig,
    decision: Decision,
    decod: DecodabilityResult | None,
    efficacy: dict[str, EfficacyResult] | None,
    controls: ControlsResult | None,
    bundle_path: str | None = None,
    prereg_check=None,
    necessity=None,
    leakage=None,
    multilayer=None,
    deployment=None,
    oracle=None,
    soft_gaps: list[str] | None = None,
) -> AuditCard:
    scope = scope_sentence(bundle)
    # Causal intervention(s) actually run; causal verdicts are bounded to these so
    # the verdict can never be quoted as a method-transcending claim.
    tested_interventions = (
        ["single-layer additive steering"]
        if (efficacy is not None or controls is not None)
        else []
    )
    necessity_claims: list[str] = []
    necessity_risks: list[str] = []
    if necessity is not None:
        if necessity.inconclusive:
            necessity_risks.append(
                "Necessity (ablation) evidence was provided but could not be "
                "adjudicated: " + "; ".join(necessity.notes)
            )
        else:
            tested_interventions = tested_interventions + ["ablation"]
            if necessity.necessary:
                necessity_claims.append(
                    "Under ablation, the signal IS necessary: removing it degrades "
                    "the behavior more than an ablate_random control. A "
                    "not_causally_sufficient verdict here is consistent with a "
                    "distributed/multi-layer mechanism that single-layer additive "
                    "steering cannot induce — the signal is NOT causally inert."
                )
            else:
                necessity_claims.append(
                    "Under ablation, the signal is also NOT necessary: removing it "
                    "degrades the behavior no more than an ablate_random control — "
                    "convergent evidence of a limited causal role under the tested "
                    "interventions."
                )
    # --- joint multi-layer ablation (committee / distributed-mechanism test) ---
    multilayer_claims: list[str] = []
    multilayer_risks: list[str] = []
    sl_not_necessary = (
        necessity is not None
        and not necessity.inconclusive
        and not necessity.necessary
    )
    if multilayer is not None:
        if multilayer.inconclusive:
            multilayer_risks.append(
                "Joint multi-layer ablation evidence was provided but could not be "
                "adjudicated: " + "; ".join(multilayer.notes)
            )
        else:
            tested_interventions = tested_interventions + [
                f"joint multi-layer ablation (layers {multilayer.layers})"
            ]
            if multilayer.necessary and sl_not_necessary:
                multilayer_claims.append(
                    "DISTRIBUTED MECHANISM: joint ablation across layers "
                    f"{multilayer.layers} IS necessary while single-layer ablation "
                    "was NOT. Direct evidence the signal is causally load-bearing "
                    "across layers — the false-negative a single-layer (steering- "
                    "or ablation-only) verdict cannot see. It is NOT causally inert."
                )
            elif multilayer.necessary:
                multilayer_claims.append(
                    f"Under joint ablation across layers {multilayer.layers}, the "
                    "signal is necessary: removing it jointly degrades the behavior "
                    "more than an ablate_random control — consistent with the "
                    "single-layer necessity finding."
                )
            else:
                multilayer_claims.append(
                    f"Even under joint ablation across layers {multilayer.layers}, "
                    "the signal is NOT necessary: a layer-robust null that single-"
                    "layer ablation alone could not establish."
                )
    # Over-read guard: a single-layer not-necessary null must NOT be read as
    # "no causal role" unless the distributed case was actually tested.
    if sl_not_necessary and (multilayer is None or multilayer.inconclusive):
        multilayer_risks.append(
            "Necessity was tested at single layer(s) only; a distributed, "
            "multi-layer ('committee') mechanism was NOT tested and cannot be ruled "
            "out by this null. Supply multi-layer ablation evidence to close this gap."
        )

    # --- oracle (activation-patching) calibration ---
    oracle_claims: list[str] = []
    oracle_risks: list[str] = []
    if oracle is not None:
        if oracle.inconclusive:
            oracle_risks.append(
                "Oracle (patching) calibration evidence was provided but could not "
                "be adjudicated: " + "; ".join(oracle.notes)
            )
        else:
            tested_interventions = tested_interventions + [
                f"activation patching (layers {oracle.layers})"
            ]
            rec = oracle.recovered_fraction
            if oracle.faithful:
                oracle_claims.append(
                    f"Oracle calibration: patching the audited direction recovers "
                    f"{rec.point:.0%} (95% CI {rec.lo:.0%}–{rec.hi:.0%}) of the "
                    f"full-site causal effect and beats a random-patch control — the "
                    "direction is a FAITHFUL coordinate of the site's mechanism, not "
                    "merely a correlate."
                )
            else:
                oracle_risks.append(
                    f"DIRECTION UNFAITHFUL (oracle/patching): the site is causal but "
                    f"patching the audited direction recovers only {rec.point:.0%} "
                    f"(95% CI {rec.lo:.0%}–{rec.hi:.0%}) of the full-site effect "
                    "(or does not beat the random-patch control). The probe reads a "
                    "correlate of the mechanism, not the coordinate the mechanism "
                    "runs through — necessity/sufficiency can still mislead a monitor."
                )

    interventions_str = ", ".join(tested_interventions) or "none (causal stage not run)"
    config_hash = _canonical_hash({"config": cfg.to_dict(), "protocol_version": "0.1"})
    bundle_hash = _canonical_hash(bundle.to_dict())

    profile = cfg.profile_status()
    diagnostics: dict = {
        "decision_reasons": decision.reasons,
        # the full protocol config is part of the card: a weakened threshold
        # can never hide behind a hash
        "config": cfg.to_dict(),
        "config_nondefault": cfg.nondefault_fields(),
        "profile": profile,
    }
    if prereg_check is not None:
        diagnostics["preregistration"] = prereg_check.to_dict()
        if not prereg_check.matches:
            risks_prereg = (
                "PRE-REGISTRATION MISMATCH: the run deviated from the committed "
                f"plan ({prereg_check.declared_hash[:16]}); the pre-registration "
                "claim does not hold. Diffs: " + "; ".join(prereg_check.diffs)
            )
        else:
            risks_prereg = None
    else:
        risks_prereg = None
    if decod is not None:
        diagnostics["decodability"] = decod.to_dict()
    if efficacy is not None:
        diagnostics["efficacy"] = {arm: res.to_dict() for arm, res in efficacy.items()}
    if controls is not None:
        diagnostics["controls"] = controls.to_dict()
    if necessity is not None:
        diagnostics["necessity"] = necessity.to_dict()
    if multilayer is not None:
        diagnostics["multilayer"] = multilayer.to_dict()
    if leakage is not None:
        diagnostics["leakage"] = leakage.to_dict()
    if oracle is not None:
        diagnostics["oracle"] = oracle.to_dict()
    if deployment is not None:
        diagnostics["deployment"] = deployment.to_dict()
    diagnostics["causal_summary"] = _causal_summary(
        decision.verdict, necessity, multilayer, oracle
    )
    headline = _headline_label(
        decision.verdict, decision.status, necessity, leakage, multilayer
    )

    if decision.verdict is not None:
        allowed = [
            c.format(scope=scope, interventions=interventions_str)
            for c in ALLOWED_CLAIMS[decision.verdict]
        ]
        disallowed = DISALLOWED_CLAIMS[decision.verdict] + DISALLOWED_CLAIMS_ALWAYS
    else:
        # protocol incomplete: no causal claim — but a cleanly-passed
        # decodability stage still licenses its own (correlational) claims,
        # and only at the strict bar (a loosened margin could have produced
        # the "beats baselines" pass unjustly)
        decod_clean = (
            decod is not None
            and decod.beats_chance
            and decod.beats_baselines
            and not decod.protocol_violations
            and profile["status"] != "loosened"
        )
        if decod_clean:
            allowed = [
                f"Under {scope}, the signal is linearly decodable on held-out "
                "prompt families and beats surface (text-statistics) baselines.",
                "NO causal or monitor-validation claim is licensed: the causal "
                "stages of the protocol were not run (see decision reasons).",
            ]
        else:
            allowed = [
                "None. The protocol was incomplete; no claim is licensed by this audit."
            ]
        disallowed = ["Any safety or causal claim."] + DISALLOWED_CLAIMS_ALWAYS

    allowed = allowed + necessity_claims + multilayer_claims + oracle_claims

    risks = _base_residual_risks(efficacy, controls, necessity, multilayer)
    if soft_gaps:
        for gap in soft_gaps:
            risks.insert(0, f"[provided but invalid] {gap}")
    if risks_prereg:
        risks.insert(0, risks_prereg)
    if profile["status"] == "loosened":
        risks.insert(
            0,
            "Config is LOOSENED relative to the frozen standard "
            f"({profile['profile']}); positive sub-claims are not licensed at "
            f"the standard bar. Loosened field(s): {profile['loosened']}.",
        )
    for res in (decod, controls):
        if res is not None:
            risks.extend(getattr(res, "notes", []))
    if efficacy is not None:
        for arm, res in sorted(efficacy.items()):
            risks.extend(f"[{arm}] {n}" for n in res.notes)
    risks.extend(necessity_risks)
    risks.extend(multilayer_risks)
    risks.extend(oracle_risks)
    if deployment is not None:
        risks.extend(f"[deployment] {n}" for n in deployment.notes)
    if leakage is not None and leakage.leaky:
        risks.insert(
            0,
            "LEAKY: probe AUROC collapses when giveaway spans are removed (and "
            "more than under random-span removal) — it reads textual evidence, "
            "not internal state; expect false negatives where the behavior isn't "
            "spelled out in the text.",
        )

    return AuditCard(
        model=bundle.model,
        revision=bundle.revision,
        layers=bundle.layers,
        direction_source=bundle.direction_source,
        prompt_distribution=bundle.prompt_distribution,
        prompt_license=bundle.prompt_license,
        n_prompts=decod.n_examples if decod else 0,
        alpha_grid=bundle.alpha_grid,
        behavioral_metrics=bundle.behavioral_metrics,
        judges=bundle.judge_names,
        controls=bundle.steering_arms,
        seed=cfg.seed,
        tested_interventions=tested_interventions,
        diagnostics=diagnostics,
        verdict=decision.verdict,
        status=decision.status,
        label=headline,
        allowed_claims=allowed,
        disallowed_claims=disallowed,
        residual_risks=risks,
        protocol_version="0.1",
        config_hash=config_hash,
        bundle_hash=bundle_hash,
        rerun_command=(
            f"sieve audit --bundle {bundle_path} --seed {cfg.seed}" if bundle_path else None
        ),
        preregistration=prereg_check.to_dict() if prereg_check is not None else None,
    )


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def card_to_json(card: AuditCard) -> str:
    d = asdict(card)
    d["verdict"] = card.verdict.value if card.verdict else None
    return json.dumps(d, indent=1, default=str)


def _fmt_ci(d: dict) -> str:
    return f"{d['point']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"


def _fmt_stat(x: float) -> str:
    """Format a possibly-NaN agreement statistic; NaN renders as 'n/a'."""
    return f"{x:.2f}" if x == x else "n/a"


_PROFILE_BADGE = {
    "strict": "✅ {name} (the standard bar)",
    "stricter": "✅ stricter than {name}",
    "loosened": "⚠️ CUSTOM — LOOSENED below {name} (positive verdicts void)",
}


def _profile_line(card: AuditCard) -> str:
    profile = card.diagnostics.get("profile") or {}
    status = profile.get("status", "strict")
    name = profile.get("profile", "SIEVE-v0.1-strict")
    badge = _PROFILE_BADGE.get(status, status).format(name=name)
    extra = ""
    if status == "loosened" and profile.get("loosened"):
        extra = f" — loosened: {', '.join(profile['loosened'])}"
    elif status == "stricter" and profile.get("tightened"):
        extra = f" — tightened: {', '.join(profile['tightened'])}"
    return f"**Profile:** {badge}{extra}"


def card_to_markdown(card: AuditCard) -> str:
    verdict_str = card.label or (card.verdict.value if card.verdict else card.status)
    interv = ", ".join(card.tested_interventions)
    untested = []
    if "ablation" not in card.tested_interventions:
        untested.append("necessity (ablation)")
    if not any(i.startswith("joint multi-layer ablation") for i in card.tested_interventions):
        untested.append("distributed/multi-layer mechanisms")
    caveat = "  ·  causal verdicts are bounded to these" + (
        f"; {' and '.join(untested)} not tested" if untested else ""
    )
    lines = [
        f"# SIEVE audit card — `{verdict_str}`",
        "",
        f"> **Verdict: {verdict_str}**"
        + (f" — under {interv}" if interv else "")
        + f" (protocol v{card.protocol_version}, "
        f"config `{card.config_hash}`, bundle `{card.bundle_hash}`)",
        ">",
        f"> **Tested intervention(s):** {interv or '— (causal stage not run)'}"
        + (caveat if interv else ""),
        ">",
        f"> {_profile_line(card)}",
    ]
    if card.preregistration is not None:
        pre = card.preregistration
        if pre["matches"]:
            lines.append(
                f"> **Pre-registered:** ✅ matches `{pre['declared_hash'][:16]}` "
                "(config + scope committed before results)"
            )
        else:
            lines.append(
                f"> **Pre-registered:** ⚠️ MISMATCH vs `{pre['declared_hash'][:16]}` "
                "— the run deviated from the committed plan (see residual risks)"
            )
    lines += [
        "",
        "## Scope (what was actually tested)",
        "",
        f"- **Model:** {card.model}" + (f" @ {card.revision}" if card.revision else ""),
        f"- **Layer(s):** {card.layers}",
        f"- **Direction:** {card.direction_source}",
        "- **Prompts:** "
        # strip any "(license: ...)" suffix that pod_run.sh may have embedded in the
        # distribution string (older bundles baked license into the distribution field)
        + _strip_license_suffix(card.prompt_distribution)
        + f" (license: {card.prompt_license}, n={card.n_prompts})",
        f"- **Alpha grid:** {card.alpha_grid}",
        f"- **Behavioral metric(s):** {', '.join(card.behavioral_metrics) or '—'}",
        f"- **Judges:** {', '.join(card.judges) or '—'}",
        f"- **Steering arms:** {', '.join(card.controls) or '—'}",
        f"- **Seed:** {card.seed}",
        "",
        "## Diagnostics",
        "",
    ]
    dec = card.diagnostics.get("decodability")
    if dec:
        lines += [
            f"- Probe AUROC: **{_fmt_ci(dec['probe_auroc'])}** "
            f"({dec['held_out_scheme']}, n={dec['n_examples']}, "
            f"{dec['n_families']} families)",
        ]
        for name, a in dec["baseline_aurocs"].items():
            diff = dec["probe_vs_baseline"][name]
            lines.append(
                f"- Surface baseline `{name}`: AUROC {a:.3f}; "
                f"probe − baseline = {_fmt_ci(diff)}"
            )
    eff = card.diagnostics.get("efficacy")
    if eff:
        for arm, e in sorted(eff.items()):
            tag = "Efficacy gate (probe)" if arm == "probe" else f"Control-arm movement ({arm})"
            lines.append(
                f"- {tag}: **{'passed' if e['effective'] else 'FAILED'}** "
                f"(hook_correct={e['hook_correct']}, "
                f"median rel. residual delta @|α|={e['max_alpha']:g}: "
                f"{e['median_rel_delta_at_max']:.4f}, "
                f"output changed: {e['any_output_changed_at_max']})"
            )
    ctrl = card.diagnostics.get("controls")
    if ctrl:
        ja = ctrl["judge"]
        raw = ja.get("raw_agreement", float("nan"))
        raw_str = f", raw={raw:.2f}" if raw == raw else ""  # raw==raw is False only for NaN
        floor_str = (
            "  [one-sided judged behavior: judges agree, one outcome dominates]"
            if ja.get("behavior_floor") else ""
        )
        lines.append(
            f"- Dose-response: rho={ctrl['dose_rho']:.2f} (p={ctrl['dose_p']:.4f}); "
            f"judge agreement: spearman={_fmt_stat(ja['min_pairwise_spearman'])}, "
            f"kappa={_fmt_stat(ja['min_pairwise_kappa'])}{raw_str}{floor_str}"
        )
        for alpha, by_control in ctrl["probe_vs_controls"].items():
            for control, diff in by_control.items():
                lines.append(
                    f"- |probe| − |{control}| @α={alpha}: {_fmt_ci(diff)}"
                )
    nec = card.diagnostics.get("necessity")
    if nec:
        if nec["inconclusive"]:
            lines.append(
                "- Necessity (ablation): inconclusive — " + "; ".join(nec["notes"])
            )
        else:
            lines.append(
                f"- Necessity (ablation): "
                f"{'NECESSARY' if nec['necessary'] else 'not necessary'} "
                f"(probe-ablation drop {_fmt_ci(nec['probe_drop'])}, "
                f"random-ablation drop {_fmt_ci(nec['random_drop'])}, "
                f"probe−random excess {_fmt_ci(nec['probe_vs_random_drop'])})"
            )
    ml = card.diagnostics.get("multilayer")
    if ml:
        if ml["inconclusive"]:
            lines.append(
                f"- Multi-layer ablation (joint layers {ml['layers']}): "
                "inconclusive — " + "; ".join(ml["notes"])
            )
        else:
            mn = ml["necessity"]
            lines.append(
                f"- Multi-layer ablation (joint layers {ml['layers']}): "
                f"{'NECESSARY (joint)' if ml['necessary'] else 'not necessary (joint)'} "
                f"(joint-ablation drop {_fmt_ci(mn['probe_drop'])}, "
                f"random-ablation drop {_fmt_ci(mn['random_drop'])}, "
                f"probe−random excess {_fmt_ci(mn['probe_vs_random_drop'])})"
            )
    orc = card.diagnostics.get("oracle")
    if orc:
        if orc["inconclusive"]:
            lines.append(
                f"- Oracle (patching, layers {orc['layers']}): inconclusive — "
                + "; ".join(orc["notes"])
            )
        else:
            rec = orc["recovered_fraction"]
            comp = orc.get("patch_completeness")
            comp_str = (
                f"; full-site patch completeness {comp['point']:.0%}"
                if comp else ""
            )
            lines.append(
                f"- Oracle (patching, layers {orc['layers']}): "
                f"{'FAITHFUL' if orc['faithful'] else 'UNFAITHFUL'} — direction "
                f"recovers {_fmt_ci(rec)} of the full-site effect; "
                f"direction vs random-patch {_fmt_ci(orc['direction_vs_random'])}"
                f"{comp_str}"
            )
    cs = card.diagnostics.get("causal_summary")
    if cs:
        ml_part = f", multilayer={cs['multilayer']}" if cs.get("multilayer") else ""
        orc_part = f", oracle={cs['oracle']}" if cs.get("oracle") else ""
        lines.append(
            f"- **Causal summary:** sufficiency={cs['sufficiency']}, "
            f"necessity={cs['necessity']}{ml_part}{orc_part} → {cs['combined']}"
        )
    lines += ["", "### Decision reasons", ""]
    lines += [f"- {r}" for r in card.diagnostics.get("decision_reasons", [])]

    dep = card.diagnostics.get("deployment")
    if dep:
        lines += ["", "## Deployment lens (practitioner FP/FN view)", ""]
        lines += [f"- {p}" for p in dep["plain_language"]]
        lines += [
            "",
            "| Condition | FPR budget | Recall (95% CI) |",
            "| --- | --- | --- |",
        ]
        for name, pts in dep["operating_points"].items():
            for p in pts:
                r = p["recall"]
                lines.append(
                    f"| {name} | {p['fpr_target'] * 100:.0f}% | "
                    f"{r['point'] * 100:.0f}% "
                    f"[{r['lo'] * 100:.0f}%, {r['hi'] * 100:.0f}%] |"
                )
        lines += [
            "",
            "AUROC by condition: "
            + ", ".join(f"{c['name']}={c['auroc']:.3f}" for c in dep["curves"]),
            "",
            "_ROC curves: see the `*.roc.svg` chart and `*.html` / `*.pdf` report "
            "written alongside this card._",
        ]

    lines += ["", "## Allowed claims (scope-bound; do not detach)", ""]
    lines += [f"- {c}" for c in card.allowed_claims]
    lines += ["", "## Disallowed claims", ""]
    lines += [f"- ~~{c}~~" for c in card.disallowed_claims]
    lines += ["", "## Residual risks", ""]
    lines += [f"- {r}" for r in card.residual_risks]
    profile = card.diagnostics.get("profile") or {}
    nondefault = card.diagnostics.get("config_nondefault") or {}
    lines += ["", "## Protocol config", "", f"- {_profile_line(card)}"]
    if profile.get("status") == "loosened":
        lines.append(
            "  - A loosened config voids `causally_sufficient` and positive "
            "decodability claims; only the strict bar licenses them."
        )
    if nondefault:
        lines.append("- threshold overrides vs the strict profile:")
        for k, v in sorted(nondefault.items()):
            tag = ""
            if k in profile.get("loosened", []):
                tag = "  ⚠️ LOOSER"
            elif k in profile.get("tightened", []):
                tag = "  (stricter)"
            lines.append(f"  - `{k} = {v}`{tag}")
    cfg_d = card.diagnostics.get("config") or {}
    if cfg_d:
        lines.append(
            "- full config: "
            + ", ".join(f"`{k}={v}`" for k, v in sorted(cfg_d.items()))
        )
    lines += [
        "",
        "## Reproducibility",
        "",
        f"- Protocol: v{card.protocol_version}; config hash `{card.config_hash}`; "
        f"bundle hash `{card.bundle_hash}`",
        f"- Re-run: `{card.rerun_command or 'n/a'}`",
        "",
    ]
    return "\n".join(lines)


def card_to_html(card: AuditCard) -> str:
    """A self-contained deployment report: verdict header + plain-language lens +
    operating-point table + inline ROC chart. Opens in any browser and prints to
    PDF (File → Print → Save as PDF) with no dependency."""
    verdict_str = card.label or (card.verdict.value if card.verdict else card.status)
    dep = card.diagnostics.get("deployment")
    svg = roc_svg(dep["curves"]) if dep else ""
    plain = "".join(f"<li>{p}</li>" for p in (dep["plain_language"] if dep else []))
    rows = ""
    if dep:
        for name, pts in dep["operating_points"].items():
            for p in pts:
                r = p["recall"]
                rows += (
                    f"<tr><td>{name}</td><td>{p['fpr_target'] * 100:.0f}%</td>"
                    f"<td>{r['point'] * 100:.0f}% "
                    f"[{r['lo'] * 100:.0f}%, {r['hi'] * 100:.0f}%]</td></tr>"
                )
    scope = (
        f"{card.model}"
        + (f" @ {card.revision}" if card.revision else "")
        + f" · layer(s) {card.layers} · {card.direction_source} · "
        f"{card.prompt_distribution}"
    )
    return f"""<!doctype html><meta charset="utf-8">
<title>SIEVE deployment report — {verdict_str}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 760px; margin: 2rem auto;
          color: #111; line-height: 1.5; padding: 0 1rem; }}
  h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.1rem; margin-top: 1.6rem; }}
  .verdict {{ background:#f3f4f6; border-left:4px solid #2563eb; padding:.6rem .9rem;
             border-radius:4px; font-weight:600; }}
  .scope {{ color:#555; font-size:.9rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-top:.5rem; }}
  th, td {{ border: 1px solid #ddd; padding: .35rem .6rem; text-align: left; font-size:.9rem; }}
  th {{ background:#f9fafb; }}
  .hint {{ color:#666; font-size:.85rem; }}
  @media print {{ body {{ margin: 0; }} }}
</style>
<h1>SIEVE deployment report</h1>
<p class="verdict">Verdict: {verdict_str}</p>
<p class="scope">{scope}</p>
<h2>What this means for a deployer</h2>
<ul>{plain}</ul>
<h2>Operating points</h2>
<table><tr><th>Condition</th><th>FPR budget</th><th>Recall (95% CI)</th></tr>{rows}</table>
<h2>ROC curves</h2>
{svg}
<p class="hint">Pick a threshold from the curve: lower it to miss fewer cases
(more false alarms), raise it to cut false alarms (more misses). To save as PDF,
use your browser's Print → Save as PDF.</p>
"""


def write_card(card: AuditCard, out_dir: str | Path, stem: str) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{stem}.json"
    md_path = out / f"{stem}.md"
    json_path.write_text(card_to_json(card))
    md_path.write_text(card_to_markdown(card))
    # deployment artifacts: a standalone ROC chart and a printable HTML report
    if card.diagnostics.get("deployment"):
        (out / f"{stem}.roc.svg").write_text(
            roc_svg(card.diagnostics["deployment"]["curves"])
        )
        (out / f"{stem}.html").write_text(card_to_html(card))
    return json_path, md_path
