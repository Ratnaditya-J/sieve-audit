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
from .decodability import DecodabilityResult
from .efficacy import EfficacyResult
from .verdict import (
    ALLOWED_CLAIMS,
    DISALLOWED_CLAIMS,
    DISALLOWED_CLAIMS_ALWAYS,
    RESIDUAL_RISKS_COMMON,
    AuditCard,
    Decision,
    Verdict,
)


def _causal_summary(verdict, necessity) -> dict:
    """Cross-method agreement (#4): integrate sufficiency (steering) and
    necessity (ablation) into one read, flagging — never hiding — disagreement
    or single-method coverage. No single method's result stands in for the
    whole causal picture."""
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

    if suff == "sufficient" and nec == "necessary":
        combined = ("sufficient AND necessary — strong causal evidence under the "
                    "tested interventions")
    elif suff == "not_sufficient" and nec == "necessary":
        combined = ("necessary but NOT sufficient — distributed/partial-mechanism "
                    "signature; the signal is causally involved, not inert")
    elif suff == "sufficient" and nec == "not_necessary":
        combined = ("sufficient but NOT necessary — likely one of several redundant "
                    "pathways")
    elif suff == "not_sufficient" and nec == "not_necessary":
        combined = ("neither sufficient nor necessary under the tested interventions "
                    "— limited causal role")
    else:
        combined = (f"single-method evidence only (sufficiency={suff}, "
                    f"necessity={nec}); cross-method agreement not established")
    return {"sufficiency": suff, "necessity": nec, "combined": combined}


def _canonical_hash(obj: object) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def scope_sentence(bundle: EvidenceBundle) -> str:
    layers = ",".join(str(l) for l in bundle.layers)
    return (
        f"[model={bundle.model}"
        + (f"@{bundle.revision}" if bundle.revision else "")
        + f", layer(s)={layers}, direction={bundle.direction_source}, "
        f"prompts={bundle.prompt_distribution}, "
        f"metrics={'/'.join(bundle.behavioral_metrics)}, "
        f"single-layer additive steering]"
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
    diagnostics["causal_summary"] = _causal_summary(decision.verdict, necessity)

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

    allowed = allowed + necessity_claims

    risks = list(RESIDUAL_RISKS_COMMON)
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
    verdict_str = card.verdict.value if card.verdict else card.status
    interv = ", ".join(card.tested_interventions)
    lines = [
        f"# SIEVE audit card — `{verdict_str}`",
        "",
        f"> **Verdict: {verdict_str}**"
        + (f" — under {interv}" if interv else "")
        + f" (protocol v{card.protocol_version}, "
        f"config `{card.config_hash}`, bundle `{card.bundle_hash}`)",
        ">",
        f"> **Tested intervention(s):** {interv or '— (causal stage not run)'}"
        + (
            "  ·  causal verdicts are bounded to these; necessity (ablation) and "
            "distributed/multi-layer mechanisms were not tested"
            if interv
            else ""
        ),
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
        f"- **Prompts:** {card.prompt_distribution} "
        f"(license: {card.prompt_license}, n={card.n_prompts})",
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
        lines.append(
            f"- Dose-response: rho={ctrl['dose_rho']:.2f} (p={ctrl['dose_p']:.4f}); "
            f"judge agreement: spearman={ctrl['judge']['min_pairwise_spearman']:.2f}, kappa={ctrl['judge']['min_pairwise_kappa']:.2f}"
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
                f"(probe-ablation drop {_fmt_ci(nec['probe_drop'])}; "
                f"vs ablate_random {_fmt_ci(nec['probe_vs_random_drop'])})"
            )
    cs = card.diagnostics.get("causal_summary")
    if cs:
        lines.append(
            f"- **Causal summary:** sufficiency={cs['sufficiency']}, "
            f"necessity={cs['necessity']} → {cs['combined']}"
        )
    lines += ["", "### Decision reasons", ""]
    lines += [f"- {r}" for r in card.diagnostics.get("decision_reasons", [])]

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


def write_card(card: AuditCard, out_dir: str | Path, stem: str) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{stem}.json"
    md_path = out / f"{stem}.md"
    json_path.write_text(card_to_json(card))
    md_path.write_text(card_to_markdown(card))
    return json_path, md_path
