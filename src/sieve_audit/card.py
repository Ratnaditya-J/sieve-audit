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
)


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
) -> AuditCard:
    scope = scope_sentence(bundle)
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
    if decod is not None:
        diagnostics["decodability"] = decod.to_dict()
    if efficacy is not None:
        diagnostics["efficacy"] = {arm: res.to_dict() for arm, res in efficacy.items()}
    if controls is not None:
        diagnostics["controls"] = controls.to_dict()

    if decision.verdict is not None:
        allowed = [c.format(scope=scope) for c in ALLOWED_CLAIMS[decision.verdict]]
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

    risks = list(RESIDUAL_RISKS_COMMON)
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
    lines = [
        f"# SIEVE audit card — `{verdict_str}`",
        "",
        f"> **Verdict: {verdict_str}** (protocol v{card.protocol_version}, "
        f"config `{card.config_hash}`, bundle `{card.bundle_hash}`)",
        ">",
        f"> {_profile_line(card)}",
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
