"""SIEVE command-line interface.

- ``sieve audit --bundle bundle.json``: full audit, card to reports/
- ``sieve selftest``: run the six rigged ground-truth scenarios and verify the
  engine returns exactly the rigged verdicts (SIEVE auditing itself)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .bundle import EvidenceBundle
from .calibration import run_calibration
from .card import write_card
from .config import AuditConfig
from .engine import run_audit
from .synth import SCENARIOS
from .verdict import INSUFFICIENT_PROTOCOL


def _cmd_audit(args: argparse.Namespace) -> int:
    cfg = AuditConfig(seed=args.seed)
    bundle = EvidenceBundle.load(args.bundle)
    prereg = None
    if args.prereg:
        from .prereg import PreRegistration

        prereg = PreRegistration.load(args.prereg)
    result = run_audit(bundle, cfg, bundle_path=str(args.bundle), prereg=prereg)
    stem = args.name or Path(args.bundle).stem
    json_path, md_path = write_card(result.card, args.out, stem)
    verdict = result.card.label or (
        result.card.verdict.value if result.card.verdict else result.card.status
    )
    print(f"[sieve] verdict: {verdict}")
    print(f"[sieve] card: {md_path} (+ {json_path.name})")
    if result.card.diagnostics.get("deployment"):
        print(f"[sieve] deployment report: {Path(args.out) / (stem + '.html')} "
              f"(+ {stem}.roc.svg)")
        if args.pdf:
            from .report import write_pdf

            pdf_path = write_pdf(result.card, Path(args.out) / f"{stem}.pdf")
            print(f"[sieve] deployment PDF: {pdf_path}")
    pre = result.card.preregistration
    if pre is not None:
        print(
            f"[sieve] pre-registration: {'MATCHES' if pre['matches'] else 'MISMATCH'} "
            f"({pre['declared_hash'][:16]})"
        )
        for d in pre["diffs"]:
            print(f"  - {d}")
    if result.card.status == INSUFFICIENT_PROTOCOL:
        print("[sieve] protocol incomplete - no causal verdict was issued:")
        for r in result.card.diagnostics["decision_reasons"]:
            print(f"  - {r}")
    return 0


def _cmd_prereg(args: argparse.Namespace) -> int:
    from .prereg import build_prereg

    cfg = AuditConfig(seed=args.seed)
    bundle = EvidenceBundle.load(args.bundle)
    prereg = build_prereg(bundle, cfg, note=args.note)
    prereg.save(args.out)
    print(f"[sieve] pre-registration hash: {prereg.prereg_hash}")
    print(f"[sieve] committed config + scope -> {args.out}")
    print("[sieve] publish this hash before running; audit with "
          "`sieve audit --bundle <b> --prereg " + str(args.out) + "`")
    return 0


def _cmd_selftest(args: argparse.Namespace) -> int:
    cfg = AuditConfig(seed=args.seed)
    failures = []
    for expected, make in SCENARIOS.items():
        bundle = make(seed=args.seed)
        result = run_audit(bundle, cfg)
        got = (
            result.card.verdict.value
            if result.card.verdict
            else result.card.status
        )
        ok = got == expected
        print(f"[sieve] {'PASS' if ok else 'FAIL'}  {expected:28s} -> {got}")
        if not ok:
            failures.append((expected, got))
        if args.out:
            write_card(result.card, args.out, f"selftest_{expected}")
    if failures:
        print(f"[sieve] selftest FAILED: {failures}")
        return 1
    print("[sieve] selftest passed: 6/6 rigged scenarios returned the rigged verdict")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    cfg = AuditConfig(seed=args.seed)
    rep = run_calibration(cfg=cfg)
    print(
        f"[sieve] calibration on {rep.n} ground-truth cases: "
        f"accuracy {rep.accuracy:.0%} ({rep.n_correct}/{rep.n}); "
        f"false negatives (truly-causal -> not_causally_sufficient): "
        f"{rep.false_negatives}"
    )
    for expected, got in rep.confusion.items():
        print(f"  expected {expected:28s} -> {got}")
    for m in rep.mismatches:
        print(f"  MISMATCH {m['name']}: expected {m['expected']} got {m['got']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sieve",
        description="SIEVE - validity checks for AI safety signals (see DESIGN.md).",
    )
    sub = parser.add_subparsers(dest="command")

    p_audit = sub.add_parser("audit", help="run a full validity audit on an evidence bundle")
    p_audit.add_argument("--bundle", required=True, help="path to an evidence bundle JSON")
    p_audit.add_argument("--out", default="reports", help="output directory for the audit card")
    p_audit.add_argument("--name", default=None, help="card filename stem (default: bundle stem)")
    p_audit.add_argument("--seed", type=int, default=0)
    p_audit.add_argument(
        "--prereg",
        default=None,
        help="verify the run against a pre-registration JSON (from `sieve prereg`)",
    )
    p_audit.add_argument(
        "--pdf",
        action="store_true",
        help="also render a PDF deployment report (needs matplotlib)",
    )
    p_audit.set_defaults(func=_cmd_audit)

    p_pre = sub.add_parser(
        "prereg",
        help="freeze config + scope to a hash BEFORE results (anti-retrofitting)",
    )
    p_pre.add_argument("--bundle", required=True, help="a scope bundle (results may be empty)")
    p_pre.add_argument("--out", required=True, help="output pre-registration JSON")
    p_pre.add_argument("--note", default=None, help="optional free-text note")
    p_pre.add_argument("--seed", type=int, default=0)
    p_pre.set_defaults(func=_cmd_prereg)

    p_self = sub.add_parser(
        "selftest", help="audit six rigged ground-truth scenarios; verdicts must match"
    )
    p_self.add_argument("--out", default=None, help="also write the six audit cards here")
    p_self.add_argument("--seed", type=int, default=0)
    p_self.set_defaults(func=_cmd_selftest)

    p_cal = sub.add_parser(
        "calibrate",
        help="report the verdict's error rate vs ground-truth cases (#3)",
    )
    p_cal.add_argument("--seed", type=int, default=0)
    p_cal.set_defaults(func=_cmd_calibrate)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
