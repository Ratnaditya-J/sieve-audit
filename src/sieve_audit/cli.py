"""SIEVE command-line interface (stubs for v0.1; engine TODO)."""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sieve",
        description="SIEVE - validity checks for AI safety signals (see DESIGN.md).",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("audit", help="run a full validity audit end-to-end (TODO)")
    sub.add_parser("steer-controls", help="run matched-control steering (TODO)")
    sub.add_parser("baselines", help="run surface baselines (length/TF-IDF/template) (TODO)")
    sub.add_parser("report", help="render an audit card from results (TODO)")
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0
    print(f"[sieve] '{args.command}' is not implemented yet - see DESIGN.md for the v0.1 plan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
