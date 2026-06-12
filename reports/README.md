# Reports

Generated audit cards land here.

- [`apollo_audit_summary.md`](apollo_audit_summary.md) — the first real-world
  audit: both published Apollo deception probes (arXiv 2502.03407), from
  released artifacts only.
- `apollo_pairs_probe.{md,json}`, `apollo_rp_probe.{md,json}` — the full cards.
- `selftest_*.{md,json}` — cards for the six rigged ground-truth scenarios
  (`sieve selftest --out reports/selftest`).

Planned: the Qwen3-32B L55 steering demo (`qwen3_l55_demo_report.md`) — the
canonical full-protocol worked example referenced in the README and DESIGN
(requires GPU time).

Audit cards contain aggregate results only (no prompt text), consistent with the
data policy in the root `.gitignore`.
