# Auditing the Apollo deception probes (arXiv 2502.03407)

> **Note on config hash:** the audit cards in `reports/` were generated with
> an earlier `sieve-audit` version.  The strict profile config hash has since
> changed (a `min_random_controls` field was added to `AuditConfig`, shifting
> the hash even though the value is the default 1).  The **verdict is
> unchanged** (`insufficient_protocol` for both probes — SIEVE correctly
> refuses a causal verdict when no steering evidence exists).  Re-run the
> commands below to produce cards with the current hash.

The first real-world SIEVE audit: the two published Apollo Research deception
probes for Llama-3.3-70B-Instruct layer 22 —

- **pairs_probe** (trained on instructed pairs)
- **rp_probe** (trained on roleplaying)

— audited from *released artifacts only* (no GPU, no model inference), using
the per-token probe scores shipped in Parrack et al.'s
[black_to_white_boosts](https://github.com/carlo-leonardo-attubato/black_to_white_boosts)
benchmark dumps (arXiv 2507.12691), which embed the Apollo probes' scores on
five evaluation datasets.

## What this audit can and cannot test

| SIEVE stage | Runnable from released artifacts? |
|---|---|
| Decodability (probe vs chance) | **yes** |
| Surface baselines (length / TF-IDF must-beat) | **yes** |
| Efficacy gate | no — no steering evidence was ever published |
| Matched-control steering | no — same |
| Causal-sufficiency verdict | **refused: insufficient protocol** |

The refusal is the point: **no published deception-probe release ships the
evidence a causal verdict requires.** The audit cards in
[`../../reports/`](../../reports/) record exactly how far the published
evidence goes and which claims it licenses.

## Reproduce

```bash
# 1. download the dumps (~1 GB; not redistributed here - no license on either repo)
mkdir -p /tmp/sieve-targets && cd /tmp/sieve-targets
for f in ai_liar apps_backdoors sandbagging_wmdp_mmlu insider_trading_full; do
  curl -L -O "https://media.githubusercontent.com/media/carlo-leonardo-attubato/black_to_white_boosts/master/results/$f.jsonl"
done

# 2. build evidence bundles (one per probe)
cd <sieve-audit>
python -m sieve_audit.adapters.parrack_b2w /tmp/sieve-targets/{ai_liar,apps_backdoors,sandbagging_wmdp_mmlu,insider_trading_full}.jsonl \
    --probe pairs_probe --out /tmp/sieve-targets/bundle_pairs_probe.json
python -m sieve_audit.adapters.parrack_b2w /tmp/sieve-targets/{ai_liar,apps_backdoors,sandbagging_wmdp_mmlu,insider_trading_full}.jsonl \
    --probe rp_probe --out /tmp/sieve-targets/bundle_rp_probe.json

# 3. audit
sieve audit --bundle /tmp/sieve-targets/bundle_pairs_probe.json --name apollo_pairs_probe
sieve audit --bundle /tmp/sieve-targets/bundle_rp_probe.json --name apollo_rp_probe
```

## Expected output

Both audits return **`insufficient_protocol`** — SIEVE refuses a causal verdict
because no steering/control evidence exists in the release. What it *can*
certify (the decodability stage) it does:

| Probe | Decodable (pooled, held-out) | Best surface baseline | Probe − baseline | Verdict |
|---|---|---|---|---|
| `pairs_probe` | **0.798** [0.780, 0.815] | 0.462 | **+0.336** | `insufficient_protocol` |
| `rp_probe` | **0.723** [0.702, 0.743] | 0.462 | **+0.261** | `insufficient_protocol` |

So the decodability claims survive a fair cross-dataset surface baseline (the
probes clear it decisively), but the *causal* reading the probes are sometimes
assumed to license is **not evaluable from what was published**. Full cards:
[`reports/apollo_pairs_probe.md`](../../reports/apollo_pairs_probe.md),
[`reports/apollo_rp_probe.md`](../../reports/apollo_rp_probe.md); narrative in
[`reports/apollo_audit_summary.md`](../../reports/apollo_audit_summary.md).

## Method notes

- Probe score per example: mean per-token score over the assistant's content
  tokens (Apollo's aggregation, on the response).
- Surface-baseline input: the full rendered conversation — the same context
  the probe's activations attend over. If a TF-IDF reader of the dialogue
  matches the probe, the *evaluation* cannot distinguish "reads deception from
  activations" from "reads the scenario from the prompt" (the
  Smith/Chughtai/Nanda critique, arXiv 2511.22662, made operational).
- Families = source datasets, so the leave-one-family-out baseline split
  tests cross-dataset generalization — the probes' own headline claim. The
  family id comes from each sample's `metadata.dataset`, so the downloaded
  `insider_trading_full.jsonl` yields the family **`insider_trading__upscale`**
  (the id recorded in the published cards); `reports/apollo_audit_summary.md`
  shortens it to `insider_trading` in prose. Same dataset, three surface names.
- The probes' training sets (instructed-pairs, roleplaying) are disjoint from
  all four evaluation dumps used here.
