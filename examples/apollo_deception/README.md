# Auditing the Apollo deception probes (arXiv 2502.03407)

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

## Method notes

- Probe score per example: mean per-token score over the assistant's content
  tokens (Apollo's aggregation, on the response).
- Surface-baseline input: the full rendered conversation — the same context
  the probe's activations attend over. If a TF-IDF reader of the dialogue
  matches the probe, the *evaluation* cannot distinguish "reads deception from
  activations" from "reads the scenario from the prompt" (the
  Smith/Chughtai/Nanda critique, arXiv 2511.22662, made operational).
- Families = source datasets, so the leave-one-family-out baseline split
  tests cross-dataset generalization — the probes' own headline claim.
- The probes' training sets (instructed-pairs, roleplaying) are disjoint from
  all five evaluation dumps.
