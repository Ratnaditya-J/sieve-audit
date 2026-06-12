# SIEVE evidence-bundle format (v0.1)

The bundle is the contract between *whoever can run the model* and *whoever
wants to check the claim*. An adapter (vendor script, nnsight/TransformerLens
harness, Inspect task) records what happened; SIEVE audits the record. Bundles
contain scores and flags — never raw activations — so they are small,
diffable, and hashable. Reference implementation:
[`src/sieve_audit/bundle.py`](../src/sieve_audit/bundle.py).

A bundle is one JSON object:

```jsonc
{
  "model": "Qwen/Qwen3-32B",
  "revision": "main",                  // pin it
  "layers": [55],
  "direction_source": "mean-diff (eval - deploy), z-scored, unit norm",
  "prompt_distribution": "controlled_holdout_v2 (4 families, length-matched pairs)",
  "prompt_license": "CC-BY-4.0",
  "behavioral_metrics": ["refusal", "sandbagging", "hedging"],
  "adapter": "my_adapter:0.3",
  "bundle_version": "0.1",
  "decodability": { ... },             // stage 1 evidence
  "efficacy": [ ... ],                 // stage 2 evidence
  "steering": [ ... ]                  // stage 3 evidence
}
```

## `decodability` — can the signal be read at all?

```jsonc
{
  "texts":  ["raw prompt text", ...],      // the text the activations attend over
  "labels": [0, 1, ...],                   // ground-truth condition per example
  "probe_scores": [0.93, 0.12, ...],       // the audited signal, one score per example
  "families": ["direct", "implicit", ...], // prompt-family id per example
  "probe_scores_out_of_sample": true       // attestation: no example trained the probe
}
```

Requirements the engine enforces:
- every family needs both classes (≥ `min_family_class_n` each); a near-single-class
  family silences the surface baselines and voids the audit;
- `probe_scores_out_of_sample` must be attested `true`, or any positive
  decodability finding is refused;
- ≥ 2 families gives the leave-one-family-out generalization test; 1 family
  downgrades to k-fold and records "family generalization untested".

## `efficacy` — did the intervention take effect? (one record per forward pass)

```jsonc
{
  "arm": "probe",              // also required for EVERY control arm
  "alpha": 20.0,
  "prompt_id": "p17",
  "resid_delta_norm": 19.4,    // ||h_steered - h_base|| at the intervened layer
  "resid_base_norm": 410.2,
  "expected_delta_norm": 20.0, // |alpha| * ||w||
  "output_changed": true       // any generated token differs from alpha=0
}
```

Requirements:
- include `alpha = 0` records (the no-op check) and the largest |alpha| from
  the steering grid, on the same prompts the steering stage judged
  (≥ `min_shared_efficacy_prompts` overlap);
- records for every control arm too — a control that never moved the stream
  is degenerate and voids the comparison;
- a failed probe-arm gate alongside significant behavioral deltas is rejected
  as internally inconsistent (you cannot sandbag the gate).

## `steering` — one record per judged steered generation

```jsonc
{
  "arm": "probe",              // "probe" | "random" | "orthogonal" | "wrong_layer"
  "alpha": -20.0,
  "prompt_id": "p17",
  "judge_scores": {"judge_opus": 0.8, "judge_gpt": 0.7}   // each in [0,1]
}
```

Requirements:
- all of `random`, `orthogonal`, `wrong_layer` control arms (matched: same
  norm, same layer except wrong_layer, same prompts, same alphas);
- a sign-symmetric alpha grid including 0 and ≥ 2 nonzero magnitudes
  (dose-response needs ≥ 3 distinct alphas; the primary test points are
  ±max);
- ≥ `min_steered_prompts` prompts per arm per alpha (default 20);
- ≥ 2 *independent* judges on every record — identical score vectors are
  flagged as duplication;
- no duplicate `(arm, alpha, prompt_id)` records (retakes are rejected).

## What the engine returns

`sieve audit --bundle bundle.json` emits an audit card (markdown + JSON) with
one of five verdicts — `not_decodable`, `surface_confounded`,
`intervention_ineffective`, `not_causally_sufficient`,
`causally_sufficient` — or refuses with `insufficient_protocol`, listing
exactly which requirement above was not met. Claims on the card are bound to
the scope, the full config (deviations from defaults are bannered), and
SHA-256 hashes of both config and bundle.

Partial bundles are legitimate: a decodability-only bundle (like the
[Apollo probe audit](../examples/apollo_deception/README.md)) yields the
decodability diagnostics, licenses only correlational claims, and documents
the refusal of causal ones.
