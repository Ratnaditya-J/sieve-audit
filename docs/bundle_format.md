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

## `leakage` - Tier-2 span-removal check (optional)

```jsonc
{
  "labels": [0, 1, ...],
  "probe_scores_full": [0.93, 0.12, ...],          // unmodified input
  "probe_scores_leak_removed": [0.51, 0.49, ...],  // giveaway spans removed
  "probe_scores_random_removed": [0.90, 0.15, ...],// matched random spans removed (control)
  // optional named `cot` span category (the verbalizer-vs-CoT question):
  "probe_scores_cot_removed": [0.52, 0.50, ...],       // only the model's CoT removed
  "probe_scores_cot_random_removed": [0.91, 0.14, ...] // matched-length random control
}
```

`leaky` fires when AUROC collapses under leak-removal but not under the random
control. The `cot` category is adjudicated with the same rule and reported as
its own delta: `cot_leaky` means the signal was reading the CoT text
(CoT-parroting); `cot_survives` means it retains **above-chance** (CI lower
bound) discrimination without the CoT - the only regime in which the signal
tells you something the CoT does not. If `probe_scores_cot_random_removed` is
absent, the generic random control is used and the sharing is noted.

## `verbalization` - the audited signal is an activation verbalizer's claims (optional)

```jsonc
{
  "target_model": "Qwen/Qwen2.5-7B-Instruct",   // whose activations were verbalized
  "verbalizer": "patchscopes:identity@L20",     // verbalizer method + version
  "layer": 20,
  "token_selection": "last",                    // which position(s) were read
  "property_tested": "sandbagging",             // the behavioral property Y
  "texts": ["raw input prompt", ...],
  "cot_texts": ["the model's CoT per example", ...],  // the Tier-2 `cot` span
  "labels": [0, 1, ...],                        // or null-filled when label-free
  "verbalizer_claim_scores": [0.91, 0.08, ...], // P(claim asserts Y), in [0,1]
  "families": ["direct", "implicit", ...],
  "claim_scores_out_of_sample": true
}
```

Requirements the engine enforces:
- when `decodability` is also present, `verbalizer_claim_scores` must **equal**
  `probe_scores` (and the labels must match): the verdict is provably about the
  verbalizer's claims, not a substituted signal;
- claim scores lie in [0, 1] (they are scalarized claims, not raw logits).

See [`verbalizer_faithfulness.md`](verbalizer_faithfulness.md) for how the
existing gates adjudicate confabulation, CoT-redundancy, and faithfulness, and
`sieve selftest --verbalizer` for the rigged ground-truth scenarios.

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
