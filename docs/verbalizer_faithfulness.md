# Auditing activation verbalizers with SIEVE

**Question:** when does verbalizing a model's internal activations into English
tell you something you could not get by reading its chain-of-thought (CoT)?

Activation verbalizers (Patchscopes, LatentQA, activation-explainer setups)
translate a hidden state *h* at (layer L, token position t) into a
natural-language claim about what the model is "thinking". The central validity
threat is **text inversion / confabulation** (arXiv 2509.13316): the verbalizer
reproduces its own parametric priors, or paraphrases the input text or the CoT,
rather than reading the target model's internals. A verbalization that restates
the prompt or the CoT is worthless as a safety signal - you already had both.

SIEVE's verdict contract is signal-agnostic (DESIGN.md §1), so verbalizer
faithfulness reduces to a standard audit with one adapter
([`adapters/verbalizer.py`](../src/sieve_audit/adapters/verbalizer.py)): each
claim is scalarized into `P(claim asserts Y)` in [0, 1] (two-scorer discipline,
mirroring the two-judge rule) and fed through the gates SIEVE already has.

## The mapping (reuse, don't reinvent)

| Failure mode | Existing SIEVE gate | Verdict / flag |
| --- | --- | --- |
| Text inversion: claims predictable from the raw prompt text | decodability vs surface baselines (`baselines.py`) | `surface_confounded` |
| Nothing decodable at all | decodability vs chance | `not_decodable` |
| **CoT-parroting**: claims restate the model's verbalized reasoning | Tier-2 leakage, named `cot` span category (`leakage.py`) | `cot-leaky` on the card |
| **Claims read what the CoT hides** (the payoff regime) | same gate, survival side | `survives-cot-removal` (requires above-chance AUROC without the CoT, CI lower bound) |
| Claims decodable but the named direction is epiphenomenal | matched-control steering suite | `not_causally_sufficient` |
| Claims track a causally load-bearing property | full protocol | `causally_sufficient` |

**The CoT removal *is* the CoT-comparison.** The Tier-2 gate re-scores the same
examples with the CoT stripped, against a *matched-length random-removal
control* (removing spans pushes the input off-distribution; the control
separates "removed the CoT" from "removed tokens"). If the claim signal
collapses under CoT removal but not under the control, the verbalizer was
reading the CoT - it adds nothing over the transcript. If it survives (and the
post-removal AUROC's CI lower bound still beats chance - survival is earned,
never defaulted), the verbalizer is reading something the CoT does not expose.
That surviving region is where verbalization earns its keep.

## The causal stage and its caveat

Decodability + leakage catch confabulation *correlationally*. The causal stage
needs a direction, and a verbalizer does not hand you one. The adapter's honest
default (`recover_claim_direction`) is the **mean-difference of the target
model's activations at (layer, position), grouped by the verbalizer's claim**
(asserts-Y vs not), z-scored, unit-norm - "the direction that predicts what the
verbalizer says" - fed as the probe arm to the existing
random/orthogonal/wrong-layer steering controls.

Printed on every card that carries steering evidence for a verbalizer:

> The steered direction was recovered from the verbalizer's CLAIMS, not from
> the verbalizer's mechanism. A `not_causally_sufficient` verdict is bounded to
> that recovery method and the tested intervention (single-layer additive
> steering); it does **not** prove the verbalizer confabulated -
> distributed/multi-layer mechanisms are untested.

The stronger alternative for a located-read verbalizer is activation patching:
SIEVE's `PatchingRecord` arms (`clean|corrupt|patch_full|patch_direction|
patch_random`) apply unchanged - patch the verbalizer-named site and ask
whether the claimed behavior appears. The adapter's vectors are compatible with
the HF runner's patching stage.

## Anti-detachment

When a bundle carries `verbalization` evidence alongside `decodability`, the
engine enforces that the decodability `probe_scores` **are** the
`verbalizer_claim_scores` (and the labels match). A verdict can therefore never
be earned by one signal and quoted for the verbalizer.

## Selftest

`sieve selftest --verbalizer` audits four rigged scenarios whose correct
outcome is known by construction (`synth.VERBALIZER_SCENARIOS`), pinning the
verdict **and** the `cot` gate flags:

1. **faithful** → `causally_sufficient`, `survives-cot-removal`;
2. **confabulating** (claims from prompt text alone) → `surface_confounded`;
3. **CoT-parroting** (claims track the CoT, not the internals) → `cot-leaky`
   fires, the random control does not;
4. **decodable-but-epiphenomenal** → `not_causally_sufficient`, CoT gate clean.

`tests/test_verbalizer.py` asserts all four across seeds.

## Running it

```bash
# GPU-free, from recorded verbalizer outputs (claim records JSONL):
python -m sieve_audit.adapters.verbalizer bundle \
  --records claims.jsonl --model <target> --verbalizer patchscopes:identity@L20 \
  --layer 20 --property sandbagging \
  --prompt-distribution my_eval_v1 --prompt-license CC-BY-4.0 \
  --attest-out-of-sample --out bundle.json
sieve audit --bundle bundle.json

# with model access ([runner] extras): verbalize -> vectors -> steer/judge -> bundle
python -m sieve_audit.adapters.verbalizer verbalize --model <target> \
  --prompts prompts.jsonl --layer 20 --out claims.jsonl --activations acts.npz
python -m sieve_audit.adapters.verbalizer vectors --records claims.jsonl \
  --activations acts.npz --property sandbagging --out vectors.npz
# then hf_steering_runner steer/judge with vectors.npz, and `bundle --steer --judged`
```

Activations stay local (data policy: scores and flags only, never raw
activations).
