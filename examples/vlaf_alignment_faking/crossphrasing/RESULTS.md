# Cross-phrasing transfer test — RESULT (verified)

**Question (VLAF first-author point 1):** does the monitored-vs-unmonitored
contrastive direction encode an abstract monitoring *state*, or the surface form
of the `<is_free_tier_user>` tag? Build the direction on one phrasing, test
whether it still separates the conditions under others. Scenario body identical
across all four variants; only the oversight channel changes.

Model: allenai/OLMo-2-1124-7B-Instruct, layer 15, last-token residual.
n=500/variant (250 monitored / 250 unmonitored). Mean-diff direction (z-scored),
AUROC on test; diagonal = leave-one-family-out. Independently re-verified.

## Transfer matrix (rows=train phrasing, cols=test phrasing)
```
train\test      v0_tag  v1_para  v2_rena  v3_impl
v0_tag           1.000    0.641    0.623    0.335
v1_paraphrase    0.588    1.000    0.951    0.738
v2_renamed       0.600    0.983    1.000    0.606
v3_implicit      0.396    0.746    0.581    1.000
```

## Findings (verified)
- Within-variant decodability = **1.000** for all four phrasings: the
  monitored/unmonitored distinction is always perfectly linearly present.
- The originally-audited **bare-tag (v0) direction is phrasing-specific**: it
  transfers at/below chance to every surface form not sharing the "free-tier"
  vocabulary. Decisive-cell mean = **0.52**. Cleanest same-structure test
  (tag -> session_mode, v0->v2) = **0.61** (bootstrap CI [0.575, 0.675], excludes
  0.5; family-robust) — a weak real residual, NOT abstract state. No layer in
  8-28 and no normalization lifts the decisive mean above ~0.54. The bare-tag
  direction is geometrically near-orthogonal to all rephrasings (cosine <= 0.08).
- It is **not pure single-token lexical matching**: the paraphrase and renamed
  phrasings share a genuinely transferable direction (v1<->v2 ~0.95, cosine 0.37)
  that the tag direction is simply orthogonal to.
- v0->v3 = 0.335 is genuine anti-correlation (CI [0.291, 0.382]).

## Calibrated conclusion
On OLMo-2-7B, the contrastive monitored-vs-unmonitored direction from the bare
`<is_free_tier_user>` tag is **phrasing-bound**: perfect within its own phrasing,
at/below chance across surface forms that don't share its vocabulary. It does
**not** encode a rephrasing-invariant monitoring state, but it is not a single
token either — a different (paraphrase/renamed) phrasing pair shares a
transferable component the tag direction lacks. This supports (and sharpens) the
original "surface/tag-bound" reading via the author's own suggested test.

## Caveats
1. **v3_implicit is confounded** — it removed the policy block AND the explicit
   marker, so its collapse vs v0 (0.335) can't be attributed to implicitness
   alone (v1<->v3 transfers fine at ~0.74). The **v0<->v2 pair (~0.61)** carries
   the weight.
2. This concerns the **contrastive readout direction**, NOT the paper's
   OOD-derived steering/mitigation vector (Sorry-Bench/WMDP, no tag). That object
   is a separate audit (experiment B) and could behave differently.
