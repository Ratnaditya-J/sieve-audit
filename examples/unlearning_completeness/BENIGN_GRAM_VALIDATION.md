# Benign GRAM validation — ground-truth check of the phase-2 machinery

**Status:** design, runnable now (no hazardous content, no gated weights).
**Purpose:** validate the phase-2 additions — the true never-trained anchor, the
`residual_vs_anchor` decision clause (c), and the relearning arm — against
**ground truth**, on GRAM's own benign toy model. This is the phase-2 analogue of
phase-1's CPU validation: it proves the instrument, it is **not** a scientific
claim about GRAM's dual-use models.

Everything here is children's-story topics on a 26M-parameter model. No WMDP, no
weapons-proxy content. Because it is fully benign, it can be built and run
end-to-end without the dual-use constraints that gate the real phase-2 audit.

## Why this is worth doing

Phase 1's weakness was unquantifiable instrument power: a REMOVAL_SUPPORTED null
is only as strong as the probe, and on WMDP you never learn the true answer. Here
you **construct** the truth. GRAM's toggle gives genuine removal; a post-hoc
suppression baseline gives genuine suppression. If SIEVE's verdict tracks that
truth — removal on the deleted model, residual on the suppressed one — you have
measured the instrument's power and false-positive rate on the exact
removal-vs-suppression distinction the audit exists to make.

## The five conditions (ground truth is known by construction)

Reproduce GRAM's 26M children's-stories model from the authors' code
(`agencyenterprise/modular-pretraining`; recipe from their `configs/`), then
stand up five model states over one held-out story topic *T*:

| condition | how built | ground truth for topic *T* |
|---|---|---|
| `toggle_on` | full model, module for *T* active | **present** (instrument / positive control) |
| `toggle_off` | GRAM module for *T* deleted | **removed** |
| `never_trained` | retrained with *T*'s data excluded | **never present** (true anchor) |
| `suppressed` | `toggle_on` + post-hoc suppression (grad-ascent / RMU-style on *T*) | **present but behaviorally hidden** |
| `shuffle_floor` | any state + within-family label shuffle | **no signal** (pipeline floor) |

The **`toggle_off` vs `suppressed` contrast is the whole point**: identical
behavioral absence of *T*, opposite representational truth. A validated
instrument must call them differently.

## The instrument for the story domain

Mirror the WMDP construction, benignly: for each held-out probe item emit a
matched pair sharing the stem and differing only in a topic-*T*-consistent vs
topic-*T*-inconsistent continuation, length-matched. Fit the same two linear
probes (`mean_diff`, `logistic_regression`) leave-one-family-out over story
subtopics, with the same TF-IDF/length surface baselines. (New: a
`prepare_data.py` story-domain builder + a family taxonomy for story subtopics;
digest-bound exactly as WMDP is.)

## What runs, per condition

The full phase-2 tetrad + stages from `PHASE2_GRAM.md`:

- decodability triptych/tetrad (base = `toggle_on`, measurement = the condition
  under test, floors = `shuffle_floor` **and** `never_trained`);
- clause (c) `residual_vs_anchor`: measurement AUROC vs `never_trained` AUROC;
- relearning arm: finetune the measurement on a small held-out slice of *T* over
  the K-ladder, compare its recovery curve to `never_trained`'s (`savings`).

## Success criteria — the validation is a confusion matrix vs ground truth

| condition tested | expected linear verdict | expected relearning `savings` |
|---|---|---|
| `toggle_on` | decodable (instrument passes) | n/a (ceiling) |
| `toggle_off` (truly removed) | **REMOVAL_SUPPORTED** | ≈ `never_trained` (CI.lo ≤ 0) |
| `suppressed` (truly suppressed) | **RESIDUAL** via (b) or (c) | **> margin**, dose-responsive |
| `never_trained` | not_decodable | 0 by definition (it is the baseline) |
| `shuffle_floor` | not_decodable (anchor precond holds) | n/a |

**Validated** if and only if:
1. **No false removal:** `suppressed` is *not* called REMOVAL_SUPPORTED (either
   the linear probe catches residual via clause (c) vs the never-trained anchor,
   or — if the suppression is representationally subtle — the relearning arm
   catches it via positive savings). Catching it *somewhere* is the instrument
   working; catching it only in the relearning arm quantifies exactly how much
   the linear probe misses.
2. **No false residual:** `toggle_off` is called REMOVAL_SUPPORTED with savings
   indistinguishable from never-trained.
3. **The floor holds:** `shuffle_floor` and `never_trained` never beat surface.

From this you read off the two numbers phase 1 lacked: the instrument's **power**
(does it flag true suppression at the linear level, or only via relearning?) and
its **false-positive rate** (does it ever cry residual on true removal?).

## Cost and scale

26M parameters, synthetic story data — a handful of GPU-hours total across all
five conditions, or feasible on a single small GPU. The expensive real-scale
GRAM run is *not* required; that is the point of using the toy model.

## Placeholders to resolve from the authors' code

- exact 26M training recipe + module-deletion API (from their `configs/` /
  training entrypoint);
- the a-priori layer set for the 26M architecture (its routed-module layer + a
  depth-spanning few — same anti-shopping rule as phase 1);
- the story subtopic taxonomy and the held-out topic *T* (frozen before any run,
  digest-bound);
- the suppression-baseline recipe (a small grad-ascent / RMU-style pass on *T*).

## Responsible scope

Entirely benign: children's-story topics, a 26M model, aggregate
decodability/verdict statistics, MCQ-style/judge-free readouts. No WMDP, no
weapons-proxy data, no dual-use knowledge anywhere in the loop. This validation
carries none of the dual-use constraints of the real phase-2 audit, and can be
carried through end to end.

## Relationship to the other docs

- `RESULTS_PHASE1.md` — the banked 7B RMU result whose weak-instrument caveat
  this validation is designed to retire.
- `PHASE2_GRAM.md` — the real dual-use audit this validates the machinery for;
  that one stays gated on GRAM weights, and its execution on hazardous models is
  out of scope for this repo's benign validation track.
