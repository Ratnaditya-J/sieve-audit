# SIEVE audit card — `causally_sufficient`

> **Verdict: causally_sufficient** (protocol v0.1, config `be7d2a694c14992e`, bundle `650cdaef8f894050`)

## Scope (what was actually tested)

- **Model:** synthetic/ground-truth-causally-sufficient @ v1
- **Layer(s):** [7]
- **Direction:** rigged contrastive direction (synthetic)
- **Prompts:** synthetic-families-v1 (license: MIT (generated), n=240)
- **Alpha grid:** [-20.0, -10.0, 0.0, 10.0, 20.0]
- **Behavioral metric(s):** eval_behavior_score
- **Judges:** judge_a, judge_b
- **Steering arms:** orthogonal, probe, random, wrong_layer
- **Seed:** 0

## Diagnostics

- Probe AUROC: **0.956 [0.932, 0.975]** (leave-one-family-out, n=240, 4 families)
- Surface baseline `length`: AUROC 0.000; probe − baseline = 0.956 [0.932, 0.975]
- Surface baseline `tfidf`: AUROC 0.003; probe − baseline = 0.953 [0.929, 0.974]
- Efficacy gate: **passed** (hook_correct=True, median rel. residual delta @|α|=20: 0.1984, output changed: True)
- Dose-response: rho=0.97 (p=0.0000); judge agreement: spearman=0.68, kappa=1.00
- |probe| − |random| @α=-20.0: 0.325 [0.306, 0.346]
- |probe| − |orthogonal| @α=-20.0: 0.340 [0.318, 0.358]
- |probe| − |wrong_layer| @α=-20.0: 0.349 [0.331, 0.364]
- |probe| − |random| @α=20.0: 0.321 [0.301, 0.341]
- |probe| − |orthogonal| @α=20.0: 0.323 [0.306, 0.341]
- |probe| − |wrong_layer| @α=20.0: 0.311 [0.292, 0.329]

### Decision reasons

- probe effect exceeds all matched controls, dose-responsive, judge-agreed

## Allowed claims (scope-bound; do not detach)

- Under [model=synthetic/ground-truth-causally-sufficient@v1, layer(s)=7, direction=rigged contrastive direction (synthetic), prompts=synthetic-families-v1, metrics=eval_behavior_score, single-layer additive steering], steering along the signal changed behavior more than matched random/orthogonal/wrong-layer controls, with monotone dose-response and judge agreement.
- Under [model=synthetic/ground-truth-causally-sufficient@v1, layer(s)=7, direction=rigged contrastive direction (synthetic), prompts=synthetic-families-v1, metrics=eval_behavior_score, single-layer additive steering], the signal is causally sufficient for the measured behavioral metric(s).

## Disallowed claims

- ~~The signal is causally necessary (ablation/necessity was not tested).~~
- ~~The signal is causally sufficient outside the audited scope (other layers, prompts, models, alphas).~~
- ~~The model is safe / not deceptive / not eval-aware.~~
- ~~This signal is a reliable deployment monitor without further validation.~~
- ~~This audit certifies anything outside its scope block.~~

## Residual risks

- Single-layer additive steering only; distributed/multi-layer mechanisms untested.
- Sufficiency-style evidence only; necessity (ablation) untested.
- Results are specific to the audited prompt distribution and may not transfer.
- Behavioral metrics depend on judge quality; judge agreement is reported, not guaranteed.

## Reproducibility

- Protocol: v0.1; config hash `be7d2a694c14992e`; bundle hash `650cdaef8f894050`
- Re-run: `n/a`
