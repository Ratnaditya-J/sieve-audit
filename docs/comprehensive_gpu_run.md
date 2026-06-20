# Comprehensive GPU run — plan + result

> **DONE (2026-06-20).** Executed on Qwen2.5-1.5B-Instruct (A40, ~$0.54).
> Verdict **`surface_confounded · necessary`** — all eight axes captured, no
> insufficient_protocol. Worked example, bundle and reports:
> [examples/comprehensive_refusal/](../examples/comprehensive_refusal/). The
> runner gaps below were closed; the adapter now emits every axis. The oracle
> came back honestly *inconclusive* (single-site patching did not move the
> model's refusal — no oracle effect to calibrate against).

---

## Original plan (resume reference)

Goal: **one** real model run that exercises **every** SIEVE axis end-to-end and
produces a single audit card with **no incomplete / insufficient verdict** —
decodability, both leakage tiers, efficacy, sufficiency (steering + matched
controls), necessity (single-layer ablation), multi-layer (joint) ablation,
oracle (activation patching), and the deployment lens.

Status at pause (2026-06-20): the GPU-free **core + adjudication** for all four
new axes is shipped, tested (123 passing), and pushed on
`sieve-validity-bounding`. What remains is **GPU-adapter work** so the runner
emits evidence for the new axes, then the run itself.

## Target

- Model: `Qwen/Qwen2.5-1.5B-Instruct` (same family as the refusal_necessity demo;
  small enough for a single A100, big enough to be non-trivial).
- Behavior: refusal (contrastive harmful/benign prompts; AdvBench harmful set via
  the public `llm-attacks` CSV — walledai/AdvBench is gated).
- Direction: contrastive mean-difference probe at a mid layer; joint set for the
  multi-layer axis = {mid-2, mid, mid+2}.
- Judges: OpenRouter `openai/gpt-4o-mini` + `openai/gpt-4.1-mini` (2 finite judges
  required; gemini route 404'd last run — keep 3 configured for resilience).

## Runner status vs. what each axis needs

Existing subcommands (`adapters/hf_steering_runner.py`): `vectors`, `decode`,
`decode-lofo`, `steer`, `judge`, `bundle`, `ablate`, `bundle-ablation`.

| Axis | Evidence section | Runner today | To add |
| --- | --- | --- | --- |
| Decodability + TF-IDF leakage | `decodability` | ✅ `decode` | — |
| Efficacy + sufficiency | `efficacy`, `steering` | ✅ `steer`/`judge` | — |
| Necessity (single-layer) | `ablation` | ✅ `ablate` | — |
| **Tier-2 leakage** | `leakage` | ❌ | re-score holdout with leak spans removed (elicitation prompt + verbalized reasoning) **and** a random-span-removal control → `probe_scores_full/leak_removed/random_removed` |
| **Multi-layer** | `multilayer` | ❌ | ablation hook applied at the **joint** layer set at once; arms baseline/probe/ablate_random; judged |
| **Oracle (patching)** | `patching` | ❌ | run clean + corrupt; patch full residual at site (oracle); patch only the direction's component; patch a random direction's component; judged |
| **Off-distribution** | `deployment` | ❌ | score the probe on a second prompt mix (e.g. a different harmful set) → labels + probe_scores |

All new evidence shapes are additive JSON; the core already validates and
adjudicates them (`leakage.py`, `multilayer.py`, `oracle.py`, `deployment.py`).

## Adapter work (local, no GPU) — do first

1. `cmd_leakage`: reuse `decode`'s scoring path twice on the holdout — once with
   leak spans stripped, once with equal-length random spans stripped — emit
   `LeakageEvidence`.
2. `cmd_multilayer`: generalize `AblationHook` to a list of layers; emit
   `MultiLayerRecord`s with `layers=[...]`.
3. `cmd_patching`: add a patch hook that caches clean-run residuals and writes
   them into the corrupt run (full / direction-projected / random-projected);
   emit `PatchingRecord`s.
4. `cmd_deploy_decode`: `decode` on a second prompt set → `DeploymentEvidence`.
5. Extend `bundle`/`bundle-ablation` (or a new `bundle-all`) to fold every
   section into one bundle.

## Run (GPU) — after adapter work, on the user's green light

Cost envelope: A100 community + public IP, ~1–1.5h ≈ **$2–4**. Follow the
GPU-capture-then-analyze-local rule: on the pod do only the forward passes +
judging, `scp` the bundle down, terminate the pod, run `sieve audit` locally.

```
# on pod: vectors → decode → steer → judge → ablate → multilayer → patching
#         → leakage → deploy-decode → bundle-all  (writes bundle_all.json)
# local:
sieve audit --bundle bundle_all.json --pdf --name comprehensive
```

Acceptance: the card's verdict is a real causal verdict (not
`insufficient_protocol`), the causal summary reports all of
sufficiency/necessity/multilayer/oracle, the deployment lens has ROC curves for
in-distribution (+ off-distribution + leakage-removed), and every axis shows
either a conclusive result or an *honest* inconclusive with its reason — no
silent gaps.

## Teardown safety

`runpodctl` with `ssh info` → use `ip`/`port` (NOT `host`). Trap-based teardown +
`--terminate-after` backstop; confirm `runpodctl get pod` shows none running and
check spend before declaring done.
