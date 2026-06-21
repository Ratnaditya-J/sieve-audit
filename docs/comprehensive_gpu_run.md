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

| Axis | Evidence section | Runner subcommand |
| --- | --- | --- |
| Decodability + TF-IDF leakage | `decodability` | `decode-lofo` |
| Efficacy + sufficiency | `efficacy`, `steering` | `steer` / `judge` |
| Necessity (single-layer) | `ablation` | `ablate` / `judge` |
| Tier-2 leakage | `leakage` | `leakage` |
| Multi-layer | `multilayer` | `multilayer` / `judge` |
| Oracle (patching) | `patching` | `patching` / `judge` |
| Off-distribution | `deployment` | `decode` (on ood prompts) |

All eight axes are implemented in `adapters/hf_steering_runner.py`; `pod_run.sh`
exercises every subcommand end-to-end.

## Adapter work — DONE

All eight subcommands are implemented.  The `bundle` command accepts all axes
via flags (`--judged-ablation`, `--judged-multilayer`, `--judged-patching`,
`--leakage`, `--decode-ood`) and folds them into a single `bundle_all.json`.

## Run — DONE (2026-06-21)

A40 48 GB pod (Stockholm), ~44 min, ~$0.32.  All eight subcommands ran via
`pod_run.sh`; bundle downloaded and pod terminated.

```bash
# reproduce locally (bundle already checked in):
sieve audit --bundle examples/comprehensive_refusal/bundle_all.json \
            --out examples/comprehensive_refusal/reports/ --name comprehensive
```

Acceptance criteria met: all axes conclusive or honestly inconclusive; no
`insufficient_protocol`; multi-draw null (3 random controls) embedded in bundle.

## Next: harder target

The refusal direction on a 1.5B model is surface-confounded — the verdict
`surface_confounded · necessary` is an honest finding, not a failure.  To
prove the methodology in the *positive* direction (causal verdict that is NOT
surface-confounded), the natural next run is a subtler behavior on a larger
model — deception, sandbagging, or a non-obvious monitor setting — where the
probe beats surface baselines and the full causal picture is interpretable.

Candidate targets:
- Llama-3.3-70B-Instruct, deception direction (Apollo probes provide a
  starting point; build a direction via contrastive pairs, then run the full
  pipeline with steering evidence)
- A sandbagging direction on a capability eval (Gemma-2-27B or similar)
- A refusal direction on a model fine-tuned to be robustly non-surface-confounded
