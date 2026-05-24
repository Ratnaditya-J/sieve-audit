# SIEVE

**Safety Indicator Evidence Validation Engine** — validity checks for AI safety signals.

> Does your safety signal survive controls, or is it just decodable?

SIEVE audits whether an activation-based safety **signal** (a probe direction, SAE
feature, monitor score, …) is *merely decodable* or actually *causally
load-bearing* — and emits a scoped, caveat-bound **evidence card** you can cite.
It is a **validity layer upstream of AI control**: vet a signal *before* it is
trusted in a control protocol or a public safety claim.

Its job is to **prevent overclaiming from activation evidence** — not to prove
models safe.

## Verdict, in one glance

A SIEVE audit returns one of five verdicts:

1. **not decodable** — no better than surface baselines
2. **surface-confounded** — decodable, but a length/TF-IDF baseline matches it
3. **intervention ineffective → inconclusive** — the steer never took effect (not a null!)
4. **not causally sufficient** — effective, but no more than random / orthogonal / wrong-layer controls
5. **causally sufficient** — exceeds controls, dose-responsive, judge-agreed

## Status

Pre-alpha. The design is specified in [`DESIGN.md`](DESIGN.md); the engine is in
progress. v0.1 targets contrastive residual-stream directions in open-weight
decoder LMs under single-layer additive steering.

## Install (planned)

```bash
pip install sieve-audit            # core
pip install "sieve-audit[inspect]" # + Inspect integration
```

## Data policy

This repository ships **only openly-licensed prompts**. Raw activations, judged
run outputs, and non-redistributable benchmark prompts are never committed
(see `.gitignore`).

## Provenance

SIEVE generalizes the controlled causal-sufficiency protocol from the companion
paper *"Probing Is Not Enough"*
([alignment-evals](https://github.com/Ratnaditya-J/alignment-evals/blob/main/docs/probing-is-not-enough.pdf)).

## License

MIT — see [`LICENSE`](LICENSE).
