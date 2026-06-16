# SIEVE — Safety Indicator Evidence Validation Engine

**Status:** v0.1 design (pre-alpha).
**Subtitle:** Validity checks for AI safety signals — does a signal survive controls, or is it just decodable?

---

## 1. What SIEVE is

A reproducible audit that tests whether an activation-based (or other) **safety signal** is merely *decodable* or actually *causally load-bearing*, and emits a **scoped, caveat-bound evidence card** you can cite (e.g., in a system card).

SIEVE is a **validity layer upstream of AI control**: vet a signal/monitor *before* it is trusted inside a control protocol or a public safety claim. It is **not** a control protocol, **not** an eval-awareness benchmark, and **not** a safety certification.

**Why an independent tool.** The value is neutrality. A tool whose main output is often *"this signal is decodable but not causally valid"* is one that signal-vendors and labs are structurally disinclined to build and publicize. SIEVE is the third-party validity check; its brand is **preventing overclaiming from activation evidence** — not proving models safe.

A "safety indicator" is anything that claims to tell you something safety-relevant about a model: an activation probe, an SAE feature, an NLA/verbalizer output, a reasoning-trace signal, a behavioral-eval signal, or an AI-control monitor score. v0.1 implements the activation-probe case; the verdict contract is signal-agnostic.

## 2. Scope (the honest boundary)

> **v0.1:** For contrastive residual-stream directions in open-weight decoder LMs, under single-layer additive steering, SIEVE tests whether linear decodability survives surface baselines and matched steering controls, and reports a scoped causal-sufficiency verdict.

Anything outside that sentence is out of scope for v0.1 (see §8).

## 3. Verdict taxonomy (the intellectual core)

1. **not_decodable** — probe ≈ length/TF-IDF baseline.
2. **surface_confounded** — probe ≈ surface baselines on held-out families.
3. **intervention_ineffective** — steering did not perceptibly move the residual stream/outputs (magnitude/quantization). **Inconclusive on causality, NOT a null.** *(e.g., gpt-oss-120b L34.)*
4. **not_causally_sufficient** — intervention effective, but effect ≤ random / orthogonal / wrong-layer controls. *(e.g., Qwen3-32B L55, gpt-oss-120b L15.)*
5. **causally_sufficient** — exceeds all controls, dose-responsive, and judge-agreed.

## 4. Pipeline — own vs. wrap

**Own (the differentiated spine):**
- the **efficacy gate** (intervention-correctness + "did anything move?"),
- **matched-control steering** (probe / random / orthogonal / wrong-layer),
- **surface-baseline** checks (length, TF-IDF, template),
- **two-judge** behavioral scoring + inter-rater agreement,
- the **verdict logic**, and
- the **audit-card generator**.

**Wrap (do not rebuild):** activation extraction & probe training (nnsight / TransformerLens / repeng / Inspect), and the inference stack.

## 5. Efficacy gate (correctness + anti-gaming)

Before any causal verdict, verify the intervention actually took effect:
- hook-correctness checks (α=0 is a no-op; α≠0 changes the residual stream by ≈ α·w at prefill positions; hook is removed after);
- an **efficacy check**: did the residual stream move, and did *any* output change at the tested α?

If the intervention did not take effect, the verdict is **intervention_ineffective (inconclusive)** — never "no causal effect." This is both a correctness requirement (the gpt-oss L34 lesson) and the cheapest anti-gaming defense (you cannot claim "no effect → safe" by steering where steering cannot bite). On open weights, run a layer sweep so a dead layer cannot be cherry-picked.

## 6. Audit card (claim calibration)

Every audit emits one inseparable record:

- **Scope block:** model + revision, layer(s), direction & derivation, prompt distribution + source + license + n, α grid, behavioral metric(s), judges + agreement, controls run, seed.
- **Diagnostics:** probe AUROC vs baselines; surface-confound result; efficacy-gate result; per-control steering deltas ± CI; dose-response; judge agreement.
- **Verdict:** one of the five (§3).
- **Allowed claims** (templated, scope-bound), e.g.: "the signal is linearly decodable"; "generalizes across held-out families"; "did not pass causal-sufficiency controls under [scope]"; "treat as diagnostic, not a validated monitor."
- **Disallowed claims** (printed explicitly): "the model is safe / not eval-aware / not deceptive"; "this probe is a reliable monitor without further validation."
- **Residual risks** (printed): single layer; sufficiency only, not necessity; prompt-family X only; distributed/multi-layer mechanisms untested; closed-model mitigations not independently verified.
- **Reproducibility:** protocol version + **config hash** + exact re-run command.

## 7. Anti-safetywashing guarantees

The risk: an anti-overclaiming tool can be turned into a laundering tool. Defenses, baked in:
1. **Claims are non-detachable from scope + caveats.** The output is a versioned, config-hashed card, not a quotable sentence. A system card cites the hash; re-running the hash reproduces the result.
2. **No causal verdict without the full protocol** — controls present, efficacy gate passed, adequate n, ≥2 judges with agreement reported. Otherwise: *inconclusive / insufficient protocol.*
3. **Efficacy gate blocks "dead-layer → safe"** (§5); layer sweep on open weights; report the distribution and all layers/configs tested.
4. **Honesty about closed models:** on closed weights the auditee self-runs SIEVE, so it provides a standardized, honest, reproducible *format* — not independent audit. True independence requires open weights or an evaluator with white-box access. (This is precisely the argument for deeper white-box access for evaluators.)
5. **The bar is a frozen, named profile, and loosening is a hard gate — not a knob.** The defaults *are* `SIEVE-v0.1-strict`; "passed SIEVE" means "passed that exact object," identified by hash. Thresholds remain configurable for research, but the verdict logic is asymmetric: you may only make the bar *stricter* and keep a positive verdict. Loosening any threshold — or changing a knob whose "stricter" direction is ambiguous — **voids** `causally_sufficient` and any positive-decodability claim, downgrading them to *insufficient protocol*; it never voids a verdict that went against the probe (same asymmetry as §7.2). The card states the profile as a binary an outsider can read (`strict` / `stricter` / `CUSTOM — LOOSENED`), so a weakened bar can neither masquerade as the standard nor quietly buy a stronger verdict. This is what makes SIEVE a *standard* rather than a parameterized script: the bar is a specific frozen artifact, and configurability is walled off from the headline claim.

## 8. MVP (v0.1) vs. roadmap

- **v0.1 ships:** verdict taxonomy, efficacy gate, control-suite steering, surface baselines, two-judge scoring, audit card, CLI + an Inspect-compatible scorer/report. Models: Qwen3-32B (dense), gpt-oss-120b (MoE).
- **Roadmap:** necessity/ablation (the other half of causality), multi-layer/distributed interventions, more models (Llama / Kimi / GLM), pre-registration mode, non-activation signal types (trace, behavioral, monitor scores).

## 9. Flagship demo (three cards, not two)

- **Qwen3-32B L55** → *effective, not causally sufficient* (full controls).
- **gpt-oss-120b L15** → *effective, probe-direction null* (corroborating).
- **gpt-oss-120b L34** → *intervention ineffective → inconclusive* (demonstrates the efficacy gate and its anti-gaming value).

## 10. Inspect compatibility

Package the audit as an Inspect task/scorer so the verdict + card slot into the Inspect ecosystem and, later, `inspect_evals`. This is the adoption highway and the mechanism by which SIEVE becomes a *cited* protocol rather than a one-off script.

## 11. Non-goals

Not a control protocol; not an eval-awareness leaderboard; not a claim that any signal *is* causal; not a safety certification.

## 12. Provenance

SIEVE implements and generalizes the controlled causal-sufficiency protocol from *"Probing Is Not Enough: Eval/Deploy Directions Are Decodable but Not Causally Sufficient for Behavior"* (companion research). The paper establishes the claim ("probing is not enough"); SIEVE is the reusable validity layer.
