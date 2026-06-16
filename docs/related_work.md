# Related work & landscape: where SIEVE sits

*Compiled 2026-06-16 from a deep-research sweep (20 sources, adversarial
verification) plus a direct source audit of the UK AISI Inspect codebases and
skeptical product audits of the named commercial players. Dated claims and
URLs below; uncertainty is flagged where it could not be closed.*

SIEVE's positioning claim is that **no shipped, reusable, third-party-runnable
*probe-validity* harness exists** — nothing bundles a surface-baseline
must-beat + matched-control steering (random / orthogonal / wrong-layer) + an
efficacy gate into a single pass/refuse verdict on whether an activation
signal can be trusted. The landscape is densely populated *around* that niche
with probe **builders**, feature **benchmarks**, **red-teamers**,
**guardrails**, and **LLM-judges** — but the niche itself is empty at every
location checked. This document records what each adjacent player ships and
why it is not the same thing.

## How to read the categories

- **(a) Direct competitor** — ships a probe-validity verdict engine. *None found.*
- **(b) Adjacent player** — builds/deploys/red-teams probes, features, or
  monitors, but does not adjudicate probe validity.
- **(c) Academic prior art** — the methods SIEVE operationalizes, established
  but never packaged as a third-party tool.

---

## (a) Direct competitors — none found

No company, open-source repo, PyPI package, or framework module audited below
occupies the niche. The two artifacts whose *names* most resemble SIEVE were
confirmed on inspection to be something else (see Haize "Verdict" and Robust
Intelligence "AI Validation" below) — the near-misses are decisively near.

---

## (b) Adjacent players

### Interpretability platforms (closest in spirit)

**Transluce** — Observatory / Monitor / Docent.
[github.com/TransluceAI/observatory](https://github.com/TransluceAI/observatory)
(MIT), Monitor announced 2024-10-23
([transluce.org/observability-interface](https://transluce.org/observability-interface)).
A toolkit "for describing model features and intervening on those features to
steer behavior." Monitor's AI "linter" *flags* candidate spurious concepts but
**explicitly defers causality to the user** ("does not explain how models make
decisions"), and Transluce's own write-up concedes it "reveals correlations
rather than definitively establishing causality." No surface baseline, no
matched-control steering, no verdict. → *steering/diagnosis platform.*

**Goodfire AI** — Ember (hosted interpretability API, public demo deprecated
as of early 2026) → Silico (model-design platform, early access 2026-04).
[goodfire.ai](https://www.goodfire.ai/),
[github.com/goodfire-ai](https://github.com/goodfire-ai). Inspect/steer SAE
features; debug and design models. Causal intervention appears as a *research
method in prose* (a "feature validation stage" in their papers) but is **not a
shipped verdict-emitting product**. → *steering/diagnosis platform.*

### Feature-quality benchmark (standardized, but not probe certification)

**SAEBench** — arXiv [2503.09532](https://arxiv.org/abs/2503.09532) (2025-03),
interface at neuronpedia.org/sae-bench. Eight metrics over 200+ SAEs; proof
that standardized validation infrastructure *can* exist and be adopted in this
space — but it benchmarks **SAE quality/comparison**, not the causal validity
of a deployment safety probe, and runs no matched-control steering or
pass/refuse verdict. → *feature benchmark.*

### Probe builders / reproductions

**Apollo Research** — "Detecting Strategic Deception Using Linear Probes,"
arXiv [2502.03407](https://arxiv.org/abs/2502.03407) (2025-02),
[github.com/ApolloResearch/deception-detection](https://github.com/ApolloResearch/deception-detection).
Trains linear deception probes on Llama-3.3-70B (AUROC 0.96–0.999) — and the
authors *themselves* conclude performance is "insufficient as a robust defence
against deception." A probe **builder**; precisely the kind of claim a
validity harness scrutinizes. (SIEVE's first audit target — see
[`../examples/apollo_deception/`](../examples/apollo_deception/).)

The wider GitHub population (`deception-probe*` repos, incl. FAR AI's
`AlignmentResearch/obfuscation-atlas`, 2026-06-12) are likewise **builders or
research studies**, not reusable validity engines. Concept searches for the
niche itself return nothing relevant: "probe certification" surfaces
*industrial temperature sensors*; "causal sufficiency steering" and "monitor
validation" return nothing; "interpretability audit" returns SHAP / medical-XAI
tools.

### AI-security companies (black-box; not probe validity)

| Company | Ships | Operates on | Classification |
|---|---|---|---|
| **Gray Swan** | Shade (red-team), Arena, **Cygnal** (runtime guardrail) | prompts/responses/tool-calls | red-team + guardrail builder |
| **Haize Labs** | Haizing/Sphynx/Cascade (red-team), **Verdict** (LLM-judge lib) | text outputs | red-team + LLM-judge tooling |
| **Lakera** | Guard (guardrail), Red, Gandalf | text I/O | runtime guardrail/firewall |
| **Robust Intelligence** | "AI Validation," AI Firewall → Cisco AI Defense | black-box model I/O | black-box model red-teaming |
| **METR** | capability/threat evals, behavioral monitor red-teaming | behavior | behavioral evals |

Sources: [grayswan.ai](https://www.grayswan.ai/),
[haizelabs.com](https://www.haizelabs.com/) + Verdict
([github.com/haizelabs/verdict](https://github.com/haizelabs/verdict), arXiv
[2502.18018](https://arxiv.org/abs/2502.18018)),
[lakera.ai](https://www.lakera.ai/),
[robustintelligence.com](https://www.robustintelligence.com/platform/ai-validation),
[metr.org](https://metr.org/). All operate at the text-I/O / behavioral layer;
none reads activations to adjudicate a probe's causal validity.

### "Sounds like SIEVE but isn't" — disambiguated

- **Haize "Verdict"** — name and pass/fail framing look adjacent, but it judges
  **text outputs only** (compound LLM-as-a-judge). Its use of "interpretability"
  means making *the judge's own reasoning* interpretable, not validating
  mechanistic probes. Zero activations/layers/steering.
- **Robust Intelligence "AI Validation"** — the phrase most likely to be
  confused with SIEVE. On inspection it is **black-box model stress-testing**
  (adversarial inputs → output susceptibility → guardrail recommendations),
  explicitly "without access to model internals." It validates *behavioral
  robustness*, never *a probe's causal validity*.

### Institutional infrastructure (the adoption highway, not a competitor)

**UK AISI — Inspect / `inspect_evals`.**
[github.com/UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai),
[inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals) (137 evals,
audited at `main`, 2026-06). A direct source audit found **zero** hits for
`residual_stream`, `linear_probe`, `wrong_layer`, `surface_baseline`,
`causally`, or `white_box`. Inspect is a **black-box behavioral eval
framework**: its providers are API backends doing `generate → score`. The one
activation-capable provider, `nnterp.py`, merely *exposes* `hidden_states` as a
raw model-output primitive — the role SIEVE's *adapter* plays — with no
steering, baseline, control, or verdict logic. Inspect is therefore SIEVE's
**adoption path** (package the audit as an Inspect scorer — DESIGN.md §10), not
a competitor in the niche.

---

## (c) Academic prior art (operationalized by SIEVE, never packaged)

The methods SIEVE composes are mature and well-established. SIEVE's contribution
is *packaging them into a frozen, third-party-runnable verdict engine*, not
inventing them. Cite these as the foundation:

- **Hewitt & Liang 2019**, "Designing and Interpreting Probes with Control
  Tasks" ([D19-1275](https://aclanthology.org/D19-1275/)) — control tasks &
  selectivity; direct ancestor of the decodability-vs-chance and
  surface-baseline stages.
- **Elazar et al. 2021**, "Amnesic Probing" (TACL) — INLP-based causal removal;
  the causal-vs-correlational move.
- **Belinkov 2022**, "Probing Classifiers: Promises, Shortcomings, and
  Advances" — the validity survey.
- **Parrack, Attubato & Heimersheim 2025**, "Benchmarking Deception Probes via
  Black-to-White Performance Boosts," arXiv
  [2507.12691](https://arxiv.org/abs/2507.12691) (2025-07, rev. 2026-01) — the
  **closest published analog to SIEVE's surface-baseline must-beat**: benchmark
  a probe by how much white-box access beats a black-box text-only monitor.
  It is a benchmark *methodology in a paper*, missing the steering /
  causal-sufficiency stages, and unpackaged. SIEVE generalizes and operationalizes it.
- **Smith, Chughtai & Nanda 2025**, "Difficulties with Evaluating a Deception
  Detector for AIs," arXiv [2511.22662](https://arxiv.org/abs/2511.22662)
  (2025-11) — establishes that probe validation is a *recognized, unsolved* open
  problem ("we currently lack the necessary examples"). SIEVE is a constructive,
  operational response.
- **"The Rogue Scalpel" 2025**, arXiv
  [2509.22067](https://arxiv.org/abs/2509.22067) (2025-09) — random steering
  produces 1–13% behavioral effects (up from 0%), empirically confirming the
  confound SIEVE's matched-control stage targets. (Note: this paper demonstrates
  the confound; it does **not** itself ship SIEVE-style matched controls — a
  stronger reading was refuted under adversarial verification.)

---

## The genuine whitespace

A **frozen-control, third-party-runnable validity verdict engine** for activation
probes does not exist as a shipped artifact, while the *demand* is explicitly
recognized by the field (Smith/Chughtai/Nanda; labs conceding their own
high-AUROC probes are not robust defenses). SIEVE occupies that gap. Its
contribution is **infrastructure-and-synthesis on an openly-unsolved problem**,
not a new method — and the standard-setting move (a frozen `SIEVE-v0.1-strict`
profile, DESIGN.md §7.5) is what distinguishes it from a parameterized script.

## Residual uncertainty (what could not be closed)

- **Private lab tooling is unknowable** from outside — no external search can
  settle whether a frontier lab runs an internal equivalent.
- **Goodfire** docs are not publicly browsable and Silico is early-access; a
  future/partner validation feature cannot be fully ruled out.
- Some vendor pages (Cisco, Haize docs) **403-blocked** automated fetch;
  corroborated via GitHub/arXiv/secondary sources, consistent but not
  verbatim-primary in every case.
- `gh` code search is **not exhaustive** (indexing/rate limits); absence of a
  code hit is weaker evidence than absence of a repo.
- The demand-side critiques are all from **late 2025–early 2026**, so the gap is
  real but **time-sensitive** — it could be filled imminently.
