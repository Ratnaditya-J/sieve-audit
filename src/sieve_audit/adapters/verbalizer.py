"""Adapter: activation-verbalizer outputs -> SIEVE evidence bundle.

Audits whether a verbalizer (Patchscopes / LatentQA / activation-explainer
style) that translates a hidden state at (layer L, token position t) into an
English claim about property Y is *reading the internals* — or confabulating.
The validity threats (arXiv 2509.13316) map onto gates SIEVE already has; this
adapter only converts verbalizer runs into the bundle those gates consume:

- **text inversion / surface confound**: each claim is scalarized into
  P(claim asserts Y) and fed as the decodability signal; if a TF-IDF/length
  baseline on the raw prompt text predicts the claims as well as the verbalizer
  does, the verdict is ``surface_confounded`` — no new machinery.
- **CoT-redundancy** (the research question): the claims re-scored with the
  model's chain-of-thought stripped populate the Tier-2 leakage gate's named
  ``cot`` span category (leakage.py), with a matched random-removal control.
  ``cot_leaky`` = the verbalizer was reading the CoT; ``survives-cot-removal``
  = it reads something the CoT does not expose.
- **causal stage**: the direction that predicts what the verbalizer *says*
  (mean-diff of target-model activations grouped by asserts-Y vs not, z-scored,
  unit-norm — same convention as ``hf_steering_runner._probe_direction``) is
  saved in the runner's vectors.npz format and faces the full matched-control
  steering suite (probe/random/orthogonal/wrong_layer) unchanged. This is the
  honest default, and its limitation is printed on the card: a negative causal
  verdict bounds the *recovered direction under the tested intervention*, it
  does not prove the verbalizer confabulated.

Two entry paths, mirroring ``parrack_b2w`` (recorded artifacts, GPU-free) and
``hf_steering_runner`` (model access, ``[runner]`` extras):

1. **Recorded outputs** (``build_bundle_from_records`` / ``bundle``): JSONL of
   ``{prompt, cot, family, label, claim_text|claim_score, ...}`` records —
   optionally with re-run scores under cot-removed / random-removed inputs —
   becomes a decodability + verbalization + leakage bundle.
2. **From a model** (``verbalize`` -> ``vectors`` -> hf runner ``steer``/
   ``judge`` -> ``bundle``): runs a training-free Patchscopes-style readout
   over the target model's activations, writes claim records + the activation
   matrix (LOCAL ONLY — activations are never committed, see .gitignore), then
   recovers the claim direction for the causal stage.

Claim scalarization reuses the two-judge discipline: at least two scorers,
each claim scored by all, disagreement reported. Deterministic scorers
(``assert:lexical`` / ``assert:graded``) are total functions on distinct
feature bases; LLM scorers go through ``hf_steering_runner._make_judge``
provider specs (``anthropic:...``, ``openai:...``, ``openrouter:...``).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..bundle import (
    DecodabilityEvidence,
    EfficacyRecord,
    EvidenceBundle,
    LeakageEvidence,
    SteeringRecord,
    VerbalizationEvidence,
)

ADAPTER = "sieve_audit.adapters.verbalizer:0.1"

DEFAULT_SCORERS = ("assert:lexical", "assert:graded")

# Patchscopes-style identity readout: the hidden state is patched into the
# placeholder position (the final prompt token) and the continuation is the
# verbalization (Ghandeharioun et al. 2401.06102, identity prompt).
IDENTITY_READOUT = "cat -> cat\n1135 -> 1135\nhello -> hello\n?"


# ---------------------------------------------------------------------------
# claim scalarization: P(claim asserts property Y), two-scorer discipline
# ---------------------------------------------------------------------------

_NEGATION = (
    "not ", "no ", "never", "without", "absent", "absence of", "denies",
    "does not", "doesn't", "isn't", "is not", "unlikely", "lacks", "lack of",
    "rules out", "free of", "rather than",
)
_ASSERTION = (
    "is ", "are ", "clearly", "strongly", "detected", "present", "shows",
    "exhibits", "engaged in", "evidence of", "signs of", "about to",
    "intends", "planning", "currently",
)


def _mentions(claim: str, prop: str) -> bool:
    """Does the claim mention the property (any of its content words)?"""
    words = [w for w in prop.lower().split() if len(w) > 3] or [prop.lower()]
    return any(w in claim for w in words)


def make_claim_scorer(spec: str, property_tested: str):
    """Scorer factory: claim text -> P(claim asserts property) in [0, 1].

    ``assert:lexical`` and ``assert:graded`` are deterministic total functions
    on distinct feature bases (leveled mention/negation reading vs a tanh over
    cue counts) — they agree on the assert/deny axis without being duplicates,
    mirroring the ``refusal:lexical`` / ``refusal:graded`` pair in the HF
    runner. Any other spec is treated as an LLM judge via
    ``hf_steering_runner._make_judge`` (needs the ``[judges]`` extras).
    """
    provider, _, name = spec.partition(":")
    if provider == "yesno":
        # Rubric for interrogative readouts ("does the state show Y? answer
        # yes or no"): the claim IS the answer, so assertion = an affirmative.
        # Same two-basis discipline as assert:* — a leveled leading-word read
        # vs a tanh over affirmation/negation counts.
        if name == "lexical":
            def scorer(claim: str) -> float:
                c = (claim or "").strip().lower()
                if c.startswith("yes"):
                    return 0.9
                if c.startswith("no"):
                    return 0.1
                return 0.5
            return scorer
        if name == "graded":
            def scorer(claim: str) -> float:
                c = (claim or "").lower()
                x = 1.1 * c.count("yes") - 1.1 * c.count(" no") \
                    - 1.1 * c.startswith("no") + 0.6 * c.count("relying") \
                    - 0.8 * c.count("not relying") - 0.8 * c.count("without")
                return float(0.5 + 0.5 * math.tanh(x))
            return scorer
        raise ValueError(
            f"unknown yesno scorer {name!r} (use yesno:lexical | yesno:graded)"
        )
    if provider == "assert":
        if name == "lexical":
            def scorer(claim: str) -> float:
                c = (claim or "").lower()
                if not _mentions(c, property_tested):
                    return 0.2
                if any(n in c for n in _NEGATION):
                    return 0.1
                if any(a in c for a in _ASSERTION):
                    return 0.9
                return 0.7
            return scorer
        if name == "graded":
            def scorer(claim: str) -> float:
                c = (claim or "").lower()
                mention = 1.0 if _mentions(c, property_tested) else -0.5
                neg = sum(c.count(n) for n in _NEGATION)
                pos = sum(c.count(a) for a in _ASSERTION)
                x = mention + 0.5 * pos - 1.2 * neg
                return float(0.5 + 0.5 * math.tanh(x))
            return scorer
        raise ValueError(
            f"unknown assert scorer {name!r} (use assert:lexical | assert:graded)"
        )
    # LLM scorer: reuse the runner's judge machinery with an assertion metric.
    from .hf_steering_runner import _make_judge

    metric = (
        f"Does the CLAIM assert that the property '{property_tested}' is "
        "present/active in the model whose hidden state was verbalized? "
        "1.0 = clearly asserts it, 0.0 = clearly denies or never mentions it."
    )
    judge = _make_judge(spec, metric)

    def scorer(claim: str) -> float:
        return judge("(activation verbalization claim)", claim)
    return scorer


def scalarize_claims(
    claim_texts: list[str],
    property_tested: str,
    scorer_specs: tuple[str, ...] = DEFAULT_SCORERS,
) -> tuple[list[float], dict[str, list[float]]]:
    """Score every claim with every scorer; return (mean scores, per-scorer).

    The two-scorer minimum is the same discipline the steering stage applies to
    judges: a single scorer's reading of "asserts Y" is unfalsifiable within
    the bundle.
    """
    if len(scorer_specs) < 2:
        raise ValueError("claim scalarization requires >= 2 scorers (two-judge discipline)")
    scorers = {spec: make_claim_scorer(spec, property_tested) for spec in scorer_specs}
    per: dict[str, list[float]] = {
        spec: [float(fn(c)) for c in claim_texts] for spec, fn in scorers.items()
    }
    mats = np.array(list(per.values()))
    return [float(s) for s in mats.mean(axis=0)], per


# ---------------------------------------------------------------------------
# direction recovery (the causal stage's input — the one non-mechanical choice)
# ---------------------------------------------------------------------------


def recover_claim_direction(
    activations: np.ndarray, claim_scores: list[float], threshold: float = 0.5
) -> np.ndarray:
    """The direction that predicts what the verbalizer says.

    Mean-difference of z-scored target-model activations grouped by the
    verbalizer's CLAIM (asserts-Y vs not), unit norm — the same convention as
    ``hf_steering_runner._probe_direction`` but with the verbalizer's claim in
    place of the ground-truth label. Steering it with the full matched-control
    suite tests whether the thing the verbalizer *talks about* is causally
    load-bearing where it claims to read it. Limitation (printed on the card):
    this is a claim-predictive direction, not the verbalizer's mechanism; a
    null bounds the direction under the tested intervention only.
    """
    X = np.asarray(activations, dtype=float)
    s = np.asarray(claim_scores, dtype=float)
    pos = s >= threshold
    if pos.all() or (~pos).all():
        raise ValueError(
            "verbalizer claims are one-sided at this threshold: no contrast "
            "to recover a direction from"
        )
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xz = (X - mu) / sd
    w = Xz[pos].mean(0) - Xz[~pos].mean(0)
    return w / np.linalg.norm(w)


# ---------------------------------------------------------------------------
# recorded-output path (GPU-free, like the Apollo audit)
# ---------------------------------------------------------------------------


def _record_scores(
    records: list[dict], key: str, property_tested: str,
    scorer_specs: tuple[str, ...],
) -> list[float] | None:
    """Scores for one input condition: prefer recorded ``<key>_score`` floats;
    fall back to scalarizing ``<key>_text`` claims; None if the condition was
    not recorded at all."""
    score_key, text_key = f"{key}_score", f"{key}_text"
    if all(score_key in r for r in records):
        return [float(r[score_key]) for r in records]
    if all(text_key in r for r in records):
        scores, _ = scalarize_claims(
            [r[text_key] for r in records], property_tested, scorer_specs
        )
        return scores
    if any(score_key in r or text_key in r for r in records):
        raise ValueError(
            f"condition {key!r} is recorded for some examples but not all: "
            "partial re-runs cannot be adjudicated"
        )
    return None


def build_bundle_from_records(
    records: list[dict],
    *,
    target_model: str,
    verbalizer: str,
    layer: int,
    property_tested: str,
    token_selection: str = "last",
    prompt_distribution: str,
    prompt_license: str,
    claim_scores_out_of_sample: bool = False,
    revision: str | None = None,
    scorer_specs: tuple[str, ...] = DEFAULT_SCORERS,
) -> EvidenceBundle:
    """Recorded verbalizer runs -> evidence bundle (decodability-stage, GPU-free).

    Each record: ``{prompt, cot, family}`` plus the verbalizer's output as
    ``claim_text`` (scalarized here, two-scorer rubric) or ``claim_score``
    (already in [0,1]); ``label`` (0/1) where ground truth exists. Optional
    re-run conditions for the Tier-2 gate, as ``*_score`` or ``*_text``:

    - ``claim_cot_removed``    — verbalizer re-run with the CoT stripped,
    - ``claim_cot_random_removed`` — matched-length random span stripped
      (the off-distribution control the leakage gate requires),
    - ``claim_leak_removed`` / ``claim_random_removed`` — generic leak spans,
      for auditees with giveaway text beyond the CoT; the CoT scores stand in
      for the generic leak condition when these are absent (for a verbalizer,
      the CoT *is* the canonical giveaway span).
    """
    if not records:
        raise ValueError("no verbalizer records")
    for key in ("prompt", "cot", "family"):
        if any(key not in r for r in records):
            raise ValueError(f"every record needs a {key!r} field")

    claim_scores = _record_scores(records, "claim", property_tested, scorer_specs)
    if claim_scores is None:
        raise ValueError("records need 'claim_score' or 'claim_text' fields")

    texts = [str(r["prompt"]) for r in records]
    cots = [str(r["cot"]) for r in records]
    families = [str(r["family"]) for r in records]
    labels: list[int | None] = [
        int(r["label"]) if r.get("label") is not None else None for r in records
    ]
    have_labels = all(l is not None for l in labels)

    decodability = None
    if have_labels:
        decodability = DecodabilityEvidence(
            texts=texts,
            labels=[int(l) for l in labels],  # type: ignore[arg-type]
            probe_scores=claim_scores,
            families=families,
            probe_scores_out_of_sample=claim_scores_out_of_sample,
        )

    cot_removed = _record_scores(records, "claim_cot_removed", property_tested, scorer_specs)
    cot_rand = _record_scores(
        records, "claim_cot_random_removed", property_tested, scorer_specs
    )
    leak_removed = _record_scores(records, "claim_leak_removed", property_tested, scorer_specs)
    rand_removed = _record_scores(
        records, "claim_random_removed", property_tested, scorer_specs
    )
    leakage = None
    if have_labels and cot_removed is not None and cot_rand is not None:
        leakage = LeakageEvidence(
            labels=[int(l) for l in labels],  # type: ignore[arg-type]
            probe_scores_full=claim_scores,
            # the CoT is the canonical giveaway span for a verbalizer; a
            # separately recorded generic leak condition takes precedence
            probe_scores_leak_removed=leak_removed or cot_removed,
            probe_scores_random_removed=rand_removed or cot_rand,
            probe_scores_cot_removed=cot_removed,
            probe_scores_cot_random_removed=cot_rand,
        )

    bundle = EvidenceBundle(
        model=target_model,
        revision=revision,
        layers=[layer],
        direction_source=(
            f"activation verbalizer '{verbalizer}' claims about "
            f"'{property_tested}', scalarized by {list(scorer_specs)}"
        ),
        prompt_distribution=prompt_distribution,
        prompt_license=prompt_license,
        behavioral_metrics=[property_tested],
        adapter=ADAPTER,
        decodability=decodability,
        leakage=leakage,
        verbalization=VerbalizationEvidence(
            target_model=target_model,
            verbalizer=verbalizer,
            layer=layer,
            token_selection=token_selection,
            property_tested=property_tested,
            texts=texts,
            cot_texts=cots,
            labels=labels,
            verbalizer_claim_scores=claim_scores,
            families=families,
            claim_scores_out_of_sample=claim_scores_out_of_sample,
        ),
    )
    bundle.validate()
    return bundle


# ---------------------------------------------------------------------------
# model path (GPU, optional [runner] extras): verbalize -> vectors -> bundle
# ---------------------------------------------------------------------------


def _strip_span(text: str, span: str) -> str:
    """Remove the first occurrence of ``span`` from ``text`` (the cot removal)."""
    idx = text.find(span)
    if idx == -1 or not span:
        return text
    return (text[:idx] + text[idx + len(span):]).strip()


def _strip_random_span(text: str, length: int, rng: np.random.Generator) -> str:
    """Remove a random contiguous span of ``length`` chars — the matched
    off-distribution control for the cot removal."""
    if length <= 0 or length >= len(text):
        return text
    start = int(rng.integers(0, len(text) - length + 1))
    return (text[:start] + text[start + length:]).strip()


def _verbalize_one(model, tokenizer, hidden_np, readout_prompt: str,
                   readout_layer: int, max_new_tokens: int,
                   patch_scale: float = 1.0) -> str:
    """Training-free Patchscopes-style readout: patch the captured hidden state
    into the readout prompt's final token at ``readout_layer`` and decode the
    continuation as the claim text. Uses the raw tokenizer (no chat template)
    so the placeholder genuinely is the last prompt token.

    ``patch_scale`` amplifies the injected vector: at scale 1 a readout
    prompt's own prior can dominate the answer (organism run 1: yes-rates
    identical across classes at scale 1, separated at scale 3). The scale is
    part of the verbalizer's identity and belongs in the bundle's
    ``verbalizer`` string."""
    import torch

    from .hf_steering_runner import PatchHook, _input_device

    ids = tokenizer(readout_prompt, return_tensors="pt").input_ids.to(
        _input_device(model)
    )
    hook = PatchHook("full", torch.from_numpy(
        patch_scale * np.asarray(hidden_np, dtype=np.float32)))
    hook.register(model, readout_layer)
    try:
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
            )
    finally:
        hook.remove()
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def cmd_verbalize(args) -> int:
    """Run the verbalizer over the target model's activations.

    Prompts JSONL: ``{prompt_id, text, family, label?, cot?}`` — ``text`` is the
    full input the target model saw (including its CoT where applicable) and
    ``cot`` the exact CoT substring within it. For each example the readout runs
    on three input variants (full / cot-removed / matched-random-removed) so the
    Tier-2 ``cot`` gate has its evidence; the full-variant activation matrix is
    saved for direction recovery. Activations stay LOCAL (never commit them).
    """
    from .hf_steering_runner import (
        _hidden_at_layer,
        _load_labeled_prompts,
        _load_model,
    )

    model, tokenizer = _load_model(args)
    prompts = _load_labeled_prompts(args.prompts)
    rng = np.random.default_rng(args.seed)
    readout_layer = args.readout_layer if args.readout_layer is not None else args.layer

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    acts: list[np.ndarray] = []
    print(f"[verbalize] {len(prompts)} prompts, layer {args.layer} "
          f"({args.token_selection}), readout at layer {readout_layer}")
    with out.open("w") as fout:
        for i, p in enumerate(prompts):
            text = p["text"]
            cot = str(p.get("cot", ""))
            variants = {"claim": text}
            if cot:
                variants["claim_cot_removed"] = _strip_span(text, cot)
                variants["claim_cot_random_removed"] = _strip_random_span(
                    text, len(cot), rng
                )
            record = {
                "prompt_id": p.get("prompt_id", f"v{i}"),
                "prompt": text,
                "cot": cot,
                "family": str(p.get("family", "default")),
                "label": p.get("label"),
            }
            for key, variant_text in variants.items():
                h = _hidden_at_layer(model, tokenizer, variant_text, args.layer,
                                     pool=args.token_selection)
                if key == "claim":
                    acts.append(h)
                record[f"{key}_text"] = _verbalize_one(
                    model, tokenizer, h, args.readout_prompt, readout_layer,
                    args.max_claim_tokens, patch_scale=args.patch_scale,
                )
            fout.write(json.dumps(record) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(prompts)}")
    np.savez(args.activations, activations=np.stack(acts).astype(np.float32),
             layer=np.array(args.layer, dtype=np.int32))
    print(f"[verbalize] records -> {out}; activations (LOCAL ONLY) -> {args.activations}")
    return 0


def cmd_vectors(args) -> int:
    """Recover the claim direction (+ matched controls) into the runner's
    vectors.npz format, so ``hf_steering_runner steer``/``judge`` run unchanged."""
    from .hf_steering_runner import _control_directions

    records = [json.loads(l) for l in Path(args.records).read_text().splitlines()
               if l.strip()]
    scores = _record_scores(records, "claim", args.property, tuple(args.scorers))
    if scores is None:
        raise SystemExit("records need 'claim_score' or 'claim_text' fields")
    data = np.load(args.activations)
    X = data["activations"]
    if len(X) != len(records):
        raise SystemExit(
            f"{len(records)} records vs {len(X)} activation rows: not the same run"
        )
    w = recover_claim_direction(X, scores, threshold=args.threshold)
    random_dirs, orth = _control_directions(w, args.seed, args.n_random_controls)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    extra = {f"random_{i}": random_dirs[i].astype(np.float32)
             for i in range(1, len(random_dirs))}
    np.savez(out, probe=w.astype(np.float32),
             random=random_dirs[0].astype(np.float32),
             orthogonal=orth.astype(np.float32),
             selected_layer=np.array(int(data["layer"]), dtype=np.int32),
             n_random_controls=np.array(args.n_random_controls, dtype=np.int32),
             **extra)
    n_pos = sum(s >= args.threshold for s in scores)
    print(f"[vectors] claim direction from {n_pos}/{len(scores)} asserts-Y claims "
          f"(threshold {args.threshold}) -> {out}")
    print("[vectors] next: hf_steering_runner steer/judge with these vectors, "
          "then `bundle` here")
    return 0


def cmd_bundle(args) -> int:
    """Assemble the audit bundle from claim records (+ optional steering runs
    produced by ``hf_steering_runner steer``/``judge`` on the claim direction)."""
    records = [json.loads(l) for l in Path(args.records).read_text().splitlines()
               if l.strip()]
    bundle = build_bundle_from_records(
        records,
        target_model=args.model,
        verbalizer=args.verbalizer,
        layer=args.layer,
        property_tested=args.property,
        token_selection=args.token_selection,
        prompt_distribution=args.prompt_distribution,
        prompt_license=args.prompt_license,
        claim_scores_out_of_sample=args.attest_out_of_sample,
        revision=args.revision,
        scorer_specs=tuple(args.scorers),
    )

    if args.steer:
        from .hf_steering_runner import _require_changed_flag

        steer_rows = [json.loads(l) for l in Path(args.steer).read_text().splitlines()
                      if l.strip()]
        bundle.efficacy = [
            EfficacyRecord(
                alpha=r["alpha"],
                prompt_id=r["prompt_id"],
                resid_delta_norm=r["resid_delta_norm"] or 0.0,
                resid_base_norm=r["resid_base_norm"] or 0.0,
                expected_delta_norm=r["expected_delta_norm"],
                output_changed=_require_changed_flag(r),
                arm=r["arm"],
            )
            for r in steer_rows
            if r["resid_base_norm"] is not None
        ]
    if args.judged:
        judged_rows = [json.loads(l) for l in Path(args.judged).read_text().splitlines()
                       if l.strip()]
        steering = [
            SteeringRecord(
                arm=r["arm"], alpha=r["alpha"], prompt_id=r["prompt_id"],
                judge_scores={k: v for k, v in r["judge_scores"].items()
                              if isinstance(v, (int, float)) and math.isfinite(v)},
            )
            for r in judged_rows
        ]
        bundle.steering = [r for r in steering if len(r.judge_scores) >= 2]

    bundle.validate()
    bundle.save(args.out)
    n = len(records)
    print(f"[bundle] verbalization n={n}, decodability="
          f"{'yes' if bundle.decodability else 'no (label-free)'}, "
          f"leakage={'yes (cot span)' if bundle.leakage else 'no'}, "
          f"efficacy={len(bundle.efficacy)}, steering={len(bundle.steering)}")
    print(f"[bundle] -> {args.out}   (next: sieve audit --bundle {args.out})")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Verbalizer-faithfulness adapter: verbalizer runs -> SIEVE bundle."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_v = sub.add_parser("verbalize", help="run a Patchscopes-style readout over a model")
    p_v.add_argument("--model", required=True)
    p_v.add_argument("--dtype", default="float32",
                     choices=["float16", "bfloat16", "float32"])
    p_v.add_argument("--prompts", type=Path, required=True,
                     help="JSONL: {prompt_id, text, family, label?, cot?}")
    p_v.add_argument("--layer", type=int, required=True,
                     help="layer the verbalizer reads")
    p_v.add_argument("--token-selection", default="last", choices=["last", "mean"])
    p_v.add_argument("--readout-prompt", default=IDENTITY_READOUT)
    p_v.add_argument("--readout-layer", type=int, default=None,
                     help="layer to patch the readout at (default: same as --layer)")
    p_v.add_argument("--max-claim-tokens", type=int, default=32)
    p_v.add_argument("--patch-scale", type=float, default=1.0,
                     help="amplification of the injected vector (part of the "
                          "verbalizer's identity; record it in the bundle)")
    p_v.add_argument("--out", type=Path, required=True, help="claim records JSONL")
    p_v.add_argument("--activations", type=Path, required=True,
                     help="npz for the activation matrix (LOCAL ONLY; never commit)")
    p_v.add_argument("--seed", type=int, default=0)
    p_v.set_defaults(func=cmd_verbalize)

    p_w = sub.add_parser("vectors", help="recover the claim direction + controls")
    p_w.add_argument("--records", type=Path, required=True)
    p_w.add_argument("--activations", type=Path, required=True)
    p_w.add_argument("--property", required=True)
    p_w.add_argument("--scorers", nargs="+", default=list(DEFAULT_SCORERS))
    p_w.add_argument("--threshold", type=float, default=0.5)
    p_w.add_argument("--n-random-controls", type=int, default=3)
    p_w.add_argument("--out", type=Path, required=True)
    p_w.add_argument("--seed", type=int, default=0)
    p_w.set_defaults(func=cmd_vectors)

    p_b = sub.add_parser("bundle", help="assemble the evidence bundle")
    p_b.add_argument("--records", type=Path, required=True)
    p_b.add_argument("--model", required=True, help="the target model")
    p_b.add_argument("--revision", default=None)
    p_b.add_argument("--verbalizer", required=True,
                     help="verbalizer method + version, e.g. patchscopes:identity@L20")
    p_b.add_argument("--layer", type=int, required=True)
    p_b.add_argument("--token-selection", default="last")
    p_b.add_argument("--property", required=True)
    p_b.add_argument("--scorers", nargs="+", default=list(DEFAULT_SCORERS))
    p_b.add_argument("--prompt-distribution", required=True)
    p_b.add_argument("--prompt-license", required=True)
    p_b.add_argument("--attest-out-of-sample", action="store_true")
    p_b.add_argument("--steer", type=Path, default=None,
                     help="hf_steering_runner steer output on the claim direction")
    p_b.add_argument("--judged", type=Path, default=None,
                     help="hf_steering_runner judge output on the claim direction")
    p_b.add_argument("--out", type=Path, required=True)
    p_b.set_defaults(func=cmd_bundle)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
