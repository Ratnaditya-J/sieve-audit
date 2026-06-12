"""Adapter: Parrack, Attubato & Heimersheim's black_to_white_boosts dumps.

Source: https://github.com/carlo-leonardo-attubato/black_to_white_boosts
(arXiv 2507.12691). Their ``results/*.jsonl`` files contain, per sample, the
full conversation, per-token activations scores for the two Apollo Research
deception probes of arXiv 2502.03407 (Goldowsky-Dill et al.) —
``pairs_probe`` (trained on instructed pairs) and ``rp_probe`` (trained on
roleplaying) — both reading layer 22 of Llama-3.3-70B-Instruct, plus
honest/deceptive labels.

This adapter builds a *decodability-stage* evidence bundle:

- probe score per example = mean per-token probe score over the assistant's
  content tokens (Apollo's mean-aggregation, applied to the response);
- text per example = the full rendered conversation (the same context the
  probe's activations attend over — the fair input for surface baselines);
- family per example = source dataset (ai_liar, insider_trading, ...), so the
  leave-one-family-out baseline split tests cross-dataset generalization,
  which is exactly the probes' headline claim.

The probes were trained on Apollo's instructed-pairs / roleplaying training
sets, none of which appear in these evaluation dumps, so
``probe_scores_out_of_sample=True`` is attested.

No steering/efficacy evidence exists for these probes (none was ever
published), so the causal stages cannot run: SIEVE will report the
decodability diagnostics and refuse a causal verdict. That refusal is the
finding.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..bundle import DecodabilityEvidence, EvidenceBundle

PROBES = ("pairs_probe", "rp_probe")

# labels in the dumps; ambiguous/other samples are dropped
_LABEL_MAP = {"honest": 0, "deceptive": 1}


def _sample_text(conv: dict) -> str:
    return "\n".join(m["content"] for m in conv["messages"])


def _sample_score(conv: dict, probe: str) -> float | None:
    scores = conv.get("probe_scores", {}).get(probe)
    if not scores:
        return None
    idx = [
        i
        for i in conv.get("assistant_indices", [])
        if i not in set(conv.get("non_content_indices", []))
    ]
    if not idx:
        return None
    return float(sum(scores[i] for i in idx) / len(idx))


def load_examples(
    jsonl_paths: list[str | Path], probe: str, branch: str = "main"
) -> tuple[list[str], list[int], list[float], list[str]]:
    """Per-example (text, label, probe score, family) across the given dumps."""
    texts: list[str] = []
    labels: list[int] = []
    scores: list[float] = []
    families: list[str] = []
    for path in jsonl_paths:
        path = Path(path)
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                label = _LABEL_MAP.get(str(sample["metadata"].get("label", "")).lower())
                if label is None:
                    continue
                conv = sample.get("conversations", {}).get(branch)
                if conv is None:
                    continue
                score = _sample_score(conv, probe)
                if score is None:
                    continue
                texts.append(_sample_text(conv))
                labels.append(label)
                scores.append(score)
                families.append(sample["metadata"].get("dataset", path.stem))
    return texts, labels, scores, families


def build_bundle(jsonl_paths: list[str | Path], probe: str) -> EvidenceBundle:
    if probe not in PROBES:
        raise ValueError(f"unknown probe {probe!r}; expected one of {PROBES}")
    texts, labels, scores, families = load_examples(jsonl_paths, probe)
    if not texts:
        raise ValueError("no labeled samples found in the given dumps")
    trained_on = "instructed-pairs" if probe == "pairs_probe" else "roleplaying"
    return EvidenceBundle(
        model="meta-llama/Llama-3.3-70B-Instruct",
        revision=None,
        layers=[22],
        direction_source=(
            f"Apollo Research logistic-regression deception probe "
            f"('{probe}', trained on {trained_on}; arXiv 2502.03407), "
            f"scores via arXiv 2507.12691 dumps"
        ),
        prompt_distribution="black_to_white_boosts eval dumps: "
        + ", ".join(sorted(set(families))),
        prompt_license="unreleased license (repos ship no LICENSE); data not redistributed",
        behavioral_metrics=["deception (honest=0 / deceptive=1)"],
        adapter="sieve_audit.adapters.parrack_b2w:0.1",
        decodability=DecodabilityEvidence(
            texts=texts,
            labels=labels,
            probe_scores=scores,
            families=families,
            # the probes were trained on Apollo's instructed-pairs/roleplaying
            # training sets; these evaluation dumps are disjoint from both
            probe_scores_out_of_sample=True,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a SIEVE evidence bundle from black_to_white_boosts dumps."
    )
    parser.add_argument("jsonls", nargs="+", help="paths to results/*.jsonl dumps")
    parser.add_argument("--probe", choices=PROBES, required=True)
    parser.add_argument("--out", required=True, help="output bundle JSON path")
    args = parser.parse_args(argv)

    bundle = build_bundle(args.jsonls, args.probe)
    bundle.save(args.out)
    n = bundle.decodability.labels
    print(
        f"[sieve-adapter] {args.probe}: {len(n)} examples "
        f"({sum(n)} deceptive / {len(n) - sum(n)} honest), "
        f"families: {sorted(set(bundle.decodability.families))}"
    )
    print(f"[sieve-adapter] bundle -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
