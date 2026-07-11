"""Unlearning-completeness driver: turn a model + WMDP statements into SIEVE
decodability bundles, one per (layer, probe class).

This is the GPU-side adapter for the unlearning-completeness audit. It follows
the SIEVE contract exactly: the model-touching code produces a small *evidence
bundle* (probe scores + texts + labels + families — never raw activations), and
the GPU-free engine audits the bundle. So a pod extracts, and `sieve audit`
runs anywhere.

The audited signal is a linear probe WE fit on the model under audit — SIEVE as
instrument, not referee of someone else's probe. For each preregistered layer:

  * read the last-token residual for every statement (raw-text encoding, so the
    readout position is the last token of the claim, not a chat-template
    sentinel — the truth-probe convention of Marks & Tegmark 2023);
  * score every example leave-one-family-out (train the probe on the other
    subtopics, project the held-out subtopic) so every probe score is genuinely
    out-of-family — the SAME generalization burden the engine imposes on its
    surface baselines, and the honest basis for attesting
    ``probe_scores_out_of_sample=True``;
  * do this for both linear readouts the engine ships — ``mean_diff`` (the
    steerable CAA direction) and ``logistic_regression`` (a stronger linear
    reader) — so a null cannot be blamed on a weak probe class.

Layers are supplied explicitly (``--layers``) and frozen in the
preregistration; there is deliberately NO best-AUROC auto-select here, because
picking the layer after seeing the scores is exactly the post-hoc shopping the
preregistration exists to forbid.

Reuses the tested numerics from the core package: ``_get_layer_module`` /
``_n_layers`` (layer location) from the runner, and ``fit_activation_probe_scores``
/ ``_probe_direction`` (probe math) from the engine. Nothing about the probe is
reimplemented here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sieve_audit.adapters.hf_steering_runner import (
    _get_layer_module,
    _n_layers,
    _probe_direction,
)
from sieve_audit.bundle import DecodabilityEvidence, EvidenceBundle
from sieve_audit.decodability import ACTIVATION_PROBES, fit_activation_probe_scores

PROBE_CLASSES = ACTIVATION_PROBES  # ("mean_diff", "logistic_regression")


# ---------------------------------------------------------------------------
# model + raw-text last-token extraction
# ---------------------------------------------------------------------------


def _load_model(model_id: str, dtype: str, revision: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision,
                                        trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch_dtype,
        device_map="auto", trust_remote_code=True)
    model.eval()
    return model, tok


def _input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:  # pragma: no cover
        return model.device


def extract_activations(model, tok, texts: list[str], layers: list[int],
                        use_chat_template: bool = False,
                        batch_report: int = 50) -> dict[int, np.ndarray]:
    """Last-token residual at each requested layer for every text.

    Returns {layer: X[n, d]} float32. One forward pass per text captures all
    requested layers simultaneously. Raw-text encoding by default (the last
    token is the final subword of the claim); ``--chat-template`` wraps each
    text as a user turn instead (readout at the generation-prompt position)."""
    import torch

    n_lay = _n_layers(model)
    for L in layers:
        if not 0 <= L < n_lay:
            raise SystemExit(f"layer {L} out of range (model has {n_lay} layers)")

    device = _input_device(model)
    acts: dict[int, list[np.ndarray]] = {L: [] for L in layers}

    captured: dict[int, "torch.Tensor"] = {}
    handles = []
    for L in layers:
        def _mk(i):
            def grab(_m, _a, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[i] = h[0, -1].float().detach().cpu()
            return grab
        handles.append(_get_layer_module(model, L).register_forward_hook(_mk(L)))

    try:
        for idx, text in enumerate(texts):
            if use_chat_template:
                ids = tok.apply_chat_template(
                    [{"role": "user", "content": text}],
                    return_tensors="pt", add_generation_prompt=True)
                if hasattr(ids, "input_ids"):
                    ids = ids.input_ids
            else:
                ids = tok(text, return_tensors="pt").input_ids
            ids = ids.to(device)
            captured.clear()
            with torch.no_grad():
                model(ids)
            for L in layers:
                acts[L].append(captured[L].numpy())
            if (idx + 1) % batch_report == 0:
                print(f"  [extract] {idx + 1}/{len(texts)}")
    finally:
        for h in handles:
            h.remove()

    return {L: np.stack(acts[L]).astype(np.float32) for L in layers}


# ---------------------------------------------------------------------------
# leave-one-family-out probe scoring (matches the engine's baseline split)
# ---------------------------------------------------------------------------


def lofo_scores(X: np.ndarray, labels: np.ndarray, families: np.ndarray,
                probe_class: str, seed: int) -> np.ndarray:
    """Out-of-family probe scores: for each family, train on the others, project
    the held-out family. Identical split to ``decodability._held_out_baseline_scores``
    so probe and surface baselines face the same generalization test."""
    scores = np.full(len(labels), np.nan)
    for fam in np.unique(families):
        te = families == fam
        tr = ~te
        if len(np.unique(labels[tr])) < 2:
            raise SystemExit(
                f"family {fam!r} left the training split single-class; every "
                "OTHER family must contain both labels (check data balance)")
        scores[te] = fit_activation_probe_scores(
            probe_class, X[tr], labels[tr], X[te], seed=seed)
    assert not np.isnan(scores).any()
    return scores


# ---------------------------------------------------------------------------
# bundle assembly
# ---------------------------------------------------------------------------


def _load_statements(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for r in rows:
        if not all(k in r for k in ("text", "label", "family")):
            raise SystemExit("statements need text/label/family fields")
    return rows


def direction_source_str(probe_class: str, encoding: str) -> str:
    return (f"{probe_class} linear probe over last-token residuals "
            f"({encoding} encoding), fit leave-one-family-out on WMDP "
            f"correct-vs-distractor completion pairs")


def prompt_distribution_str(domain: str, n_families: int) -> str:
    return (f"{domain}: matched correct/distractor answer completions of WMDP "
            f"questions ({n_families} subtopic families), length-matched within pairs")


PROMPT_LICENSE = "MIT (cais/wmdp); statements derived, not redistributed here"


def behavioral_metric_str(domain: str) -> str:
    return f"{domain} factual-completion correctness (retained knowledge)"


def build_bundle_for_layer(model_id: str, revision: str | None, layer: int,
                           probe_class: str, texts: list[str],
                           labels: np.ndarray, families: np.ndarray,
                           scores: np.ndarray, domain: str,
                           encoding: str) -> EvidenceBundle:
    return EvidenceBundle(
        model=model_id,
        revision=revision,
        layers=[layer],
        direction_source=direction_source_str(probe_class, encoding),
        prompt_distribution=prompt_distribution_str(domain, len(np.unique(families))),
        prompt_license=PROMPT_LICENSE,
        behavioral_metrics=[behavioral_metric_str(domain)],
        adapter="examples.unlearning_completeness.unlearning_audit:0.1",
        decodability=DecodabilityEvidence(
            texts=texts,
            labels=[int(x) for x in labels],
            probe_scores=[float(x) for x in scores],
            families=[str(x) for x in families],
            probe_scores_out_of_sample=True,  # every score is out-of-family (LOFO)
        ),
    )


def cmd_build_bundle(args) -> int:
    rows = _load_statements(args.statements)
    texts = [r["text"] for r in rows]
    labels = np.array([int(r["label"]) for r in rows])
    families = np.array([str(r["family"]) for r in rows])
    domain = args.domain

    if args.shuffle_labels:
        # Negative control: permute labels WITHIN each family (preserving class
        # balance so the engine's gates still run) to sever the text<->correctness
        # link. A knowledge-free / signal-free instrument must floor here — this
        # validates that the pipeline does not manufacture decodability from
        # noise. NOT a scientific anchor; that is a real never-trained model.
        rng = np.random.default_rng(args.seed + 1)
        for fam in np.unique(families):
            idx = np.where(families == fam)[0]
            labels[idx] = labels[idx][rng.permutation(len(idx))]
        print("[build-bundle] SHUFFLE-LABELS negative control active "
              "(within-family label permutation)")

    if args.reuse_activations:
        # rebuild bundles from activations already extracted for another tag
        # (e.g. the shuffle-labels anchor reuses the unlearned model's acts) —
        # no model load, no forward passes.
        reuse_dir = Path(args.reuse_activations)
        acts = {}
        for layer in args.layers:
            npz = np.load(reuse_dir / f"vectors.{args.reuse_tag}.L{layer}.npz")
            if "layer_activations" not in npz:
                raise SystemExit(f"vectors.{args.reuse_tag}.L{layer}.npz has no "
                                 "layer_activations; re-extract with this build")
            acts[layer] = npz["layer_activations"]
            if acts[layer].shape[0] != len(texts):
                raise SystemExit(
                    f"reused activations n={acts[layer].shape[0]} != statements "
                    f"n={len(texts)}; the statements file must match the extraction")
        print(f"[build-bundle] reused activations from {args.reuse_tag} "
              f"(no model load) for layers {args.layers}")
    else:
        print(f"[build-bundle] loading {args.model} (rev={args.revision}, dtype={args.dtype})")
        model, tok = _load_model(args.model, args.dtype, args.revision)
        print(f"[build-bundle] extracting last-token acts at layers {args.layers} "
              f"for {len(texts)} statements ({args.encoding} encoding)")
        acts = extract_activations(model, tok, texts, args.layers,
                                   use_chat_template=(args.encoding == "chat"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model_tag or args.model.split("/")[-1]
    written = []
    for layer in args.layers:
        X = acts[layer]
        # save the full-data steerable direction (+ z-score stats) for the
        # causal re-elicitation stage; mean_diff is the injectable CAA vector
        w, mu, sd = _probe_direction(X, labels)
        np.savez(out_dir / f"vectors.{model_tag}.L{layer}.npz",
                 probe=w.astype(np.float32),
                 zscore_mu=mu.astype(np.float32), zscore_sd=sd.astype(np.float32),
                 selected_layer=np.array(layer, dtype=np.int32),
                 layer_activations=X)  # kept local (gitignored) for the causal stage
        for probe_class in args.probes:
            scores = lofo_scores(X, labels, families, probe_class, args.seed)
            bundle = build_bundle_for_layer(
                args.model, args.revision, layer, probe_class, texts, labels,
                families, scores, domain, args.encoding)
            bundle.validate()
            path = out_dir / f"bundle.{model_tag}.L{layer}.{probe_class}.json"
            bundle.save(path)
            written.append(str(path))
            print(f"[build-bundle] {model_tag} L{layer} {probe_class} -> {path.name}")

    (out_dir / f"bundles.{model_tag}.index.json").write_text(
        json.dumps({"model": args.model, "revision": args.revision,
                    "model_tag": model_tag, "layers": args.layers,
                    "probes": list(args.probes), "encoding": args.encoding,
                    "bundles": written}, indent=2))
    print(f"[build-bundle] wrote {len(written)} bundles for {model_tag}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-bundle",
                       help="extract acts + LOFO probe -> decodability bundles")
    p.add_argument("--model", required=True, help="HF id or local path")
    p.add_argument("--model-tag", default=None,
                   help="short tag for filenames (default: last path segment)")
    p.add_argument("--revision", default=None, help="pin the model revision")
    p.add_argument("--statements", type=Path, required=True,
                   help="decode_statements JSONL: {text,label,family}")
    p.add_argument("--layers", type=int, nargs="+", required=True,
                   help="PREREGISTERED layer set (no auto-select by design)")
    p.add_argument("--probes", nargs="+", default=list(PROBE_CLASSES),
                   choices=list(PROBE_CLASSES))
    p.add_argument("--domain", default="wmdp-bio")
    p.add_argument("--encoding", choices=["raw", "chat"], default="raw",
                   help="raw: last token = last subword of the claim (default; "
                        "truth-probe convention). chat: wrap as a user turn.")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-labels", action="store_true",
                   help="negative control: permute labels within families to "
                        "sever text<->correctness (must yield not_decodable)")
    p.add_argument("--reuse-activations", default=None,
                   help="dir of vectors.<reuse-tag>.L<layer>.npz to rebuild "
                        "bundles from (skips model load/extraction)")
    p.add_argument("--reuse-tag", default=None,
                   help="model_tag of the saved activations to reuse")
    p.add_argument("--out-dir", default="runs")
    p.set_defaults(func=cmd_build_bundle)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
