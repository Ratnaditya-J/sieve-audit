#!/usr/bin/env python3
"""Reconstruct all prompt JSONL files from the existing bundle_all.json.

Writes into ./prompts/ (relative to this script):
  holdout.jsonl        — 120 {prompt_id,text,label,family} for decode-lofo + leakage
  steer.jsonl          — 25 {prompt_id,text,label}  harmful prompts for steer/ablate/multilayer
  ood.jsonl            — 120 {prompt_id,text,label} off-distribution decode
  patching_pairs.jsonl — 25 {prompt_id,clean_text,corrupt_text} for oracle patching

Run from this directory:  python prepare_prompts.py
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).parent
BUNDLE = HERE / "bundle_all.json"
OUT = HERE / "prompts"
OUT.mkdir(exist_ok=True)

b = json.loads(BUNDLE.read_text())
d = b["decodability"]
texts = d["texts"]
labels = d["labels"]
families = d["families"]

# ---- holdout (decode-lofo + leakage) ----
# The harmful instruction text itself is the surface giveaway (leak_span).
holdout = []
for i, (text, label, family) in enumerate(zip(texts, labels, families)):
    row: dict = {
        "prompt_id": f"h{i}",
        "text": text,
        "label": label,
        "family": family,
    }
    if label == 1:
        # leak_span = the text itself (full harmful instruction is the giveaway);
        # the leakage runner will remove it and rescore.
        row["leak_span"] = text
    holdout.append(row)
(OUT / "holdout.jsonl").write_text("\n".join(json.dumps(r) for r in holdout))
print(f"holdout: {len(holdout)} prompts -> {OUT}/holdout.jsonl")

# ---- steer / ablate / multilayer prompts ----
# First 25 harmful prompts, IDs e0-e24.
harmful = [(i, text) for i, (text, label) in enumerate(zip(texts, labels)) if label == 1]
steer = [{"prompt_id": f"e{j}", "text": text, "label": 1}
         for j, (_, text) in enumerate(harmful[:25])]
(OUT / "steer.jsonl").write_text("\n".join(json.dumps(r) for r in steer))
print(f"steer: {len(steer)} prompts -> {OUT}/steer.jsonl")

# ---- off-distribution decode ----
# Rephrase each prompt as "Question: <harmful>" (label=1) or "Answer: <benign>" (label=0).
# This is the "AdvBench/benign reworded (Question:...)" distribution from the original run.
benign = [text for text, label in zip(texts, labels) if label == 0]
ood = []
for i, (text, label, family) in enumerate(zip(texts, labels, families)):
    if label == 1:
        ood_text = f"Question: {text}"
    else:
        ood_text = f"How to: {text}"
    ood.append({"prompt_id": f"ood{i}", "text": ood_text, "label": label, "family": family})
(OUT / "ood.jsonl").write_text("\n".join(json.dumps(r) for r in ood))
print(f"ood: {len(ood)} prompts -> {OUT}/ood.jsonl")

# ---- patching pairs ----
# clean = benign version of the same slot, corrupt = harmful instruction.
benign_texts = [text for text, label in zip(texts, labels) if label == 0]
pairs = []
for j, (_, harmful_text) in enumerate(harmful[:25]):
    clean = benign_texts[j % len(benign_texts)]
    pairs.append({
        "prompt_id": f"e{j}",
        "clean_text": clean,
        "corrupt_text": harmful_text,
    })
(OUT / "patching_pairs.jsonl").write_text("\n".join(json.dumps(r) for r in pairs))
print(f"patching pairs: {len(pairs)} -> {OUT}/patching_pairs.jsonl")

print("done — all prompt files written to", OUT)
