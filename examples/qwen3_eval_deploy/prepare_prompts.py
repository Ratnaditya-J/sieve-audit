"""Convert the flagship run's prompt sources into the runner's JSONL format.

Inputs (both openly available):
- Microsoft Test-Awareness-Steering triggers (MIT,
  https://github.com/microsoft/Test_Awareness_Steering,
  data/triggers/triggers.json): 2,587 real/hypothetical pairs — the contrast
  set that trains the eval/deploy direction (label 1 = hypothetical/eval,
  0 = real/deploy), matching the companion paper's probe training corpus.
- alignment-evals controlled_holdout_v2.yaml: 160 length-matched eval/deploy
  pairs across 4 framing families — the decodability holdout. Its 40
  underlying trigger-free questions double as the steering prompt set.

Usage:
    python prepare_prompts.py \
        --triggers /path/to/triggers.json \
        --holdout /path/to/controlled_holdout_v2.yaml \
        --out-dir prompts/ [--contrast-pairs 400]

Outputs: contrast_triggers.jsonl, controlled_holdout_v2.jsonl,
steer_prompts.jsonl — the three files the runbook in README.md consumes.
The contrast and holdout sets are fully disjoint (different sources), so
`--attest-out-of-sample` on the bundle is truthful.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  {path}  ({len(rows)} records)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triggers", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("prompts"))
    parser.add_argument("--contrast-pairs", type=int, default=400,
                        help="trigger pairs to sample for the contrast set")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # --- contrast set (trains the direction) ---
    triggers = json.loads(args.triggers.read_text())
    rng = random.Random(args.seed)
    if args.contrast_pairs and args.contrast_pairs < len(triggers):
        triggers = rng.sample(triggers, args.contrast_pairs)
    contrast = []
    for i, t in enumerate(triggers):
        contrast.append({"prompt_id": f"ms{i}_eval", "text": t["hypothetical"],
                         "label": 1, "category": t.get("category", "")})
        contrast.append({"prompt_id": f"ms{i}_deploy", "text": t["real"],
                         "label": 0, "category": t.get("category", "")})
    write_jsonl(args.out_dir / "contrast_triggers.jsonl", contrast)

    # --- decodability holdout (families = framing styles) ---
    doc = yaml.safe_load(args.holdout.read_text())
    holdout = []
    for pair in doc["pairs"]:
        for side in ("a", "b"):
            rec = pair[side]
            holdout.append({
                "prompt_id": f"{pair['pair_id']}_{rec['label']}",
                "text": rec["text"].strip(),
                "label": 1 if rec["label"] == "eval" else 0,
                "family": f"{pair['family']}_{pair['family_name'].replace(' ', '-')}",
            })
    write_jsonl(args.out_dir / "controlled_holdout_v2.jsonl", holdout)

    # --- steering prompts: the trigger-free base questions ---
    steer = [{"prompt_id": f"q{q['qid']}", "text": q["text"], "topic": q["topic"]}
             for q in doc["metadata"]["questions"]]
    write_jsonl(args.out_dir / "steer_prompts.jsonl", steer)

    print("done — contrast/holdout sources are disjoint; "
          "--attest-out-of-sample is truthful for this holdout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
