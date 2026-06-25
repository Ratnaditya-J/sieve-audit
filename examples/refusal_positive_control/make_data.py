#!/usr/bin/env python3
"""
Build the three prompt sets for the SIEVE refusal positive control from XSTest
(Rottger et al. 2024). XSTest is ideal here: its "safe" prompts are SURFACE-MATCHED
to genuinely "unsafe" ones (same trigger words), which handicaps the TF-IDF surface
baseline and gives the activation direction a fair shot at beating it under LOFO; and
its safe prompts are naturally borderline for refusal (models over-refuse them), the
candidate pool for the two-sided steer set.

Outputs (disjoint splits; steer/ablate prompts never train the direction):
  prompts/decode.jsonl      clear harmful(1)+harmless(0), family=XSTest type, both labels/family
  prompts/ablate.jsonl      clear harmful(1) for necessity (high baseline refusal)
  prompts/steer_pool.jsonl  borderline candidates (safe, over-refusal-prone) -> calibrate on GPU
Record schema matches the adapter: {prompt_id, text(JSON chat msgs), scenario, label, family}.
"""
import csv, io, json, random, sys, urllib.request
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "prompts"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0
URLS = [
    "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
    "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_v2_prompts.csv",
    "https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/main/xstest_v2_prompts.csv",
]


def fetch_csv():
    last = None
    for u in URLS:
        try:
            with urllib.request.urlopen(u, timeout=30) as r:
                txt = r.read().decode("utf-8")
            rows = list(csv.DictReader(io.StringIO(txt)))
            if rows:
                print(f"[data] fetched {len(rows)} rows from {u}", file=sys.stderr)
                return rows
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"[data] failed {u}: {e}", file=sys.stderr)
    raise SystemExit(f"could not fetch XSTest CSV: {last}")


# XSTest safe-type and unsafe-contrast-type names don't match 1:1; remap to
# shared semantic families so every family carries BOTH labels (LOFO needs it).
FAMILY_MAP = {
    "privacy_fictional": "privacy", "privacy_public": "privacy", "privacy": "privacy",
    "real_group_nons_discr": "discrimination", "nons_group_real_discr": "discrimination",
    "discr": "discrimination",
}


def to_rec(prompt_id, prompt, label, family):
    msgs = [{"role": "user", "content": prompt}]
    return {"prompt_id": prompt_id, "text": json.dumps(msgs),
            "scenario": prompt, "label": int(label), "family": family}


def main():
    rows = fetch_csv()
    # derive label + family from 'type' (contrast_* = unsafe); tolerate a 'label' col
    safe, unsafe = [], []
    for i, row in enumerate(rows):
        typ = (row.get("type") or "").strip()
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            continue
        lab_col = (row.get("label") or "").strip().lower()
        is_unsafe = lab_col == "unsafe" or typ.startswith("contrast")
        base = typ.replace("contrast_", "") or "misc"
        family = FAMILY_MAP.get(base, base)
        rec = to_rec(f"xs_{i}", prompt, 1 if is_unsafe else 0, family)
        (unsafe if is_unsafe else safe).append(rec)
    print(f"[data] safe(0)={len(safe)} unsafe(1)={len(unsafe)}", file=sys.stderr)

    rng = random.Random(SEED)
    rng.shuffle(safe); rng.shuffle(unsafe)

    # hold out disjoint steer-pool (safe) and ablate (unsafe) BEFORE decode
    steer_pool = safe[:120]                # borderline candidates -> calibrate to centered set on GPU
    decode_safe = safe[120:]
    ablate = unsafe[:50]                   # clear harmful for necessity
    decode_unsafe = unsafe[50:]
    decode = decode_safe + decode_unsafe
    rng.shuffle(decode)

    # sanity: per-family per-class counts in decode (need >=5 each; LOFO needs both labels)
    fam = defaultdict(lambda: [0, 0])
    for r in decode:
        fam[r["family"]][r["label"]] += 1
    weak = {f: c for f, c in fam.items() if min(c) < 5}
    print(f"[data] decode families (label0,label1): {dict(fam)}", file=sys.stderr)
    if weak:
        print(f"[data] WARNING families with <5 in a class (LOFO risk): {weak}", file=sys.stderr)

    for name, recs in [("decode", decode), ("ablate", ablate), ("steer_pool", steer_pool)]:
        p = OUT / f"{name}.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in recs))
        n1 = sum(r["label"] for r in recs)
        print(f"[data] wrote {p} n={len(recs)} (harmful={n1}, harmless={len(recs)-n1})", file=sys.stderr)
    print("[data] done", file=sys.stderr)


if __name__ == "__main__":
    main()
