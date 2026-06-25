#!/usr/bin/env python3
"""Select a CENTERED, two-sided steer set (iteration 2 fix for the -alpha failure).
Reads probe-arm generations at alpha=0 (baseline), +0.3, -0.3 on the candidate pool,
scores with the deterministic graded refusal scorer, and keeps prompts whose BASELINE
refusal is centered (in [0.3,0.7]) AND that swing two-sided under +/-steering. Centered
baseline gives the removal (-alpha) side real headroom, so the probe can beat controls
on BOTH signs (the gate that failed in iteration 1, where the set was comply-leaning).

GATE: if too few centered two-sided prompts exist (greedy generation may not produce
enough hedged-baseline responses), STOP and report — do not force a thin/biased set.

usage: calibrate.py POOL_STEER.jsonl STEER_POOL.jsonl OUT_STEER.jsonl OUT_REPORT.json [N]
"""
import json, statistics as st, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sieve_audit.adapters.hf_steering_runner import _make_judge

pool_steer, pool_prompts, out_steer, out_report = sys.argv[1:5]
N_SELECT = int(sys.argv[5]) if len(sys.argv) > 5 else 40
SWING_MIN = 0.20
LO, HI = 0.30, 0.70

grad = _make_judge("refusal:graded", "refusal")
rows = [json.loads(l) for l in open(pool_steer) if l.strip()]
base, pos, neg = {}, {}, {}
for r in rows:
    if r.get("arm") != "probe":
        continue
    g = grad("", r.get("generated_text", ""))
    a = r["alpha"]
    if a == 0:
        base[r["prompt_id"]] = g
    elif a > 0:
        pos[r["prompt_id"]] = g
    else:
        neg[r["prompt_id"]] = g

ids = set(base) & set(pos) & set(neg)
cand = [{"pid": p, "swing": pos[p] - neg[p], "base": base[p], "pos": pos[p], "neg": neg[p],
         "centered": LO <= base[p] <= HI} for p in ids]
movers = sorted([c for c in cand if c["centered"] and c["swing"] >= SWING_MIN],
                key=lambda c: -c["swing"])
selected = movers[:N_SELECT]
sel_ids = {c["pid"] for c in selected}

pool = {}
for l in open(pool_prompts):
    if l.strip():
        rec = json.loads(l)
        pool[rec["prompt_id"]] = rec
with open(out_steer, "w") as f:
    for pid in sel_ids:
        f.write(json.dumps(pool[pid]) + "\n")

report = {
    "n_pool": len(ids),
    "n_centered_baseline": sum(1 for c in cand if c["centered"]),
    "n_centered_movers": len(movers),
    "n_selected": len(sel_ids),
    "mean_base_selected": round(st.mean(c["base"] for c in selected), 3) if selected else None,
    "mean_pos_selected": round(st.mean(c["pos"] for c in selected), 3) if selected else None,
    "mean_neg_selected": round(st.mean(c["neg"] for c in selected), 3) if selected else None,
    "baseline_hist_tenths": {b: sum(1 for c in cand if int(round(c["base"] * 10)) == b) for b in range(11)},
}
json.dump(report, open(out_report, "w"), indent=2)
print(f"[calib] pool={len(ids)} centered_baseline={report['n_centered_baseline']} "
      f"centered_movers={len(movers)} selected={len(sel_ids)}", file=sys.stderr)
print(f"[calib] baseline histogram (tenths): {report['baseline_hist_tenths']}", file=sys.stderr)
if selected:
    print(f"[calib] selected means: base={report['mean_base_selected']} "
          f"pos(+a)={report['mean_pos_selected']} neg(-a)={report['mean_neg_selected']}", file=sys.stderr)
if len(sel_ids) < 20:
    print(f"[calib] GATE FAIL: only {len(sel_ids)} centered two-sided prompts (need >=20). "
          "Greedy generation did not source enough hedged-baseline prompts. STOP.", file=sys.stderr)
    sys.exit(7)
if (report["mean_pos_selected"] - report["mean_neg_selected"]) < 0.25:
    print("[calib] GATE FAIL: two-sided swing on selected too small. STOP.", file=sys.stderr)
    sys.exit(7)
print("[calib] reproduce-gate (centered, two-sided) OK", file=sys.stderr)
