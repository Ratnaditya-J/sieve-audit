# First real necessity audit — refusal direction (Qwen2.5-1.5B)

Produced end-to-end with [`docs/necessity_runner.md`](../../docs/necessity_runner.md)
on a RunPod A100 (2026-06-20), then audited locally (GPU-free).

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`, layer 14, last-token contrastive direction
- **Direction:** refusal mean-diff — 30 AdvBench harmful vs 30 benign instructions
- **Eval:** 32 AdvBench harmful prompts; behavior = refusal; judges = 2 OpenRouter
  models (`openai/gpt-4o-mini`, `openai/gpt-4.1-mini`)
- **Arms:** baseline / probe-ablated / ablate_random (32 each)

## Result (necessity gate)

| | value |
|---|---|
| headline label | **necessary · sufficiency not established** (the necessity finding is surfaced, not buried) |
| machine verdict / status | `verdict=None` / `status=insufficient_protocol` (ablation-only bundle; no steering → no full causal verdict) |
| necessity | **necessary = True** (n_paired = 32) |
| probe-ablation refusal drop | **0.07 [0.011, 0.155]** (95% CI excludes 0) |
| probe drop − ablate_random drop | **0.068 [0.006, 0.147]** (95% CI excludes 0) |
| causal_summary | *single-method evidence only (sufficiency untested, necessity = necessary); cross-method agreement not established* |

**Reading:** ablating the refusal direction reduces refusal *more than* ablating a
random direction of equal norm — the direction is **causally necessary** under
single-layer ablation, matching the known result (Arditi et al. 2024). A modest
effect, as expected for single-layer ablation of a small model, but statistically
clean and directionally correct. SIEVE still returns `insufficient_protocol`
because the bundle carries only ablation evidence — it reports the necessity
finding without overclaiming a full causal verdict.

This is an end-to-end validation that the runner (`ablate` → `judge` →
`bundle-ablation`) and the necessity gate agree with ground truth. The bundle
(`ablation_bundle.json`) is text-free (arm / prompt_id / judge_scores only).
Re-audit: `sieve audit --bundle ablation_bundle.json`.
