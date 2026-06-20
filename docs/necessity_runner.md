# Producing real necessity (ablation) evidence on a GPU pod

The `hf_steering_runner` adapter now has an **ablation** path (#2): generate
baseline / probe-ablated / ablate-random outputs, judge them, and assemble an
`EvidenceBundle` whose **necessity gate** (`sieve audit`) tells you whether the
direction is *causally necessary* — the complement to the steering-sufficiency
test.

Recommended validation concept: **refusal.** The refusal direction is known to
be causally necessary for refusal (Arditi et al.), so a correct pipeline should
return **necessary** — a good end-to-end sanity check that the runner + SIEVE
agree with a known result. Bring your own prompts (a standard harmful/harmless
refusal set); this repo ships no prompt data.

## On the pod

```bash
# deps + keys
pip install "sieve-audit[runner] @ git+https://github.com/Ratnaditya-J/sieve-audit"
export HF_TOKEN=...                 # gated models (from ~/.cache/huggingface/token)
export OPENROUTER_API_KEY=...       # judging (from ~/.config/ramp/openrouter_key)

MODEL=Qwen/Qwen2.5-7B-Instruct      # small = cheap; any HF causal LM works
L=14                                 # a mid layer

# 1. direction (+ random control) from contrast prompts
#    contrast.jsonl rows: {"prompt_id","text","label"}  label 1=refuse-class, 0=benign
python -m sieve_audit.adapters.hf_steering_runner vectors \
  --model $MODEL --layer $L --contrast-prompts contrast.jsonl --out vecs.npz

# 2. correctness check (A1-A3), then the real generations
#    eval.jsonl rows: {"prompt_id","text"}  prompts that elicit refusal
python -m sieve_audit.adapters.hf_steering_runner ablate \
  --model $MODEL --layer $L --vectors vecs.npz --eval-prompts eval.jsonl --test
python -m sieve_audit.adapters.hf_steering_runner ablate \
  --model $MODEL --layer $L --vectors vecs.npz \
  --eval-prompts eval.jsonl --max-new-tokens 200 --out ablate_gen.jsonl

# 3. two independent judges (one OpenRouter key, two routes = two judges)
python -m sieve_audit.adapters.hf_steering_runner judge \
  --generations ablate_gen.jsonl --steer-prompts eval.jsonl --metric refusal \
  --judge openrouter:anthropic/claude-3.5-sonnet \
  --judge openrouter:openai/gpt-4.1-mini \
  --out ablate_judged.jsonl

# 4. assemble the ablation bundle (small JSON; scp it back)
python -m sieve_audit.adapters.hf_steering_runner bundle-ablation \
  --judged ablate_judged.jsonl --model $MODEL --layer $L \
  --direction-source "refusal mean-diff @L$L" \
  --prompt-distribution "your-refusal-set" --prompt-license "see source" \
  --metric refusal --out ablation_bundle.json
```

## Back on your laptop (GPU-free)

```bash
sieve audit --bundle ablation_bundle.json
```

The card's necessity block reports `necessary` / `not necessary` / `inconclusive`
with paired bootstrap CIs (probe-ablation drop vs the ablate-random control) and
per-judge direction agreement. Need ≥ ~30 eval prompts and 2 judges, or the gate
returns `inconclusive` (by design — absence of evidence is never a free verdict).

For a *full* card (decodability + sufficiency + necessity on one direction), also
run the `decode-lofo` and `steer`/`judge`/`bundle` steps, then merge the
`ablation` list into that bundle JSON before auditing.
