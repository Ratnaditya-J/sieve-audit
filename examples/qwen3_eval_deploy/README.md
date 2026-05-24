# Example: Qwen3-32B eval/deploy (L55)

Flagship demo (planned). Reproduces the canonical SIEVE audit on the eval/deploy
direction at Qwen3-32B L55: linearly decodable (AUROC 0.917), but **not causally
sufficient** under matched random / orthogonal / wrong-layer controls.

Companion demos:
- gpt-oss-120b **L15** -> effective, probe-direction null (corroborating).
- gpt-oss-120b **L34** -> intervention ineffective -> inconclusive (shows the efficacy gate).

The generated audit card will be written to [`../../reports/`](../../reports/).
