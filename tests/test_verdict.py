from sieve_audit import AuditCard, Verdict, __version__


def test_version():
    assert __version__


def test_five_verdict_states():
    assert len(list(Verdict)) == 5


def test_audit_card_minimal():
    card = AuditCard(
        model="Qwen/Qwen3-32B",
        revision=None,
        layers=[55],
        direction_source="mean_diff_zscored",
        prompt_distribution="demo",
        prompt_license="open",
        n_prompts=100,
        alpha_grid=[-20.0, 0.0, 20.0],
        behavioral_metrics=["refusal", "sandbagging", "hedging"],
        judges=["judge_a", "judge_b"],
        controls=["random", "orthogonal", "wrong_layer"],
        seed=0,
    )
    assert card.protocol_version == "0.1"
    assert card.verdict is None  # not run yet
