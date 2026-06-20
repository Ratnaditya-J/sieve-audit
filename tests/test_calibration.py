"""#3 calibration scaffold: the criterion's error rate against ground truth.

On the rigged scenarios (whose true verdict is known by construction), the
criterion must recover every label with zero false negatives — a measured,
reported error rate, not an assumed-correct criterion.
"""
from sieve_audit.calibration import run_calibration


def test_calibration_recovers_ground_truth():
    rep = run_calibration()
    assert rep.n >= 6
    assert rep.accuracy == 1.0
    assert rep.false_negatives == 0
    assert rep.mismatches == []


def test_calibration_report_serializes():
    d = run_calibration().to_dict()
    assert set(d) >= {"n", "accuracy", "false_negatives", "confusion", "mismatches"}
