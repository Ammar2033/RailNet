"""Compute-cost model: multiplies drop, adds and total ops rise."""

from railnet.analysis import linear_compute


def test_multiply_reduction_is_one_minus_rails_over_in():
    c = linear_compute(out_f=1024, in_f=1152, rail_count=96, avg_terms=2.3)
    assert abs(c["mul_reduction"] - (1 - 96 / 1152)) < 1e-9
    assert c["mul_reduction"] > 0.9


def test_rail_does_more_adds_and_more_total_ops():
    c = linear_compute(out_f=1024, in_f=1152, rail_count=96, avg_terms=2.3)
    assert c["add_ratio"] > 2.0
    assert c["total_op_ratio"] > 1.0  # not a total-arithmetic win
    assert c["weight_bytes_ratio"] == 1.0  # route-id map == dense weight bytes


def test_tokens_scale_linearly():
    a = linear_compute(64, 48, 16, 2.0, tokens=1)
    b = linear_compute(64, 48, 16, 2.0, tokens=10)
    assert b["rail_ops"] == 10 * a["rail_ops"]
    assert b["dense_ops"] == 10 * a["dense_ops"]
