"""Analytical FPGA datapath model — sanity of the resource trade."""

from railnet.analysis import match_dense_throughput
from railnet.analysis.fpga import dense_tile, rail_tile


def test_matched_throughput_trades_dsp_for_fabric():
    m = match_dense_throughput(out_f=1024, in_f=1152, rail_count=96, avg_terms=2.3, dense_dsps=64)
    assert m["rail"]["cycles_per_token"] <= m["target_cycles_per_token"]
    assert m["ratios"]["dsp"] < 0.2  # far fewer DSPs
    assert m["ratios"]["lut"] > 1.5  # more fabric
    assert m["rail"]["adders"] > m["dense"]["dsps"]


def test_dsp_ratio_is_roughly_scale_invariant():
    ratios = [
        match_dense_throughput(1024, 1152, 96, 2.3, d)["ratios"]["dsp"] for d in (16, 64, 256)
    ]
    assert max(ratios) - min(ratios) < 0.05


def test_rail_stage_b_scales_with_rail_count():
    a = rail_tile(512, 512, 32, 2.0, adders=200, mul_dsps=8)
    b = rail_tile(512, 512, 96, 2.0, adders=200, mul_dsps=8)
    assert b["stage_b_cycles"] > a["stage_b_cycles"]


def test_dense_cycles_scale_inverse_with_dsps():
    assert (
        dense_tile(1024, 1152, 32)["cycles_per_token"]
        == 2 * dense_tile(1024, 1152, 64)["cycles_per_token"]
    )
