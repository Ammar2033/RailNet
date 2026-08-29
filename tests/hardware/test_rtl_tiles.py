"""Run the hardware/rtl/ Amaranth functional checks under pytest.

Skipped unless the `[rtl]` extra (amaranth) is installed.
"""

import pytest

pytest.importorskip("amaranth")

from hardware.rtl.test_tiles import (
    test_dense_inner_matches_dot,
    test_stage_a_bram_matches_golden,
    test_stage_a_bram_matches_reg_on_many,
    test_stage_a_reg_matches_golden,
)
