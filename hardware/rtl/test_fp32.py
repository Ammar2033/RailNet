"""Test pure-Amaranth FP32/BF16 arithmetic units against Python float."""

import struct
import numpy as np
import pytest
from amaranth.sim import Simulator

from hardware.rtl.fp32 import Bf16ToFp32, Fp32Multiplier, Fp32ToBf16, PipelinedFp32Adder


def float_to_fp32_bits(f: float) -> int:
    return struct.unpack(">I", struct.pack(">f", float(f)))[0]


def fp32_bits_to_float(bits: int) -> float:
    return struct.unpack(">f", struct.pack(">I", bits & 0xFFFFFFFF))[0]


def float_to_bf16_bits(f: float) -> int:
    u32 = float_to_fp32_bits(f)
    return (u32 >> 16) & 0xFFFF


def bf16_bits_to_float(bits: int) -> float:
    u32 = (bits & 0xFFFF) << 16
    return fp32_bits_to_float(u32)


def test_bf16_to_fp32():
    dut = Bf16ToFp32()
    test_vals = [0.0, 1.0, -1.0, 0.5, -0.25, 3.140625, 128.0, -0.00390625]

    for val in test_vals:
        b16 = float_to_bf16_bits(val)
        expected_f = bf16_bits_to_float(b16)

        got_bits = {}

        async def tb(ctx):
            ctx.set(dut.bf16, b16)
            await ctx.delay(1e-9)
            got_bits["val"] = ctx.get(dut.fp32)

        sim = Simulator(dut)
        sim.add_testbench(tb)
        sim.run()

        got_f = fp32_bits_to_float(got_bits["val"])
        assert got_f == expected_f, f"Mismatch: expected {expected_f}, got {got_f}"


def test_fp32_to_bf16():
    dut = Fp32ToBf16()
    test_vals = [0.0, 1.0, -1.0, 0.5, -0.25, 3.1415926, 128.0]

    for val in test_vals:
        u32 = float_to_fp32_bits(val)
        # Expected BF16 with round to nearest even:
        rnd = (u32 >> 16) + (1 if ((u32 & 0x8000) and ((u32 & 0x7FFF) or (u32 & 0x10000))) else 0)
        expected_b16 = rnd & 0xFFFF

        got_bits = {}

        async def tb(ctx):
            ctx.set(dut.fp32, u32)
            await ctx.delay(1e-9)
            got_bits["val"] = ctx.get(dut.bf16)

        sim = Simulator(dut)
        sim.add_testbench(tb)
        sim.run()

        assert got_bits["val"] == expected_b16, f"For {val}: expected {hex(expected_b16)}, got {hex(got_bits['val'])}"


def test_pipelined_fp32_adder():
    dut = PipelinedFp32Adder(tag_w=7)

    test_pairs = [
        (1.0, 2.0, 0),      # 1 + 2 = 3
        (5.5, -2.5, 0),     # 5.5 + (-2.5) = 3.0
        (10.0, 3.0, 1),     # 10 - 3 = 7.0
        (0.0, 4.25, 0),     # 0 + 4.25 = 4.25
        (7.5, 0.0, 0),      # 7.5 + 0 = 7.5
        (1.0, -1.0, 0),     # 1 - 1 = 0
        (100.0, 0.03125, 0),# 100.03125
    ]

    results = []

    async def tb(ctx):
        for idx, (a_f, b_f, sub) in enumerate(test_pairs):
            ctx.set(dut.a, float_to_fp32_bits(a_f))
            ctx.set(dut.b, float_to_fp32_bits(b_f))
            ctx.set(dut.sub, sub)
            ctx.set(dut.tag_in, idx)
            ctx.set(dut.valid_in, 1)
            await ctx.tick()
            if ctx.get(dut.valid_out):
                results.append((ctx.get(dut.tag_out), fp32_bits_to_float(ctx.get(dut.res))))

        ctx.set(dut.valid_in, 0)
        # Flush remaining pipeline stages (2 stages)
        for _ in range(4):
            await ctx.tick()
            if ctx.get(dut.valid_out):
                results.append((ctx.get(dut.tag_out), fp32_bits_to_float(ctx.get(dut.res))))

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    for idx, (a_f, b_f, sub) in enumerate(test_pairs):
        expected = a_f - b_f if sub else a_f + b_f
        # Find matching tag
        matched = [res for tag, res in results if tag == idx]
        assert len(matched) == 1, f"Missing result for tag {idx}"
        got = matched[0]
        assert np.isclose(got, expected, atol=1e-5), f"Op {idx} ({a_f}, {b_f}, sub={sub}): expected {expected}, got {got}"


def test_fp32_multiplier():
    dut = Fp32Multiplier()

    test_pairs = [
        (2.0, 3.0),
        (-1.5, 4.0),
        (0.0, 10.0),
        (2.5, 0.5),
        (-3.0, -3.0),
    ]

    results = []

    async def tb(ctx):
        for a_f, b_f in test_pairs:
            ctx.set(dut.a, float_to_fp32_bits(a_f))
            ctx.set(dut.b, float_to_fp32_bits(b_f))
            ctx.set(dut.valid_in, 1)
            await ctx.tick()
            if ctx.get(dut.valid_out):
                results.append(fp32_bits_to_float(ctx.get(dut.res)))

        ctx.set(dut.valid_in, 0)
        for _ in range(3):
            await ctx.tick()
            if ctx.get(dut.valid_out):
                results.append(fp32_bits_to_float(ctx.get(dut.res)))

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    for (a_f, b_f), got in zip(test_pairs, results):
        expected = a_f * b_f
        assert np.isclose(got, expected, atol=1e-5), f"Mul ({a_f}, {b_f}): expected {expected}, got {got}"
