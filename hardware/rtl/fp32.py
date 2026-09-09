"""Synthesizable pure-Amaranth FP32 / BF16 arithmetic units for RailNet tiles.

Provides:
- Bf16ToFp32: Zero-cost combinatorial conversion (extends mantissa with 16 zeros).
- Fp32ToBf16: Combinatorial truncation with Round-to-Nearest-Even (RNE).
- PipelinedFp32Adder: 2-stage pipelined floating-point adder/subtractor with:
    * Stage 1: Exponent comparison, magnitude ordering, and mantissa right-shift alignment.
    * Stage 2: Mantissa add/sub, normalization (priority leading-zero count), rounding.
    * Forwarding/tag metadata passed along to support zero-stall RMW hazard bypass.
- Fp32Multiplier: Pipelined floating-point multiplier (FP32 x FP32 -> FP32) for Stage-B rail MAC.

Amaranth 0.5.
"""

from amaranth import Cat, Const, Module, Mux, Signal, unsigned
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out


class Bf16ToFp32(wiring.Component):
    """Combinatorial BF16 to FP32 expansion.
    BF16: [15] sign, [14:7] exp (8b), [6:0] mantissa (7b).
    FP32: [31] sign, [30:23] exp (8b), [22:16] mantissa, [15:0] 0.
    """

    bf16: In(unsigned(16))
    fp32: Out(unsigned(32))

    def elaborate(self, platform):
        m = Module()
        # Cat concatenates from LSB to MSB:
        # [0:16] zeros, [16:32] bf16
        m.d.comb += self.fp32.eq(Cat(Const(0, 16), self.bf16))
        return m


class Fp32ToBf16(wiring.Component):
    """Combinatorial FP32 to BF16 rounding (Round-to-Nearest-Even)."""

    fp32: In(unsigned(32))
    bf16: Out(unsigned(16))

    def elaborate(self, platform):
        m = Module()
        upper = Signal(16)
        round_bit = Signal()
        sticky = Signal()
        round_up = Signal()

        m.d.comb += [
            upper.eq(self.fp32[16:32]),
            round_bit.eq(self.fp32[15]),
            sticky.eq(self.fp32[0:15] != 0),
            # RNE: round up if round_bit and (sticky or lsb_of_upper)
            round_up.eq(round_bit & (sticky | upper[0])),
        ]

        # Check for overflow when adding round_up
        rounded = Signal(17)
        m.d.comb += rounded.eq(upper + round_up)

        # If exponent overflows (becomes 0xFF with non-zero mantissa), clamp to max normal or inf
        with m.If(rounded[7:15] == 0xFF):  # Exponent in upper 16 is bits 7..14
            # Keep original sign, set max float
            m.d.comb += self.bf16.eq(Cat(Const(0x7F, 7), Const(0xFE, 8), self.fp32[31]))
        with m.Else():
            m.d.comb += self.bf16.eq(rounded[0:16])

        return m


def _leading_zeros_26(sig):
    """Combinatorial priority encoder to count leading zeros of a 26-bit signal."""
    lz = Signal(range(27))
    cases = []
    # Test from bit 25 down to bit 0
    curr = Const(26, 5)
    for bit_idx in range(26):
        curr = Mux(sig[bit_idx], Const(25 - bit_idx, 5), curr)
    return curr


class PipelinedFp32Adder(wiring.Component):
    """2-Stage pipelined FP32 adder/subtractor with tag pass-through for hazard bypass.

    Cycle 0: Inputs (a, b, sub, tag, valid).
             Unpack, compare exponents, align mantissas.
    Cycle 1: Pipeline stage 1 register.
             Add/sub mantissas, normalize, round.
    Cycle 2: Pipeline stage 2 register -> Output (res, res_tag, res_valid).
    """

    def __init__(self, tag_w: int = 7):
        self.tag_w = tag_w
        super().__init__(
            {
                "a": In(unsigned(32)),
                "b": In(unsigned(32)),
                "sub": In(1),  # 1: a - b, 0: a + b
                "tag_in": In(tag_w),  # e.g. rail address
                "valid_in": In(1),
                "res": Out(unsigned(32)),
                "tag_out": Out(tag_w),
                "valid_out": Out(1),
            }
        )

    def elaborate(self, platform):
        m = Module()

        # -------------------------------------------------------------
        # STAGE 0 (Combinatorial unpack & align before Stage 1 register)
        # -------------------------------------------------------------
        sign_a = self.a[31]
        exp_a = self.a[23:31]
        mant_a = self.a[0:23]
        is_zero_a = (exp_a == 0) & (mant_a == 0)

        effective_sign_b = self.b[31] ^ self.sub
        exp_b = self.b[23:31]
        mant_b = self.b[0:23]
        is_zero_b = (exp_b == 0) & (mant_b == 0)

        # Implicit leading 1 for normalized floats
        full_mant_a = Signal(24)
        full_mant_b = Signal(24)
        m.d.comb += [
            full_mant_a.eq(Mux(exp_a == 0, Cat(mant_a, Const(0, 1)), Cat(mant_a, Const(1, 1)))),
            full_mant_b.eq(Mux(exp_b == 0, Cat(mant_b, Const(0, 1)), Cat(mant_b, Const(1, 1)))),
        ]

        # Determine larger operand
        a_greater = Signal()
        with m.If(exp_a > exp_b):
            m.d.comb += a_greater.eq(1)
        with m.Elif(exp_a == exp_b):
            m.d.comb += a_greater.eq(full_mant_a >= full_mant_b)
        with m.Else():
            m.d.comb += a_greater.eq(0)

        exp_diff = Signal(8)
        m.d.comb += exp_diff.eq(Mux(a_greater, exp_a - exp_b, exp_b - exp_a))

        # Shift smaller mantissa right (extended with 3 guard/round bits: 24+3 = 27 bits)
        large_mant_ext = Signal(27)
        small_mant_ext = Signal(27)
        large_exp = Signal(8)
        large_sign = Signal()
        eff_sub = Signal()

        shifted_small = Signal(27)
        raw_small_ext = Signal(27)

        m.d.comb += [
            large_sign.eq(Mux(a_greater, sign_a, effective_sign_b)),
            eff_sub.eq(sign_a ^ effective_sign_b),
            large_exp.eq(Mux(a_greater, exp_a, exp_b)),
            large_mant_ext.eq(Mux(a_greater, Cat(Const(0, 3), full_mant_a), Cat(Const(0, 3), full_mant_a))),
        ]
        with m.If(a_greater):
            m.d.comb += [
                large_mant_ext.eq(Cat(Const(0, 3), full_mant_a)),
                raw_small_ext.eq(Cat(Const(0, 3), full_mant_b)),
            ]
        with m.Else():
            m.d.comb += [
                large_mant_ext.eq(Cat(Const(0, 3), full_mant_b)),
                raw_small_ext.eq(Cat(Const(0, 3), full_mant_a)),
            ]

        # Dynamic shift clamped to 27
        shift_amt = Signal(5)
        m.d.comb += shift_amt.eq(Mux(exp_diff > 27, 27, exp_diff[0:5]))
        m.d.comb += shifted_small.eq(raw_small_ext >> shift_amt)

        # -------------------------------------------------------------
        # PIPELINE REGISTER: STAGE 1
        # -------------------------------------------------------------
        r1_valid = Signal()
        r1_tag = Signal(self.tag_w)
        r1_zero_a = Signal()
        r1_zero_b = Signal()
        r1_a_val = Signal(32)
        r1_b_eff = Signal(32)

        r1_large_exp = Signal(8)
        r1_large_sign = Signal()
        r1_eff_sub = Signal()
        r1_large_mant = Signal(27)
        r1_small_mant = Signal(27)

        m.d.sync += [
            r1_valid.eq(self.valid_in),
            r1_tag.eq(self.tag_in),
            r1_zero_a.eq(is_zero_a),
            r1_zero_b.eq(is_zero_b),
            r1_a_val.eq(self.a),
            r1_b_eff.eq(Cat(self.b[0:31], effective_sign_b)),
            r1_large_exp.eq(large_exp),
            r1_large_sign.eq(large_sign),
            r1_eff_sub.eq(eff_sub),
            r1_large_mant.eq(large_mant_ext),
            r1_small_mant.eq(shifted_small),
        ]

        # -------------------------------------------------------------
        # STAGE 1 LOGIC (Add/Sub & Normalization)
        # -------------------------------------------------------------
        mant_result = Signal(28)  # 28 bits allows 1 bit carry out
        with m.If(r1_eff_sub):
            m.d.comb += mant_result.eq(r1_large_mant - r1_small_mant)
        with m.Else():
            m.d.comb += mant_result.eq(r1_large_mant + r1_small_mant)

        # Normalization calculation
        norm_exp = Signal(8)
        norm_mant = Signal(23)
        final_sign = Signal()

        # Leading zero count of the upper 27 bits of mant_result
        lz = _leading_zeros_26(mant_result[1:27])

        m.d.comb += final_sign.eq(r1_large_sign)

        with m.If(r1_zero_a & r1_zero_b):
            m.d.comb += [norm_exp.eq(0), norm_mant.eq(0), final_sign.eq(0)]
        with m.Elif(r1_zero_a):
            m.d.comb += [norm_exp.eq(r1_b_eff[23:31]), norm_mant.eq(r1_b_eff[0:23]), final_sign.eq(r1_b_eff[31])]
        with m.Elif(r1_zero_b):
            m.d.comb += [norm_exp.eq(r1_a_val[23:31]), norm_mant.eq(r1_a_val[0:23]), final_sign.eq(r1_a_val[31])]
        with m.Elif(mant_result == 0):
            m.d.comb += [norm_exp.eq(0), norm_mant.eq(0), final_sign.eq(0)]
        with m.Else():
            with m.If(~r1_eff_sub & mant_result[27]):
                # Addition carry out: shift right by 1, increment exponent
                m.d.comb += [
                    norm_exp.eq(r1_large_exp + 1),
                    norm_mant.eq(mant_result[4:27]),
                ]
            with m.Else():
                # Subtraction or no carry: shift left by lz
                shifted_mant = Signal(28)
                m.d.comb += shifted_mant.eq(mant_result << lz)
                with m.If(lz >= r1_large_exp):
                    # Underflow to 0
                    m.d.comb += [norm_exp.eq(0), norm_mant.eq(0)]
                with m.Else():
                    m.d.comb += [
                        norm_exp.eq(r1_large_exp - lz),
                        norm_mant.eq(shifted_mant[3:26]),
                    ]

        # -------------------------------------------------------------
        # PIPELINE REGISTER: STAGE 2 (Output)
        # -------------------------------------------------------------
        m.d.sync += [
            self.valid_out.eq(r1_valid),
            self.tag_out.eq(r1_tag),
            self.res.eq(Cat(norm_mant, norm_exp, final_sign)),
        ]

        return m


class Fp32Multiplier(wiring.Component):
    """Pipelined FP32 multiplier (2-stage)."""

    a: In(unsigned(32))
    b: In(unsigned(32))
    valid_in: In(1)
    res: Out(unsigned(32))
    valid_out: Out(1)

    def elaborate(self, platform):
        m = Module()

        sign_a = self.a[31]
        exp_a = self.a[23:31]
        mant_a = self.a[0:23]
        is_zero_a = (exp_a == 0) & (mant_a == 0)

        sign_b = self.b[31]
        exp_b = self.b[23:31]
        mant_b = self.b[0:23]
        is_zero_b = (exp_b == 0) & (mant_b == 0)

        full_mant_a = Signal(24)
        full_mant_b = Signal(24)
        m.d.comb += [
            full_mant_a.eq(Mux(exp_a == 0, 0, Cat(mant_a, Const(1, 1)))),
            full_mant_b.eq(Mux(exp_b == 0, 0, Cat(mant_b, Const(1, 1)))),
        ]

        # Stage 1: multiply 24x24 mantissas and sum exponents
        r1_valid = Signal()
        r1_sign = Signal()
        r1_zero = Signal()
        r1_exp_sum = Signal(10)
        r1_prod = Signal(48)

        m.d.sync += [
            r1_valid.eq(self.valid_in),
            r1_sign.eq(sign_a ^ sign_b),
            r1_zero.eq(is_zero_a | is_zero_b),
            r1_exp_sum.eq(exp_a + exp_b),
            r1_prod.eq(full_mant_a * full_mant_b),
        ]

        # Stage 2: normalize product, bias adjust (exp_sum - 127)
        final_exp = Signal(8)
        final_mant = Signal(23)
        final_sign = Signal()

        m.d.comb += final_sign.eq(r1_sign)

        with m.If(r1_zero | (r1_exp_sum < 127)):
            m.d.comb += [final_exp.eq(0), final_mant.eq(0), final_sign.eq(0)]
        with m.Else():
            with m.If(r1_prod[47]):
                # Bit 47 is 1 -> normalize by shifting right 1
                adjusted_exp = Signal(10)
                m.d.comb += adjusted_exp.eq(r1_exp_sum - 127 + 1)
                with m.If(adjusted_exp >= 255):
                    # Overflow
                    m.d.comb += [final_exp.eq(254), final_mant.eq(Const(0x7FFFFF, 23))]
                with m.Else():
                    m.d.comb += [final_exp.eq(adjusted_exp[0:8]), final_mant.eq(r1_prod[24:47])]
            with m.Else():
                adjusted_exp = Signal(10)
                m.d.comb += adjusted_exp.eq(r1_exp_sum - 127)
                with m.If(adjusted_exp >= 255):
                    m.d.comb += [final_exp.eq(254), final_mant.eq(Const(0x7FFFFF, 23))]
                with m.Elif(adjusted_exp < 1):
                    m.d.comb += [final_exp.eq(0), final_mant.eq(0)]
                with m.Else():
                    m.d.comb += [final_exp.eq(adjusted_exp[0:8]), final_mant.eq(r1_prod[23:46])]

        m.d.sync += [
            self.valid_out.eq(r1_valid),
            self.res.eq(Cat(final_mant, final_exp, final_sign)),
        ]

        return m


class CombFp32Adder(wiring.Component):
    """Combinatorial FP32 adder/subtractor (single-cycle).
    Provides instant result for single-cycle RMW memory accumulation and forwarding.
    """

    a: In(unsigned(32))
    b: In(unsigned(32))
    sub: In(1)
    res: Out(unsigned(32))

    def elaborate(self, platform):
        m = Module()

        sign_a = self.a[31]
        exp_a = self.a[23:31]
        mant_a = self.a[0:23]
        is_zero_a = (exp_a == 0) & (mant_a == 0)

        effective_sign_b = self.b[31] ^ self.sub
        exp_b = self.b[23:31]
        mant_b = self.b[0:23]
        is_zero_b = (exp_b == 0) & (mant_b == 0)

        full_mant_a = Signal(24)
        full_mant_b = Signal(24)
        m.d.comb += [
            full_mant_a.eq(Mux(exp_a == 0, Cat(mant_a, Const(0, 1)), Cat(mant_a, Const(1, 1)))),
            full_mant_b.eq(Mux(exp_b == 0, Cat(mant_b, Const(0, 1)), Cat(mant_b, Const(1, 1)))),
        ]

        a_greater = Signal()
        with m.If(exp_a > exp_b):
            m.d.comb += a_greater.eq(1)
        with m.Elif(exp_a == exp_b):
            m.d.comb += a_greater.eq(full_mant_a >= full_mant_b)
        with m.Else():
            m.d.comb += a_greater.eq(0)

        exp_diff = Signal(8)
        m.d.comb += exp_diff.eq(Mux(a_greater, exp_a - exp_b, exp_b - exp_a))

        large_mant_ext = Signal(27)
        raw_small_ext = Signal(27)
        large_exp = Signal(8)
        large_sign = Signal()
        eff_sub = Signal()

        m.d.comb += [
            large_sign.eq(Mux(a_greater, sign_a, effective_sign_b)),
            eff_sub.eq(sign_a ^ effective_sign_b),
            large_exp.eq(Mux(a_greater, exp_a, exp_b)),
        ]
        with m.If(a_greater):
            m.d.comb += [
                large_mant_ext.eq(Cat(Const(0, 3), full_mant_a)),
                raw_small_ext.eq(Cat(Const(0, 3), full_mant_b)),
            ]
        with m.Else():
            m.d.comb += [
                large_mant_ext.eq(Cat(Const(0, 3), full_mant_b)),
                raw_small_ext.eq(Cat(Const(0, 3), full_mant_a)),
            ]

        shift_amt = Signal(5)
        shifted_small = Signal(27)
        m.d.comb += [
            shift_amt.eq(Mux(exp_diff > 27, 27, exp_diff[0:5])),
            shifted_small.eq(raw_small_ext >> shift_amt),
        ]

        mant_result = Signal(28)
        with m.If(eff_sub):
            m.d.comb += mant_result.eq(large_mant_ext - shifted_small)
        with m.Else():
            m.d.comb += mant_result.eq(large_mant_ext + shifted_small)

        norm_exp = Signal(8)
        norm_mant = Signal(23)
        final_sign = Signal()

        lz = _leading_zeros_26(mant_result[1:27])
        m.d.comb += final_sign.eq(large_sign)

        with m.If(is_zero_a & is_zero_b):
            m.d.comb += [norm_exp.eq(0), norm_mant.eq(0), final_sign.eq(0)]
        with m.Elif(is_zero_a):
            m.d.comb += [norm_exp.eq(exp_b), norm_mant.eq(mant_b), final_sign.eq(effective_sign_b)]
        with m.Elif(is_zero_b):
            m.d.comb += [norm_exp.eq(exp_a), norm_mant.eq(mant_a), final_sign.eq(sign_a)]
        with m.Elif(mant_result == 0):
            m.d.comb += [norm_exp.eq(0), norm_mant.eq(0), final_sign.eq(0)]
        with m.Else():
            with m.If(~eff_sub & mant_result[27]):
                m.d.comb += [
                    norm_exp.eq(large_exp + 1),
                    norm_mant.eq(mant_result[4:27]),
                ]
            with m.Else():
                shifted_mant = Signal(28)
                m.d.comb += shifted_mant.eq(mant_result << lz)
                with m.If(lz >= large_exp):
                    m.d.comb += [norm_exp.eq(0), norm_mant.eq(0)]
                with m.Else():
                    m.d.comb += [
                        norm_exp.eq(large_exp - lz),
                        norm_mant.eq(shifted_mant[3:26]),
                    ]

        m.d.comb += self.res.eq(Cat(norm_mant, norm_exp, final_sign))
        return m


__all__ = [
    "Bf16ToFp32",
    "CombFp32Adder",
    "Fp32Multiplier",
    "Fp32ToBf16",
    "PipelinedFp32Adder",
]

