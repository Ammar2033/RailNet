"""BF16 bit helpers (delegates to the proven compiler module)."""
import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(
        name, str(_HERE / fname)
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RN = _load("rn_compiler_core", "04_bf16_learned_basis.py")

bf16_bits_to_float32 = RN.bf16_bits_to_float32
bf16_array_to_float32 = RN.bf16_array_to_float32
float32_to_bf16_bits = RN.float32_to_bf16_bits
fp32_array_to_bf16_bits = RN.fp32_array_to_bf16_bits
bf16_bitwise_equal = RN.bf16_bitwise_equal
