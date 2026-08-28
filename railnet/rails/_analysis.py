import numpy as np
import math
from railnet.dtypes.bf16 import (
    bf16_bits_to_float32, bf16_array_to_float32, 
    float32_to_bf16_bits, fp32_array_to_bf16_bits, bf16_bitwise_equal
)

def analyze_unique_values(
    raw
):

    unique_values, counts = np.unique(
        raw,
        return_counts=True
    )

    values_float = (
        bf16_array_to_float32(
            unique_values
        ).astype(
            np.float64
        )
    )

    return (
        unique_values,
        counts.astype(
            np.float64
        ),
        values_float
    )


# ============================================================
# WEIGHTED 1D K-MEANS INITIALIZATION
# ============================================================


