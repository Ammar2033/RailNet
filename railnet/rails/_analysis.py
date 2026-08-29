import numpy as np

from railnet.dtypes.bf16 import (
    bf16_array_to_float32,
)


def analyze_unique_values(raw):

    unique_values, counts = np.unique(raw, return_counts=True)

    values_float = bf16_array_to_float32(unique_values).astype(np.float64)

    return (unique_values, counts.astype(np.float64), values_float)


# ============================================================
# WEIGHTED 1D K-MEANS INITIALIZATION
# ============================================================
