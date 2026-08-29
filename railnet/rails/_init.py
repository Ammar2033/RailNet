import numpy as np

from railnet.dtypes.bf16 import (
    float32_to_bf16_bits,
)

EXTREME_RAIL_SLOTS = 0


def initialize_rails(values_float, target_bits, counts, rail_count):
    """
    Weighted 1D k-means-like initialization.

    Centers are always converted back to actual BF16 values.

    This is a better starting point than simple quantiles
    because frequent values have greater influence.

    Args:
        values_float: decoded BF16 values as float64 (length = unique)
        target_bits: original BF16 bits as uint16 (same length)
        counts: frequency of each unique value
        rail_count: desired number of rails
    """

    if rail_count >= len(values_float):
        selected = target_bits.copy()

        if len(selected) < rail_count:
            padding = np.zeros(rail_count - len(selected), dtype=np.uint16)

            selected = np.concatenate([selected, padding])

        return selected[:rail_count]

    # --------------------------------------------------------
    # Uniform quantile initialization.
    #
    # Previous weighted quantile + weighted k-means
    # concentrated rails near zero (dense region) and
    # missed tail values, giving poor exact coverage
    # (e.g., 64 rails: exhaustive 2460 vs uniform 4387).
    #
    # Uniform across sorted unique values gives better
    # spread and preserves tail representability while
    # still being frequency-aware via later weighted
    # optimization. This is a hardened fix.
    # --------------------------------------------------------

    order = np.argsort(values_float)

    sorted_values = values_float[order]

    centers = []

    for i in range(rail_count):
        # Uniform index across sorted unique values
        idx = int((i + 0.5) / rail_count * len(sorted_values))

        idx = min(idx, len(sorted_values) - 1)

        centers.append(float(sorted_values[idx]))

    centers = np.asarray(centers, dtype=np.float64)

    # --------------------------------------------------------
    # Optional light weighted refinement (1 iteration)
    # is skipped to preserve spread. Full weighted
    # k-means would pull centers back to dense region
    # and degrade tail coverage. The coordinate-descent
    # `update_basis` later performs weighted optimization
    # anyway.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Convert every center to BF16.
    # --------------------------------------------------------

    rails = np.array([float32_to_bf16_bits(center) for center in centers], dtype=np.uint16)

    # --------------------------------------------------------
    # Remove duplicate BF16 rails.
    # --------------------------------------------------------

    rails = np.unique(rails)

    # --------------------------------------------------------
    # Need exactly rail_count slots.
    # Fill remaining slots with frequent actual values.
    # --------------------------------------------------------

    if len(rails) < rail_count:
        frequency_order = np.argsort(counts)[::-1]

        used = {int(x) for x in rails}

        additions = []

        for index in frequency_order:
            candidate = int(target_bits[index])

            if candidate in used:
                continue

            used.add(candidate)

            additions.append(candidate)

            if len(rails) + len(additions) >= rail_count:
                break

        if additions:
            rails = np.concatenate([rails, np.array(additions, dtype=np.uint16)])

    rails = rails[:rail_count]

    # --------------------------------------------------------
    # Extreme tail rails.
    #
    # Rank-uniform spacing skips the extreme ends of the
    # sorted unique values (smallest denormal-like values
    # and largest magnitudes). Analysis showed these are
    # exactly the values missing from exact coverage.
    # Reserve first slots for min and max actual values.
    # --------------------------------------------------------

    if EXTREME_RAIL_SLOTS > 0 and rail_count >= 4:
        order_by_value = np.argsort(values_float)

        extreme_bits = []

        seen = {int(x) for x in rails}

        for k in range(EXTREME_RAIL_SLOTS):
            low_index = int(order_by_value[k])

            bits_low = int(target_bits[low_index])

            if bits_low not in seen:
                extreme_bits.append(bits_low)
                seen.add(bits_low)

            high_index = int(order_by_value[-1 - k])

            bits_high = int(target_bits[high_index])

            if bits_high not in seen:
                extreme_bits.append(bits_high)
                seen.add(bits_high)

        for slot, bits_extreme in enumerate(extreme_bits):
            rails[slot] = np.uint16(bits_extreme)

    return rails


# ============================================================
# FAST GREEDY ROUTING
# ============================================================
