import json
import mmap
import struct
from pathlib import Path
from collections import Counter

import numpy as np


# ============================================================
# RailNet-1B
# BF16 TENSOR ANALYSIS
#
# IMPORTANT:
#   - Does NOT load the whole model.
#   - Reads raw BF16 bytes directly from Safetensors.
#   - Preserves exact 16-bit BF16 representation.
#
# Default tensor:
#
#   model.layers.0.mlp.up_proj.weight
#
# ~7.96M parameters
# ============================================================


MODEL_DIR = Path("model_data")
MODEL_FILE = MODEL_DIR / "model.safetensors"
CONFIG_FILE = MODEL_DIR / "config.json"


TARGET_TENSOR = (
    "model.layers.0.mlp.up_proj.weight"
)


# Number of most frequent BF16 values to show.
TOP_VALUES = 30


# ============================================================
# HUMAN SIZE
# ============================================================

def human_size(nbytes: int) -> str:

    value = float(nbytes)

    for unit in ["B", "KB", "MB", "GB", "TB"]:

        if value < 1024.0:

            return (
                f"{value:.2f} {unit}"
            )

        value /= 1024.0

    return (
        f"{value:.2f} PB"
    )


# ============================================================
# SAFETENSORS HEADER
# ============================================================

def read_safetensors_header():

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Model bulunamadı: {MODEL_FILE}"
        )

    with open(
        MODEL_FILE,
        "rb"
    ) as f:

        # ----------------------------------------------------
        # Safetensors format:
        #
        # first 8 bytes = header length uint64
        # ----------------------------------------------------

        header_len_bytes = f.read(8)

        if len(header_len_bytes) != 8:

            raise RuntimeError(
                "Geçersiz safetensors dosyası."
            )

        header_len = struct.unpack(
            "<Q",
            header_len_bytes
        )[0]

        header_bytes = f.read(
            header_len
        )

        if len(header_bytes) != header_len:

            raise RuntimeError(
                "Safetensors header eksik."
            )

    header = json.loads(
        header_bytes.decode(
            "utf-8"
        )
    )

    return (
        header,
        8 + header_len
    )


# ============================================================
# TENSOR METADATA
# ============================================================

def get_tensor_metadata(
    header,
    tensor_name
):

    if tensor_name not in header:

        raise KeyError(
            f"Tensor bulunamadı: {tensor_name}"
        )

    metadata = header[
        tensor_name
    ]

    if not isinstance(
        metadata,
        dict
    ):

        raise RuntimeError(
            f"Geçersiz tensor metadata: "
            f"{tensor_name}"
        )

    return metadata


# ============================================================
# BF16 -> FP32
# ============================================================

def bf16_to_float32(
    raw_uint16
):

    """
    Exact BF16 bit pattern → FP32 numerical value.

    BF16 stores the upper 16 bits of IEEE FP32.

    Example:

        BF16 bits:
            ABCD

        FP32 bits:
            ABCD0000
    """

    raw_uint16 = np.asarray(
        raw_uint16,
        dtype=np.uint16
    )

    fp32_bits = (
        raw_uint16.astype(
            np.uint32
        )
        << 16
    )

    return fp32_bits.view(
        np.float32
    )


# ============================================================
# READ RAW BF16 TENSOR
# ============================================================

def read_bf16_tensor(
    metadata,
    data_base_offset
):
    dtype = metadata.get("dtype")

    if dtype != "BF16":
        raise TypeError(
            f"Beklenen dtype BF16, gelen: {dtype}"
        )

    shape = tuple(
        int(x)
        for x in metadata["shape"]
    )

    offsets = metadata["data_offsets"]

    if not isinstance(offsets, list) or len(offsets) != 2:
        raise RuntimeError(
            "Geçersiz data_offsets."
        )

    start, end = (
        int(offsets[0]),
        int(offsets[1])
    )

    byte_length = end - start

    expected_elements = 1
    for dim in shape:
        expected_elements *= dim

    expected_bytes = expected_elements * 2

    if byte_length != expected_bytes:
        raise RuntimeError(
            "Tensor byte size uyuşmuyor.\n"
            f"Expected: {expected_bytes}\n"
            f"Actual  : {byte_length}"
        )

    absolute_start = (
        data_base_offset + start
    )

    # --------------------------------------------------------
    # mmap yaşam süresini sadece raw byte kopyalama ile
    # sınırlandırıyoruz.
    # --------------------------------------------------------

    with open(MODEL_FILE, "rb") as f:
        with mmap.mmap(
            f.fileno(),
            length=0,
            access=mmap.ACCESS_READ
        ) as mm:

            raw_bytes = mm[
                absolute_start:
                absolute_start + byte_length
            ]

    # mmap artık kapalı.
    # NumPy dizisi bağımsız memory'de oluşturuluyor.
    values = np.frombuffer(
        raw_bytes,
        dtype=np.uint16
    ).copy()

    return (
        values,
        shape,
        byte_length
    )
# ============================================================
# TOP VALUE ANALYSIS
# ============================================================

def top_bf16_values(
    raw,
    top_n=TOP_VALUES
):

    unique_values, counts = (
        np.unique(
            raw,
            return_counts=True
        )
    )

    order = np.argsort(
        counts
    )[::-1]

    result = []

    for index in order[
        :top_n
    ]:

        bits = int(
            unique_values[
                index
            ]
        )

        count = int(
            counts[
                index
            ]
        )

        result.append(
            (
                bits,
                count
            )
        )

    return (
        unique_values,
        counts,
        result
    )


# ============================================================
# SIGN / EXPONENT / MANTISSA ANALYSIS
# ============================================================

def analyze_bf16_bits(
    raw
):

    raw32 = (
        raw.astype(
            np.uint32
        )
        << 16
    )

    sign = (
        raw32
        >> 31
    ) & 1

    exponent = (
        raw32
        >> 23
    ) & 0xFF

    mantissa = (
        raw32
        &
        0x7FFFFF
    )

    return {
        "negative_count":
            int(
                np.count_nonzero(
                    sign
                )
            ),

        "positive_count":
            int(
                np.count_nonzero(
                    sign == 0
                )
            ),

        "zero_count":
            int(
                np.count_nonzero(
                    raw == 0
                )
            ),

        "nonzero_count":
            int(
                np.count_nonzero(
                    raw != 0
                )
            ),

        "unique_count":
            int(
                len(
                    np.unique(
                        raw
                    )
                )
            ),

        "exponent_unique":
            int(
                len(
                    np.unique(
                        exponent
                    )
                )
            ),

        "mantissa_unique":
            int(
                len(
                    np.unique(
                        mantissa
                    )
                )
            ),
    }


# ============================================================
# RAIL CAPACITY ESTIMATES
# ============================================================

def estimate_single_rail_exact_capacity(
    unique_count
):

    """
    If one weight must map to exactly one rail,
    then at least unique_count distinct rails are
    required for exact representation of these values.
    """

    return unique_count


def estimate_multi_rail_binary_capacity(
    rail_count,
    max_terms
):

    """
    Upper bound on number of signed combinations.

    This is NOT the achievable number after collision/
    cancellation constraints. It is only a combinatorial
    upper-bound indicator.

    Each rail:
        {-1, 0, +1}

    Maximum combinations:
        3^N

    With at most K active rails:
        Σ C(N,k)*2^k
    """

    total = 1

    for k in range(
        1,
        max_terms + 1
    ):

        if k > rail_count:
            break

        # C(n,k)
        combinations = (
            math_comb(
                rail_count,
                k
            )
        )

        total += (
            combinations
            *
            (2 ** k)
        )

    return total


def math_comb(
    n,
    k
):

    if k < 0 or k > n:

        return 0

    k = min(
        k,
        n - k
    )

    result = 1

    for i in range(
        1,
        k + 1
    ):

        result = (
            result
            *
            (n - k + i)
            //
            i
        )

    return result


# ============================================================
# CONFIG ANALYSIS
# ============================================================

def inspect_config():

    if not CONFIG_FILE.exists():

        print()
        print(
            "config.json bulunamadı."
        )

        return

    print()
    print(
        "MODEL CONFIG"
    )

    print(
        "-" * 80
    )

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        config = json.load(
            f
        )

    keys = [
        "model_type",
        "architectures",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
        "torch_dtype",
        "dtype",
    ]

    for key in keys:

        if key in config:

            print(
                f"{key:30s}: "
                f"{config[key]}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "RAILNET-1B BF16 TENSOR ANALYSIS"
    )
    print("=" * 80)

    print()
    print(
        f"Target tensor:"
    )

    print(
        f"  {TARGET_TENSOR}"
    )

    print()

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    header, data_base_offset = (
        read_safetensors_header()
    )

    metadata = get_tensor_metadata(
        header,
        TARGET_TENSOR
    )

    print(
        "TENSOR METADATA"
    )

    print(
        "-" * 80
    )

    print(
        f"dtype       : "
        f"{metadata['dtype']}"
    )

    print(
        f"shape       : "
        f"{tuple(metadata['shape'])}"
    )

    number_of_elements = 1

    for dimension in metadata[
        "shape"
    ]:

        number_of_elements *= int(
            dimension
        )

    print(
        f"parameters  : "
        f"{number_of_elements:,}"
    )

    offsets = metadata[
        "data_offsets"
    ]

    byte_size = (
        int(offsets[1])
        -
        int(offsets[0])
    )

    print(
        f"raw size    : "
        f"{human_size(byte_size)}"
    )

    # --------------------------------------------------------
    # Read raw BF16
    # --------------------------------------------------------

    print()
    print(
        "READING RAW BF16"
    )

    print(
        "-" * 80
    )

    start_time = (
        __import__(
            "time"
        ).perf_counter()
    )

    raw, shape, byte_size = (
        read_bf16_tensor(
            metadata,
            data_base_offset
        )
    )

    elapsed = (
        __import__(
            "time"
        ).perf_counter()
        -
        start_time
    )

    print(
        f"Read time   : "
        f"{elapsed:.6f}s"
    )

    print(
        f"Elements    : "
        f"{len(raw):,}"
    )

    # --------------------------------------------------------
    # Exact BF16 -> FP32
    # --------------------------------------------------------

    values = bf16_to_float32(
        raw
    )

    # --------------------------------------------------------
    # Bit analysis
    # --------------------------------------------------------

    stats = analyze_bf16_bits(
        raw
    )

    unique_values, counts, top_values = (
        top_bf16_values(
            raw
        )
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print(
        "BF16 DISTRIBUTION"
    )

    print(
        "-" * 80
    )

    print(
        f"Unique BF16 values : "
        f"{stats['unique_count']:,}"
    )

    print(
        f"Unique ratio       : "
        f"{stats['unique_count'] / len(raw):.8%}"
    )

    print(
        f"Repeated ratio     : "
        f"{1.0 - stats['unique_count'] / len(raw):.8%}"
    )

    print(
        f"Zero values        : "
        f"{stats['zero_count']:,}"
    )

    print(
        f"Non-zero values    : "
        f"{stats['nonzero_count']:,}"
    )

    print(
        f"Negative values    : "
        f"{stats['negative_count']:,}"
    )

    print(
        f"Positive values    : "
        f"{stats['positive_count']:,}"
    )

    print(
        f"Unique exponents   : "
        f"{stats['exponent_unique']}"
    )

    print(
        f"Unique mantissas   : "
        f"{stats['mantissa_unique']:,}"
    )

    print()
    print(
        f"Minimum            : "
        f"{float(np.min(values)):.10e}"
    )

    print(
        f"Maximum            : "
        f"{float(np.max(values)):.10e}"
    )

    print(
        f"Mean               : "
        f"{float(np.mean(values)):.10e}"
    )

    print(
        f"Std                : "
        f"{float(np.std(values)):.10e}"
    )

    # --------------------------------------------------------
    # Most frequent values
    # --------------------------------------------------------

    print()
    print(
        f"TOP {TOP_VALUES} BF16 VALUES"
    )

    print(
        "-" * 80
    )

    for rank, (
        bits,
        count
    ) in enumerate(
        top_values,
        start=1
    ):

        value = bf16_to_float32(
    np.array([bits], dtype=np.uint16)
)[0]

        percentage = (
            count
            /
            len(raw)
            *
            100
        )

        print(
            f"{rank:2d}. "
            f"value={float(value): .8e}  "
            f"bits=0x{bits:04X}  "
            f"count={count:10,}  "
            f"share={percentage:8.5f}%"
        )

    # --------------------------------------------------------
    # Rail estimates
    # --------------------------------------------------------

    print()
    print(
        "RAIL REPRESENTATION ESTIMATES"
    )

    print(
        "-" * 80
    )

    print(
        "Single-rail exact requirement:"
    )

    print(
        f"  Required distinct rails : "
        f"{stats['unique_count']:,}"
    )

    print()

    print(
        "Compositional {-1,0,+1} upper bounds:"
    )

    for rail_count in [
        16,
        32,
        64,
        128,
        256
    ]:

        print()
        print(
            f"  Rails = {rail_count}"
        )

        for max_terms in [
            1,
            2,
            3,
            4,
            6,
            8
        ]:

            if max_terms > rail_count:
                continue

            capacity = (
                estimate_multi_rail_binary_capacity(
                    rail_count,
                    max_terms
                )
            )

            print(
                f"    max_terms={max_terms:2d} "
                f"upper_bound={capacity:,}"
            )

    # --------------------------------------------------------
    # Model config
    # --------------------------------------------------------

    inspect_config()

    # --------------------------------------------------------
    # Final interpretation
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "ANALYSIS COMPLETE"
    )
    print("=" * 80)

    print()
    print(
        "Next critical question:"
    )

    print(
        "Can this real BF16 tensor be represented exactly"
    )

    print(
        "using a small shared rail basis + topology-only routing?"
    )

    print()
    print(
        "The unique-value count above is the key baseline."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()