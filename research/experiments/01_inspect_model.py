import json
from pathlib import Path

from safetensors import safe_open


# ============================================================
# RailNet-1B
# MODEL INSPECTOR
#
# Does NOT load the complete model into RAM.
# Supports BF16 Safetensors metadata inspection.
# ============================================================


MODEL_DIR = Path("model_data")

MODEL_FILE = MODEL_DIR / "model.safetensors"
TOKENIZER_FILE = MODEL_DIR / "tokenizer.json"
CONFIG_FILE = MODEL_DIR / "config.json"


def human_size(bytes_count: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]

    value = float(bytes_count)

    for unit in units:
        if value < 1024.0:
            return f"{value:.2f} {unit}"

        value /= 1024.0

    return f"{value:.2f} PB"


def tensor_numel(shape) -> int:
    result = 1

    for dim in shape:
        result *= int(dim)

    return result


def inspect_safetensors():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model bulunamadı: {MODEL_FILE}"
        )

    print("=" * 80)
    print("RAILNET-1B GEMMA MODEL INSPECTOR")
    print("=" * 80)

    print()
    print(
        f"Model file : {MODEL_FILE}"
    )

    file_size = MODEL_FILE.stat().st_size

    print(
        f"File size  : "
        f"{human_size(file_size)}"
    )

    print()
    print("SAFETENSORS")
    print("-" * 80)

    total_parameters = 0
    dtype_counts = {}
    tensor_infos = []

    # ========================================================
    # ONLY READ METADATA / SLICE INFORMATION
    #
    # IMPORTANT:
    #
    # get_tensor() is NOT used.
    # This allows BF16 models to be inspected even when
    # NumPy itself cannot materialize bfloat16.
    # ========================================================

    with safe_open(
        str(MODEL_FILE),
        framework="numpy",
        device="cpu",
    ) as f:

        keys = list(f.keys())

        print(
            f"Tensor count : "
            f"{len(keys)}"
        )

        metadata = f.metadata()

        if metadata:

            print()
            print("Metadata:")

            for key, value in metadata.items():

                print(
                    f"  {key}: {value}"
                )

        for key in keys:

            # ------------------------------------------------
            # IMPORTANT:
            #
            # get_slice gives us tensor metadata without
            # converting BF16 to NumPy.
            # ------------------------------------------------

            tensor_slice = f.get_slice(
                key
            )

            shape = tuple(
                int(x)
                for x in tensor_slice.get_shape()
            )

            dtype = str(
                tensor_slice.get_dtype()
            )

            numel = tensor_numel(
                shape
            )

            total_parameters += numel

            dtype_counts[dtype] = (
                dtype_counts.get(
                    dtype,
                    0
                )
                + numel
            )

            tensor_infos.append(
                {
                    "name": key,
                    "shape": shape,
                    "dtype": dtype,
                    "parameters": numel,
                }
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("MODEL SUMMARY")
    print("-" * 80)

    print(
        f"Total parameters : "
        f"{total_parameters:,}"
    )

    print(
        f"Approx. billions : "
        f"{total_parameters / 1_000_000_000:.6f} B"
    )

    print()
    print("DTYPE DISTRIBUTION")
    print("-" * 80)

    for dtype, count in sorted(
        dtype_counts.items(),
        key=lambda x: x[0]
    ):

        percentage = (
            count
            /
            total_parameters
            *
            100
        )

        print(
            f"{dtype:16s}"
            f"{count:15,} params "
            f"({percentage:6.2f}%)"
        )

    # ========================================================
    # TENSORS
    # ========================================================

    print()
    print("TENSORS")
    print("-" * 80)

    for info in tensor_infos:

        print(
            info["name"]
        )

        print(
            f"    shape      : "
            f"{info['shape']}"
        )

        print(
            f"    dtype      : "
            f"{info['dtype']}"
        )

        print(
            f"    parameters : "
            f"{info['parameters']:,}"
        )

    # ========================================================
    # TOKENIZER
    # ========================================================

    print()
    print("TOKENIZER")
    print("-" * 80)

    if TOKENIZER_FILE.exists():

        tokenizer_size = (
            TOKENIZER_FILE.stat().st_size
        )

        print(
            f"Found: "
            f"{TOKENIZER_FILE}"
        )

        print(
            f"Size : "
            f"{human_size(tokenizer_size)}"
        )

        try:

            with open(
                TOKENIZER_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                tokenizer_data = json.load(
                    f
                )

            model_section = (
                tokenizer_data.get(
                    "model",
                    {}
                )
            )

            vocab = (
                model_section.get(
                    "vocab",
                    {}
                )
            )

            if isinstance(
                vocab,
                dict
            ):

                print(
                    f"Vocabulary size: "
                    f"{len(vocab):,}"
                )

        except Exception as exc:

            print(
                f"Tokenizer JSON okunamadı: "
                f"{exc}"
            )

    else:

        print(
            f"Tokenizer bulunamadı: "
            f"{TOKENIZER_FILE}"
        )

    # ========================================================
    # CONFIG
    # ========================================================

    print()
    print("CONFIG")
    print("-" * 80)

    if CONFIG_FILE.exists():

        print(
            f"Found: "
            f"{CONFIG_FILE}"
        )

        try:

            with open(
                CONFIG_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                config = json.load(
                    f
                )

            important_keys = [
                "model_type",
                "architectures",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "vocab_size",
                "max_position_embeddings",
                "head_dim",
                "torch_dtype",
                "dtype",
            ]

            for key in important_keys:

                if key in config:

                    print(
                        f"{key:28s}: "
                        f"{config[key]}"
                    )

        except Exception as exc:

            print(
                f"Config okunamadı: "
                f"{exc}"
            )

    else:

        print(
            f"config.json bulunamadı: "
            f"{CONFIG_FILE}"
        )

    # ========================================================
    # TENSOR GROUPS
    # ========================================================

    print()
    print("IMPORTANT TENSOR GROUPS")
    print("-" * 80)

    groups = {
        "embedding": [],
        "attention": [],
        "mlp": [],
        "norm": [],
        "lm_head": [],
    }

    for info in tensor_infos:

        name = info["name"].lower()

        if (
            "embed" in name
            or
            "embedding" in name
        ):

            groups["embedding"].append(
                info
            )

        if (
            "attn" in name
            or
            "attention" in name
            or
            "q_proj" in name
            or
            "k_proj" in name
            or
            "v_proj" in name
            or
            "o_proj" in name
        ):

            groups["attention"].append(
                info
            )

        if (
            "mlp" in name
            or
            "feed_forward" in name
            or
            "up_proj" in name
            or
            "down_proj" in name
            or
            "gate_proj" in name
        ):

            groups["mlp"].append(
                info
            )

        if "norm" in name:

            groups["norm"].append(
                info
            )

        if (
            "lm_head" in name
            or
            "output" in name
        ):

            groups["lm_head"].append(
                info
            )

    for group_name, items in groups.items():

        parameter_count = sum(
            item["parameters"]
            for item in items
        )

        print(
            f"{group_name:12s}: "
            f"{len(items):4d} tensors, "
            f"{parameter_count:,} params"
        )

    print()
    print("=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    inspect_safetensors()