"""Unit tests for INT8 compiler, quantization, and integer rail routing."""

import numpy as np
import pytest

from railnet.compiler.int8 import (
    CANONICAL_INT8_BASIS,
    compile_int8_routes,
    compile_int8_tensor,
    dequantize_from_int8,
    learn_int8_basis,
    quantize_to_int8,
)
from railnet.core.tensor import RailTensor


def test_quantize_and_dequantize():
    rng = np.random.default_rng(42)
    weights = rng.normal(0, 0.05, size=(64, 128)).astype(np.float32)

    int8_w, scale = quantize_to_int8(weights)

    assert int8_w.dtype == np.int8
    assert scale > 0
    assert np.all(int8_w >= -128)
    assert np.all(int8_w <= 127)

    dequant = dequantize_from_int8(int8_w, scale)
    # Cosine similarity should be > 0.999
    cos_sim = np.dot(weights.flatten(), dequant.flatten()) / (
        np.linalg.norm(weights.flatten()) * np.linalg.norm(dequant.flatten())
    )
    assert cos_sim > 0.999


def test_learn_int8_basis_coverage():
    rng = np.random.default_rng(123)
    int8_w = rng.integers(-128, 128, size=(128, 128), dtype=np.int8)

    learned = learn_int8_basis(int8_w, rails=32, max_terms=3)
    rails = learned["rails"]

    assert len(rails) <= 32
    assert len(rails) >= len(CANONICAL_INT8_BASIS)

    # Compile routes for all unique values
    routes = compile_int8_routes(learned["unique_vals"], rails, max_terms=3)

    # Every unique value must reconstruct exactly
    for val in learned["unique_vals"]:
        v = int(val)
        key = v & 0xFF
        assert key in routes
        terms = routes[key]
        assert len(terms) <= 3
        reconstructed = sum(-rails[r] if s else rails[r] for r, s in terms)
        assert reconstructed == v, f"Mismatch for val={v}: got {reconstructed} with route {terms}"


def test_compile_int8_tensor():
    rng = np.random.default_rng(999)
    raw = rng.normal(0, 0.02, size=(32, 64)).astype(np.float32)

    tensor = compile_int8_tensor(
        raw=raw,
        rails=32,
        max_terms=3,
        name="test_int8_layer",
        shape=(32, 64),
    )

    assert isinstance(tensor, RailTensor)
    assert tensor.dtype == "int8"
    assert tensor.shape.dims == (32, 64)
    assert tensor.rail_count <= 32
    assert tensor.max_terms == 3
    assert "scale" in tensor.metadata
    assert tensor.metadata["scale"] > 0

    # Serialization check
    d = tensor.to_dict()
    assert d["name"] == "test_int8_layer"
    assert d["dtype"] == "int8"
    assert "scale" in d
    assert isinstance(d["rails"], list)
    assert isinstance(d["routes"], dict)
