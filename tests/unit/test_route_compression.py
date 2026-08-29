"""Route-map compression schemes: every structural scheme must be lossless and
its bit count must be sane."""

import numpy as np
import pytest

from railnet.analysis.route_compression import benchmark_route_map, compressed_route_map_bits

SHAPES = [(64, 96), (128, 128), (256, 48)]


@pytest.fixture
def clustered_map():
    """A route map with block structure (what real route maps look like)."""
    rng = np.random.default_rng(0)
    m, n = 256, 192
    ids = np.empty((m, n), dtype=np.uint16)
    for bi in range(0, m, 16):
        for bj in range(0, n, 16):
            local = rng.integers(0, 12, size=8).astype(np.uint16)  # small local palette
            ids[bi : bi + 16, bj : bj + 16] = rng.choice(local, size=(16, 16))
    return ids


def test_all_schemes_lossless(clustered_map):
    b = benchmark_route_map(clustered_map)
    for name, r in b.items():
        if name.startswith("_"):
            continue
        assert r["lossless"], name
        assert r["bits"] > 0


def test_raw_equals_dense(clustered_map):
    b = benchmark_route_map(clustered_map)
    assert b["raw_uint16"]["ratio_vs_dense"] == 1.0
    assert b["raw_uint16"]["bits"] == clustered_map.size * 16


def test_block_structure_compresses(clustered_map):
    """On genuinely block-clustered data some scheme should beat dense."""
    b = benchmark_route_map(clustered_map)
    best = min(r["ratio_vs_dense"] for k, r in b.items() if not k.startswith("_"))
    assert best < 0.9


def test_block_palette_actually_round_trips(clustered_map):
    """The bit count assumes lossless — confirm the concrete codec does recover."""
    from railnet.rails.compression.palette import BlockCompressedRouteMap

    packed = BlockCompressedRouteMap.compress(
        clustered_map.reshape(-1), clustered_map.shape, block_size=16
    )
    back = packed.decompress().reshape(clustered_map.shape)
    assert np.array_equal(back, clustered_map)


@pytest.mark.parametrize("shape", SHAPES)
def test_flat_input_with_shape(shape):
    rng = np.random.default_rng(1)
    flat = rng.integers(0, 200, size=shape[0] * shape[1]).astype(np.uint16)
    b = benchmark_route_map(flat, shape=shape)
    assert all(r["lossless"] for k, r in b.items() if not k.startswith("_"))
    assert compressed_route_map_bits(flat, shape, "global_minwidth") == b["global_minwidth"]["bits"]
