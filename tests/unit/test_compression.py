import numpy as np

from railnet.artifacts.compression import compress_route_ids, decompress_route_ids


def test_entropy_compression_lossless():
    """Verify that zlib correctly compresses and losslessly decompresses route ids."""
    # Create a realistic route map (lots of zeros/small numbers and a few rare combinations)
    route_ids = np.random.choice(
        [0, 1, 2, 5, 10, 100, 200], size=1024 * 1024, p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.025, 0.025]
    ).astype(np.uint16)

    encoded = compress_route_ids(route_ids)

    assert isinstance(encoded, str)
    assert len(encoded) > 0

    # Verify the compression ratio is at least somewhat effective on patterned data
    uncompressed_size = route_ids.nbytes
    compressed_size = len(encoded)

    assert compressed_size < uncompressed_size, (
        f"Compression failed to reduce size: {compressed_size} >= {uncompressed_size}"
    )

    decoded = decompress_route_ids(encoded)

    # Exactness verification
    assert decoded.dtype == np.uint16
    assert decoded.shape == route_ids.shape
    np.testing.assert_array_equal(decoded, route_ids)


def test_entropy_compression_random_noise():
    """Verify worst-case (uniform random) still losslessly decompresses."""
    route_ids = np.random.randint(0, 65535, size=10000, dtype=np.uint16)

    encoded = compress_route_ids(route_ids)
    decoded = decompress_route_ids(encoded)

    np.testing.assert_array_equal(decoded, route_ids)
