import numpy as np
import pytest

from railnet.rails.compression.palette import BlockCompressedRouteMap

def test_block_palette_lossless():
    """Verify that Block Palette Compression losslessly decompresses route ids."""
    # Create a realistic route map with some spatial locality
    shape = (1024, 1024)
    # Generate 16x16 blocks that use only a handful of values
    route_ids = np.zeros(shape, dtype=np.uint16)
    
    for i in range(0, shape[0], 16):
        for j in range(0, shape[1], 16):
            # Pick 5 random unique values for this block
            local_palette = np.random.randint(0, 65535, size=5, dtype=np.uint16)
            # Fill the block
            route_ids[i:i+16, j:j+16] = np.random.choice(local_palette, size=(16, 16))
            
    # Flatten it like the compiler provides
    flat_route_ids = route_ids.reshape(-1)
    
    # Compress
    compressed = BlockCompressedRouteMap.compress(flat_route_ids, shape, block_size=16)
    
    # Check that palettes are truncated
    assert compressed.palettes.shape[1] == 5, "Palette max size should exactly match the 5 unique values per block we generated"
    
    # Decompress
    decompressed = compressed.decompress()
    
    # Exactness verification
    assert decompressed.dtype == np.uint16
    assert decompressed.shape == flat_route_ids.shape
    np.testing.assert_array_equal(decompressed, flat_route_ids)

def test_block_palette_padding():
    """Verify compression works with shapes not perfectly divisible by block size."""
    shape = (100, 200) # not divisible by 16
    route_ids = np.random.randint(0, 65535, size=shape, dtype=np.uint16).flatten()
    
    compressed = BlockCompressedRouteMap.compress(route_ids, shape, block_size=16)
    decompressed = compressed.decompress()
    
    np.testing.assert_array_equal(decompressed, route_ids)
