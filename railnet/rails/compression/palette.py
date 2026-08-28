import numpy as np
from dataclasses import dataclass
from railnet.core.shape import Shape

@dataclass
class BlockCompressedRouteMap:
    """
    A runtime-friendly spatial compression scheme for Route Maps.
    Divides the original (M, N) matrix into small blocks (e.g., 16x16).
    For each block, extracts a local palette of unique 16-bit route IDs.
    The block's spatial elements are then stored as small 4-bit or 8-bit indices pointing to the palette.
    """
    original_shape: tuple
    block_size: int
    palettes: np.ndarray        # (num_blocks, max_palette_size) uint16
    local_indices: np.ndarray   # (num_blocks, block_size, block_size) uint8
    palette_sizes: np.ndarray   # (num_blocks,) uint16 - how many actual unique routes in each block

    @classmethod
    def compress(cls, route_ids: np.ndarray, shape: tuple, block_size: int = 16) -> 'BlockCompressedRouteMap':
        """
        Compresses a dense 1D or 2D route_ids array into a block palette representation.
        """
        if len(shape) != 2:
            raise ValueError(f"Block compression only supports 2D matrices, got {shape}")
            
        M, N = shape
        
        # We need to pad the matrix if it doesn't divide evenly
        pad_m = (block_size - (M % block_size)) % block_size
        pad_n = (block_size - (N % block_size)) % block_size
        
        dense_2d = route_ids.reshape(shape)
        if pad_m > 0 or pad_n > 0:
            dense_2d = np.pad(dense_2d, ((0, pad_m), (0, pad_n)), mode='constant', constant_values=0)
            
        P_M, P_N = dense_2d.shape
        blocks_m = P_M // block_size
        blocks_n = P_N // block_size
        num_blocks = blocks_m * blocks_n
        
        # Reshape into blocks: (blocks_m, block_size, blocks_n, block_size) -> (num_blocks, block_size, block_size)
        blocks = dense_2d.reshape(blocks_m, block_size, blocks_n, block_size).swapaxes(1, 2).reshape(num_blocks, block_size, block_size)
        
        # Preallocate outputs (worst case: all block elements are unique -> 256 for 16x16)
        max_possible_palette = block_size * block_size
        palettes = np.zeros((num_blocks, max_possible_palette), dtype=np.uint16)
        local_indices = np.zeros((num_blocks, block_size, block_size), dtype=np.uint8)
        palette_sizes = np.zeros(num_blocks, dtype=np.uint16)
        
        for i in range(num_blocks):
            block = blocks[i]
            # Find unique route IDs in this block
            unique_routes, indices = np.unique(block, return_inverse=True)
            
            p_size = len(unique_routes)
            palette_sizes[i] = p_size
            palettes[i, :p_size] = unique_routes
            local_indices[i] = indices.reshape(block_size, block_size)
            
        # Optional: truncate palettes array to the absolute maximum found across all blocks to save space
        global_max_palette = int(np.max(palette_sizes))
        palettes = palettes[:, :global_max_palette]
            
        return cls(
            original_shape=shape,
            block_size=block_size,
            palettes=palettes,
            local_indices=local_indices,
            palette_sizes=palette_sizes
        )
        
    def decompress(self) -> np.ndarray:
        """
        Decompresses back to the flat dense 1D uint16 route_ids array expected by the runtime.
        """
        M, N = self.original_shape
        P_M = M + (self.block_size - (M % self.block_size)) % self.block_size
        P_N = N + (self.block_size - (N % self.block_size)) % self.block_size
        
        blocks_m = P_M // self.block_size
        blocks_n = P_N // self.block_size
        num_blocks = blocks_m * blocks_n
        
        # Fast vectorization: use advanced indexing
        # For each block i, we want palettes[i, local_indices[i]]
        # We can do this with np.take_along_axis or advanced indexing
        
        # Create a block index array (num_blocks, block_size, block_size)
        block_idx = np.arange(num_blocks)[:, None, None]
        
        # Fetch the original route IDs
        reconstructed_blocks = self.palettes[block_idx, self.local_indices]
        
        # Reshape back to 2D
        reconstructed_2d = reconstructed_blocks.reshape(blocks_m, blocks_n, self.block_size, self.block_size).swapaxes(1, 2).reshape(P_M, P_N)
        
        # Crop padding
        final_2d = reconstructed_2d[:M, :N]
        return final_2d.reshape(-1)
