# MEMORY AND STORAGE IN RAILNET

The defining characteristic of RailNet is the absolute removal of standard dense weight tensors from the runtime memory footprint.

## Storage Hierarchy

A conventional neural network stores an `(out_features, in_features)` matrix of parameters (e.g., in BF16 or FP32).
A compiled RailNet tensor abandons this entirely. Instead, the storage is partitioned into two components:

1. **The Shared Rails (O(1))**
   - A highly compressed dictionary of canonical weight values.
   - For a standard Gemma 1B dense layer, the RailNet configuration typically extracts exactly $N=96$ shared rails.
   - Storage Cost: 96 $\times$ 16 bits = 192 bytes.
   - This pool is completely disconnected from the dimensions of the layer.

2. **The Route Map (Topology)**
   - Replaces the dense matrix with an integer mapping.
   - For every element $(i, j)$ in the dense matrix, RailNet simply stores the *index* of the route ID that corresponds to the linear combination of rails required to reconstruct that parameter.
   - Since route IDs correspond exactly to the `uint16` bit-patterns of the distinct values, the route map structurally is an `int32` or `uint16` array matching the shape of the tensor.
   
## The Primary Bottleneck: Route Map Dominance

Currently, the topology/routing map represents **>99%** of the physical storage artifact (`_GLOBAL_layerX.json`). 

Because RailNet maps dense matrices directly to index maps without structural compression, the memory complexity of a compiled tensor remains $O(N \times M)$ where $N$ and $M$ are the dimensions of the target matrix. While the *unique values* (the entropy of the model) have been successfully decoupled and condensed into 96 Rails, the *spatial distribution* (the map) has not.

### Active Research Directives

1. **Route Map Compression**: Future iterations of RailNet must address the $O(N \times M)$ route map size. Strategies being explored include:
   - Run-length encoding (RLE) for repeated routes.
   - Low-rank factorization of the routing table.
   - Spatially coherent block assignment.
2. **Artifact Serialization**: Moving from `JSON` artifacts to raw binary formats (`.rnmodel`) will drastically reduce the physical footprint on disk, which is currently bloated by stringified integers.

## Runtime Memory Footprint (Honest Reporting)

RailNet implements strict `MemoryBudget` checking to assert the reality of its memory consumption:

```python
from railnet.storage.memory import MemoryBudget

budget = MemoryBudget(out_features, in_features)
budget.assert_compliance(dense_active=False, route_map_type="uint16", rails=96)
# Returns honest structural byte counts for the compiled artifact
report = budget.get_honest_report()
```

If the dense parameters are accidentally materialized in the runtime execution path, the `MemoryBudget` assertion will fail, preventing false claims about RailNet's memory efficiency.
