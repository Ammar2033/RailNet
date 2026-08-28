# HARDWARE RESEARCH DIRECTIVES

RailNet's ultimate ambition extends beyond CPU/GPU runtime abstractions. Because RailNet maps continuous floating-point spaces into discrete combinatorial routing structures, it naturally translates into specialized hardware architectures.

## ASIC / FPGA Roadmap

The mathematical operations required for a RailNet forward pass are uniquely suited for spatial computing (FPGAs) or Application-Specific Integrated Circuits (ASICs). The core transformation replaces Dense Matrix Multiply-Accumulate (MAC) grids with **Routing Fabrics**.

### 1. The Routing Fabric

In traditional architectures, weights are loaded from HBM/DRAM into SRAM, where they are multiplied by inputs. 

In a RailNet Architecture (`RailFabric` concept), the **Shared Rails** are permanently stored in ultra-fast local registers (e.g., 96 float registers on-die). The "weight matrix" is replaced by a massive routing switch.

When an input vector $X$ arrives, the hardware uses the compressed **Route Map** (which dictates which Rails apply to which $(i, j)$ element) to physically route the inputs to summing junctions.
- No `fma` (fused multiply-add) instructions are executed across the bulk matrix.
- Instead, inputs are purely routed and accumulated (adds/subtracts based on the term sign).
- The final sum at each router node is multiplied *once* by the canonical Rail value.

### 2. Multiplier-less Inference

By pushing the complexity into the routing topology, RailNet effectively removes millions of multiplier circuits from the hardware pipeline, replacing them with cheaper multiplexers and adders.

- Dense layer: $O(N \times M)$ Multiplications
- RailNet layer: $O(N \times M)$ Additions + $O(N)$ Multiplications

### 3. Latency and Power Target

If fully realized in hardware, RailNet projects massive reductions in both dynamic power (fewer multipliers, zero dense-weight DRAM fetching) and static power (smaller footprint).

## Current Emulation (`railnet.hardware`)

The `railnet.hardware` module provides early software emulation for these physical concepts:
- `MemorySubsystem`: Simulates constrained bandwidth and distinct memory pools (Shared Registers vs HBM).
- `RoutingFabric`: Simulates the routing of inputs based on topology matrices, measuring theoretical crossbar latency.
- `ComputeFabric`: Simulates the isolated phase of multiplying the accumulated values with the shared rails.

These modules serve as the blueprint for eventual hardware translation.
