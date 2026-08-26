# MEMORY — Honest Accounting

Dense BF16: 2 bytes/param.
RailNet: rail_bits (rail_count*16) + route_map (numel*16 for naive uint16 route_ids).

Layer-0 example: dense 51.19 MiB → honest full representation ~41.6 MiB (1.23×).
Do not report dictionary-only bits.

Route-map compression is the primary open problem.
