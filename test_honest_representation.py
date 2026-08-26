import json
import math
from pathlib import Path

# ============================================================
# HONEST FULL REPRESENTATION ACCOUNTING (spec 20 / 42 / 60)
#
# Components that MUST be counted:
#   A. rail primitives          : R x 16
#   B. unique route descriptions: sum_terms x (id_bits+sign)
#   C. per-element route map    : N_total x id_bits   <- DOMINANT
#   D. tensor boundaries/meta   : small constant
#
# The previous "Full representation = json_file_size*8"
# reported ONLY A+B+D and omitted C entirely -> wildly
# optimistic. This script recomputes honestly from the saved
# GLOBAL artifact.
# ============================================================

art_path = Path("compiled/layer0/_GLOBAL_layer0.json")
art = json.load(open(art_path))

rails = art["rails"]
R = art["rail_count"]
routes = art["routes"]
n_routes = len(routes)

TENSOR_SIZES = {
    "model.layers.0.mlp.up_proj.weight": 6912 * 1152,
    "model.layers.0.mlp.gate_proj.weight": 6912 * 1152,
    "model.layers.0.mlp.down_proj.weight": 1152 * 6912,
    "model.layers.0.self_attn.q_proj.weight": 1024 * 1152,
    "model.layers.0.self_attn.k_proj.weight": 256 * 1152,
    "model.layers.0.self_attn.v_proj.weight": 256 * 1152,
    "model.layers.0.self_attn.o_proj.weight": 1152 * 1024,
}

N_total = sum(TENSOR_SIZES.values())
orig_bits = N_total * 16

# A. rails
rail_bits_A = R * 16

# B. route descriptions
id_bits_per_term = math.ceil(math.log2(R))
active_terms = sum(len(t) for t in routes.values())
desc_bits_B = active_terms * (id_bits_per_term + 1)

# C. per-element route map (the dominant term!)
#    Encoding A: global route index  -> ceil(log2(n_routes))
#    Encoding B: raw BF16 key        -> 16 bits (current
#       runtime convention: route id IS the bit pattern)
map_bits_C_13 = N_total * math.ceil(math.log2(n_routes))
map_bits_C_16 = N_total * 16

# D. metadata (shapes, boundaries, checksum) - fixed estimate
meta_bits_D = 4096

total_honest_13bit = rail_bits_A + desc_bits_B + map_bits_C_13 + meta_bits_D
total_honest_16bit = rail_bits_A + desc_bits_B + map_bits_C_16 + meta_bits_D

print("=" * 78)
print("HONEST FULL REPRESENTATION - LAYER-0 GLOBAL BASIS")
print("=" * 78)
print(f"Tensors                : {len(TENSOR_SIZES)}")
print(f"N_total elements       : {N_total:,}")
print(f"Original (BF16 dense)  : {orig_bits:,} bits ({orig_bits/8/1024/1024:.2f} MiB)")
print()
print(f"A rails                : {rail_bits_A:,} bits")
print(f"B route descriptions   : {desc_bits_B:,} bits "
      f"({n_routes} routes, {active_terms:,} terms, {id_bits_per_term}+1 b)")
print(f"C map (13-bit ids)     : {map_bits_C_13:,} bits  <-- DOMINANT")
print(f"C map (16-bit bf16key) : {map_bits_C_16:,} bits")
print(f"D metadata             : {meta_bits_D:,} bits")
print()
print(f"HONEST TOTAL (13-bit)  : {total_honest_13bit:,} bits "
      f"({total_honest_13bit/8/1024/1024:.2f} MiB)"
      )
print(f"  storage compression  : {orig_bits/total_honest_13bit:.3f}x")
print()
print(f"HONEST TOTAL (16-bit)  : {total_honest_16bit:,} bits "
      f"({total_honest_16bit/8/1024/1024:.2f} MiB)"
      )
print(f"  storage compression  : {orig_bits/total_honest_16bit:.4f}x")
print()
dict_only = art_path.stat().st_size * 8
print(f"[OLD WRONG METRIC]     : {dict_only:,} bits (dictionary only!)")
print("=" * 78)

report = {
    "tensor_scope": "model.layers.0 (7 tensors)",
    "N_total_elements": N_total,
    "original_bits": orig_bits,
    "components": {
        "rails_bits": rail_bits_A,
        "route_description_bits": desc_bits_B,
        "element_map_bits_13bit_ids": map_bits_C_13,
        "element_map_bits_16bit_keys": map_bits_C_16,
        "metadata_bits_estimate": meta_bits_D,
    },
    "honest_total_bits_best_case_13bit": total_honest_13bit,
    "honest_compression_best_case_13bit": round(orig_bits / total_honest_13bit, 4),
    "honest_total_bits_raw_key_mapping_16bit": total_honest_16bit,
    "honest_compression_raw_key_mapping": round(orig_bits / total_honest_16bit, 6),
    "note": (
        "Dictionary-only metric (previous report) omitted the "
        "per-element route map, which dominates storage. RailNet's "
        "proven wins: runtime weight-array elimination (rails+topology "
        f"resident ~= {(rail_bits_A+desc_bits_B)/8/1024:.1f} KiB) and shared "
        "computation reduction (-95.97%). Bit-storage compression is "
        "~1.23x best-case and ~1.00x under raw-BF16-key mapping."
    ),
}

out = Path("results/full_representation_report.json")
json.dump(report, open(out, "w"), indent=2)
print(f"\nSaved: {out}")
