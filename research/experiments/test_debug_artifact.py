import importlib.util, json, numpy as np
spec = importlib.util.spec_from_file_location("mod", "E:/Ammqr/Railnet/04_bf16_learned_basis.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

art = json.load(open("E:/Ammqr/Railnet/railnet_lossless_basis_lossless.json"))
rails = np.array(art["rail_bits"], dtype=np.uint16)
keys_art = set(int(k) for k in art["routes"].keys())

raw, _ = mod.read_target_tensor()
uniq_bits, counts, uniq_vals = mod.analyze_unique_values(raw)
uniq_set = set(int(b) for b in uniq_bits)

print("tensor uniques:", len(uniq_set))
print("artifact keys :", len(keys_art))
print("missing from artifact:", len(uniq_set - keys_art))
print("extra in artifact    :", len(keys_art - uniq_set))

# Recompile fresh from saved rails and compare key sets
table_fresh = mod.compile_exact_routes_exhaustive(np.array(sorted(uniq_set), dtype=np.uint16), rails, 4)
keys_fresh = set(table_fresh.keys())
print("fresh compile keys   :", len(keys_fresh))
print("missing from FRESH   :", len(uniq_set - keys_fresh))
print("fresh == artifact keys?", keys_fresh == keys_art)
if keys_fresh != keys_art:
    print("only in fresh:", len(keys_fresh - keys_art), "only in artifact:", len(keys_art - keys_fresh))

# Show a few missing examples with their float values
miss = sorted(uniq_set - keys_art)[:10]
for b in miss:
    v = float(mod.bf16_bits_to_float32(b))
    print(f"  missing {b:04X} v={v:+.9f}")
