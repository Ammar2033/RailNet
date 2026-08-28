import importlib.util, numpy as np, struct
spec = importlib.util.spec_from_file_location("mod", "E:/Ammqr/Railnet/04_bf16_learned_basis.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
spec3 = importlib.util.spec_from_file_location("mod3", "E:/Ammqr/Railnet/03_bf16_rail_compile.py")
mod3 = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(mod3)

raw, _ = mod.read_target_tensor()
uniq_bits, counts, uniq_vals = mod.analyze_unique_values(raw)
# get quantile rails from 03
unique_vals_03, counts_03 = mod3.unique_values_with_counts(raw)
rails_q = mod3.select_rails(unique_vals_03, counts_03, 64)
print("quantile rails decoded range", min([float(mod3.bf16_bits_to_fp32(int(b))) for b in rails_q]), max([float(mod3.bf16_bits_to_fp32(int(b))) for b in rails_q]))

# greedy with those rails
for mt in [2,4]:
    routes, signs, _, _ = mod.greedy_routes(uniq_vals, rails_q, mt)
    rec = mod.reconstruct_routes(routes, signs, rails_q)
    obj = mod.calculate_objective(uniq_vals, uniq_bits, counts, rec)
    print(f"greedy quantile 64/{mt} exact {obj['exact_unique']}")

# exhaustive via 03
for mt in [2,4]:
    route_map = mod3.compile_exact_routes(mod3.np.array(uniq_bits), rails_q, mt)
    # Actually unique_vals_03 is bits sorted by freq, but compile expects unique_values array of bits (any order)
    # Use uniq_bits directly
    uniq_bits_np = np.array(uniq_bits, dtype=np.uint16)
    # need to ensure function works with that
    routes2, exact = mod3.build_route_table_for_unique_values(uniq_bits_np, rails_q, mt)
    print(f"exhaustive quantile 64/{mt} exact {exact}")

# also test our fixed rails with exhaustive
rails_fixed = mod.initialize_rails(uniq_vals, uniq_bits, counts, 64)
print("fixed rails range", min([float(mod.bf16_bits_to_float32(int(b))) for b in rails_fixed]), max([float(mod.bf16_bits_to_float32(int(b))) for b in rails_fixed]))
for mt in [2,4]:
    routes, signs, _, _ = mod.greedy_routes(uniq_vals, rails_fixed, mt)
    rec = mod.reconstruct_routes(routes, signs, rails_fixed)
    obj = mod.calculate_objective(uniq_vals, uniq_bits, counts, rec)
    print(f"greedy fixed 64/{mt} exact {obj['exact_unique']}")
    # exhaustive
    routes2, exact = mod3.build_route_table_for_unique_values(np.array(uniq_bits, dtype=np.uint16), rails_fixed, mt)
    print(f"exhaustive fixed 64/{mt} exact {exact}")
