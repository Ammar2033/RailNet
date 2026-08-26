import importlib.util, numpy as np, time, struct
spec = importlib.util.spec_from_file_location("mod", "E:/Ammqr/Railnet/04_bf16_learned_basis.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
spec3 = importlib.util.spec_from_file_location("mod3", "E:/Ammqr/Railnet/03_bf16_rail_compile.py")
mod3 = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(mod3)

raw, _ = mod.read_target_tensor()
uniq_bits, counts, uniq_vals = mod.analyze_unique_values(raw)
print(f"loaded {len(uniq_bits)}")

def exhaustive_exact(rails, max_terms):
    uniq_bits_np = np.array(uniq_bits, dtype=np.uint16)
    _, exact = mod3.build_route_table_for_unique_values(uniq_bits_np, rails, max_terms)
    return exact

# initial fixed rails
rails = mod.initialize_rails(uniq_vals, uniq_bits, counts, 64)
print(f"init greedy exact: {mod.calculate_objective(uniq_vals, uniq_bits, counts, mod.reconstruct_routes(*mod.greedy_routes(uniq_vals, rails, 4)[:2], rails))['exact_unique']}")
print(f"init exhaustive exact: {exhaustive_exact(rails,4)}")

# Run 5 iterations of update_basis using greedy
cur_rails = rails.copy()
for it in range(1,6):
    routes, signs, residual, _ = mod.greedy_routes(uniq_vals, cur_rails, 4)
    # update
    cur_rails = mod.update_basis(uniq_vals, counts, cur_rails.copy(), routes, signs, np.zeros(len(uniq_vals), dtype=bool))
    # evaluate greedy
    rg, sg, _, _ = mod.greedy_routes(uniq_vals, cur_rails, 4)
    rec = mod.reconstruct_routes(rg, sg, cur_rails)
    obj = mod.calculate_objective(uniq_vals, uniq_bits, counts, rec)
    ex = exhaustive_exact(cur_rails,4)
    print(f"iter {it} greedy exact {obj['exact_unique']} weighted {obj['weighted_exact']:.3%} rmse {obj['rmse']:.3e} exhaustive {ex}")

