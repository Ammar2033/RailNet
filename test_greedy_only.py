import time, struct, json, mmap
import numpy as np, importlib.util
spec = importlib.util.spec_from_file_location("mod", "E:/Ammqr/Railnet/04_bf16_learned_basis.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

raw, shape = mod.read_target_tensor()
uniq_bits, counts, uniq_vals = mod.analyze_unique_values(raw)
print(f"unique {len(uniq_bits)}")

for rc in [32,64]:
    for mt in [2,4]:
        rails = mod.initialize_rails(uniq_vals, uniq_bits, counts, rc)
        print(f"\nRAILS {rc} MT {mt} rails sample", [float(mod.bf16_bits_to_float32(b)) for b in rails[:3]])
        start=time.perf_counter()
        routes, signs, residual, active = mod.greedy_routes(uniq_vals, rails, mt)
        rec = mod.reconstruct_routes(routes, signs, rails)
        obj = mod.calculate_objective(uniq_vals, uniq_bits, counts, rec)
        print(f" greedy exact {obj['exact_unique']}/{len(uniq_bits)} weighted {obj['weighted_exact']:.3%} rmse {obj['rmse']:.3e} active {np.sum(active)} time {time.perf_counter()-start:.3f}s")
