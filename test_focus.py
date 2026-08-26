import importlib.util, numpy as np, time
spec = importlib.util.spec_from_file_location("mod", "E:/Ammqr/Railnet/04_bf16_learned_basis.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

raw, _ = mod.read_target_tensor()
uniq_bits, counts, uniq_vals = mod.analyze_unique_values(raw)
N = len(uniq_bits)

# Fast check: init with extremes only (no learning)
for rc in [64, 128]:
    rails = mod.initialize_rails(uniq_vals, uniq_bits, counts, rc)
    table = mod.compile_exact_routes_exhaustive(uniq_bits, rails, 4)
    cnt = sum(1 for b in uniq_bits if int(b) in table)
    print(f"init+extremes {rc}/4 exhaustive: {cnt}/{N} ({cnt/N:.2%})", flush=True)

print(flush=True)

# Full learn for the two key configs
for rc, mt in [(64, 4), (128, 4)]:
    start = time.perf_counter()
    learned = mod.learn_basis(uniq_vals, uniq_bits, counts, rc, mt)
    elapsed = time.perf_counter() - start
    obj = learned["objective"]
    table = mod.compile_exact_routes_exhaustive(uniq_bits, learned["rails"], mt)
    ex = sum(1 for b in uniq_bits if int(b) in table)
    print(f"learn {rc}/{mt}: greedy_exact={obj['exact_unique']} EXHAUSTIVE={ex}/{N} ({ex/N:.2%}) time={elapsed:.1f}s", flush=True)
