import importlib.util, numpy as np, time
spec = importlib.util.spec_from_file_location("mod", "E:/Ammqr/Railnet/04_bf16_learned_basis.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

raw, _ = mod.read_target_tensor()
uniq_bits, counts, uniq_vals = mod.analyze_unique_values(raw)
N = len(uniq_bits)

for rc, mt in [(128, 4), (64, 4)]:
    start = time.perf_counter()
    learned = mod.learn_basis(uniq_vals, uniq_bits, counts, rc, mt)
    elapsed = time.perf_counter() - start
    obj = learned["objective"]
    rails = learned["rails"]
    table = mod.compile_exact_routes_exhaustive(uniq_bits, rails, mt)
    missing = [int(b) for b in uniq_bits if int(b) not in table]
    ex = N - len(missing)
    print(f"learn {rc}/{mt}: greedy={obj['exact_unique']} EXHAUSTIVE={ex}/{N} ({ex/N:.3%}) missing={len(missing)} time={elapsed:.1f}s", flush=True)
    if missing and len(missing) <= 20:
        for b in missing:
            v = float(mod.bf16_bits_to_float32(b))
            print(f"   missing bits={b:04X} v={v:+.9f}", flush=True)
    np.save(f"E:/Ammqr/Railnet/learned_rails_{rc}_{mt}.npy", rails)
