import importlib.util, numpy as np, struct
spec = importlib.util.spec_from_file_location("mod", "E:/Ammqr/Railnet/04_bf16_learned_basis.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

for name in [
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
]:
    globals()["TARGET_TENSOR"] = name
    mod.TARGET_TENSOR = name
    raw, shape = mod.read_target_tensor()
    bits, counts, vals = mod.analyze_unique_values(raw)
    print(f"\n=== {name} ===")
    print(f"shape {shape} params {len(raw):,}")
    print(f"unique {len(bits):,} ratio {len(bits)/len(raw):.6%}")
    print(f"range {vals.min():+.5f} .. {vals.max():+.5f}")
    print(f"zeros {int(np.count_nonzero(raw==0)):,}")
    top = np.argsort(-counts)[:3]
    for t in top:
        print(f"  top: bits={bits[t]:04X} v={vals[t]:+.6f} count={int(counts[t]):,}")
    # uniform init 64/4 exhaustive capacity probe (fast)
    rails64 = mod.initialize_rails(vals, bits, counts, 64)
    table = mod.compile_exact_routes_exhaustive(bits, rails64, 4)
    cov = sum(1 for b in bits if int(b) in table)
    print(f"uniform-init 64/4 exhaustive probe: {cov}/{len(bits)} ({cov/len(bits):.2%})")
