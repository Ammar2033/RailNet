import importlib.util, numpy as np, time, struct
spec = importlib.util.spec_from_file_location("mod", "E:/Ammqr/Railnet/04_bf16_learned_basis.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

raw, _ = mod.read_target_tensor()
uniq_bits, counts, uniq_vals = mod.analyze_unique_values(raw)
N = len(uniq_bits)
print(f"unique {N}")

for rc in [64, 128]:
    rails = mod.initialize_rails(uniq_vals, uniq_bits, counts, rc)
    rail_vals = np.array([float(mod.bf16_bits_to_float32(int(b))) for b in rails])
    print(f"\n=== RAILS {rc} ===")
    print(f"rail range: {rail_vals.min():.6f} .. {rail_vals.max():.6f}")
    start = time.perf_counter()
    table = mod.compile_exact_routes_exhaustive(uniq_bits, rails, 4)
    elapsed = time.perf_counter() - start
    missing = []
    for i in range(N):
        b = int(uniq_bits[i])
        if b not in table:
            missing.append((b, float(uniq_vals[i]), int(counts[i])))
    print(f"exhaustive compile {elapsed:.2f}s  exact {N-len(missing)}/{N}  missing {len(missing)}")
    mvals = np.array([m[1] for m in missing])
    mcnts = np.array([m[2] for m in missing])
    print(f"missing |v|: min {np.abs(mvals).min():.4f} max {np.abs(mvals).max():.4f} mean {np.abs(mvals).mean():.4f}")
    print(f"missing |v| > max|rail|: {int(np.sum(np.abs(mvals) > np.abs(rail_vals).max()))}")
    print(f"missing |v| < min|rail|: {int(np.sum(np.abs(mvals) < np.abs(rail_vals).min()))} (min|rail|={np.abs(rail_vals).min():.2e})")
    print(f"missing total usage count: {int(mcnts.sum())} of {int(counts.sum())}")
    # sort by magnitude
    order = np.argsort(-np.abs(mvals))
    print("missing by |v| desc (top 20):")
    for idx in order[:20]:
        b, v, c = missing[idx]
        print(f"  bits={b:04X} v={v:+.6f} count={c}")
