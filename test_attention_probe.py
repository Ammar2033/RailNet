import importlib.util, numpy as np, time, json
spec = importlib.util.spec_from_file_location("mod", "E:/Ammqr/Railnet/04_bf16_learned_basis.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# List all layer-0 tensor names + shapes from header
header, base = mod.read_safetensors_header()
print("LAYER-0 TENSORS:")
for k in sorted(header.keys()):
    if ".layers.0." in k:
        m = header[k]
        print(f"  {k:55s} {m['dtype']:6s} {tuple(m['shape'])}")

ATT = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
]

for name in ATT:
    globals()["TARGET_TENSOR"] = name
    mod.TARGET_TENSOR = name
    raw, shape = mod.read_target_tensor()
    bits, counts, vals = mod.analyze_unique_values(raw)
    print(f"\n=== {name} ===")
    print(f"shape {shape} params {len(raw):,}")
    print(f"unique {len(bits):,} range {vals.min():+.5f}..{vals.max():+.5f} zeros {int(np.count_nonzero(raw==0)):,}")
    # capacity probes (uniform init only, cheap exhaustive)
    for rc in [32, 64, 96, 128]:
        rails = mod.initialize_rails(vals, bits, counts, rc)
        row = []
        for mt in [2, 3, 4]:
            t0 = time.perf_counter()
            table = mod.compile_exact_routes_exhaustive(bits, rails, mt)
            cov = sum(1 for b in bits if int(b) in table)
            el = time.perf_counter() - t0
            mark = "*" if cov == len(bits) else ""
            row.append(f"{mt}t:{cov}/{len(bits)}{mark}({el:.2f}s)")
        print(f"  rails={rc:3d}  " + "  ".join(row))
