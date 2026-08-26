import importlib.util, numpy as np
spec = importlib.util.spec_from_file_location("mod", "E:/Ammqr/Railnet/04_bf16_learned_basis.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
spec3 = importlib.util.spec_from_file_location("mod3", "E:/Ammqr/Railnet/03_bf16_rail_compile.py")
mod3 = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(mod3)

raw, _ = mod.read_target_tensor()
uniq_bits, counts, uniq_vals = mod.analyze_unique_values(raw)
minv, maxv = uniq_vals.min(), uniq_vals.max()
print(f"range {minv} to {maxv}")

def linear_init(rail_count):
    centers = [minv + (maxv-minv)*(i+0.5)/rail_count for i in range(rail_count)]
    rails = np.array([mod.float32_to_bf16_bits(c) for c in centers], dtype=np.uint16)
    rails=np.unique(rails)
    if len(rails)<rail_count:
        # fill with frequent
        order=np.argsort(counts)[::-1]
        used=set(int(x) for x in rails)
        adds=[]
        for idx in order:
            cand=int(uniq_bits[idx])
            if cand in used: continue
            used.add(cand)
            adds.append(cand)
            if len(rails)+len(adds)>=rail_count: break
        if adds:
            rails=np.concatenate([rails, np.array(adds,dtype=np.uint16)])
    return rails[:rail_count]

for rc in [32,64,128]:
    rails = linear_init(rc)
    decoded=[float(mod.bf16_bits_to_float32(b)) for b in rails]
    print(f"\nLinear {rc} range {min(decoded):.4f} {max(decoded):.4f} sample {decoded[:5]} ... {decoded[-5:]}")
    for mt in [4]:
        _, exact = mod3.build_route_table_for_unique_values(np.array(uniq_bits,dtype=np.uint16), rails, mt)
        print(f"Linear exhaustive {rc}/{mt} exact {exact} {exact/4494:.2%}")
        # greedy
        r,s,_,_=mod.greedy_routes(uniq_vals, rails, mt)
        rec=mod.reconstruct_routes(r,s,rails)
        obj=mod.calculate_objective(uniq_vals, uniq_bits, counts, rec)
        print(f" greedy {obj['exact_unique']}")

# also test quantil + extremes
def quantile_plus_extremes(rail_count):
    # weighted quantile for most, plus extremes
    n_extreme=4
    n_quant = rail_count - n_extreme
    order = np.argsort(uniq_vals)
    sorted_vals = uniq_vals[order]
    sorted_counts = counts[order]
    cum=np.cumsum(sorted_counts)
    total=cum[-1]
    centers=[]
    for i in range(n_quant):
        q=(i+0.5)/n_quant
        target=q*total
        idx=int(np.searchsorted(cum, target))
        idx=min(idx, len(sorted_vals)-1)
        centers.append(float(sorted_vals[idx]))
    # add extremes: min, max, and maybe second min/max
    sorted_unique=np.sort(uniq_vals)
    extremes=[sorted_unique[0], sorted_unique[-1], sorted_unique[1], sorted_unique[-2]]
    centers=centers+extremes[:n_extreme]
    rails=np.array([mod.float32_to_bf16_bits(c) for c in centers], dtype=np.uint16)
    rails=np.unique(rails)
    if len(rails)<rail_count:
        order_f=np.argsort(counts)[::-1]
        used=set(int(x) for x in rails)
        adds=[]
        for idx in order_f:
            cand=int(uniq_bits[idx])
            if cand in used: continue
            used.add(cand)
            adds.append(cand)
            if len(rails)+len(adds)>=rail_count: break
        if adds:
            rails=np.concatenate([rails, np.array(adds,dtype=np.uint16)])
    return rails[:rail_count]

for rc in [64]:
    rails = quantile_plus_extremes(rc)
    print(f"\nQuant+ext {rc} ", [float(mod.bf16_bits_to_float32(b)) for b in rails[:5]], "...", [float(mod.bf16_bits_to_float32(b)) for b in rails[-5:]])
    _, exact = mod3.build_route_table_for_unique_values(np.array(uniq_bits,dtype=np.uint16), rails, 4)
    print(f"Quant+ext exhaustive 64/4 exact {exact}")
