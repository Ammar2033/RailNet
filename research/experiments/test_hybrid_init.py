import importlib.util, numpy as np, struct
spec = importlib.util.spec_from_file_location("mod", "E:/Ammqr/Railnet/04_bf16_learned_basis.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
spec3 = importlib.util.spec_from_file_location("mod3", "E:/Ammqr/Railnet/03_bf16_rail_compile.py")
mod3 = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(mod3)

raw, _ = mod.read_target_tensor()
uniq_bits, counts, uniq_vals = mod.analyze_unique_values(raw)

def hybrid_init(values_float, bits, counts, rail_count, weighted_ratio=0.5):
    # half weighted quantile before kmeans, half uniform
    n_weighted = int(rail_count * weighted_ratio)
    n_uniform = rail_count - n_weighted
    # weighted picks as before (quantile)
    order = np.argsort(values_float)
    sorted_vals = values_float[order]
    sorted_counts = counts[order]
    cum = np.cumsum(sorted_counts)
    total=cum[-1]
    weighted_centers=[]
    for i in range(n_weighted):
        q=(i+0.5)/n_weighted
        target=q*total
        idx=int(np.searchsorted(cum, target))
        idx=min(idx, len(sorted_vals)-1)
        weighted_centers.append(float(sorted_vals[idx]))
    # uniform picks from sorted unique values
    sorted_unique = np.sort(values_float)
    uniform_centers=[]
    for i in range(n_uniform):
        idx=int((i+0.5)/n_uniform * len(sorted_unique))
        idx=min(idx, len(sorted_unique)-1)
        uniform_centers.append(float(sorted_unique[idx]))
    centers = np.array(weighted_centers + uniform_centers, dtype=np.float64)
    # convert to BF16
    rails = np.array([mod.float32_to_bf16_bits(c) for c in centers], dtype=np.uint16)
    rails = np.unique(rails)
    # fill remaining if duplicates
    if len(rails) < rail_count:
        # fill with frequent
        order_f = np.argsort(counts)[::-1]
        used=set(int(x) for x in rails)
        adds=[]
        for idx in order_f:
            cand=int(bits[idx])
            if cand in used: continue
            used.add(cand)
            adds.append(cand)
            if len(rails)+len(adds)>=rail_count: break
        if adds:
            rails=np.concatenate([rails, np.array(adds, dtype=np.uint16)])
    return rails[:rail_count]

for rc in [64,128]:
    for mt in [4,6]:
        rails_h = hybrid_init(uniq_vals, uniq_bits, counts, rc, 0.5)
        decoded = [float(mod.bf16_bits_to_float32(b)) for b in rails_h]
        print(f"\nHybrid {rc} range {min(decoded):.4f} to {max(decoded):.4f}")
        # exhaustive
        _, exact = mod3.build_route_table_for_unique_values(np.array(uniq_bits, dtype=np.uint16), rails_h, mt)
        print(f"Hybrid exhaustive {rc}/{mt} exact {exact}/4494 {exact/4494:.2%}")
        # greedy
        r,s,_,_ = mod.greedy_routes(uniq_vals, rails_h, mt)
        rec = mod.reconstruct_routes(r,s, rails_h)
        obj = mod.calculate_objective(uniq_vals, uniq_bits, counts, rec)
        print(f"Hybrid greedy {rc}/{mt} exact {obj['exact_unique']} {obj['exact_unique']/4494:.2%}")

# also test pure uniform
def uniform_init(values_float, bits, counts, rail_count):
    sorted_unique = np.sort(values_float)
    centers=[]
    for i in range(rail_count):
        idx=int((i+0.5)/rail_count * len(sorted_unique))
        idx=min(idx, len(sorted_unique)-1)
        centers.append(float(sorted_unique[idx]))
    rails = np.array([mod.float32_to_bf16_bits(c) for c in centers], dtype=np.uint16)
    rails=np.unique(rails)
    if len(rails)<rail_count:
        order_f=np.argsort(counts)[::-1]
        used=set(int(x) for x in rails)
        adds=[]
        for idx in order_f:
            cand=int(bits[idx])
            if cand in used: continue
            used.add(cand)
            adds.append(cand)
            if len(rails)+len(adds)>=rail_count: break
        if adds:
            rails=np.concatenate([rails, np.array(adds, dtype=np.uint16)])
    return rails[:rail_count]

for rc in [64]:
    rails_u = uniform_init(uniq_vals, uniq_bits, counts, rc)
    decoded = [float(mod.bf16_bits_to_float32(b)) for b in rails_u]
    print(f"\nUniform {rc} range {min(decoded):.4f} to {max(decoded):.4f}")
    for mt in [4,6]:
        _, exact = mod3.build_route_table_for_unique_values(np.array(uniq_bits, dtype=np.uint16), rails_u, mt)
        print(f"Uniform exhaustive {rc}/{mt} exact {exact}")
