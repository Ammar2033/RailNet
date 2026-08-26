import importlib.util, numpy as np, struct, json, mmap
from pathlib import Path
spec = importlib.util.spec_from_file_location("mod", "E:/Ammqr/Railnet/04_bf16_learned_basis.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
raw, shape = mod.read_target_tensor()
uniq_bits, counts, uniq_vals = mod.analyze_unique_values(raw)
print(f"uniq {len(uniq_bits)} vals min {uniq_vals.min():.4f} max {uniq_vals.max():.4f} mean {np.average(uniq_vals, weights=counts):.6f}")

# test quantile init from 03
import importlib.util as iu
spec3 = iu.spec_from_file_location("mod3", "E:/Ammqr/Railnet/03_bf16_rail_compile.py")
mod3 = iu.module_from_spec(spec3)
spec3.loader.exec_module(mod3)

# Need to get unique values with counts sorted by frequency? 03 uses unique_values,counts from raw but sorted by freq
# It then select_rails with TOP_ANCHORS
# Let's replicate 03 select
unique_vals_03, counts_03 = mod3.unique_values_with_counts(raw)
print(f"03 unique {len(unique_vals_03)}")
rails_q = mod3.select_rails(unique_vals_03, counts_03, 64)
print("03 quantile rails (64) decoded:")
print([float(mod.bf16_bits_to_fp32(b)) if hasattr(mod,'bf16_bits_to_fp32') else float(mod3.bf16_bits_to_fp32(b)) for b in rails_q[:10]])
# decode via mod3 helper
decoded_q = [float(mod3.bf16_bits_to_fp32(int(b))) for b in rails_q]
print(decoded_q[:20])
print("decoded_q min", min(decoded_q), "max", max(decoded_q))
# sort decoded for view
print(sorted(decoded_q)[:10], sorted(decoded_q)[-10:])

# Now our fixed init
rails_fixed = mod.initialize_rails(uniq_vals, uniq_bits, counts, 64)
decoded_f = [float(mod.bf16_bits_to_float32(int(b))) for b in rails_fixed]
print("\n fixed rails decoded 64:")
print(decoded_f[:20])
print("min", min(decoded_f), "max", max(decoded_f))
print(sorted(decoded_f)[:10], sorted(decoded_f)[-10:])

# also check centers before kmeans
# replicate initial quantile part
order = np.argsort(uniq_vals)
sorted_vals = uniq_vals[order]
sorted_counts = counts[order]
cum = np.cumsum(sorted_counts)
total=cum[-1]
centers=[]
for i in range(64):
    q=(i+0.5)/64
    target=q*total
    idx=int(np.searchsorted(cum, target, side="left"))
    idx=min(idx, len(sorted_vals)-1)
    centers.append(float(sorted_vals[idx]))
print("\n initial quantile centers (before kmeans) sample:")
print(centers[:10], centers[-10:])
# after kmeans 8 iters
centers=np.array(centers, dtype=np.float64)
for it in range(8):
    dist=np.abs(uniq_vals[:,None]-centers[None,:])
    assign=np.argmin(dist, axis=1)
    new=centers.copy()
    for k in range(64):
        mask=assign==k
        if not np.any(mask): continue
        new[k]=np.sum(uniq_vals[mask]*counts[mask])/max(np.sum(counts[mask]),1.0)
    centers=new
    print(f"iter {it} centers sample", centers[:5], "...", centers[-5:])
print("final centers before BF16", centers[:10])
print("final decoded after BF16", [float(mod.bf16_bits_to_float32(mod.float32_to_bf16_bits(c))) for c in centers[:10]])
