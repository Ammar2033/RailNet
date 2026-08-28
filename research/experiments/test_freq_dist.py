import importlib.util, numpy as np
spec = importlib.util.spec_from_file_location("mod", "E:/Ammqr/Railnet/04_bf16_learned_basis.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
raw, _ = mod.read_target_tensor()
uniq_bits, counts, uniq_vals = mod.analyze_unique_values(raw)
# sort by frequency
order = np.argsort(counts)[::-1]
print("Top 20 frequent values:")
for i in order[:20]:
    print(f"bits {uniq_bits[i]:04X} val {uniq_vals[i]:.6f} count {int(counts[i])}")
# also sort by value
order_v = np.argsort(uniq_vals)
print("\nSorted by value extremes:")
for i in list(order_v[:10]) + list(order_v[-10:]):
    print(f"val {uniq_vals[i]:.6f} count {int(counts[i])} bits {uniq_bits[i]:04X}")
# histogram
import math
print("\nValue range", uniq_vals.min(), uniq_vals.max())
print("std", np.sqrt(np.average((uniq_vals - np.average(uniq_vals, weights=counts))**2, weights=counts)))
# Check distribution of quantiles vs uniform
# Weighted quantile picks based on cumulative counts, so dense region dominates
# Uniform quantile picks based on sorted unique values equally spaced
# Let's see uniform picks for 64
uniq_sorted = np.sort(uniq_vals)
uniform_picks = [uniq_sorted[int(len(uniq_sorted)*i/64)] for i in range(64)]
print("\nUniform picks (sorted unique spaced):", uniform_picks[:10], uniform_picks[-10:])
# Weighted picks (our initial quantile before kmeans)
order = np.argsort(uniq_vals)
sorted_vals = uniq_vals[order]
sorted_counts = counts[order]
cum = np.cumsum(sorted_counts)
total=cum[-1]
weighted_picks=[]
for i in range(64):
    q=(i+0.5)/64
    target=q*total
    idx=int(np.searchsorted(cum, target))
    weighted_picks.append(float(sorted_vals[idx]))
print("Weighted picks:", weighted_picks[:10], weighted_picks[-10:])
print("Weighted min", min(weighted_picks), "max", max(weighted_picks))
print("Uniform min", min(uniform_picks), "max", max(uniform_picks))
