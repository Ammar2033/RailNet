import time, struct, json, mmap, math
from pathlib import Path
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("mod", "E:/Ammqr/Railnet/04_bf16_learned_basis.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
raw, shape = mod.read_target_tensor()
uniq_bits, counts, uniq_vals = mod.analyze_unique_values(raw)
print(f"loaded unique {len(uniq_bits)}")

rc, mt = 64, 4
print(f"Testing RAILS={rc} MAXTERMS={mt}")
start=time.perf_counter()
learned = mod.learn_basis(uniq_vals, uniq_bits, counts, rc, mt)
elapsed=time.perf_counter()-start
obj=learned["objective"]
print(f" exact_unique {obj['exact_unique']}/{len(uniq_bits)}")
print(f" weighted_exact {obj['weighted_exact']:.6f}")
print(f" weighted_rmse {obj['rmse']:.3e}")
print(f" history {learned['history']}")
print(f" time {elapsed:.2f}s")
# also check monotonic: reconstruct routes
# Test sparsity
routes=learned["routes"]
print(f" routes shape {routes.shape} active per unique {np.count_nonzero(routes, axis=1)[:10]}")
# try decode rails
print([float(mod.bf16_bits_to_float32(b)) for b in learned["rails"][:5]])
