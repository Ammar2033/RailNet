import time, json, struct, mmap
from pathlib import Path
import numpy as np
# import the hardened module
import importlib.util, sys
spec = importlib.util.spec_from_file_location("mod", "E:/Ammqr/Railnet/04_bf16_learned_basis.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

print("Testing hardened initialize and greedy")

# Load tensor quickly
raw, shape = mod.read_target_tensor()
uniq_bits, counts, uniq_vals = mod.analyze_unique_values(raw)
print(f"unique {len(uniq_bits)}")

# Test single config 64/4
for rc, mt in [(32,2),(32,4),(64,2),(64,4),(128,4)]:
    print("\n"+"="*40)
    print(f"RAILS={rc} MAXTERMS={mt}")
    start=time.perf_counter()
    learned = mod.learn_basis(uniq_vals, uniq_bits, counts, rc, mt)
    elapsed=time.perf_counter()-start
    obj=learned["objective"]
    rails=learned["rails"]
    routes=learned["routes"]
    signs=learned["signs"]
    rep=mod.representation_bits(rc, len(raw), len(uniq_bits), routes, signs)
    print(f" exact_unique {obj['exact_unique']}/{len(uniq_bits)} ratio {obj['exact_unique']/len(uniq_bits):.2%}")
    print(f" weighted_exact {obj['weighted_exact']:.4%}")
    print(f" weighted_rmse {obj['rmse']:.3e} max_error {obj['max_error']:.3e}")
    print(f" active_terms {rep['active_terms']} vs max possible {len(uniq_bits)*mt}")
    print(f" full_compression {rep['full_compression']:.3f}x")
    print(f" time {elapsed:.2f}s history len {len(learned['history'])}")
    for h in learned["history"]:
        print(f"  iter {h['iteration']} exact {h['exact_unique']} w_exact {h['weighted_exact']:.3%} mse {h['weighted_mse']:.3e}")
    # Check monotonic: history exact should be non-decreasing for best? Actually per iteration current, not best
    # Also test greedy sparsity
    print(" rails decoded sample:", [float(mod.bf16_bits_to_float32(b)) for b in rails[:5]])
