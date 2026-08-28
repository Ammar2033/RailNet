import os

def get_function_source(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        source = f.read()
    
    lines = source.split('\n')
    funcs = {}
    current_func = None
    current_lines = []
    
    for line in lines:
        if line.startswith("def "):
            if current_func:
                funcs[current_func] = '\n'.join(current_lines)
            current_func = line.split("def ")[1].split("(")[0]
            current_lines = [line]
        elif current_func is not None:
            current_lines.append(line)
                
    if current_func:
        funcs[current_func] = '\n'.join(current_lines)
        
    return funcs

funcs = get_function_source("e:\\Ammqr\\Railnet\\04_bf16_learned_basis.py")

BASE_IMPORTS = """import numpy as np
import math
from railnet.dtypes.bf16 import (
    bf16_bits_to_float32, bf16_array_to_float32, 
    float32_to_bf16_bits, fp32_array_to_bf16_bits, bf16_bitwise_equal
)
"""

FILES = {
    "_analysis.py": ["analyze_unique_values"],
    "_init.py": ["initialize_rails"],
    "_compile.py": [
        "greedy_routes", "compile_exact_routes_exhaustive", "exhaustive_exact_count",
        "_add_candidate", "_build_pair_sum_table", "_route_has_unique_rails", "_route_value",
        "_bf16_to_float64_bits", "_exact_bf16_equal", "reconstruct_routes", "reconstructed_to_bf16",
        "exact_mask", "calculate_objective"
    ],
    "_repair.py": ["repair_missing_values", "repair_safe_slots"],
    "_optimize.py": ["update_basis", "score_objective", "repair_duplicate_rails", "try_residual_repairs", "learn_basis"]
}

os.makedirs("e:\\Ammqr\\Railnet\\railnet\\rails", exist_ok=True)

for fname, fn_names in FILES.items():
    content = BASE_IMPORTS + "\n"
    if fname == "_init.py":
        content += "EXTREME_RAIL_SLOTS = 0\n\n"
    if fname == "_compile.py":
        pass
    if fname == "_repair.py":
        content += "from ._compile import compile_exact_routes_exhaustive\n"
        content += "SAFE_SCAN_MAX_MISSING = 48\nSAFE_SCAN_LIMIT = 160\nRESIDUAL_CANDIDATES = 64\n\n"
    if fname == "_optimize.py":
        content += "from ._compile import compile_exact_routes_exhaustive, calculate_objective\n"
        content += "from ._repair import repair_missing_values, repair_safe_slots\n"
        content += "from ._init import initialize_rails\n"
        content += "import time\n"
        content += "MAX_RAIL_REPAIRS_PER_ITER = 4\nRESIDUAL_CANDIDATES = 64\nREPAIR_COMPILE_BUDGET = 96\n\n"

    for fn in fn_names:
        if fn in funcs:
            content += funcs[fn] + "\n\n"
        else:
            print(f"Warning: function {fn} not found!")
            
    with open(os.path.join("e:\\Ammqr\\Railnet\\railnet\\rails", fname), "w", encoding="utf-8") as f:
        f.write(content)
        
print("Successfully generated submodules.")
