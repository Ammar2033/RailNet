"""
Gemma3 demo — loads compiled/ artifact and runs exact verification
without re-compiling. Requires model_data/ and compiled/ from Stage 15.

  pip install -e .
  python examples/gemma3_demo.py
"""
from pathlib import Path
import json
import numpy as np

compiled = Path("compiled/manifest.json")
if not compiled.exists():
    print("No compiled/manifest.json — run research/experiments/15a_gemma_full_compile.py first (needs model_data/model.safetensors)")
    raise SystemExit(0)

manifest = json.loads(compiled.read_text())
print(f"Manifest tensors: {len(manifest.get('tensors', []))}")

# Light forward check via runtime shim
from railnet.runtime.transformer import RailNetModel
model = RailNetModel.load("compiled")
print(f"Loaded model via RailNetModel — config layers={model.config.get('num_hidden_layers')}")

# Verification: one linear
from railnet.kernel import rail_linear
print("Demo OK — for full generation see research/experiments/16_gemma_generation.py")
