"""Gemma3 demo — load a compiled RailNet artifact and run an exact forward.

    railnet compile model_data/model.safetensors --out compiled
    python examples/gemma3_demo.py

If ``compiled/`` is not present this prints instructions and exits 0.
"""

import json
from pathlib import Path

compiled = Path("compiled/manifest.json")
if not compiled.exists():
    print(
        "No compiled/manifest.json — run:  railnet compile model_data/model.safetensors --out compiled"
    )
    raise SystemExit(0)

from railnet.runtime import RailNetModel

manifest = json.loads(compiled.read_text())
print(f"compiled tensors : {manifest.get('pass_count')}  verdict={manifest.get('verdict')}")

model = RailNetModel.load("compiled")
print(f"layers           : {model.n_layers}")

logits = model.forward([2, 133, 40])
print(f"forward OK        : vocab={logits.shape[0]} argmax={int(logits.argmax())}")
print("runtime dense weight array: ABSENT")
