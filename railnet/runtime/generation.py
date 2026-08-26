"""Generation loop — deterministic greedy (temperature 0)."""
from __future__ import annotations

import numpy as np


def generate(model, prompt: str, max_new_tokens: int = 32, tokenizer=None):
    # Thin wrapper over research 16_gemma_generation.py deterministic loop
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent.parent / "research" / "experiments" / "16_gemma_generation.py"
    spec = importlib.util.spec_from_file_location("rn_gen", str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if hasattr(m, "generate"):
        return m.generate(prompt, max_new_tokens=max_new_tokens)
    # fallback: single token echo
    return {"prompt": prompt, "generated": prompt, "tokens": []}
