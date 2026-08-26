"""
Transformer runtime — loads compiled artifact and runs exact forward.
Delegates to proven 13/14/15b logic via thin wrapper.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from railnet.transformer import block_forward, init_from_config


class RailNetModel:
    def __init__(self, manifest: dict, compiled_dir: Path, device=None):
        self.manifest = manifest
        self.compiled_dir = Path(compiled_dir)
        self.device = device
        self.config = manifest.get("config") or json.loads((self.compiled_dir / ".." / "model_data" / "config.json").read_text()) if (self.compiled_dir / ".." / "model_data" / "config.json").exists() else {}
        # lazy load adapter
        init_from_config(self.config) if self.config else None

    @classmethod
    def load(cls, artifact_path: str, device=None):
        p = Path(artifact_path)
        if p.is_dir():
            # compiled/ directory
            manifest_path = p / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
            else:
                manifest = {"compiled_dir": str(p)}
            return cls(manifest, p, device=device)
        elif p.suffix == ".rnmodel":
            # future single-file artifact — unpack via artifacts reader
            from railnet.artifacts.reader import read_rnmodel
            return read_rnmodel(str(p), device=device)
        else:
            # assume json manifest path
            manifest = json.loads(p.read_text())
            return cls(manifest, p.parent, device=device)

    def forward(self, input_ids: np.ndarray):
        # Delegate to proven Stage 15b forward via import
        import importlib.util
        f = Path(__file__).resolve().parent.parent.parent / "research" / "experiments" / "15b_gemma_full_forward.py"
        spec = importlib.util.spec_from_file_location("rn_forward", str(f))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        if hasattr(m, "rail_forward"):
            return m.rail_forward(input_ids)
        raise NotImplementedError("forward not available without proven 15b module")

    def generate(self, prompt: str, max_new_tokens: int = 64, **kwargs):
        from .generation import generate
        return generate(self, prompt, max_new_tokens=max_new_tokens, **kwargs)
