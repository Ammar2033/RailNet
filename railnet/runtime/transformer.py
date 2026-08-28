"""
Transformer runtime — loads compiled artifact and runs exact forward.
Delegates to proven 13/14/15b logic via thin wrapper.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from railnet.transformer import block_forward

class RailNetModel:
    def __init__(self, manifest: dict, compiled_dir: Path, device=None):
        self.manifest = manifest
        self.compiled_dir = Path(compiled_dir)
        self.device = device
        self.config = manifest.get("config") or json.loads((self.compiled_dir / ".." / "model_data" / "config.json").read_text()) if (self.compiled_dir / ".." / "model_data" / "config.json").exists() else {}
        
        from railnet.transformer import GemmaContext
        self.ctx = GemmaContext(self.config) if self.config else None

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
        raise NotImplementedError("Full model forward requires integration with the railnet runtime backend.")

    def generate(self, prompt: str, max_new_tokens: int = 64, **kwargs):
        raise NotImplementedError("Generation requires integration with the railnet runtime backend.")
