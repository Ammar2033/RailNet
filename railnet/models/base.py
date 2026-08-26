"""ModelAdapter base — model-specific compile/runtime hooks."""
from __future__ import annotations

import abc


class ModelAdapter(abc.ABC):
    name: str = "base"
    dtype: str = "bf16"
    architecture: str = "generic"

    @abc.abstractmethod
    def inspect(self, safetensors_path: str) -> dict:
        ...

    @abc.abstractmethod
    def compile_tensor(self, raw, tensor_name: str, **kwargs) -> dict:
        ...

    @abc.abstractmethod
    def build_graph(self) -> dict:
        ...

    @abc.abstractmethod
    def build_runtime(self, compiled_dir: str, device=None):
        ...
