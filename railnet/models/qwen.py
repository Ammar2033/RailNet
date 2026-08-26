"""Qwen adapter — PLANNED."""
from .base import ModelAdapter

class QwenAdapter(ModelAdapter):
    name = "qwen"
    dtype = "bf16"
    architecture = "qwen-generic"

    def inspect(self, safetensors_path: str) -> dict:
        raise NotImplementedError("QwenAdapter PLANNED")

    def compile_tensor(self, raw, tensor_name: str, **kwargs):
        raise NotImplementedError("Qwen compile PLANNED")

    def build_graph(self) -> dict:
        return {"architecture": self.architecture, "status": "PLANNED"}

    def build_runtime(self, compiled_dir: str, device=None):
        raise NotImplementedError("Qwen runtime PLANNED")
