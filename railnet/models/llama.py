"""Llama adapter — PLANNED."""

from .base import ModelAdapter


class LlamaAdapter(ModelAdapter):
    name = "llama"
    dtype = "bf16"
    architecture = "llama-generic"

    def inspect(self, safetensors_path: str) -> dict:
        raise NotImplementedError("LlamaAdapter PLANNED — not yet proven")

    def compile_tensor(self, raw, tensor_name: str, **kwargs):
        raise NotImplementedError("Llama compile PLANNED")

    def build_graph(self) -> dict:
        return {"architecture": self.architecture, "status": "PLANNED"}

    def build_runtime(self, compiled_dir: str, device=None):
        raise NotImplementedError("Llama runtime PLANNED")
