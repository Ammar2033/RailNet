from __future__ import annotations

from .base import ModelAdapter
from .gemma import GemmaAdapter
from .llama import LlamaAdapter
from .qwen import QwenAdapter

ADAPTERS = {
    "gemma3": GemmaAdapter,
    "gemma": GemmaAdapter,
    "llama": LlamaAdapter,
    "llama3": LlamaAdapter,
    "llama-3": LlamaAdapter,
    "llama-3.2": LlamaAdapter,
    "qwen": QwenAdapter,
    "qwen2": QwenAdapter,
    "qwen2.5": QwenAdapter,
    "qwen-2.5": QwenAdapter,
}


def get_adapter(name: str) -> ModelAdapter:
    key = name.lower()
    if key not in ADAPTERS:
        raise KeyError(f"Unknown model adapter '{name}'. Available: {list(ADAPTERS)}")
    return ADAPTERS[key]()


def get_adapter_for_config(config: dict, default: str = "gemma3") -> ModelAdapter:
    """Detect and return the appropriate ModelAdapter based on config.json dictionary."""
    model_type = str(config.get("model_type", "")).lower()
    for key, adapter_cls in ADAPTERS.items():
        if key in model_type:
            return adapter_cls()

    if "rope_scaling" in config or float(config.get("rope_theta", 0)) == 500000.0:
        return LlamaAdapter()
    if float(config.get("rope_theta", 0)) == 1000000.0:
        return QwenAdapter()

    return get_adapter(default)


__all__ = [
    "ADAPTERS",
    "GemmaAdapter",
    "LlamaAdapter",
    "ModelAdapter",
    "QwenAdapter",
    "get_adapter",
    "get_adapter_for_config",
]
