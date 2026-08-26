from .base import ModelAdapter
from .gemma import GemmaAdapter
from .llama import LlamaAdapter
from .qwen import QwenAdapter

ADAPTERS = {"gemma3": GemmaAdapter, "gemma": GemmaAdapter, "llama": LlamaAdapter, "qwen": QwenAdapter}

def get_adapter(name: str) -> ModelAdapter:
    key = name.lower()
    if key not in ADAPTERS:
        raise KeyError(f"Unknown model adapter '{name}'. Available: {list(ADAPTERS)}")
    return ADAPTERS[key]()

__all__ = ["ModelAdapter", "GemmaAdapter", "LlamaAdapter", "QwenAdapter", "get_adapter"]
