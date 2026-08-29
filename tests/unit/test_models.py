"""Unit tests for model adapters."""

import pytest

from railnet.models import get_adapter
from railnet.models.gemma import GemmaAdapter
from railnet.models.llama import LlamaAdapter
from railnet.models.qwen import QwenAdapter

# ── Registry ──────────────────────────────────────────────


class TestAdapterRegistry:
    def test_gemma_lookup(self):
        adapter = get_adapter("gemma3")
        assert isinstance(adapter, GemmaAdapter)

    def test_gemma_alias(self):
        adapter = get_adapter("gemma")
        assert isinstance(adapter, GemmaAdapter)

    def test_llama_lookup(self):
        adapter = get_adapter("llama")
        assert isinstance(adapter, LlamaAdapter)

    def test_qwen_lookup(self):
        adapter = get_adapter("qwen")
        assert isinstance(adapter, QwenAdapter)

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown model adapter"):
            get_adapter("gpt4")


# ── GemmaAdapter ──────────────────────────────────────────


class TestGemmaAdapter:
    def test_properties(self):
        a = GemmaAdapter()
        assert a.name == "gemma3"
        assert a.dtype == "bf16"
        assert a.architecture == "gemma3-1b"

    def test_config(self):
        a = GemmaAdapter()
        assert a.config["hidden_size"] == 1152
        assert a.config["num_hidden_layers"] == 26
        assert a.config["vocab_size"] == 262144

    def test_build_graph(self):
        a = GemmaAdapter()
        g = a.build_graph()
        assert g["architecture"] == "gemma3-1b"
        assert g["layers"] == 26


# ── LlamaAdapter ──────────────────────────────────────────


class TestLlamaAdapter:
    def test_properties(self):
        a = LlamaAdapter()
        assert a.name == "llama"
        assert a.architecture == "llama-generic"

    def test_inspect_raises(self):
        a = LlamaAdapter()
        with pytest.raises(NotImplementedError, match="PLANNED"):
            a.inspect("nonexistent.safetensors")

    def test_compile_tensor_raises(self):
        a = LlamaAdapter()
        with pytest.raises(NotImplementedError):
            a.compile_tensor(None, "test")

    def test_build_graph_planned(self):
        a = LlamaAdapter()
        g = a.build_graph()
        assert g["status"] == "PLANNED"

    def test_build_runtime_raises(self):
        a = LlamaAdapter()
        with pytest.raises(NotImplementedError):
            a.build_runtime("compiled")


# ── QwenAdapter ───────────────────────────────────────────


class TestQwenAdapter:
    def test_properties(self):
        a = QwenAdapter()
        assert a.name == "qwen"

    def test_all_methods_raise(self):
        a = QwenAdapter()
        with pytest.raises(NotImplementedError):
            a.inspect("x")
        with pytest.raises(NotImplementedError):
            a.compile_tensor(None, "x")
        with pytest.raises(NotImplementedError):
            a.build_runtime("x")
