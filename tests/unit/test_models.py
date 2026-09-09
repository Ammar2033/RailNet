"""Unit tests for model adapters."""

import pytest

from railnet.models import get_adapter, get_adapter_for_config
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

    def test_llama_aliases(self):
        assert isinstance(get_adapter("llama3"), LlamaAdapter)
        assert isinstance(get_adapter("llama-3.2"), LlamaAdapter)

    def test_qwen_lookup(self):
        adapter = get_adapter("qwen")
        assert isinstance(adapter, QwenAdapter)

    def test_qwen_aliases(self):
        assert isinstance(get_adapter("qwen2"), QwenAdapter)
        assert isinstance(get_adapter("qwen-2.5"), QwenAdapter)

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown model adapter"):
            get_adapter("gpt4")

    def test_auto_detection_from_config(self):
        assert isinstance(get_adapter_for_config({"model_type": "llama"}), LlamaAdapter)
        assert isinstance(get_adapter_for_config({"model_type": "qwen2"}), QwenAdapter)
        assert isinstance(get_adapter_for_config({"model_type": "gemma3"}), GemmaAdapter)
        assert isinstance(get_adapter_for_config({"rope_theta": 500000.0}), LlamaAdapter)
        assert isinstance(get_adapter_for_config({"rope_theta": 1000000.0}), QwenAdapter)


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
        assert a.architecture == "llama-3.2-1b"
        assert a.dtype == "bf16"

    def test_variant_3b(self):
        a = LlamaAdapter(variant="3b")
        assert a.architecture == "llama-3.2-3b"
        assert a.config["hidden_size"] == 3072
        assert a.config["num_hidden_layers"] == 28

    def test_config(self):
        a = LlamaAdapter()
        assert a.config["hidden_size"] == 2048
        assert a.config["num_hidden_layers"] == 16
        assert a.config["num_attention_heads"] == 32
        assert a.config["num_key_value_heads"] == 8
        assert a.config["rope_scaling"]["rope_type"] == "llama3"

    def test_build_graph(self):
        a = LlamaAdapter()
        g = a.build_graph()
        assert g["architecture"] == "llama-3.2-1b"
        assert g["layers"] == 16
        assert g["heads"] == 32
        assert g["kv_heads"] == 8
        assert g["status"] == "READY"


# ── QwenAdapter ───────────────────────────────────────────


class TestQwenAdapter:
    def test_properties(self):
        a = QwenAdapter()
        assert a.name == "qwen"
        assert a.architecture == "qwen-2.5-0.5b"
        assert a.dtype == "bf16"

    def test_variant_1_5b(self):
        a = QwenAdapter(variant="1.5b")
        assert a.architecture == "qwen-2.5-1.5b"
        assert a.config["hidden_size"] == 1536
        assert a.config["num_hidden_layers"] == 28

    def test_config(self):
        a = QwenAdapter()
        assert a.config["hidden_size"] == 896
        assert a.config["num_hidden_layers"] == 24
        assert a.config["num_attention_heads"] == 14
        assert a.config["num_key_value_heads"] == 2
        assert a.config["rope_theta"] == 1000000.0

    def test_build_graph(self):
        a = QwenAdapter()
        g = a.build_graph()
        assert g["architecture"] == "qwen-2.5-0.5b"
        assert g["layers"] == 24
        assert g["has_qkv_bias"] is True
        assert g["status"] == "READY"
