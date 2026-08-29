"""Transformer runtime — loads a compiled RailNet artifact and runs an
exact forward / greedy generation.

No dense weight array is ever materialized on the rail path: linear layers run
through the bit-pattern-indexed rail kernel; norms and the tied embedding are
read row by row from the source safetensors (spec: not compiled, not claimed).

A ``"dense"`` backend (streaming the original weights from the safetensors) is
provided purely as the verification reference — it is never used for inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from railnet.dtypes.bf16 import bf16_array_to_float32
from railnet.embedding import MmapRowLookup
from railnet.kernel import CompiledTensor, rail_linear_fast
from railnet.safetensors_reader import read_tensor_raw
from railnet.transformer import GemmaContext, block_forward, rms_norm

_LAYER_NORM_KEYS = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
)
_ATTN_ROLES = ("q_proj", "k_proj", "v_proj", "o_proj")


def _bf16_bits_to_f64(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def _weight_name(layer: int, role: str) -> str:
    kind = "self_attn" if role in _ATTN_ROLES else "mlp"
    return f"model.layers.{layer}.{kind}.{role}.weight"


class RailNetModel:
    def __init__(self, manifest: dict, compiled_dir: Path, device=None):
        self.manifest = manifest
        self.compiled_dir = Path(compiled_dir)
        self.device = device

        self.config = manifest.get("config") or {}
        if not self.config:
            raise ValueError(
                "compiled manifest has no 'config' — recompile with "
                "railnet.compiler.model.compile_model"
            )
        self.ctx = GemmaContext(self.config)
        self.n_layers = int(manifest.get("num_hidden_layers") or self.config["num_hidden_layers"])

        self.source_model = self._resolve(manifest.get("source_model"))
        if self.source_model is None or not self.source_model.exists():
            raise FileNotFoundError(
                f"source safetensors not found ({manifest.get('source_model')!r}); "
                "norms + embedding are read from it at runtime"
            )
        self.tokenizer_path = self._resolve(manifest.get("tokenizer"))

        consts = manifest["constants"]
        self._emb = MmapRowLookup(consts["embedding"], model_file=self.source_model)
        self._final_norm = _bf16_bits_to_f64(
            read_tensor_raw(consts["final_norm"], model_file=self.source_model)[0]
        )
        self._norms: list[dict] = [self._load_layer_norms(b) for b in range(self.n_layers)]
        self._linears: list[dict] = [self._load_layer_linears(b) for b in range(self.n_layers)]
        self._dense_cache: dict[str, np.ndarray] = {}

    # ---- construction helpers ----------------------------------------

    def _resolve(self, p: str | None) -> Path | None:
        if not p:
            return None
        cand = Path(p)
        if cand.is_absolute():
            return cand
        for base in (self.compiled_dir, self.compiled_dir.parent, Path.cwd()):
            if (base / cand).exists():
                return (base / cand).resolve()
        return cand

    def _load_layer_norms(self, b: int) -> dict:
        norms = {}
        for key in _LAYER_NORM_KEYS:
            raw, _ = read_tensor_raw(f"model.layers.{b}.{key}.weight", model_file=self.source_model)
            norms[key] = _bf16_bits_to_f64(raw)
        for key in ("q_norm", "k_norm"):
            raw, _ = read_tensor_raw(
                f"model.layers.{b}.self_attn.{key}.weight", model_file=self.source_model
            )
            norms[key] = _bf16_bits_to_f64(raw)
        return norms

    def _load_layer_linears(self, b: int) -> dict:
        compiled = {}
        for entry in self.manifest["tensors"].values():
            if entry.get("status") != "PASS" or entry.get("layer") != b:
                continue
            route_ids = np.load(self.compiled_dir / entry["route_map"])
            compiled[entry["role"]] = CompiledTensor(
                str(self.compiled_dir / entry["artifact"]),
                route_ids,
                tuple(entry["shape"]),
            )
        return compiled

    @classmethod
    def load(cls, artifact_path: str, device=None) -> RailNetModel:
        p = Path(artifact_path)
        if p.suffix == ".rnmodel":
            from railnet.artifacts.reader import read_rnmodel

            return read_rnmodel(str(p), device=device)
        if p.is_dir():
            manifest_path = p / "manifest.json"
        else:
            manifest_path = p
            p = p.parent
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest.json at {manifest_path}")
        return cls(json.loads(manifest_path.read_text()), p, device=device)

    # ---- linear backends -------------------------------------------

    def _rail_backend(self, b: int):
        comp = self._linears[b]

        def lin(short: str, x: np.ndarray) -> np.ndarray:
            c = comp[short]
            out = np.empty((x.shape[0], c.out_features), dtype=np.float64)
            for r in range(x.shape[0]):
                out[r] = rail_linear_fast(x[r].astype(np.float64), c)
            return out

        return lin

    def _dense_weight(self, layer: int, role: str) -> np.ndarray:
        """Reference only — the original BF16 weight streamed from the safetensors."""
        name = _weight_name(layer, role)
        w = self._dense_cache.get(name)
        if w is None:
            raw, shape = read_tensor_raw(name, model_file=self.source_model)
            w = bf16_array_to_float32(raw).astype(np.float64).reshape(shape)
            self._dense_cache[name] = w
        return w

    def _dense_backend(self, b: int):
        def lin(short: str, x: np.ndarray) -> np.ndarray:
            return x @ self._dense_weight(b, short).T

        return lin

    def _backend(self, name: str):
        if name == "rail":
            return self._rail_backend
        if name == "dense":
            return self._dense_backend
        raise ValueError(f"unknown backend {name!r} (expected 'rail' or 'dense')")

    # ---- execution ------------------------------------------------

    def embed(self, ids) -> np.ndarray:
        """Token rows scaled by Gemma3's BF16(sqrt(hidden)) normalizer."""
        return self._emb.rows_f64([int(t) for t in ids]) * self.ctx.embed_scale

    def run_layers(self, h, caches, pos_offset, backend="rail", capture_hidden=False):
        make_lin = self._backend(backend)
        hidden = []
        for b in range(self.n_layers):
            h, caches[b] = block_forward(
                h,
                self._norms[b],
                make_lin(b),
                self.ctx,
                cache=caches[b],
                pos_offset=pos_offset,
                layer_idx=b,
            )
            if capture_hidden:
                hidden.append(h.copy())
        return (h, caches, hidden) if capture_hidden else (h, caches)

    def forward(self, input_ids, backend="rail", capture_hidden=False):
        """Prefill over ``input_ids``; return final-token logits (vocab,) float64.

        With ``capture_hidden=True`` returns ``(logits, [per-layer hidden])``.
        """
        ids = [int(t) for t in np.asarray(input_ids).reshape(-1)]
        caches: list = [None] * self.n_layers
        out = self.run_layers(
            self.embed(ids), caches, 0, backend=backend, capture_hidden=capture_hidden
        )
        h = out[0]
        logits = self._emb.logits_chunked(rms_norm(h[-1:], self._final_norm, self.ctx)[0])
        return (logits, out[2]) if capture_hidden else logits

    def forward_dense(self, input_ids, **kw):
        """Verification reference: same ops, weights streamed dense from the safetensors."""
        return self.forward(input_ids, backend="dense", **kw)

    def generate(self, prompt, max_new_tokens: int = 64, tokenizer=None, **_kwargs) -> dict:
        from railnet.runtime.generation import generate

        return generate(self, prompt, max_new_tokens=max_new_tokens, tokenizer=tokenizer)

    # ---- tokenizer ----------------------------------------------

    def get_tokenizer(self):
        if self.tokenizer_path is None or not Path(self.tokenizer_path).exists():
            raise FileNotFoundError(
                "no tokenizer.json recorded in the manifest — pass tokenizer= to generate()"
            )
        try:
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError("pip install tokenizers  (or: pip install 'railnet[gemma]')") from e
        return Tokenizer.from_file(str(self.tokenizer_path))
