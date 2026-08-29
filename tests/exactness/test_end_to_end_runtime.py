"""End-to-end runtime proof on a tiny synthetic Gemma-shaped model.

Builds a small BF16 safetensors model, compiles every linear with the real
RailNet compiler, then checks that:

  * every tensor compiles losslessly (compiler verdict PASS),
  * ``RailNetModel.forward`` runs with no dense weight array, and
  * RailNet logits are BF16-bitwise equal to a dense reference that shares
    the exact same transformer ops (only the linear backend differs),
  * greedy generation is deterministic.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from railnet.compiler.model import compile_model
from railnet.dtypes.bf16 import bf16_array_to_float32, fp32_array_to_bf16_bits
from railnet.runtime.transformer import RailNetModel
from railnet.transformer import GemmaContext, block_forward, rms_norm

CFG = {
    "model_type": "gemma3_text",
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 4,
    "vocab_size": 24,
    "rms_norm_eps": 1e-6,
    "query_pre_attn_scalar": 4,
    "rope_local_base_freq": 10000.0,
    "eos_token_id": [23],
}


def _bf16(arr: np.ndarray) -> np.ndarray:
    """Round-trip a float array through BF16 truncation (matches the reader)."""
    return bf16_array_to_float32(fp32_array_to_bf16_bits(arr.astype(np.float32)))


def _write_safetensors(path, tensors: dict[str, np.ndarray]) -> None:
    header, blob, offset = {}, bytearray(), 0
    for name, arr in tensors.items():
        bits = fp32_array_to_bf16_bits(arr.astype(np.float32)).tobytes()
        header[name] = {
            "dtype": "BF16",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(bits)],
        }
        blob += bits
        offset += len(bits)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


def _build_model(rng, tmp_path):
    H, I = CFG["hidden_size"], CFG["intermediate_size"]
    nh, nkv, hd = (
        CFG["num_attention_heads"],
        CFG["num_key_value_heads"],
        CFG["head_dim"],
    )
    V, L = CFG["vocab_size"], CFG["num_hidden_layers"]

    def w(*shape):
        # Quantize onto a small value grid so the compiler reaches lossless
        # coverage in very few iterations (keeps the test fast).
        q = np.round(rng.standard_normal(shape) * 6.0) / 128.0
        return q.astype(np.float32)

    tensors = {"model.embed_tokens.weight": w(V, H), "model.norm.weight": w(H)}
    for b in range(L):
        p = f"model.layers.{b}"
        tensors.update(
            {
                f"{p}.self_attn.q_proj.weight": w(nh * hd, H),
                f"{p}.self_attn.k_proj.weight": w(nkv * hd, H),
                f"{p}.self_attn.v_proj.weight": w(nkv * hd, H),
                f"{p}.self_attn.o_proj.weight": w(H, nh * hd),
                f"{p}.self_attn.q_norm.weight": w(hd),
                f"{p}.self_attn.k_norm.weight": w(hd),
                f"{p}.mlp.gate_proj.weight": w(I, H),
                f"{p}.mlp.up_proj.weight": w(I, H),
                f"{p}.mlp.down_proj.weight": w(H, I),
                f"{p}.input_layernorm.weight": w(H),
                f"{p}.post_attention_layernorm.weight": w(H),
                f"{p}.pre_feedforward_layernorm.weight": w(H),
                f"{p}.post_feedforward_layernorm.weight": w(H),
            }
        )

    src = tmp_path / "model.safetensors"
    _write_safetensors(src, tensors)
    (tmp_path / "config.json").write_text(json.dumps(CFG))
    return src, tensors


def _dense_reference(tensors, input_ids):
    """Same transformer ops as RailNet, but linears are plain dense matmuls
    over the BF16-rounded weights."""
    ctx = GemmaContext(CFG)
    emb = _bf16(tensors["model.embed_tokens.weight"])
    h = emb[list(input_ids)].astype(np.float64) * ctx.embed_scale
    caches = [None] * CFG["num_hidden_layers"]
    for b in range(CFG["num_hidden_layers"]):
        p = f"model.layers.{b}"
        norms = {
            "input_layernorm": _bf16(tensors[f"{p}.input_layernorm.weight"]),
            "post_attention_layernorm": _bf16(tensors[f"{p}.post_attention_layernorm.weight"]),
            "pre_feedforward_layernorm": _bf16(tensors[f"{p}.pre_feedforward_layernorm.weight"]),
            "post_feedforward_layernorm": _bf16(tensors[f"{p}.post_feedforward_layernorm.weight"]),
            "q_norm": _bf16(tensors[f"{p}.self_attn.q_norm.weight"]),
            "k_norm": _bf16(tensors[f"{p}.self_attn.k_norm.weight"]),
        }

        def lin(short, x, p=p):
            key = (
                f"{p}.self_attn.{short}.weight"
                if short.endswith("_proj") and short[0] in "qkvo"
                else f"{p}.mlp.{short}.weight"
            )
            wgt = _bf16(tensors[key]).astype(np.float64)
            return x @ wgt.T

        h, caches[b] = block_forward(h, norms, lin, ctx, cache=caches[b], pos_offset=0, layer_idx=b)
    hf = rms_norm(h[-1:], _bf16(tensors["model.norm.weight"]), ctx)
    return emb.astype(np.float64) @ hf[0]


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("railnet_e2e")
    rng = np.random.default_rng(0)
    src, tensors = _build_model(rng, tmp)
    out = tmp / "compiled"
    manifest = compile_model(
        str(src), out_dir=str(out), rails=32, max_terms=4, max_iters=8, verbose=False
    )
    return manifest, out, tensors


def test_all_tensors_compiled_losslessly(compiled):
    manifest, _out, _tensors = compiled
    assert manifest["verdict"] == "PASS"
    assert manifest["fail_count"] == 0
    # 7 linears * 2 layers
    assert manifest["pass_count"] == 14


def test_no_dense_weight_array(compiled):
    manifest, out, _tensors = compiled
    assert manifest["runtime_weight_array"] == "ABSENT"
    model = RailNetModel.load(str(out))
    for layer in model._linears:
        for c in layer.values():
            assert not hasattr(c, "weight")
            assert not hasattr(c, "W")
            assert c.checksum_ok


def test_railnet_logits_bf16_exact_vs_dense(compiled):
    _manifest, out, tensors = compiled
    ids = [3, 1, 4, 1, 5]
    model = RailNetModel.load(str(out))

    rail_logits = model.forward(ids)
    ref_logits = _dense_reference(tensors, ids)

    assert np.all(np.isfinite(rail_logits))
    rail_bits = fp32_array_to_bf16_bits(rail_logits.astype(np.float32))
    ref_bits = fp32_array_to_bf16_bits(ref_logits.astype(np.float32))
    assert np.array_equal(rail_bits, ref_bits), (
        f"{int(np.count_nonzero(rail_bits != ref_bits))}/{rail_bits.size} logit bits differ"
    )


def test_generation_is_deterministic(compiled):
    _manifest, out, _tensors = compiled
    model = RailNetModel.load(str(out))
    a = model.generate([2, 7, 1], max_new_tokens=6)
    b = model.generate([2, 7, 1], max_new_tokens=6)
    assert a["tokens"] == b["tokens"]
    assert 1 <= len(a["tokens"]) <= 6


def test_verify_compiled_directory(compiled):
    from railnet.compiler.model import verify_compiled

    _manifest, out, _tensors = compiled
    r = verify_compiled(str(out))
    assert r["ok"], r["problems"]
    assert r["tensors_checked"] == 14
    assert r["verdict"] == "PASS"


def test_verify_forward_rail_vs_dense(compiled):
    from railnet.verification import verify_forward

    _manifest, out, _tensors = compiled
    model = RailNetModel.load(str(out))
    r = verify_forward(model, [3, 1, 4, 1, 5], per_layer=True)
    assert r["verdict"] == "PASS"
    assert r["logit_bf16_mismatch"] == 0
    assert r["all_layers_exact"], r["first_divergent_layer"]


def test_verify_generation_rail_vs_dense(compiled):
    from railnet.verification import verify_generation

    _manifest, out, _tensors = compiled
    model = RailNetModel.load(str(out))
    r = verify_generation(model, [2, 7, 1], max_new_tokens=6)
    assert r["verdict"] == "PASS"
    assert r["rail_tokens"] == r["dense_tokens"]


def test_representation_cost_is_honest(compiled):
    from railnet.analysis import representation_cost

    _manifest, out, _tensors = compiled
    r = representation_cost(str(out))
    t = r["totals"]
    assert r["tensors"] == 14
    # route-id map stored as uint16 == dense weight bits exactly
    assert t["route_id_bits_stored"] == t["dense_bits"]
    # so the full representation is strictly larger than dense
    assert t["effective_bits_stored"] > t["dense_bits"]
    assert t["ratio_stored"] > 1.0
    # no dramatic compression even at the theoretical minimum id width
    assert t["ratio_min"] > 0.3


def test_verify_kernel_rail_vs_dense(compiled):
    from railnet.verification.kernel import verify_kernel

    _manifest, out, _tensors = compiled
    model = RailNetModel.load(str(out))
    c = model._linears[0]["gate_proj"]
    rng = np.random.default_rng(2)
    x = (rng.standard_normal(c.in_features) * 0.1).astype(np.float64)
    assert verify_kernel(x, model._dense_weight(0, "gate_proj"), c)["ok"]


def test_rail_oracle_matches_dense_oracle(compiled):
    """Tightest tier: Fraction-exact rail eval == Fraction-exact dense eval."""
    from railnet.verification import dense_oracle, rail_oracle

    _manifest, out, _tensors = compiled
    model = RailNetModel.load(str(out))
    c = model._linears[0]["q_proj"]
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(c.in_features) * 0.1).astype(np.float64)

    art = json.loads((out / "layers/layer_00/q_proj.json").read_text())
    routes = {int(k): tuple(tuple(t) for t in v) for k, v in art["routes"].items()}
    dense_w = model._dense_weight(0, "q_proj")

    y_rail = rail_oracle(x, np.array(art["rails"], dtype=np.uint16), routes, c.route_ids)
    y_dense = dense_oracle(x, dense_w)
    assert np.array_equal(fp32_array_to_bf16_bits(y_rail), fp32_array_to_bf16_bits(y_dense))


def test_from_source_dense_matches_compiled_dense(compiled):
    """A dense-only model built straight from the safetensors gives the same
    dense-backend result as the compiled model."""
    _manifest, out, _tensors = compiled
    src = out.parent / "model.safetensors"

    compiled_model = RailNetModel.load(str(out))
    source_model = RailNetModel.from_source(str(src))

    ids = [3, 1, 4, 1, 5]
    a = compiled_model.forward(ids, backend="dense")
    b = source_model.forward(ids, backend="dense")
    assert np.array_equal(a, b)

    with pytest.raises(KeyError, match="not compiled"):
        source_model.forward(ids, backend="rail")


def test_compile_resume_skips_completed(tmp_path):
    from railnet.compiler.model import compile_model

    rng = np.random.default_rng(3)
    src, _ = _build_model(rng, tmp_path)
    out = tmp_path / "c"

    first = compile_model(str(src), out_dir=str(out), rails=32, max_iters=8, verbose=False)
    arts = sorted(out.glob("layers/*/*.json"))
    mtimes = {p: p.stat().st_mtime_ns for p in arts}

    second = compile_model(
        str(src), out_dir=str(out), rails=32, max_iters=8, resume=True, verbose=False
    )
    assert second["pass_count"] == first["pass_count"] == 14
    # nothing recompiled -> per-tensor artifact files untouched
    assert {p: p.stat().st_mtime_ns for p in arts} == mtimes
