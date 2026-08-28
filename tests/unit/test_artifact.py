"""Unit tests for artifact format, manifest, writer, reader."""
import hashlib
import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from railnet.artifacts.format import MAGIC, VERSION, ArtifactHeader, TensorRecord
from railnet.artifacts.manifest import build_manifest, checksum_manifest, save_manifest
from railnet.artifacts.writer import write_rnmodel, write_manifest_json
from railnet.artifacts.reader import read_rnmodel, verify_rnmodel


# ── ArtifactHeader ────────────────────────────────────────

class TestArtifactHeader:
    def test_defaults(self):
        h = ArtifactHeader()
        assert h.magic == b"RNET"
        assert h.version == 1
        assert h.tensor_count == 0

    def test_custom(self):
        h = ArtifactHeader(model="llama", dtype="fp16", tensor_count=5)
        assert h.model == "llama"
        assert h.dtype == "fp16"


# ── TensorRecord ──────────────────────────────────────────

class TestTensorRecord:
    def _make_record(self):
        return TensorRecord(
            name="layer0.q_proj.weight",
            shape=[1152, 1024],
            dtype="bf16",
            rail_count=96,
            max_terms=4,
            rails_bits=[0x3F80, 0xBF80],
            routes={"16256": [[0, 1]], "49024": [[1, -1]]},
        )

    def test_basic(self):
        r = self._make_record()
        assert r.name == "layer0.q_proj.weight"
        assert r.rail_count == 96

    def test_canonical_bytes(self):
        r = self._make_record()
        b = r.canonical_bytes()
        assert isinstance(b, bytes)
        # Must be valid JSON
        data = json.loads(b)
        assert data["name"] == "layer0.q_proj.weight"

    def test_canonical_bytes_deterministic(self):
        r1 = self._make_record()
        r2 = self._make_record()
        assert r1.canonical_bytes() == r2.canonical_bytes()

    def test_compute_checksum(self):
        r = self._make_record()
        cs = r.compute_checksum()
        assert len(cs) == 64  # SHA-256 hex
        # Deterministic
        assert cs == r.compute_checksum()


# ── Manifest ──────────────────────────────────────────────

class TestManifest:
    def test_build_manifest(self):
        tensors = [{"name": "t1", "rails": 96}]
        m = build_manifest("gemma3", "bf16", tensors)
        assert m["magic"] == "RNET"
        assert m["version"] == 1
        assert m["model"] == "gemma3"
        assert m["tensor_count"] == 1
        assert m["runtime_weight_array"] == "ABSENT"

    def test_checksum_manifest(self):
        m = build_manifest("test", "bf16", [])
        cs = checksum_manifest(m)
        assert len(cs) == 64
        # Deterministic
        assert cs == checksum_manifest(m)

    def test_checksum_excludes_self(self):
        m = build_manifest("test", "bf16", [])
        cs1 = checksum_manifest(m)
        m["checksum_sha256"] = "something_different"
        cs2 = checksum_manifest(m)
        assert cs1 == cs2  # checksum_sha256 key excluded from hash

    def test_save_manifest(self, tmp_path):
        m = build_manifest("test", "bf16", [{"name": "t1"}])
        path = str(tmp_path / "manifest.json")
        save_manifest(path, m)
        loaded = json.loads(Path(path).read_text())
        assert loaded["model"] == "test"
        assert "checksum_sha256" in loaded


# ── Writer / Reader roundtrip ─────────────────────────────

class TestRnmodelRoundtrip:
    def test_write_basic(self, tmp_path):
        out = str(tmp_path / "test.rnmodel")
        tensors = [{"name": "t1", "shape": [4, 3], "dtype": "bf16", "rails": [0x3F80]}]
        result = write_rnmodel(out, "gemma3", "bf16", tensors)
        assert Path(result).exists()
        # Check magic
        with open(result, "rb") as f:
            magic = f.read(4)
            assert magic == MAGIC

    def test_write_with_route_maps(self, tmp_path):
        out = str(tmp_path / "test.rnmodel")
        tensors = [{"name": "t1"}]
        route_maps = {"t1": np.array([0, 1, 0, 1], dtype=np.uint16)}
        result = write_rnmodel(out, "test", "bf16", tensors, route_maps=route_maps)
        assert Path(result).exists()

    def test_verify_valid(self, tmp_path):
        out = str(tmp_path / "test.rnmodel")
        tensors = [{"name": "t1"}]
        write_rnmodel(out, "test", "bf16", tensors)
        ok, info = verify_rnmodel(out)
        assert ok is True
        assert info.get("model") == "test"

    def test_verify_bad_magic(self, tmp_path):
        out = str(tmp_path / "bad.rnmodel")
        with open(out, "wb") as f:
            f.write(b"XXXX")  # bad magic
            f.write(struct.pack("<I", 1))
            f.write(struct.pack("<I", 2))
            f.write(b"{}")
        ok, info = verify_rnmodel(out)
        assert ok is False
        assert "bad magic" in info.get("error", "")


# ── write_manifest_json ───────────────────────────────────

class TestWriteManifestJson:
    def test_write_and_read(self, tmp_path):
        path = str(tmp_path / "out.json")
        data = {"model": "test", "version": 1}
        write_manifest_json(path, data)
        loaded = json.loads(Path(path).read_text())
        assert loaded["model"] == "test"
