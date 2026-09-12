"""A compiled artifact must load on a machine other than the one that built it.

Regression: compiled/manifest.json in this repo records
`/Volumes/SSD/Projects/2026/Ongoing/RailNet/model_data/model.safetensors`, an
absolute path from the macOS host that compiled it. `_resolve` returned any
absolute path unchecked, so `RailNetModel.load()` raised FileNotFoundError
everywhere else and the project's headline exactness result could not be
reproduced by anyone but the original author.
"""

from __future__ import annotations

from pathlib import Path

from railnet.runtime.transformer import RailNetModel

STALE_MAC_PATH = "/Volumes/SSD/Projects/2026/Ongoing/RailNet/model_data/model.safetensors"


def _resolver(compiled_dir: Path) -> RailNetModel:
    """A bare instance: __init__ loads a real model, which these tests do not need."""
    model = object.__new__(RailNetModel)
    model.compiled_dir = Path(compiled_dir)
    return model


def _layout(tmp_path: Path) -> tuple[Path, Path]:
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    model_data = tmp_path / "model_data"
    model_data.mkdir()
    weights = model_data / "model.safetensors"
    weights.write_bytes(b"not really safetensors")
    return compiled, weights


def test_stale_absolute_path_falls_back_to_the_repo_copy(tmp_path):
    compiled, weights = _layout(tmp_path)
    assert _resolver(compiled)._resolve(STALE_MAC_PATH) == weights.resolve()


def test_an_absolute_path_that_exists_is_used_as_is(tmp_path):
    compiled, weights = _layout(tmp_path)
    assert _resolver(compiled)._resolve(str(weights)) == weights


def test_relative_paths_still_resolve(tmp_path):
    compiled, weights = _layout(tmp_path)
    assert _resolver(compiled)._resolve("model_data/model.safetensors") == weights.resolve()


def test_an_unresolvable_path_is_returned_unchanged_not_invented(tmp_path):
    """The caller checks .exists() and raises a clear error; do not guess."""
    compiled, _ = _layout(tmp_path)
    missing = "/nowhere/at/all/ghost.safetensors"
    assert _resolver(compiled)._resolve(missing) == Path(missing)


def test_none_stays_none(tmp_path):
    compiled, _ = _layout(tmp_path)
    assert _resolver(compiled)._resolve(None) is None
