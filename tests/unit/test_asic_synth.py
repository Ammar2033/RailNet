"""Regression tests for the Sky130 standard-cell synthesis driver.

These pin the two defects that actually broke the Gate 3 run, plus the honesty
guarantees of the report. They are pure-function tests: no Docker, no Liberty
file and no Yosys invocation, so they run anywhere the package imports.
"""

from __future__ import annotations

from pathlib import Path

import pytest

synth_asic = pytest.importorskip("hardware.asic.synth_asic")


def _sources() -> list[Path]:
    return [synth_asic.ROOT / "hardware" / "rtl" / "build" / "railnet_top.v"]


def test_container_script_refers_to_every_input_by_bare_name():
    """Regression: the Liberty path must follow the same convention as the reads.

    The container stages all inputs flat into one directory. An earlier version
    made only `read_verilog` and the stat output basename-aware and left the
    Liberty as an absolute host path, so synthesis died inside the container
    with "Can't open liberty file 'F:/.../sky130_fd_sc_hd__tt_025C_1v80.lib'".
    """
    liberty = synth_asic.ROOT / "hardware" / "asic" / "pdk" / "sky130_fd_sc_hd__tt_025C_1v80.lib"
    script = synth_asic.build_script(liberty, _sources(), basenames=True)

    assert "read_verilog railnet_top.v" in script
    assert f"dfflibmap -liberty {liberty.name}" in script
    assert f"abc -fast -liberty {liberty.name}" in script
    # No host-absolute path may survive into a container script.
    assert str(synth_asic.ROOT) not in script
    assert liberty.as_posix() not in script


def test_local_script_keeps_repo_relative_paths():
    liberty = synth_asic.ROOT / "hardware" / "asic" / "pdk" / "sky130_fd_sc_hd__tt_025C_1v80.lib"
    script = synth_asic.build_script(liberty, _sources(), basenames=False)

    assert "read_verilog hardware/rtl/build/railnet_top.v" in script
    assert liberty.as_posix() in script


def test_script_targets_the_configured_top():
    liberty = synth_asic.ROOT / "hardware" / "asic" / "pdk" / "sky130_fd_sc_hd__tt_025C_1v80.lib"
    script = synth_asic.build_script(liberty, _sources(), basenames=True)
    assert f"hierarchy -check -top {synth_asic.TOP}" in script


def test_without_liberty_the_script_maps_nothing():
    """No Liberty means no mapping, and the script must not pretend otherwise."""
    script = synth_asic.build_script(None, _sources(), basenames=False)

    assert "abc" not in script
    assert "dfflibmap" not in script
    assert "stat -json" in script


def test_flatten_is_present_because_per_module_mapping_is_intractable():
    """Yosys 0.38 re-parses the 12.8 MB Liberty per module without flatten.

    Twenty re-parses were observed before the run was abandoned; flattened, the
    same design maps in about 11 seconds. Dropping `flatten` silently
    reintroduces an hours-long run, so pin it.
    """
    liberty = synth_asic.ROOT / "hardware" / "asic" / "pdk" / "sky130_fd_sc_hd__tt_025C_1v80.lib"
    script = synth_asic.build_script(liberty, _sources(), basenames=True)
    assert "\nflatten\n" in script
    # Memories stay out of the standard-cell area.
    assert "memory -nomap" in script


def test_parse_chip_area_prefers_the_top_module_total():
    log = (
        "Chip area for module '\\stage_a': 1000.5\n"
        "Chip area for module '\\stage_b': 2000.25\n"
        "Chip area for top module '\\railnet_top': 3000.75\n"
    )
    total, per_module = synth_asic.parse_chip_area(log)
    assert total == 3000.75
    assert per_module["stage_a"] == 1000.5
    assert per_module["stage_b"] == 2000.25


def test_parse_chip_area_reports_nothing_when_the_log_has_no_area():
    """Yosys 0.38 emits no "Chip area" line; the area comes from stat JSON.

    The parser must return None rather than inventing a number, so the caller
    falls back to the JSON instead of publishing a fabricated area.
    """
    total, per_module = synth_asic.parse_chip_area("Executing ABC pass.\n")
    assert total is None
    assert per_module == {}


def test_resolve_liberty_returns_none_for_a_missing_file(monkeypatch, tmp_path):
    monkeypatch.setenv("SKY130_LIB", str(tmp_path / "does_not_exist.lib"))
    assert synth_asic.resolve_liberty() is None


def test_resolve_liberty_rejects_a_truncated_download(monkeypatch, tmp_path):
    """A partial Liberty would map against an incomplete cell set and still
    produce a number, which is worse than having no Liberty at all."""
    bad = tmp_path / "truncated.lib"
    bad.write_text("<!DOCTYPE html><html>404 Not Found</html>", encoding="utf-8")
    monkeypatch.setenv("SKY130_LIB", str(bad))
    assert synth_asic.resolve_liberty() is None
