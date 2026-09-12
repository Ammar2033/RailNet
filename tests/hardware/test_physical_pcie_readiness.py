"""Physical FPGA & PCIe Hardware Prototyping Readiness Verification Suite.

Validates:
1. Hardware constraint files (.xdc and .lpf) against Verilog wrapper port definitions.
2. Timing constraints, clock declarations, and clock domain crossing false paths.
3. Host PCIe driver bridge architecture (LitePCIe, FTDI USB, Mock fallback).
4. Multi-target FPGA build engine CLI and script generation.
"""

from pathlib import Path
import re
import pytest

from railnet.runtime.pcie import (
    FtdiUsbBridge,
    LitePCIeBridge,
    MockPCIeBridge,
    RailNetPCIeDriver,
    REG_CTRL,
    REG_STATUS,
    REG_IN_FEATURES,
    REG_OUT_FEATURES,
    REG_TILE_MASK,
    REG_CYCLE_COUNT,
)
from hardware.fpga.build_fpga import build_fpga, get_default_targets

ROOT = Path(__file__).resolve().parents[2]
WRAPPER_V = ROOT / "hardware" / "fpga" / "railnet_pcie_wrapper.v"
XDC_PATH = ROOT / "hardware" / "fpga" / "qmtech_artix7_pcie.xdc"
LPF_PATH = ROOT / "hardware" / "fpga" / "ecp5_versa.lpf"


def test_wrapper_verilog_exists_and_declares_ports():
    """Verify synthesizable railnet_pcie_wrapper.v exists and has key ports."""
    assert WRAPPER_V.exists(), f"Missing wrapper at {WRAPPER_V}"
    text = WRAPPER_V.read_text(encoding="utf-8")

    expected_ports = [
        "pcie_clk",
        "pcie_rst_n",
        "core_clk",
        "core_rst_n",
        "s_axi_awaddr",
        "s_axi_wdata",
        "s_axi_rdata",
        "s_axis_dma_tdata",
        "s_axis_dma_tvalid",
        "s_axis_dma_tready",
        "m_axis_dma_tdata",
        "m_axis_dma_tvalid",
        "m_axis_dma_tready",
        "irq_out",
        "debug_leds",
    ]
    for port in expected_ports:
        assert re.search(rf"\b{port}\b", text), f"Port '{port}' missing in railnet_pcie_wrapper.v"


def test_artix7_xdc_syntax_and_port_alignment():
    """Verify QMTech Artix-7 XDC constraints match wrapper ports and declare valid timing."""
    assert XDC_PATH.exists(), f"Missing XDC constraint file at {XDC_PATH}"
    xdc_text = XDC_PATH.read_text(encoding="utf-8")

    # 1. Check fundamental reset and clock pins
    assert "get_ports pcie_rst_n" in xdc_text
    assert "get_ports sys_clk" in xdc_text
    assert "get_ports {debug_leds[" in xdc_text

    # 2. Check timing constraints exist
    assert "create_clock -period 10.000" in xdc_text  # 100 MHz PCIe refclk
    assert "create_clock -period 20.000" in xdc_text  # 50 MHz system clock
    assert "set_clock_groups -asynchronous" in xdc_text  # CDC false path

    # 3. Check IOSTANDARD definitions
    assert "IOSTANDARD LVCMOS33" in xdc_text
    assert "PULLUP true" in xdc_text  # For PERST# reset line


def test_ecp5_lpf_syntax_and_port_alignment():
    """Verify Lattice ECP5 Versa LPF preferences define pins and clocks."""
    assert LPF_PATH.exists(), f"Missing LPF file at {LPF_PATH}"
    lpf_text = LPF_PATH.read_text(encoding="utf-8")

    assert 'LOCATE COMP "clk_100_p"' in lpf_text
    assert 'LOCATE COMP "rst_n"' in lpf_text
    assert 'LOCATE COMP "pcie_perst_n"' in lpf_text
    assert 'LOCATE COMP "debug_leds[0]"' in lpf_text
    assert 'FREQUENCY PORT "clk_100_p" 100.0 MHz' in lpf_text


def test_pcie_driver_auto_discovery():
    """Test RailNetPCIeDriver automatically falls back to Mock when no hardware card is plugged in."""
    driver = RailNetPCIeDriver(backend="auto")
    assert not driver.is_hardware
    assert driver.hardware_backend == "mock"

    # Verify CSR read/write via Mock bridge
    driver.write_csr(REG_IN_FEATURES, 256)
    driver.write_csr(REG_OUT_FEATURES, 64)
    assert driver.read_csr(REG_IN_FEATURES) == 256
    assert driver.read_csr(REG_OUT_FEATURES) == 64


def test_pcie_driver_explicit_mock_backend():
    """Test explicitly specifying mock backend."""
    driver = RailNetPCIeDriver(backend="mock")
    assert not driver.is_hardware
    assert isinstance(driver.bridge, MockPCIeBridge)


def test_pcie_driver_invalid_backend_raises():
    """Test passing invalid backend raises ValueError."""
    with pytest.raises(ValueError, match="Unknown backend"):
        RailNetPCIeDriver(backend="quantum_bus")


def test_litepcie_bridge_offline_behavior():
    """Test LitePCIeBridge when /dev/litepcie node is absent (graceful exception)."""
    bridge = LitePCIeBridge(dev_path="/dev/nonexistent_litepcie_test")
    assert not bridge.is_connected

    with pytest.raises(RuntimeError, match="LitePCIe device.*not open"):
        bridge.write_csr(0x00, 0x1234)

    with pytest.raises(RuntimeError, match="LitePCIe device.*not open"):
        bridge.read_csr(0x00)


def test_ftdi_usb_bridge_packet_protocol():
    """Test FTDI USB packet creation, framing, and command encoding."""
    bridge = FtdiUsbBridge(baudrate=3000000)
    assert not bridge.is_connected

    # Test CSR Write Packet Framing: [0x01, addr_hi, addr_lo, b0, b1, b2, b3]
    pkt_w = bridge._build_csr_write_packet(0x0010, 0xAABBCCDD)
    assert len(pkt_w) == 7
    assert pkt_w[0] == bridge.CMD_WRITE_CSR
    assert pkt_w[1] == 0x00
    assert pkt_w[2] == 0x10
    assert pkt_w[3:7] == (0xAABBCCDD).to_bytes(4, byteorder="little")

    # Test CSR Read Packet Framing: [0x02, addr_hi, addr_lo]
    pkt_r = bridge._build_csr_read_packet(0x0024)
    assert len(pkt_r) == 3
    assert pkt_r[0] == bridge.CMD_READ_CSR
    assert pkt_r[1] == 0x00
    assert pkt_r[2] == 0x24

    # Test DMA H2C Packet Framing: [0x03, len_hi, len_lo, payload...]
    payload = bytes([0x11, 0x22, 0x33, 0x44])
    pkt_dma = bridge._build_dma_h2c_packet(payload)
    assert len(pkt_dma) == 3 + len(payload)
    assert pkt_dma[0] == bridge.CMD_DMA_H2C
    assert int.from_bytes(pkt_dma[1:3], byteorder="big") == len(payload)
    assert pkt_dma[3:] == payload


def test_fpga_build_cli_dryrun(tmp_path):
    """Test multi-target FPGA build engine dry-run mode for both ECP5 and Artix-7."""
    # Report goes to tmp_path: a dry run must never overwrite
    # results/fpga_pnr_results.json, which holds the real P&R evidence.
    res_ecp5 = build_fpga(target_platform="ecp5", dry_run=True, results_path=tmp_path / "ecp5.json")
    assert res_ecp5["platform_selection"] == "ecp5"
    assert res_ecp5["dry_run"] is True
    assert "railnet_top_2x2_ecp5" in res_ecp5["targets"]
    assert res_ecp5["targets"]["railnet_top_2x2_ecp5"]["status"] == "DRY_RUN_SCRIPTS_GENERATED"

    res_artix7 = build_fpga(target_platform="artix7", dry_run=True, results_path=tmp_path / "artix7.json")
    assert res_artix7["platform_selection"] == "artix7"
    assert "railnet_top_2x2_artix7" in res_artix7["targets"]
    assert res_artix7["targets"]["railnet_top_2x2_artix7"]["status"] == "DRY_RUN_SCRIPTS_GENERATED"


def test_failed_pnr_is_not_reported_as_a_completed_run():
    """A P&R that aborts must classify as PNR_FAILED.

    Regression: nextpnr signals failure by raising SystemExit, which was caught
    and ignored. An unroutable design was then recorded with the same shape as a
    clean run (and tagged FPGA-MEASURED), while the build still exited 0. The
    real failure this guards against is railnet_top_2x2 aborting with
    "ERROR: IO 's_axis_tvalid' is unconstrained in LPF".
    """
    from hardware.fpga.build_fpga import _derive_pnr_status

    unconstrained_io = ["ERROR: IO 's_axis_tvalid' is unconstrained in LPF"]

    # Tool aborted, and it logged an error.
    assert _derive_pnr_status(1, unconstrained_io, routed_success=False) == "PNR_FAILED"
    # Exit code alone is enough.
    assert _derive_pnr_status(1, [], routed_success=True) == "PNR_FAILED"
    # An ERROR line alone is enough, even on a zero exit.
    assert _derive_pnr_status(0, unconstrained_io, routed_success=True) == "PNR_FAILED"
    # No "Routing complete." in the log is enough.
    assert _derive_pnr_status(0, [], routed_success=False) == "PNR_FAILED"
    # Only a clean, routed run counts.
    assert _derive_pnr_status(0, [], routed_success=True) == "PNR_COMPLETED"
