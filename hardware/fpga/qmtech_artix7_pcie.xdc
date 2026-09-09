# =============================================================================
# RailNet FPGA Constraints for QMTech Xilinx Artix-7 XC7A35T/100T PCIe Board
# Target Package: FGG484 (-1 speed grade)
# PCIe Specification: Gen2 x1 Endpoint with LitePCIe / XDMA
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration Bank Voltage Select (CFGBVS)
# -----------------------------------------------------------------------------
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

# -----------------------------------------------------------------------------
# Onboard System Clock (50 MHz Single-Ended Oscillator)
# -----------------------------------------------------------------------------
set_property PACKAGE_PIN M21 [get_ports sys_clk]
set_property IOSTANDARD LVCMOS33 [get_ports sys_clk]
create_clock -period 20.000 -name sys_clk [get_ports sys_clk]

# -----------------------------------------------------------------------------
# PCIe Fundamental Reset (PERST# from PCIe Edge Connector)
# Active-Low, 3.3V LVCMOS with internal pull-up
# -----------------------------------------------------------------------------
set_property PACKAGE_PIN J20 [get_ports pcie_rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports pcie_rst_n]
set_property PULLUP true [get_ports pcie_rst_n]
set_false_path -from [get_ports pcie_rst_n]

# -----------------------------------------------------------------------------
# PCIe Reference Clock (100 MHz HCSL Differential Pair on GTP Quad 216)
# -----------------------------------------------------------------------------
set_property PACKAGE_PIN F6 [get_ports pcie_refclk_p]
set_property PACKAGE_PIN E6 [get_ports pcie_refclk_n]
create_clock -period 10.000 -name pcie_refclk [get_ports pcie_refclk_p]

# -----------------------------------------------------------------------------
# PCIe Gen2 Lane 0 Serial Transceiver Pins (GTP Quad 216 Channel 0)
# -----------------------------------------------------------------------------
set_property PACKAGE_PIN B4 [get_ports pcie_rx_p]
set_property PACKAGE_PIN A4 [get_ports pcie_rx_n]
set_property PACKAGE_PIN B8 [get_ports pcie_tx_p]
set_property PACKAGE_PIN A8 [get_ports pcie_tx_n]

# -----------------------------------------------------------------------------
# Status Diagnostic LEDs (Active-High, LVCMOS33)
#   debug_leds[0]: DMA H2C streaming active (data transferring into card)
#   debug_leds[1]: DMA C2H streaming active (data transferring back to host)
#   debug_leds[2]: Frame completion interrupt active
#   debug_leds[3]: PCIe link / reset status (alive heartbeat)
# -----------------------------------------------------------------------------
set_property PACKAGE_PIN E22 [get_ports {debug_leds[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {debug_leds[0]}]
set_property DRIVE 8 [get_ports {debug_leds[0]}]
set_property SLEW SLOW [get_ports {debug_leds[0]}]

set_property PACKAGE_PIN D22 [get_ports {debug_leds[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {debug_leds[1]}]
set_property DRIVE 8 [get_ports {debug_leds[1]}]
set_property SLEW SLOW [get_ports {debug_leds[1]}]

set_property PACKAGE_PIN G21 [get_ports {debug_leds[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {debug_leds[2]}]
set_property DRIVE 8 [get_ports {debug_leds[2]}]
set_property SLEW SLOW [get_ports {debug_leds[2]}]

set_property PACKAGE_PIN G22 [get_ports {debug_leds[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {debug_leds[3]}]
set_property DRIVE 8 [get_ports {debug_leds[3]}]
set_property SLEW SLOW [get_ports {debug_leds[3]}]

set_false_path -to [get_ports {debug_leds[*]}]

# -----------------------------------------------------------------------------
# Clock Domain Crossing (CDC) Timing Constraints
# pcie_clk (62.5 / 125 MHz) <-> core_clk (100 MHz) are asynchronous
# Handled safely via 47-bit Gray-coded AsyncFIFO in hardware/rtl/cdc.py
# -----------------------------------------------------------------------------
set_clock_groups -asynchronous \
    -group [get_clocks -include_generated_clocks sys_clk] \
    -group [get_clocks -include_generated_clocks pcie_refclk]

# -----------------------------------------------------------------------------
# Bitstream Generation Settings
# -----------------------------------------------------------------------------
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 33 [current_design]
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
