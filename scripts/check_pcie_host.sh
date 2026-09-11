#!/usr/bin/env bash
# RailNet PCIe Host Preparation Check (Checklist 5 - Host PC Hazırlığı)
# Run on Linux host with PCIe slot before card arrival.
# Checks: kernel, lspci, hugepages, LitePCIe driver build, BAR0 size.

set -e

echo "=== RailNet PCIe Host Check (Gate 2) ==="
echo "Kernel: $(uname -r)"
echo "lspci:"
lspci | grep -i -E "fpga|xilinx|lattice|pcie" || echo "  (no FPGA PCIe device yet - expected before card)"

echo ""
echo "Hugepages (for DMA):"
cat /proc/meminfo | grep Huge || true
echo "  Recommended: echo 32 | sudo tee /proc/sys/vm/nr_hugepages"

echo ""
echo "Checking LitePCIe driver sources..."
if [ -d "litepcie" ]; then
  echo "  litepcie/ already cloned"
else
  echo "  Cloning enjoy-digital/litepcie..."
  git clone https://github.com/enjoy-digital/litepcie.git || echo "  (clone failed - check network)"
fi

if [ -f "litepcie/litepcie.h" ]; then
  echo "  LitePCIe header found"
fi

echo ""
echo "Checking kernel headers for driver build..."
if [ -d "/lib/modules/$(uname -r)/build" ]; then
  echo "  Kernel headers present: /lib/modules/$(uname -r)/build"
else
  echo "  MISSING: sudo apt install linux-headers-$(uname -r)"
fi

echo ""
echo "PCIe BAR check (after card insertion):"
echo "  Run: sudo lspci -s 01:00.0 -vvv | grep -A 5 BAR"
echo "  Expected: BAR0 32-bit, size 4K (LitePCIe) or 1M (XDMA)"
echo "  Run: sudo dmesg | grep -i litepcie"
echo "  Run: ls -l /dev/litepcie* /dev/xdma* 2>&1"

echo ""
echo "Python driver check (no hardware, mock fallback):"
python3 -c "from railnet.runtime.pcie import RailNetPCIeDriver; d=RailNetPCIeDriver('check',backend='auto'); print(d.get_link_status()); d.close()" || echo "  Python driver check failed"

echo ""
echo "FPGA toolchain check:"
which yosys && yosys -V || echo "  yosys not found (pip install yowasp-yosys)"
which nextpnr-ecp5 && nextpnr-ecp5 --version || echo "  nextpnr-ecp5 not found (pip install yowasp-nextpnr-ecp5)"
which vivado && vivado -version || echo "  vivado not found (optional, for Artix-7 bitstream)"

echo ""
echo "=== Host check complete. Ready for Gate 2 PCIe bring-up when card arrives. ==="
