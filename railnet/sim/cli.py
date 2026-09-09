"""Command-line interface for RailNet Cycle-Accurate Virtual Accelerator Simulator."""

import argparse
import sys
from pathlib import Path

from railnet.runtime.engine import RailNetDevice
from railnet.runtime.transformer import RailNetModel
from railnet.sim.accelerator import HardwareConfig


def main():
    parser = argparse.ArgumentParser(description="RailNet Cycle-Accurate Accelerator Simulator")
    parser.add_argument("--model", type=str, required=True, help="Path to compiled RailNet model directory")
    parser.add_argument(
        "--profile",
        type=str,
        default="edge_asic",
        choices=["edge_asic", "cloud_asic", "fpga_proto"],
        help="Hardware accelerator profile",
    )
    parser.add_argument("--tokens", type=int, default=1, help="Number of tokens to simulate")
    parser.add_argument("--tiles", type=int, default=None, help="Override number of hardware tiles")
    parser.add_argument("--freq", type=float, default=None, help="Override clock frequency in MHz")

    args = parser.parse_args()

    model_dir = Path(args.model)
    if not model_dir.exists():
        print(f"Error: Model directory '{args.model}' does not exist.", file=sys.stderr)
        sys.exit(1)

    device = RailNetDevice.sim(profile=args.profile, tiles=args.tiles, freq_mhz=args.freq)
    print(f"Loading compiled model from {model_dir} into {device}...")
    model = RailNetModel.load(str(model_dir), device=device)

    dummy_tokens = [1] * args.tokens
    print(f"Simulating forward inference on {args.tokens} token(s)...")
    model.forward(dummy_tokens)

    telemetry = device.get_telemetry()
    print("\n" + telemetry.summary())


if __name__ == "__main__":
    main()
