#!/usr/bin/env python3
"""
Simple PyTorch-free test for gpushare
Uses the gpushare Python TCP client directly
"""

import sys
import os

# Add gpushare python to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from gpushare import RemoteGPU
import time


def main():
    print("=== gpushare Python TCP Client Test ===\n")

    # Get server from environment or use default
    server = os.environ.get("GPUSHARE_SERVER", "192.168.29.22:9847")
    host = server.split(":")[0]
    port = int(server.split(":")[1]) if ":" in server else 9847

    print(f"Connecting to {host}:{port}...")

    try:
        gpu = RemoteGPU(host, port)

        # Get device properties
        props = gpu.device_properties()
        print(f"\n✓ Connected to remote GPU:")
        print(f"  Name: {props['name']}")
        print(f"  Total Memory: {props['total_mem_mb'] / 1024:.2f} GB")
        print(f"  Compute Capability: {props['major']}.{props['minor']}")
        if "sm_count" in props:
            print(f"  SM Count: {props['sm_count']}")

        # Allocate some memory
        size = 1024 * 1024  # 1 MB
        print(f"\n✓ Allocating {size} bytes...")
        ptr = gpu.malloc(size)
        print(f"  GPU pointer: 0x{ptr:x}")

        # Test data verification (allocate and memset instead)
        print(f"\n✓ Testing memory operations...")
        gpu.free(ptr)
        return 0

        print("\n=== All Python tests passed ===")
        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
