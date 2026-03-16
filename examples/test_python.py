#!/usr/bin/env python3
"""
Test script for gpushare Python client.
Run the server first, then: python test_python.py [server_host]
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import gpushare


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    print(f"Connecting to gpushare server at {host}...")

    with gpushare.connect(host) as gpu:
        # Test 1: Ping
        assert gpu.ping(), "Ping failed"
        print("[PASS] Ping")

        # Test 2: Device info
        print(f"\n{gpu.info()}\n")
        print(f"[PASS] Device query")

        # Test 3: Memory alloc + transfer
        N = 1024 * 1024  # 1M floats = 4 MB
        data = np.random.randn(N).astype(np.float32)

        d_ptr = gpu.malloc(data.nbytes)
        print(f"[PASS] Allocated {data.nbytes / 1024:.0f} KB on GPU")

        # H2D transfer with timing
        t0 = time.perf_counter()
        gpu.memcpy_h2d(d_ptr, data)
        t_h2d = time.perf_counter() - t0
        bw_h2d = data.nbytes / t_h2d / 1e6
        print(f"[PASS] H2D: {t_h2d*1000:.1f} ms ({bw_h2d:.0f} MB/s)")

        # D2H transfer with timing
        result = np.empty_like(data)
        t0 = time.perf_counter()
        gpu.memcpy_d2h(result, d_ptr, data.nbytes)
        t_d2h = time.perf_counter() - t0
        bw_d2h = data.nbytes / t_d2h / 1e6
        print(f"[PASS] D2H: {t_d2h*1000:.1f} ms ({bw_d2h:.0f} MB/s)")

        # Verify
        assert np.allclose(data, result), "Data mismatch!"
        print(f"[PASS] Round-trip verified ({N} floats)")

        # Test 4: D2D copy
        d_ptr2 = gpu.malloc(data.nbytes)
        gpu.memcpy_d2d(d_ptr2, d_ptr, data.nbytes)
        result2 = np.empty_like(data)
        gpu.memcpy_d2h(result2, d_ptr2, data.nbytes)
        assert np.allclose(data, result2), "D2D mismatch!"
        print(f"[PASS] D2D copy verified")

        # Test 5: Large transfer benchmark
        sizes = [1024, 64*1024, 1024*1024, 16*1024*1024]
        print(f"\n--- Bandwidth benchmark ---")
        for size in sizes:
            buf = np.random.randn(size // 4).astype(np.float32)
            d = gpu.malloc(size)

            t0 = time.perf_counter()
            for _ in range(3):
                gpu.memcpy_h2d(d, buf)
            t = (time.perf_counter() - t0) / 3
            bw = size / t / 1e6

            gpu.free(d)
            print(f"  {size/1024:>8.0f} KB: {bw:>7.1f} MB/s  ({t*1000:.1f} ms)")

        # Cleanup
        gpu.free(d_ptr)
        gpu.free(d_ptr2)
        gpu.synchronize()

    print(f"\n=== All Python tests passed ===")


if __name__ == "__main__":
    main()
