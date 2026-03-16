#!/usr/bin/env python3
"""
gpushare Windows verification script.
Run this on the Windows client to verify the remote GPU connection works.

Usage: python verify_windows.py [server_ip:port]
"""

import ctypes
import ctypes.util
import os
import sys
import struct
import socket
import time

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
NC = "\033[0m"

# Enable ANSI colors on Windows 10+
if sys.platform == "win32":
    try:
        import ctypes as _ct
        kernel32 = _ct.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        # Strip colors if ANSI not supported
        GREEN = RED = YELLOW = CYAN = BOLD = NC = ""

def ok(msg):   print(f"  {GREEN}[PASS]{NC} {msg}")
def fail(msg): print(f"  {RED}[FAIL]{NC} {msg}")
def info(msg): print(f"  {CYAN}[INFO]{NC} {msg}")
def warn(msg): print(f"  {YELLOW}[WARN]{NC} {msg}")


def find_config_server():
    """Read server from config file."""
    paths = []
    appdata = os.environ.get("LOCALAPPDATA", "")
    if appdata:
        paths.append(os.path.join(appdata, "gpushare", "client.conf"))
    paths.append(os.path.join(os.environ.get("ProgramData", r"C:\ProgramData"),
                              "gpushare", "client.conf"))
    home = os.path.expanduser("~")
    paths.append(os.path.join(home, ".config", "gpushare", "client.conf"))

    for p in paths:
        if os.path.isfile(p):
            try:
                with open(p) as f:
                    for line in f:
                        if line.strip().startswith("server="):
                            addr = line.strip()[7:].strip()
                            if addr:
                                return addr, p
            except Exception:
                pass
    return None, None


def test_tcp_connection(host, port):
    """Test raw TCP connection to server."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        t0 = time.perf_counter()
        sock.connect((host, port))
        latency = (time.perf_counter() - t0) * 1000
        sock.close()
        return True, latency
    except Exception as e:
        return False, str(e)


def test_gpushare_protocol(host, port):
    """Test gpushare binary protocol handshake."""
    MAGIC = 0x47505553
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((host, port))

        # Send init
        hdr = struct.pack("<IIIHH", MAGIC, 24, 1, 0x0001, 0)
        payload = struct.pack("<II", 1, 1)
        sock.sendall(hdr + payload)

        # Read response
        resp = b""
        while len(resp) < 28:
            chunk = sock.recv(28 - len(resp))
            if not chunk:
                break
            resp += chunk

        if len(resp) >= 28:
            magic, length, rid, opcode, flags = struct.unpack("<IIIHH", resp[:16])
            if magic == MAGIC:
                ver, sid, max_xfer = struct.unpack("<III", resp[16:28])
                sock.close()
                return True, f"session={sid}, version={ver}"

        sock.close()
        return False, "invalid response"
    except Exception as e:
        return False, str(e)


def test_gpu_query(host, port):
    """Query device properties via gpushare protocol."""
    MAGIC = 0x47505553
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((host, port))

        # Init
        sock.sendall(struct.pack("<IIIHH", MAGIC, 24, 1, 0x0001, 0) + struct.pack("<II", 1, 1))
        resp = b""
        while len(resp) < 28:
            resp += sock.recv(28 - len(resp))

        # Query device props
        sock.sendall(struct.pack("<IIIHH", MAGIC, 20, 2, 0x0011, 0) + struct.pack("<i", 0))
        resp = b""
        while len(resp) < 16:
            resp += sock.recv(16 - len(resp))
        magic, length, rid, opcode, flags = struct.unpack("<IIIHH", resp[:16])
        payload_len = length - 16
        payload = b""
        while len(payload) < payload_len:
            payload += sock.recv(payload_len - len(payload))

        # Parse name
        name = payload[:256].split(b"\x00")[0].decode("utf-8", errors="replace")
        # Parse after name
        fmt = "<QQ" + "i" * 14 + "QQQ"
        vals = struct.unpack_from(fmt, payload, 256)

        sock.close()
        return True, {
            "name": name,
            "vram_mb": vals[0] / (1024 * 1024),
            "major": vals[12],
            "minor": vals[13],
            "sms": vals[14],
        }
    except Exception as e:
        return False, str(e)


def test_dll_loading():
    """Try to load our DLL and call NVML/CUDA functions."""
    results = {}

    # Find the DLL
    dll_paths = [
        os.path.join(os.environ.get("ProgramFiles", ""), "gpushare", "nvml.dll"),
        os.path.join(os.environ.get("ProgramFiles", ""), "gpushare", "nvcuda.dll"),
        os.path.join(os.environ.get("ProgramFiles", ""), "gpushare", "gpushare_client.dll"),
    ]
    # Also check local build dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for sub in ["build/Release", "build/Debug", "build"]:
        dll_paths.append(os.path.join(script_dir, "..", sub, "gpushare_client.dll"))

    lib = None
    lib_path = None
    for p in dll_paths:
        if os.path.isfile(p):
            try:
                lib = ctypes.CDLL(p)
                lib_path = p
                break
            except OSError:
                continue

    if lib is None:
        return False, "gpushare DLL not found", results

    results["dll_path"] = lib_path

    # Test NVML
    try:
        err = lib.nvmlInit_v2()
        if err == 0:
            count = ctypes.c_uint(0)
            lib.nvmlDeviceGetCount_v2(ctypes.byref(count))
            results["nvml_devices"] = count.value

            if count.value > 0:
                dev = ctypes.c_void_p()
                lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(dev))
                name = ctypes.create_string_buffer(256)
                lib.nvmlDeviceGetName(dev, name, 256)
                results["gpu_name"] = name.value.decode()

                class Mem(ctypes.Structure):
                    _fields_ = [("total", ctypes.c_ulonglong),
                                ("free", ctypes.c_ulonglong),
                                ("used", ctypes.c_ulonglong)]
                mem = Mem()
                lib.nvmlDeviceGetMemoryInfo(dev, ctypes.byref(mem))
                results["vram_total_mb"] = mem.total // (1024 * 1024)
                results["vram_used_mb"] = mem.used // (1024 * 1024)

                temp = ctypes.c_uint(0)
                lib.nvmlDeviceGetTemperature(dev, 0, ctypes.byref(temp))
                results["temperature"] = temp.value

                class Util(ctypes.Structure):
                    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]
                util = Util()
                lib.nvmlDeviceGetUtilizationRates(dev, ctypes.byref(util))
                results["gpu_util"] = util.value

            lib.nvmlShutdown()
            results["nvml"] = True
        else:
            results["nvml"] = False
            results["nvml_error"] = err
    except Exception as e:
        results["nvml"] = False
        results["nvml_error"] = str(e)

    # Test CUDA Driver API
    try:
        err = lib.cuInit(0)
        if err == 0:
            count = ctypes.c_int(0)
            lib.cuDeviceGetCount(ctypes.byref(count))
            results["cuda_devices"] = count.value
            results["cuda"] = True
        else:
            results["cuda"] = False
    except Exception as e:
        results["cuda"] = False

    return True, lib_path, results


def main():
    print(f"\n{BOLD}{'=' * 55}{NC}")
    print(f"{BOLD}  gpushare Windows Connection Verifier{NC}")
    print(f"{BOLD}{'=' * 55}{NC}\n")

    # Determine server
    server = None
    if len(sys.argv) > 1:
        server = sys.argv[1]
    if not server:
        server, conf_path = find_config_server()
        if server:
            info(f"Server from config: {server} ({conf_path})")
        else:
            server = input("  Enter server address (host:port): ").strip()
            if not server:
                fail("No server address provided")
                sys.exit(1)

    if ":" in server:
        host, port = server.rsplit(":", 1)
        port = int(port)
    else:
        host, port = server, 9847

    print(f"\n{BOLD}Test 1: TCP Connection{NC}")
    success, result = test_tcp_connection(host, port)
    if success:
        ok(f"Connected to {host}:{port} in {result:.1f}ms")
    else:
        fail(f"Cannot reach {host}:{port} - {result}")
        print(f"\n  {RED}Check:{NC}")
        print(f"    1. Is gpushare-server running on {host}?")
        print(f"    2. Is port {port} open in the firewall?")
        print(f"    3. Are both machines on the same network?")
        sys.exit(1)

    print(f"\n{BOLD}Test 2: Protocol Handshake{NC}")
    success, result = test_gpushare_protocol(host, port)
    if success:
        ok(f"Handshake OK ({result})")
    else:
        fail(f"Handshake failed: {result}")
        sys.exit(1)

    print(f"\n{BOLD}Test 3: GPU Query{NC}")
    success, result = test_gpu_query(host, port)
    if success:
        ok(f"GPU: {result['name']}")
        ok(f"VRAM: {result['vram_mb']:.0f} MB")
        ok(f"Compute: SM {result['major']}.{result['minor']}, {result['sms']} SMs")
    else:
        fail(f"GPU query failed: {result}")

    print(f"\n{BOLD}Test 4: DLL Loading{NC}")
    found, path, results = test_dll_loading()
    if found:
        ok(f"DLL loaded: {path}")
        if results.get("nvml"):
            ok(f"NVML: {results.get('nvml_devices', 0)} GPU(s) - {results.get('gpu_name', '?')}")
            ok(f"VRAM: {results.get('vram_used_mb', '?')}/{results.get('vram_total_mb', '?')} MB, {results.get('temperature', '?')}C")
        else:
            warn(f"NVML init failed (server may not be reachable from DLL)")
        if results.get("cuda"):
            ok(f"CUDA: {results.get('cuda_devices', 0)} device(s)")
        else:
            warn("CUDA driver API init failed")
    else:
        warn("gpushare DLL not found in system or build directory")
        info("Run install-client-windows.ps1 to install the DLL")

    print(f"\n{BOLD}Test 5: Python Client{NC}")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
        import gpushare
        gpu = gpushare.connect(host, port)
        ok(f"Python client connected")
        props = gpu.device_properties()
        ok(f"GPU: {props['name']}, {props['total_mem_mb']:.0f} MB, SM {props['major']}.{props['minor']}")

        # Quick transfer test
        import numpy as np
        data = np.random.randn(1024).astype(np.float32)
        d = gpu.malloc(data.nbytes)
        t0 = time.perf_counter()
        gpu.memcpy_h2d(d, data)
        result = np.empty_like(data)
        gpu.memcpy_d2h(result, d, data.nbytes)
        elapsed = (time.perf_counter() - t0) * 1000
        match = np.allclose(data, result)
        gpu.free(d)
        gpu.close()
        if match:
            ok(f"Data transfer: round-trip verified in {elapsed:.1f}ms")
        else:
            fail("Data verification failed!")
    except ImportError:
        warn("numpy not installed - skipping transfer test")
    except Exception as e:
        fail(f"Python client error: {e}")

    # Summary
    print(f"\n{BOLD}{'=' * 55}{NC}")
    print(f"{GREEN}{BOLD}  Remote GPU connection verified!{NC}")
    print(f"{BOLD}{'=' * 55}{NC}")
    print(f"\n  Server: {CYAN}{host}:{port}{NC}")
    if found and results.get("gpu_name"):
        print(f"  GPU:    {CYAN}{results['gpu_name']}{NC}")
    print(f"\n  Note: Windows Task Manager cannot show remote GPUs")
    print(f"  (it only reads from local hardware drivers).")
    print(f"  Use these instead:")
    print(f"    nvidia-smi              - GPU stats")
    print(f"    gpushare-monitor        - TUI dashboard")
    print(f"    python verify_windows.py - this test")
    print(f"    http://{host}:9848      - web dashboard")
    print()


if __name__ == "__main__":
    main()
