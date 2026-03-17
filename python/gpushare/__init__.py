"""
gpushare — Python client for remote GPU access.

Usage:
    import gpushare
    gpu = gpushare.connect("192.168.1.100")  # or "host:port"

    # Query device
    props = gpu.device_properties()
    print(f"Remote GPU: {props['name']} ({props['total_mem_mb']:.0f} MB)")

    # Allocate + transfer
    import numpy as np
    data = np.random.randn(1024, 1024).astype(np.float32)
    d_ptr = gpu.malloc(data.nbytes)
    gpu.memcpy_h2d(d_ptr, data)

    result = np.empty_like(data)
    gpu.memcpy_d2h(result, d_ptr, data.nbytes)
    gpu.free(d_ptr)
"""

import os
import socket
import struct
import threading
import numpy as np
from typing import Optional, Dict, Any

# Protocol constants (must match protocol.h)
_MAGIC       = 0x47505553
_HEADER_SIZE = 16
_VERSION     = 1
_DEFAULT_PORT = 9847

# Opcodes
_OP_INIT              = 0x0001
_OP_CLOSE             = 0x0002
_OP_PING              = 0x0003
_OP_GET_DEVICE_COUNT  = 0x0010
_OP_GET_DEVICE_PROPS  = 0x0011
_OP_SET_DEVICE        = 0x0012
_OP_MALLOC            = 0x0020
_OP_FREE              = 0x0021
_OP_MEMCPY_H2D        = 0x0022
_OP_MEMCPY_D2H        = 0x0023
_OP_MEMCPY_D2D        = 0x0024
_OP_MEMSET            = 0x0025
_OP_MODULE_LOAD       = 0x0030
_OP_GET_FUNCTION      = 0x0032
_OP_LAUNCH_KERNEL     = 0x0033
_OP_DEVICE_SYNC       = 0x0040
_OP_STREAM_CREATE     = 0x0041
_OP_STREAM_DESTROY    = 0x0042
_OP_STREAM_SYNC       = 0x0043
_OP_EVENT_CREATE      = 0x0044
_OP_EVENT_DESTROY     = 0x0045
_OP_EVENT_RECORD      = 0x0046
_OP_EVENT_SYNC        = 0x0047
_OP_EVENT_ELAPSED     = 0x0048

_FLAG_RESPONSE = 0x0001
_FLAG_ERROR    = 0x0002


class GPUShareError(Exception):
    pass


class RemoteGPU:
    """Connection to a remote GPU server."""

    def __init__(self, host: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self._sock.connect((host, port))
        self._lock = threading.Lock()
        self._req_id = 1
        self._session_id = None

        # Send init
        resp = self._rpc(_OP_INIT, struct.pack("<II", _VERSION, 1))
        ver, sid, max_xfer = struct.unpack("<III", resp[:12])
        self._session_id = sid

    def _send_all(self, data: bytes):
        mv = memoryview(data)
        sent = 0
        while sent < len(data):
            n = self._sock.send(mv[sent:])
            if n == 0:
                raise GPUShareError("Connection closed")
            sent += n

    def _recv_all(self, size: int) -> bytes:
        chunks = []
        received = 0
        while received < size:
            chunk = self._sock.recv(size - received)
            if not chunk:
                raise GPUShareError("Connection closed")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    def _rpc(self, opcode: int, payload: bytes = b"") -> bytes:
        with self._lock:
            rid = self._req_id
            self._req_id += 1

            total = _HEADER_SIZE + len(payload)
            header = struct.pack("<IIIHH", _MAGIC, total, rid, opcode, 0)
            self._send_all(header + payload)

            # Read response
            resp_hdr = self._recv_all(_HEADER_SIZE)
            magic, length, resp_rid, resp_op, flags = struct.unpack("<IIIHH", resp_hdr)

            if magic != _MAGIC:
                raise GPUShareError(f"Invalid response magic: 0x{magic:08x}")

            payload_len = length - _HEADER_SIZE
            resp_payload = self._recv_all(payload_len) if payload_len > 0 else b""

            return resp_payload

    def close(self):
        """Close connection to server."""
        try:
            total = _HEADER_SIZE
            header = struct.pack("<IIIHH", _MAGIC, total, 0, _OP_CLOSE, 0)
            self._send_all(header)
        except Exception:
            pass
        self._sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ── Device queries ───────────────────────────────────────

    def device_count(self) -> int:
        resp = self._rpc(_OP_GET_DEVICE_COUNT)
        return struct.unpack("<i", resp[:4])[0]

    def device_properties(self, device: int = 0) -> Dict[str, Any]:
        resp = self._rpc(_OP_GET_DEVICE_PROPS, struct.pack("<i", device))
        # Parse gs_device_props_t (packed struct, no padding)
        name = resp[:256].split(b"\x00")[0].decode("utf-8", errors="replace")
        # After name[256]: 2xQ + 14xi + 3xQ = 16+56+24 = 96 bytes
        fmt = "<QQ" + "i" * 14 + "QQQ"
        vals = struct.unpack_from(fmt, resp, 256)
        return {
            "name": name,
            "total_global_mem": vals[0],
            "total_mem_mb": vals[0] / (1024 * 1024),
            "shared_mem_per_block": vals[1],
            "regs_per_block": vals[2],
            "warp_size": vals[3],
            "max_threads_per_block": vals[4],
            "max_threads_dim": vals[5:8],
            "max_grid_size": vals[8:11],
            "clock_rate": vals[11],
            "major": vals[12],
            "minor": vals[13],
            "multi_processor_count": vals[14],
            "max_threads_per_mp": vals[15],
        }

    def set_device(self, device: int = 0):
        self._rpc(_OP_SET_DEVICE, struct.pack("<i", device))

    # ── Memory management ────────────────────────────────────

    def malloc(self, size: int) -> int:
        """Allocate GPU memory. Returns a device pointer (int handle)."""
        resp = self._rpc(_OP_MALLOC, struct.pack("<Q", size))
        ptr, err = struct.unpack("<Qi", resp[:12])
        if err != 0:
            raise GPUShareError(f"cudaMalloc failed: error {err}")
        return ptr

    def free(self, device_ptr: int):
        """Free GPU memory."""
        self._rpc(_OP_FREE, struct.pack("<Q", device_ptr))

    def memcpy_h2d(self, device_ptr: int, data) -> None:
        """Copy data from host to device. `data` can be bytes or numpy array."""
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        payload = struct.pack("<QQ", device_ptr, len(data)) + data
        self._rpc(_OP_MEMCPY_H2D, payload)

    def memcpy_d2h(self, host_buf, device_ptr: int, size: int) -> None:
        """Copy data from device to host buffer (numpy array or bytearray)."""
        resp = self._rpc(_OP_MEMCPY_D2H, struct.pack("<QQ", device_ptr, size))
        err = struct.unpack("<i", resp[:4])[0]
        if err != 0:
            raise GPUShareError(f"cudaMemcpy D2H failed: error {err}")
        raw = resp[4:4 + size]
        if isinstance(host_buf, np.ndarray):
            np.copyto(host_buf, np.frombuffer(raw, dtype=host_buf.dtype).reshape(host_buf.shape))
        elif isinstance(host_buf, (bytearray, memoryview)):
            host_buf[:size] = raw

    def memcpy_d2d(self, dst_ptr: int, src_ptr: int, size: int):
        """Copy between device pointers."""
        self._rpc(_OP_MEMCPY_D2D, struct.pack("<QQQ", dst_ptr, src_ptr, size))

    def memset(self, device_ptr: int, value: int, size: int):
        self._rpc(_OP_MEMSET, struct.pack("<QiQ", device_ptr, value, size))

    # ── Synchronization ──────────────────────────────────────

    def synchronize(self):
        """Wait for all GPU operations to complete."""
        self._rpc(_OP_DEVICE_SYNC)

    def stream_create(self) -> int:
        resp = self._rpc(_OP_STREAM_CREATE)
        return struct.unpack("<Q", resp[:8])[0]

    def stream_destroy(self, stream: int):
        self._rpc(_OP_STREAM_DESTROY, struct.pack("<Q", stream))

    def stream_synchronize(self, stream: int):
        self._rpc(_OP_STREAM_SYNC, struct.pack("<Q", stream))

    # ── Kernel execution ─────────────────────────────────────

    def load_module(self, ptx_or_cubin: bytes) -> int:
        """Load a PTX or cubin module. Returns module handle."""
        payload = struct.pack("<Q", len(ptx_or_cubin)) + ptx_or_cubin
        resp = self._rpc(_OP_MODULE_LOAD, payload)
        handle, err = struct.unpack("<Qi", resp[:12])
        if err != 0:
            raise GPUShareError(f"Module load failed: error {err}")
        return handle

    def get_function(self, module: int, name: str) -> int:
        """Get kernel function handle from loaded module."""
        name_bytes = name.encode("utf-8")[:255]
        payload = struct.pack("<Q", module) + name_bytes + b"\x00" * (256 - len(name_bytes))
        resp = self._rpc(_OP_GET_FUNCTION, payload)
        handle, err = struct.unpack("<Qi", resp[:12])
        if err != 0:
            raise GPUShareError(f"Get function '{name}' failed: error {err}")
        return handle

    def launch_kernel(self, func: int,
                      grid: tuple, block: tuple,
                      args: list,
                      shared_mem: int = 0, stream: int = 0):
        """
        Launch a kernel.

        Args:
            func: Function handle from get_function()
            grid: (x, y, z) grid dimensions
            block: (x, y, z) block dimensions
            args: List of (bytes) kernel arguments. Device pointers should be
                  packed as struct.pack("<Q", ptr).
            shared_mem: Shared memory bytes
            stream: Stream handle (0 = default)
        """
        gx, gy, gz = (grid + (1, 1))[:3]
        bx, by, bz = (block + (1, 1))[:3]

        # Serialize arguments
        args_data = b""
        for arg in args:
            if isinstance(arg, (int, np.integer)):
                arg = struct.pack("<Q", int(arg))
            elif isinstance(arg, float):
                arg = struct.pack("<f", arg)
            elif isinstance(arg, np.ndarray):
                arg = arg.tobytes()
            elif not isinstance(arg, bytes):
                arg = bytes(arg)
            args_data += struct.pack("<I", len(arg)) + arg

        header = struct.pack("<QIIIIIIIQII",
                             func,
                             gx, gy, gz,
                             bx, by, bz,
                             shared_mem,
                             stream,
                             len(args),
                             len(args_data))
        self._rpc(_OP_LAUNCH_KERNEL, header + args_data)

    # ── Convenience methods ──────────────────────────────────

    def ping(self) -> bool:
        """Test server connectivity."""
        try:
            self._rpc(_OP_PING)
            return True
        except Exception:
            return False

    def info(self) -> str:
        """Get a formatted string with GPU info."""
        props = self.device_properties()
        return (f"GPU: {props['name']}\n"
                f"  VRAM: {props['total_mem_mb']:.0f} MB\n"
                f"  SM: {props['major']}.{props['minor']}\n"
                f"  SMs: {props['multi_processor_count']}\n"
                f"  Max threads/block: {props['max_threads_per_block']}")

    def list_all_gpus(self) -> list:
        """List all available GPUs (local + remote) with properties.

        Returns a list of dicts, each with 'index', 'name', 'type' ('local'/'remote'),
        and full device properties.
        """
        gpus = []
        count = self.device_count()
        for i in range(count):
            props = self.device_properties(i)
            name = props['name']
            gpu_type = 'remote'
            if '(local)' in name:
                gpu_type = 'local'
                name = name.replace(' (local)', '')
            elif '(remote)' in name:
                name = name.replace(' (remote)', '')
            gpus.append({
                'index': i,
                'name': name,
                'type': gpu_type,
                **props,
            })
        return gpus


def _read_server_from_config() -> Optional[str]:
    """Read server address from config files."""
    import platform as _plat
    candidates = []
    home = os.path.expanduser("~")
    if _plat.system() == "Windows":
        appdata = os.environ.get("LOCALAPPDATA", "")
        if appdata:
            candidates.append(os.path.join(appdata, "gpushare", "client.conf"))
        candidates.append(r"C:\ProgramData\gpushare\client.conf")
    else:
        candidates.append(os.path.join(home, ".config", "gpushare", "client.conf"))
        candidates.append("/etc/gpushare/client.conf")

    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("server="):
                            addr = line[7:].strip()
                            if addr:
                                return addr
            except Exception:
                pass
    return None


def connect(server: Optional[str] = None, port: Optional[int] = None) -> RemoteGPU:
    """
    Connect to a gpushare server.

    If no server is specified, reads from:
      1. GPUSHARE_SERVER environment variable
      2. Config file (~/.config/gpushare/client.conf or /etc/gpushare/client.conf)
      3. Falls back to localhost:9847

    Args:
        server: Hostname or "host:port" (None = auto-detect)
        port: Port (default: 9847)
    """
    if server is None:
        # Try env var first
        env = os.environ.get("GPUSHARE_SERVER")
        if env:
            server = env
        else:
            # Try config file
            cfg = _read_server_from_config()
            if cfg:
                server = cfg
            else:
                server = "localhost"

    if ":" in server and port is None:
        parts = server.rsplit(":", 1)
        server = parts[0]
        port = int(parts[1])
    if port is None:
        port = _DEFAULT_PORT
    return RemoteGPU(server, port)
