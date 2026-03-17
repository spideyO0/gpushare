"""
gpushare PyTorch Integration Hook

Loaded at Python startup via gpushare.pth in site-packages.
Makes remote GPUs transparently available to PyTorch by:

1. Querying GPUs via ctypes to our C library (libcuda.so.1 / nvcuda.dll)
2. If ctypes finds no remote GPUs, falls back to a lightweight Python TCP
   connection to the gpushare server (essential on Windows with local GPU,
   and on the server machine itself)
3. Patching PyTorch's SM compatibility checks that reject unknown architectures
4. Providing device property fallbacks when C-level queries fail

Actual CUDA operations go through the C library at runtime.
"""

import os
import sys
import ctypes
import ctypes.util
import struct
import socket
import threading
import builtins

_patched = False
_lock = threading.Lock()
_devices = []       # [{name, major, minor, total_mem, sm_count, is_remote, ...}]
_device_count = 0
_libcuda = None


# ════════════════════════════════════════════════════════════
#  Phase 1: ctypes discovery (fast, no network overhead)
# ════════════════════════════════════════════════════════════

def _load_cuda_lib():
    """Load the CUDA driver library (our gpushare client on clients)."""
    global _libcuda
    if _libcuda is not None:
        return _libcuda

    if sys.platform == 'linux':
        names = ['libcuda.so.1', 'libcuda.so']
    elif sys.platform == 'darwin':
        names = ['libcuda.dylib', 'libcuda.1.dylib']
    elif sys.platform == 'win32':
        names = ['nvcuda.dll']
    else:
        return None

    for name in names:
        try:
            _libcuda = ctypes.CDLL(name)
            return _libcuda
        except OSError:
            continue

    # Also try gpushare-specific paths on Windows
    if sys.platform == 'win32':
        for extra in [
            os.path.join(os.environ.get('ProgramFiles', r'C:\Program Files'), 'gpushare', 'nvcuda.dll'),
            os.path.join(os.environ.get('ProgramFiles', r'C:\Program Files'), 'gpushare', 'gpushare_client.dll'),
            os.path.join(sys.prefix, 'nvcuda.dll'),
        ]:
            if os.path.isfile(extra):
                try:
                    _libcuda = ctypes.CDLL(extra)
                    return _libcuda
                except OSError:
                    continue

    path = ctypes.util.find_library('cuda')
    if path:
        try:
            _libcuda = ctypes.CDLL(path)
            return _libcuda
        except OSError:
            pass
    return None


def _query_devices_ctypes():
    """Discover GPUs via ctypes calls to the loaded CUDA library."""
    global _devices, _device_count

    lib = _load_cuda_lib()
    if lib is None:
        return

    try:
        if lib.cuInit(ctypes.c_uint(0)) != 0:
            return
    except Exception:
        return

    count = ctypes.c_int(0)
    try:
        if lib.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return
    except Exception:
        return

    _device_count = count.value
    if _device_count == 0:
        return

    for i in range(_device_count):
        dev = ctypes.c_int(0)
        try:
            lib.cuDeviceGet(ctypes.byref(dev), ctypes.c_int(i))
        except Exception:
            continue

        # Name
        name_buf = ctypes.create_string_buffer(256)
        try:
            lib.cuDeviceGetName(name_buf, ctypes.c_int(256), dev)
        except Exception:
            pass
        raw_name = name_buf.value.decode('utf-8', errors='replace')
        clean_name = raw_name.replace(' (remote)', '').replace(' (local)', '')

        # Compute capability
        major = ctypes.c_int(0)
        minor = ctypes.c_int(0)
        try:
            lib.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), dev)
        except AttributeError:
            try:
                lib.cuDeviceGetAttribute(ctypes.byref(major), ctypes.c_int(75), dev)
                lib.cuDeviceGetAttribute(ctypes.byref(minor), ctypes.c_int(76), dev)
            except Exception:
                pass

        # Total memory
        total_mem = ctypes.c_size_t(0)
        for fn_name in ('cuDeviceTotalMem_v2', 'cuDeviceTotalMem'):
            fn = getattr(lib, fn_name, None)
            if fn:
                try:
                    fn(ctypes.byref(total_mem), dev)
                    break
                except Exception:
                    pass

        # Multiprocessor count (attribute 16)
        sm_count = ctypes.c_int(0)
        try:
            lib.cuDeviceGetAttribute(ctypes.byref(sm_count), ctypes.c_int(16), dev)
        except Exception:
            pass

        _devices.append({
            'name': clean_name,
            'raw_name': raw_name,
            'major': major.value,
            'minor': minor.value,
            'total_mem': total_mem.value,
            'sm_count': sm_count.value,
            'is_remote': '(remote)' in raw_name,
        })


# ════════════════════════════════════════════════════════════
#  Phase 2: Python TCP fallback for remote GPU discovery
#  Used when ctypes loaded the real NVIDIA driver (not ours)
#  or when no CUDA library is available at all.
# ════════════════════════════════════════════════════════════

_MAGIC = 0x47505553
_HEADER_SIZE = 16
_OP_INIT = 0x0001
_OP_CLOSE = 0x0002
_OP_GET_DEVICE_COUNT = 0x0010
_OP_GET_DEVICE_PROPS = 0x0011


def _read_server_config():
    """Read gpushare server address from config files."""
    # Environment variable first
    env = os.environ.get('GPUSHARE_SERVER')
    if env:
        return env

    import platform as _plat
    candidates = []
    home = os.path.expanduser("~")
    if _plat.system() == "Windows":
        appdata = os.environ.get("LOCALAPPDATA", "")
        if appdata:
            candidates.append(os.path.join(appdata, "gpushare", "client.conf"))
        candidates.append(r"C:\ProgramData\gpushare\client.conf")
    candidates.append(os.path.join(home, ".config", "gpushare", "client.conf"))
    candidates.append("/etc/gpushare/client.conf")

    for path in candidates:
        try:
            if not os.path.isfile(path):
                continue
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


def _query_remote_gpus_tcp():
    """Connect to gpushare server via Python TCP and query remote GPUs.

    This is the fallback path for:
    - Windows with a local GPU (nvcuda.dll is the real NVIDIA driver)
    - Server machine (libcuda.so.1 is the real driver)
    - Any system where ctypes can't load the CUDA library
    """
    global _devices, _device_count

    server_addr = _read_server_config()
    if not server_addr:
        return

    # Parse host:port
    host, port = server_addr, 9847
    if ':' in server_addr:
        parts = server_addr.rsplit(':', 1)
        host = parts[0]
        try:
            port = int(parts[1])
        except ValueError:
            pass

    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3.0)  # quick timeout — don't block startup
        sock.connect((host, port))

        def _send_all(data):
            mv = memoryview(data)
            sent = 0
            while sent < len(data):
                n = sock.send(mv[sent:])
                if n == 0:
                    raise ConnectionError("closed")
                sent += n

        def _recv_all(size):
            chunks = []
            received = 0
            while received < size:
                chunk = sock.recv(size - received)
                if not chunk:
                    raise ConnectionError("closed")
                chunks.append(chunk)
                received += len(chunk)
            return b"".join(chunks)

        req_id = [1]
        def _rpc(opcode, payload=b""):
            rid = req_id[0]
            req_id[0] += 1
            total = _HEADER_SIZE + len(payload)
            header = struct.pack("<IIIHH", _MAGIC, total, rid, opcode, 0)
            _send_all(header + payload)
            resp_hdr = _recv_all(_HEADER_SIZE)
            magic, length, _, _, _ = struct.unpack("<IIIHH", resp_hdr)
            if magic != _MAGIC:
                raise ValueError("bad magic")
            plen = length - _HEADER_SIZE
            return _recv_all(plen) if plen > 0 else b""

        # INIT handshake
        _rpc(_OP_INIT, struct.pack("<II", 1, 1))

        # GET_DEVICE_COUNT
        resp = _rpc(_OP_GET_DEVICE_COUNT)
        remote_count = struct.unpack("<i", resp[:4])[0]
        if remote_count <= 0:
            return

        # GET_DEVICE_PROPS for each remote device
        for i in range(remote_count):
            resp = _rpc(_OP_GET_DEVICE_PROPS, struct.pack("<i", i))
            if len(resp) < 256 + 96:
                continue
            name = resp[:256].split(b"\x00")[0].decode("utf-8", errors="replace")
            fmt = "<QQ" + "i" * 14 + "QQQ"
            vals = struct.unpack_from(fmt, resp, 256)

            _devices.append({
                'name': name,
                'raw_name': name + ' (remote)',
                'major': vals[12],
                'minor': vals[13],
                'total_mem': vals[0],
                'sm_count': vals[14],
                'is_remote': True,
            })
            _device_count += 1

        # CLOSE
        try:
            close_hdr = struct.pack("<IIIHH", _MAGIC, _HEADER_SIZE, 0, _OP_CLOSE, 0)
            _send_all(close_hdr)
        except Exception:
            pass

    except Exception:
        pass  # Server unreachable — that's fine, just no remote GPUs
    finally:
        if sock:
            try:
                sock.close()
            except Exception:
                pass


def _ctypes_mem_info():
    """Query free/total GPU memory via ctypes."""
    lib = _load_cuda_lib()
    if lib is None:
        return (0, 0)
    free = ctypes.c_size_t(0)
    total = ctypes.c_size_t(0)
    for fn_name in ('cuMemGetInfo_v2', 'cuMemGetInfo'):
        fn = getattr(lib, fn_name, None)
        if fn:
            try:
                fn(ctypes.byref(free), ctypes.byref(total))
                return (free.value, total.value)
            except Exception:
                pass
    return (0, 0)


# ════════════════════════════════════════════════════════════
#  Device resolution helper
# ════════════════════════════════════════════════════════════

def _resolve_device(device):
    """Convert device arg (int, str, torch.device) to integer index."""
    if device is None:
        return 0
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        if device.startswith('cuda:'):
            try:
                return int(device.split(':')[1])
            except (ValueError, IndexError):
                return 0
        return 0
    idx = getattr(device, 'index', None)
    return idx if idx is not None else 0


# ════════════════════════════════════════════════════════════
#  Fake device properties (fallback when C-level fails)
# ════════════════════════════════════════════════════════════

class _DeviceProps:
    """Mimics torch.cuda._CudaDeviceProperties."""
    def __init__(self, info):
        self.name = info['name']
        self.major = info['major']
        self.minor = info['minor']
        self.total_memory = info['total_mem']
        self.multi_processor_count = info['sm_count']
        self.is_integrated = False
        self.is_multi_gpu_board = False
        self.max_threads_per_block = 1024
        self.max_threads_per_multi_processor = 2048
        self.regs_per_block = 65536
        self.regs_per_multiprocessor = 65536
        self.warp_size = 32
        self.max_block_dim = (1024, 1024, 64)
        self.max_grid_dim = (2147483647, 65535, 65535)
        self.gcnArchName = ''

    def __repr__(self):
        mb = self.total_memory // (1024 * 1024) if self.total_memory else 0
        return (f"_CudaDeviceProperties(name='{self.name}', major={self.major}, "
                f"minor={self.minor}, total_memory={mb}MB, "
                f"multi_processor_count={self.multi_processor_count})")


class _CleanNameProps:
    """Wraps real CUDADeviceProperties to strip (remote)/(local) from name."""
    __slots__ = ('_inner', 'name')

    def __init__(self, inner):
        object.__setattr__(self, '_inner', inner)
        object.__setattr__(self, 'name',
                           inner.name.replace(' (remote)', '').replace(' (local)', ''))

    def __getattr__(self, attr):
        return getattr(self._inner, attr)

    def __repr__(self):
        return repr(self._inner).replace(' (remote)', '').replace(' (local)', '')


# ════════════════════════════════════════════════════════════
#  Patch torch.cuda after it is fully loaded
# ════════════════════════════════════════════════════════════

def _do_patch():
    """Apply all PyTorch patches. Called once when torch.cuda is imported."""
    global _patched

    with _lock:
        if _patched:
            return
        _patched = True

    if _device_count == 0:
        return

    try:
        import torch
        import torch.cuda
    except ImportError:
        return

    # Collect SM architectures from all detected devices
    sm_archs = set()
    for d in _devices:
        sm_archs.add(f"sm_{d['major']}{d['minor']}")

    # ── 1. get_arch_list — include remote GPU SM architectures ──
    # Without this, PyTorch rejects GPUs whose SM isn't in its compiled arch list
    if hasattr(torch.cuda, 'get_arch_list'):
        _orig_get_arch_list = torch.cuda.get_arch_list
        def _patched_get_arch_list():
            archs = list(_orig_get_arch_list())
            for sm in sm_archs:
                if sm not in archs:
                    archs.append(sm)
            return archs
        torch.cuda.get_arch_list = _patched_get_arch_list

    # ── 2. Neutralize capability / cubin checks ──
    # These functions print warnings or raise errors for unknown SM versions
    for name in ('_check_capability', '_check_cubins'):
        if hasattr(torch.cuda, name):
            setattr(torch.cuda, name, lambda: None)

    # ── 3. is_available — True if we have any GPUs ──
    _orig_is_available = torch.cuda.is_available
    def _patched_is_available():
        if _device_count > 0:
            return True
        return _orig_is_available()
    torch.cuda.is_available = _patched_is_available

    # ── 4. device_count — correct count ──
    _orig_device_count = torch.cuda.device_count
    def _patched_device_count():
        c = _orig_device_count()
        return max(c, _device_count)
    torch.cuda.device_count = _patched_device_count

    # ── 5. get_device_capability — ctypes fallback ──
    if hasattr(torch.cuda, 'get_device_capability'):
        _orig_get_cap = torch.cuda.get_device_capability
        def _patched_get_device_capability(device=None):
            try:
                return _orig_get_cap(device)
            except Exception:
                idx = _resolve_device(device)
                if 0 <= idx < len(_devices):
                    return (_devices[idx]['major'], _devices[idx]['minor'])
                return (0, 0)
        torch.cuda.get_device_capability = _patched_get_device_capability

    # ── 6. get_device_name — strip " (remote)" / " (local)" ──
    if hasattr(torch.cuda, 'get_device_name'):
        _orig_get_name = torch.cuda.get_device_name
        def _patched_get_device_name(device=None):
            try:
                n = _orig_get_name(device)
                return n.replace(' (remote)', '').replace(' (local)', '')
            except Exception:
                idx = _resolve_device(device)
                if 0 <= idx < len(_devices):
                    return _devices[idx]['name']
                return 'Unknown GPU'
        torch.cuda.get_device_name = _patched_get_device_name

    # ── 7. get_device_properties — clean name + fallback ──
    if hasattr(torch.cuda, 'get_device_properties'):
        _orig_get_props = torch.cuda.get_device_properties
        def _patched_get_device_properties(device=None):
            try:
                p = _orig_get_props(device)
                if hasattr(p, 'name') and (' (remote)' in p.name or ' (local)' in p.name):
                    return _CleanNameProps(p)
                return p
            except Exception:
                idx = _resolve_device(device)
                if 0 <= idx < len(_devices):
                    return _DeviceProps(_devices[idx])
                raise
        torch.cuda.get_device_properties = _patched_get_device_properties

    # ── 8. mem_get_info — ctypes fallback ──
    if hasattr(torch.cuda, 'mem_get_info'):
        _orig_mem_info = torch.cuda.mem_get_info
        def _patched_mem_get_info(device=None):
            try:
                return _orig_mem_info(device)
            except Exception:
                idx = _resolve_device(device)
                if 0 <= idx < len(_devices):
                    free, total = _ctypes_mem_info()
                    if total == 0:
                        total = _devices[idx]['total_mem']
                    if free == 0:
                        free = total
                    return (free, total)
                return (0, 0)
        torch.cuda.mem_get_info = _patched_mem_get_info

    # ── 9. memory_allocated — graceful fallback ──
    if hasattr(torch.cuda, 'memory_allocated'):
        _orig_mem_alloc = torch.cuda.memory_allocated
        def _patched_memory_allocated(device=None):
            try:
                return _orig_mem_alloc(device)
            except Exception:
                return 0
        torch.cuda.memory_allocated = _patched_memory_allocated

    # ── 10. _lazy_init — recover from initialization failures ──
    if hasattr(torch.cuda, '_lazy_init'):
        _orig_lazy_init = torch.cuda._lazy_init
        _init_done = [False]
        def _patched_lazy_init():
            if _init_done[0]:
                return
            try:
                _orig_lazy_init()
                _init_done[0] = True
            except Exception:
                if _device_count > 0:
                    # Force-mark as initialized so PyTorch doesn't block
                    _init_done[0] = True
                    if hasattr(torch.cuda, '_initialized'):
                        torch.cuda._initialized = True
                    if hasattr(torch.cuda, '_queued_calls'):
                        for fn_and_args in torch.cuda._queued_calls:
                            if callable(fn_and_args):
                                try: fn_and_args()
                                except Exception: pass
                            elif isinstance(fn_and_args, (tuple, list)) and len(fn_and_args) >= 1:
                                try: fn_and_args[0](*fn_and_args[1:])
                                except Exception: pass
                        torch.cuda._queued_calls = []
                else:
                    raise
        torch.cuda._lazy_init = _patched_lazy_init

    # ── 11. Suppress SM compatibility warnings ──
    import warnings
    _orig_warn = warnings.warn
    _suppress_patterns = (
        'not compatible',
        'sm_1',
        'not currently supported',
        'Targeting sm_',
        'not in the list of supported',
        'UserWarning: CUDA initialization',
    )
    def _filtered_warn(message, *args, **kwargs):
        s = str(message)
        for pat in _suppress_patterns:
            if pat in s:
                return
        return _orig_warn(message, *args, **kwargs)
    warnings.warn = _filtered_warn

    # ── 12. torch.backends.cudnn — enable for remote GPUs ──
    try:
        import torch.backends.cudnn as _cudnn
        if hasattr(_cudnn, 'is_available'):
            _orig_cudnn_avail = _cudnn.is_available
            if not _orig_cudnn_avail():
                _cudnn.is_available = lambda: True
            if hasattr(_cudnn, 'enabled'):
                _cudnn.enabled = True
    except Exception:
        pass

    # ── 13. torch.backends.cuda.is_built ──
    try:
        import torch.backends.cuda as _cuda_be
        if hasattr(_cuda_be, 'is_built') and not _cuda_be.is_built():
            _cuda_be.is_built = lambda: True
    except Exception:
        pass

    # ── 14. torch.version.cuda — set if CPU-only build ──
    try:
        if getattr(torch.version, 'cuda', None) is None:
            torch.version.cuda = '13.1'
    except Exception:
        pass

    # ── 15. current_device — fallback if CUDA init incomplete ──
    if hasattr(torch.cuda, 'current_device'):
        _orig_current_device = torch.cuda.current_device
        def _patched_current_device():
            try:
                return _orig_current_device()
            except Exception:
                return 0
        torch.cuda.current_device = _patched_current_device

    # Status
    gpus = ', '.join(f"{d['name']} (sm_{d['major']}{d['minor']})" for d in _devices)
    sys.stderr.write(f"[gpushare] {_device_count} GPU(s): {gpus}\n")


# ════════════════════════════════════════════════════════════
#  Import hook — trigger patching when torch.cuda loads
# ════════════════════════════════════════════════════════════

def install():
    """Called from gpushare.pth at Python startup."""
    # Skip in test environments or when explicitly disabled
    if os.environ.get('GPUSHARE_NO_HOOK'):
        return

    # Phase 1: Discover GPUs via ctypes (fast, no network)
    try:
        _query_devices_ctypes()
    except Exception:
        pass

    # Phase 2: If no remote GPUs found via ctypes, try direct TCP to server.
    # This is essential for:
    #   - Windows with local GPU (nvcuda.dll is real NVIDIA, not ours)
    #   - Server machine (libcuda.so.1 is the real driver)
    #   - Any system where our C library isn't installed as the CUDA driver
    has_remote = any(d.get('is_remote') for d in _devices)
    if not has_remote:
        try:
            _query_remote_gpus_tcp()
        except Exception:
            pass

    if _device_count == 0:
        return

    # Wrap builtins.__import__ to detect torch.cuda loading
    _real_import = builtins.__import__
    _hooking = threading.local()

    def _import_hook(name, *args, **kwargs):
        module = _real_import(name, *args, **kwargs)
        if not _patched and not getattr(_hooking, 'active', False):
            # Patch once torch.cuda is fully loaded
            if 'torch.cuda' in sys.modules:
                _hooking.active = True
                try:
                    _do_patch()
                except Exception as e:
                    sys.stderr.write(f"[gpushare] Hook error: {e}\n")
                finally:
                    _hooking.active = False
        return module

    builtins.__import__ = _import_hook
