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
    """Load the CUDA driver library.

    On Windows, we MUST load our gpushare DLL by full path BEFORE the generic
    'nvcuda.dll' search, because the generic search finds the real NVIDIA driver
    in System32 first. Loading our DLL first also preloads it in the process so
    that PyTorch's C++ code (which loads nvcuda.dll by name) gets ours via
    Windows' already-loaded-DLL base-name matching.
    """
    global _libcuda
    if _libcuda is not None:
        return _libcuda

    if sys.platform == 'win32':
        # On Windows: try our gpushare DLL by FULL PATH first.
        # This ensures we load ours, not the real nvcuda.dll from System32.
        # Once loaded, Windows will reuse it for any subsequent LoadLibrary('nvcuda.dll')
        # calls (including from PyTorch's _C.pyd) because of base-name matching.
        gpushare_paths = []

        # 1. torch\lib (highest priority - PyTorch adds this via os.add_dll_directory)
        try:
            torch_mod = sys.modules.get('torch')
            if torch_mod and hasattr(torch_mod, '__file__'):
                tlib = os.path.join(os.path.dirname(torch_mod.__file__), 'lib')
                gpushare_paths.append(os.path.join(tlib, 'nvcuda.dll'))
        except Exception:
            pass
        # Also try common torch locations for MS Store Python
        for pyExe in [sys.executable]:
            if pyExe:
                # Standard Python: <python_dir>/Lib/site-packages/torch/lib
                sp = os.path.join(os.path.dirname(pyExe), 'Lib', 'site-packages', 'torch', 'lib')
                gpushare_paths.append(os.path.join(sp, 'nvcuda.dll'))
                # MS Store Python user packages
                lp = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Packages')
                if os.path.isdir(lp):
                    for pkg in os.listdir(lp):
                        if 'Python' in pkg:
                            tp = os.path.join(lp, pkg, 'LocalCache', 'local-packages')
                            for pyver in os.listdir(tp) if os.path.isdir(tp) else []:
                                sp2 = os.path.join(tp, pyver, 'site-packages', 'torch', 'lib')
                                gpushare_paths.append(os.path.join(sp2, 'nvcuda.dll'))

        # 2. gpushare install directory
        pf = os.environ.get('ProgramFiles', r'C:\Program Files')
        gpushare_paths.append(os.path.join(pf, 'gpushare', 'nvcuda.dll'))
        gpushare_paths.append(os.path.join(pf, 'gpushare', 'gpushare_client.dll'))

        # 3. Python directory
        gpushare_paths.append(os.path.join(os.path.dirname(sys.executable), 'nvcuda.dll'))

        for path in gpushare_paths:
            if not os.path.isfile(path):
                continue
            try:
                _libcuda = ctypes.CDLL(path)
                return _libcuda
            except OSError:
                continue

        # 4. Fallback: generic name (will find System32 real driver)
        try:
            _libcuda = ctypes.CDLL('nvcuda.dll')
            return _libcuda
        except OSError:
            pass

    elif sys.platform == 'linux':
        for name in ['libcuda.so.1', 'libcuda.so']:
            try:
                _libcuda = ctypes.CDLL(name)
                return _libcuda
            except OSError:
                continue

    elif sys.platform == 'darwin':
        for name in ['libcuda.dylib', 'libcuda.1.dylib']:
            try:
                _libcuda = ctypes.CDLL(name)
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


def _query_devices_ctypes_inner():
    """Inner function that does the actual ctypes GPU discovery."""
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


def _query_devices_ctypes():
    """Discover GPUs via ctypes with a timeout.

    Our DLL's cuInit() connects to the gpushare server, which can block
    indefinitely if the server is unreachable. Run in a thread with timeout
    so Python startup is never blocked.
    """
    t = threading.Thread(target=_query_devices_ctypes_inner, daemon=True)
    t.start()
    t.join(timeout=5)  # 5 second max wait
    if t.is_alive():
        sys.stderr.write("[gpushare] GPU discovery timed out (server unreachable?)\n")


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
    """Apply all PyTorch patches. Called once when torch.cuda is fully loaded."""
    global _patched

    with _lock:
        if _patched:
            return
        _patched = True

    if _device_count == 0:
        return

    # Use sys.modules directly — do NOT call `import torch` here.
    # The import-depth tracking in install() guarantees these are fully loaded.
    torch = sys.modules.get('torch')
    torch_cuda = sys.modules.get('torch.cuda')
    if torch is None or torch_cuda is None:
        _patched = False  # retry later
        return

    # Use local refs — avoids any further import calls inside _do_patch
    tc = torch_cuda

    # Determine how many GPUs are local vs remote
    _local_gpu_count = sum(1 for d in _devices if not d.get('is_remote'))
    _active_device = [0]  # track current device (local or remote)

    # Collect SM architectures from all detected devices
    sm_archs = set()
    for d in _devices:
        sm_archs.add(f"sm_{d['major']}{d['minor']}")

    # ── 1. get_arch_list — include remote GPU SM architectures ──
    if hasattr(tc, 'get_arch_list'):
        _orig_get_arch_list = tc.get_arch_list
        def _patched_get_arch_list():
            archs = list(_orig_get_arch_list())
            for sm in sm_archs:
                if sm not in archs:
                    archs.append(sm)
            return archs
        tc.get_arch_list = _patched_get_arch_list

    # ── 2. Neutralize capability / cubin checks ──
    for name in ('_check_capability', '_check_cubins'):
        if hasattr(tc, name):
            setattr(tc, name, lambda: None)

    # ── 3. is_available — True if we have any GPUs ──
    _orig_is_available = tc.is_available
    def _patched_is_available():
        if _device_count > 0:
            return True
        return _orig_is_available()
    tc.is_available = _patched_is_available

    # ── 4. device_count — correct count ──
    _orig_device_count = tc.device_count
    def _patched_device_count():
        c = _orig_device_count()
        return max(c, _device_count)
    tc.device_count = _patched_device_count

    # ── 5. get_device_capability — ctypes fallback ──
    if hasattr(tc, 'get_device_capability'):
        _orig_get_cap = tc.get_device_capability
        def _patched_get_device_capability(device=None):
            try:
                return _orig_get_cap(device)
            except Exception:
                idx = _resolve_device(device)
                if 0 <= idx < len(_devices):
                    return (_devices[idx]['major'], _devices[idx]['minor'])
                return (0, 0)
        tc.get_device_capability = _patched_get_device_capability

    # ── 6. get_device_name — strip " (remote)" / " (local)" ──
    if hasattr(tc, 'get_device_name'):
        _orig_get_name = tc.get_device_name
        def _patched_get_device_name(device=None):
            try:
                n = _orig_get_name(device)
                return n.replace(' (remote)', '').replace(' (local)', '')
            except Exception:
                idx = _resolve_device(device)
                if 0 <= idx < len(_devices):
                    return _devices[idx]['name']
                return 'Unknown GPU'
        tc.get_device_name = _patched_get_device_name

    # ── 7. get_device_properties — clean name + fallback ──
    if hasattr(tc, 'get_device_properties'):
        _orig_get_props = tc.get_device_properties
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
        tc.get_device_properties = _patched_get_device_properties

    # ── 8. mem_get_info — ctypes fallback ──
    if hasattr(tc, 'mem_get_info'):
        _orig_mem_info = tc.mem_get_info
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
        tc.mem_get_info = _patched_mem_get_info

    # ── 9. memory_allocated — graceful fallback ──
    if hasattr(tc, 'memory_allocated'):
        _orig_mem_alloc = tc.memory_allocated
        def _patched_memory_allocated(device=None):
            try:
                return _orig_mem_alloc(device)
            except Exception:
                return 0
        tc.memory_allocated = _patched_memory_allocated

    # ── 10. _lazy_init — recover from initialization failures ──
    if hasattr(tc, '_lazy_init'):
        _orig_lazy_init = tc._lazy_init
        _init_done = [False]
        def _patched_lazy_init():
            if _init_done[0]:
                return
            try:
                _orig_lazy_init()
                _init_done[0] = True
            except Exception:
                if _device_count > 0:
                    _init_done[0] = True
                    if hasattr(tc, '_initialized'):
                        tc._initialized = True
                    if hasattr(tc, '_queued_calls'):
                        for fn_and_args in tc._queued_calls:
                            if callable(fn_and_args):
                                try: fn_and_args()
                                except Exception: pass
                            elif isinstance(fn_and_args, (tuple, list)) and len(fn_and_args) >= 1:
                                try: fn_and_args[0](*fn_and_args[1:])
                                except Exception: pass
                        tc._queued_calls = []
                else:
                    raise
        tc._lazy_init = _patched_lazy_init

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
    _cudnn = sys.modules.get('torch.backends.cudnn')
    if _cudnn is not None:
        try:
            if hasattr(_cudnn, 'is_available') and not _cudnn.is_available():
                _cudnn.is_available = lambda: True
            if hasattr(_cudnn, 'enabled'):
                _cudnn.enabled = True
        except Exception:
            pass

    # ── 13. torch.backends.cuda.is_built ──
    _cuda_be = sys.modules.get('torch.backends.cuda')
    if _cuda_be is not None:
        try:
            if hasattr(_cuda_be, 'is_built') and not _cuda_be.is_built():
                _cuda_be.is_built = lambda: True
        except Exception:
            pass

    # ── 14. torch.version.cuda — set if CPU-only build ──
    try:
        torch_version = getattr(torch, 'version', None)
        if torch_version and getattr(torch_version, 'cuda', None) is None:
            torch_version.cuda = '13.1'
    except Exception:
        pass

    # ── 15. set_device — intercept remote device selection ──
    # On Windows with a local GPU, PyTorch's C++ backend only knows about
    # local GPUs. set_device(remote_idx) would call _cuda_setDevice which
    # fails with "invalid device ordinal". We intercept it here.
    if hasattr(tc, 'set_device'):
        _orig_set_device = tc.set_device
        def _patched_set_device(device):
            idx = _resolve_device(device)
            _active_device[0] = idx
            if idx < _local_gpu_count:
                # Local GPU — delegate to real PyTorch
                _orig_set_device(device)
            else:
                # Remote GPU — don't call C++ backend (it doesn't know about it).
                # The gpushare C library handles routing when it's loaded.
                # If C library is NOT the active CUDA driver (Windows with local GPU),
                # we track the selection here for Python-level queries.
                pass
        tc.set_device = _patched_set_device

    # ── 16. current_device — return tracked device ──
    if hasattr(tc, 'current_device'):
        _orig_current_device = tc.current_device
        def _patched_current_device():
            # If a remote device was selected, return it
            if _active_device[0] >= _local_gpu_count:
                return _active_device[0]
            try:
                return _orig_current_device()
            except Exception:
                return _active_device[0]
        tc.current_device = _patched_current_device

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

    # Strategy: we CANNOT patch torch.cuda during import because Python adds
    # submodules to sys.modules BEFORE they finish executing __init__.py.
    # Checking individual attrs is a race (is_available exists, device_count doesn't).
    #
    # Fix: track import nesting depth. Only patch when depth returns to 0,
    # which means ALL imports (including torch's internal sub-imports) have
    # finished. This guarantees torch.cuda is fully initialized.

    _real_import = builtins.__import__
    _depth = [0]

    def _import_hook(name, *args, **kwargs):
        _depth[0] += 1
        try:
            module = _real_import(name, *args, **kwargs)
        finally:
            _depth[0] -= 1

        # Only patch when we're back at the top level (all imports done)
        if _depth[0] == 0 and not _patched:
            if 'torch.cuda' in sys.modules:
                try:
                    _do_patch()
                except Exception as e:
                    sys.stderr.write(f"[gpushare] Hook error: {e}\n")

        return module

    builtins.__import__ = _import_hook
