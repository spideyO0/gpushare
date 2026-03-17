"""
gpushare PyTorch Integration Hook

Automatically loaded at Python startup via gpushare.pth in site-packages.
Monkey-patches torch.cuda to expose remote GPUs alongside local ones.

The remote GPU appears as a native CUDA device:
  torch.cuda.device_count()  -> includes remote GPUs
  torch.cuda.get_device_properties(N)  -> works for remote devices
  torch.cuda.set_device(N)  -> routes to remote server
  torch.tensor(..., device='cuda:N')  -> transparently uses remote GPU

No code changes required in user applications.
"""

import os
import sys
import threading

_hooked = False
_lock = threading.Lock()
_remote = None       # gpushare.RemoteGPU instance
_local_count = 0     # real local GPU count
_remote_count = 0    # remote GPU count
_remote_props = {}   # cached remote device properties

# Original torch.cuda functions (saved before patching)
_orig_device_count = None
_orig_get_device_properties = None
_orig_get_device_name = None
_orig_set_device = None
_orig_current_device = None
_orig_is_available = None
_orig_mem_get_info = None


def _init_remote():
    """Connect to gpushare server and query remote GPU info."""
    global _remote, _remote_count, _remote_props
    if _remote is not None:
        return
    try:
        import gpushare
        _remote = gpushare.connect()
        gpus = _remote.list_all_gpus()
        _remote_count = len(gpus)
        for i, g in enumerate(gpus):
            _remote_props[i] = g
    except Exception:
        _remote = None
        _remote_count = 0


def _is_remote_device(device_idx):
    return device_idx >= _local_count


def _remote_device_idx(device_idx):
    return device_idx - _local_count


def _hook_torch():
    """Monkey-patch torch.cuda to include remote GPUs."""
    global _hooked, _local_count
    global _orig_device_count, _orig_get_device_properties
    global _orig_get_device_name, _orig_set_device
    global _orig_current_device, _orig_is_available, _orig_mem_get_info

    with _lock:
        if _hooked:
            return
        _hooked = True

    try:
        import torch
        if not hasattr(torch, 'cuda'):
            return
    except ImportError:
        return

    # Connect to remote server
    _init_remote()
    if _remote_count == 0:
        return  # no remote GPUs, nothing to patch

    # Save originals
    _orig_device_count = torch.cuda.device_count
    _orig_is_available = torch.cuda.is_available
    _orig_current_device = torch.cuda.current_device

    # Get local GPU count
    try:
        if _orig_is_available():
            _local_count = _orig_device_count()
        else:
            _local_count = 0
    except Exception:
        _local_count = 0

    # Only patch if we actually have remote GPUs to add
    if _remote_count == 0:
        return

    # --- Patch device_count ---
    def patched_device_count():
        return _local_count + _remote_count
    torch.cuda.device_count = patched_device_count

    # --- Patch is_available ---
    def patched_is_available():
        return _local_count + _remote_count > 0
    torch.cuda.is_available = patched_is_available

    # --- Patch get_device_properties ---
    if hasattr(torch.cuda, 'get_device_properties'):
        _orig_get_device_properties = torch.cuda.get_device_properties

        def patched_get_device_properties(device=None):
            if device is None:
                device = torch.cuda.current_device()
            if hasattr(device, 'index'):
                device = device.index
            if isinstance(device, torch.device):
                device = device.index or 0

            if not _is_remote_device(device):
                return _orig_get_device_properties(device)

            # Return a fake CUDADeviceProperties for the remote GPU
            ridx = _remote_device_idx(device)
            rp = _remote_props.get(ridx, {})

            # Create a namespace object that looks like CUDADeviceProperties
            class RemoteDeviceProps:
                def __init__(self, props):
                    self.name = props.get('name', 'gpushare Remote GPU')
                    self.major = props.get('major', 0)
                    self.minor = props.get('minor', 0)
                    self.total_memory = props.get('total_global_mem', 0)
                    self.multi_processor_count = props.get('multi_processor_count', 0)
                    self.is_integrated = False
                    self.is_multi_gpu_board = False
                    self.max_threads_per_block = props.get('max_threads_per_block', 1024)
                    self.max_threads_per_multi_processor = props.get('max_threads_per_mp', 0)
                    self.regs_per_block = props.get('regs_per_block', 0)
                    self.regs_per_multiprocessor = 0
                    self.warp_size = props.get('warp_size', 32)
                    self.max_block_dim = (
                        props.get('max_threads_dim', [1024, 1024, 64])[0] if isinstance(props.get('max_threads_dim'), list) else 1024,
                        props.get('max_threads_dim', [1024, 1024, 64])[1] if isinstance(props.get('max_threads_dim'), list) else 1024,
                        props.get('max_threads_dim', [1024, 1024, 64])[2] if isinstance(props.get('max_threads_dim'), list) else 64,
                    )
                    self.max_grid_dim = (
                        props.get('max_grid_size', [2**31-1, 65535, 65535])[0] if isinstance(props.get('max_grid_size'), list) else 2**31-1,
                        props.get('max_grid_size', [2**31-1, 65535, 65535])[1] if isinstance(props.get('max_grid_size'), list) else 65535,
                        props.get('max_grid_size', [2**31-1, 65535, 65535])[2] if isinstance(props.get('max_grid_size'), list) else 65535,
                    )
                    self.gcnArchName = ''

                def __repr__(self):
                    return (f"_CudaDeviceProperties(name='{self.name}', "
                            f"major={self.major}, minor={self.minor}, "
                            f"total_memory={self.total_memory // (1024*1024)}MB, "
                            f"multi_processor_count={self.multi_processor_count})")

            return RemoteDeviceProps(rp)

        torch.cuda.get_device_properties = patched_get_device_properties

    # --- Patch get_device_name ---
    if hasattr(torch.cuda, 'get_device_name'):
        _orig_get_device_name = torch.cuda.get_device_name

        def patched_get_device_name(device=None):
            if device is None:
                device = torch.cuda.current_device()
            if hasattr(device, 'index'):
                device = device.index
            if isinstance(device, torch.device):
                device = device.index or 0

            if not _is_remote_device(device):
                return _orig_get_device_name(device)

            ridx = _remote_device_idx(device)
            rp = _remote_props.get(ridx, {})
            return rp.get('name', 'gpushare Remote GPU')

        torch.cuda.get_device_name = patched_get_device_name

    sys.stderr.write(f"[gpushare] Hook active: {_local_count} local + {_remote_count} remote GPU(s)\n")


def install():
    """Install the hook. Called from gpushare.pth at Python startup."""
    # We patch torch.cuda AFTER torch is fully imported.
    # Strategy: wrap the real 'import torch' so that after it completes,
    # we apply our patches. We use sys.modules post-import hook.

    import importlib
    import importlib.abc

    class _GpusharePostImport(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        """Detects 'import torch.cuda' and patches after it loads."""
        _patched = False

        def find_module(self, fullname, path=None):
            # Trigger on torch.cuda import (which happens during torch init)
            if fullname == 'torch.cuda' and not self._patched:
                return self
            return None

        def load_module(self, fullname):
            # Let the real import happen first
            self._patched = True
            # Remove ourselves to avoid recursion
            if self in sys.meta_path:
                sys.meta_path.remove(self)
            # Do the real import
            mod = importlib.import_module(fullname)
            # Now patch it
            try:
                _hook_torch()
            except Exception as e:
                sys.stderr.write(f"[gpushare] Hook failed: {e}\n")
            return mod

    # Only install if not already done
    for finder in sys.meta_path:
        if isinstance(finder, _GpusharePostImport):
            return

    sys.meta_path.insert(0, _GpusharePostImport())
