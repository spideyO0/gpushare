# gpushare — GPU over IP

## What this project is

gpushare shares an NVIDIA GPU over a TCP network. One server machine (Arch Linux, RTX 5070 12GB, CUDA 13.1, SM 12.0 Blackwell) serves GPU compute to client machines (Linux, macOS, Windows) that have no local GPU. Applications on clients see the remote GPU as native — no code changes, no wrappers, no LD_PRELOAD.

## Architecture

### Core components

**Server** (`server/server.cpp`, ~900 lines):
- C++ TCP server using pthreads (one thread per client)
- Listens on port 9847 (configurable)
- Executes real CUDA calls on behalf of clients
- Per-client resource tracking (allocated memory, streams, events, modules — all cleaned up on disconnect)
- Per-client stats (ops count, bytes in/out, memory allocated)
- LAN/WAN auto-detection: enumerates local interfaces at startup, classifies each client by subnet match, applies different TCP buffer sizes (8MB LAN, 2MB WAN)
- Live GPU status via nvidia-smi subprocess parsing
- Config file at /etc/gpushare/server.conf
- CRITICAL: each client thread calls `cudaSetDevice(g_device)` to ensure CUDA context is current (Bug #1)
- **PinnedBufferPool** in ClientSession: 4x4MB pinned buffers allocated via `cudaMallocHost`, with `is_pinned[]` tracking and automatic `malloc` fallback. Used for D2H transfers to enable async DMA.
- **PendingTransfer** tracking for async H2D/D2H operations
- Async memcpy handlers (`GS_OP_MEMCPY_H2D_ASYNC`, `GS_OP_MEMCPY_D2H_ASYNC`) with chunked transfer support (4MB chunks) for streaming large transfers without blocking

**Client library** (`client/gpushare_client.cpp`, ~900 lines):
- Single shared library (.so/.dylib/.dll) that exports three API layers:
  - CUDA Runtime API (27 functions): cudaMalloc, cudaMemcpy, cudaStreamCreate, etc.
  - CUDA Driver API (56 functions): cuInit, cuMemAlloc, cuModuleLoadData, cuLaunchKernel, etc.
  - NVML API (29 functions): nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetMemoryInfo, etc.
- Installed as symlinks for ALL CUDA libraries (libcudart.so, libcuda.so.1, libnvidia-ml.so.1, libcublas.so, libcudnn.so, libcufft.so, libcusparse.so, libcusolver.so, libcurand.so, libnvrtc.so, libnvjpeg.so)
- Auto-reads server address from config files (~/.config/gpushare/client.conf, /etc/gpushare/client.conf) — no env vars needed
- MSVC-compatible: uses PACKED_STRUCT_BEGIN/END macros, SSIZE_T typedef, DllMain for cleanup, __declspec(dllexport)
- **Pipelined RPC architecture**: replaced single `g_sock_mtx` with `g_send_mtx` (send-only lock) and a dedicated `recv_thread` that dispatches responses by `req_id` via `promise`/`future` through a `g_pending` map. Multiple threads can have concurrent in-flight RPCs.
- Stores `g_server_caps` from INIT handshake. Uses async opcodes when `GS_CAP_ASYNC` is set, chunked transfers when `GS_CAP_CHUNKED` is set, falls back to sync single-message path for old servers.
- Windows `DllMain` uses `detach()` instead of `join()` on the recv thread during process exit to avoid loader lock deadlock

**Generated stubs — two tiers:**
1. `client/generated_stubs.cpp` + `server/generated_dispatch.cpp` (from `codegen/generate_stubs.py`):
   - 133 functions with full argument-aware RPC serialization
   - Covers critical cuBLAS (GEMM, GemmEx, strided batched), cuDNN (convolution, batchnorm, pooling, softmax, activation), and additional CUDA Runtime functions
   - Server dispatch uses dlsym() to call real library functions, with handle mapping for opaque library handles
   - Uses GS_OP_LIB_CALL opcode with [lib_id:u8][func_id:u16][serialized_args]
   - Each arg has a kind: SCALAR, DEV_PTR, HANDLE, HANDLE_OUT, HOST_IN, HOST_OUT

2. `client/generated_all_stubs.cpp` + `server/generated_all_dispatch.cpp` (from `codegen/parse_headers.py`):
   - 2,432 weak symbol stubs parsed from actual CUDA headers in /opt/cuda/include/
   - Ensures applications can LINK against our library for every CUDA symbol
   - Weak symbols get overridden by tier-1 full-RPC functions
   - Server has a dlsym-based resolver for all libraries

**Total: 2,600+ exported functions across 12 CUDA libraries.**

### Protocol (`include/gpushare/protocol.h`)

Binary protocol over TCP. 16-byte header:
```
[4B magic 0x47505553 "GPUS"] [4B total_length] [4B request_id] [2B opcode] [2B flags]
```

Key opcodes:
- 0x0001 INIT, 0x0002 CLOSE, 0x0003 PING
- 0x0010-0x0012 Device management
- 0x0020-0x0025 Memory (malloc, free, memcpy H2D/D2H/D2D, memset)
- 0x0026 MEMCPY_H2D_ASYNC, 0x0027 MEMCPY_D2H_ASYNC (pipelined async transfers)
- 0x0030-0x0033 Module/kernel (load PTX, get function, launch)
- 0x0040-0x0048 Streams and events
- 0x0050-0x0052 Fat binary registration
- 0x0060 Stats, 0x0070 Live GPU status
- 0x0080 Generic library call (cuBLAS/cuDNN/etc)

Flags:
- 0x0004 GS_FLAG_CHUNKED — message is part of a chunked transfer
- 0x0008 GS_FLAG_LAST_CHUNK — final chunk in a chunked transfer

Capability bits (in `gs_init_resp_t.capabilities`):
- 0x01 GS_CAP_ASYNC — server supports async memcpy opcodes
- 0x02 GS_CAP_CHUNKED — server supports chunked transfers (4MB chunks via GS_CHUNK_SIZE)

New structs: `gs_memcpy_h2d_async_req_t`, `gs_memcpy_d2h_async_req_t`, `gs_memcpy_h2d_chunk_t`, `gs_memcpy_d2h_chunk_t`. The `gs_init_resp_t` gained a `uint32_t capabilities` field (backward-compatible -- old clients read only the first 12 bytes).

All structs use `PACKED_STRUCT_BEGIN`/`PACKED_STRUCT_END` macros for MSVC portability.

### Monitoring

- **Web dashboard** (`dashboard/app.py`): Zero-dependency Python HTTP server, connects to gpushare server via binary protocol, polls nvidia-smi, serves embedded HTML/CSS/JS. Has POST /api/server (change IP) and POST /api/service (start/stop). Client mode queries GS_OP_GET_GPU_STATUS. Displays server stats (uptime, active clients, total connections, alloc_mb). `--config` with server.conf no longer forces client mode.
- **TUI** (`tui/monitor.py`): Curses-based, stdlib only. Keys: e=edit IP, s=start, x=stop, a=auto-stop, r=reconnect, q=quit.
- **Windows tray** (`client/gpu_tray_windows.pyw`): pystray + pillow. Live temp/util icon, right-click menu with server change dialog (tkinter), service control, nvidia-smi.
- **nvidia-smi shim** (`scripts/nvidia-smi`): Python script that queries NVML via ctypes, outputs matching nvidia-smi format including CSV query mode.

### Install scripts

All scripts support: `--force` (full reinstall), upgrade detection (IS_UPGRADE), config preservation, stale CMake cache cleaning.

- `install-server-arch.sh`: Arch Linux server setup, systemd services, codegen step, firewall, creates CUDA symlinks in `/usr/local/lib/gpushare/` (reference only, real CUDA keeps priority on server), backs up real libs, installs client.conf for dual-GPU support on the server machine itself
- `install-client-linux.sh`: 9 distro families auto-detected, auto-installs deps, SELinux/AppArmor handling, creates symlinks in `/usr/lib/` (system library dir, found by dynamic linker) + installs Python startup hook to site-packages
- `install-client-macos.sh`: Homebrew deps, quarantine clearing on ALL files, launchd plist, SIP-safe shebangs, installs Python startup hook to site-packages
- `install-client-windows.ps1`: VS/MinGW/MSYS2 auto-detect, always installs DLL overrides (even with local GPU), copies nvcuda.dll into torch\lib (critical: only nvcuda.dll, NOT cudart64_*.dll), registry GPU adapter, Defender exclusion, backs up real CUDA DLLs to `C:\Program Files\gpushare\real\`, installs Python hook via `site.getsitepackages()` (works for MS Store Python), handles `--force` in both `-Force` and `--force` styles
- `uninstall.sh`: Cross-platform (Linux/macOS), dry-run support, removes symlinks from `/usr/lib/`, removes Python hook from all site-packages
- `uninstall-windows.ps1`: 11-component removal including tray process kill, registry cleanup, startup shortcut, Python hook from all site-packages (including MS Store Python), DLL overrides from torch\lib with backup restoration

## Critical gotchas (from development)

1. **CUDA context in threads**: Every client thread MUST call `cudaSetDevice()` before any driver API call (cuModuleLoadData, cuLaunchKernel). Without it: error 201.
2. **PTX version**: CUDA 13.1 = PTX ISA 9.1, SM 12.0. Use `nvcc -ptx -arch=sm_120` to generate correct PTX. Old versions (7.0, 8.7) fail with error 218.
3. **cuModuleLoadData null termination**: The server MUST null-terminate PTX data before passing to cuModuleLoadData. Copy to a std::vector and append \0.
4. **MSVC portability**: No __attribute__((packed)), no ssize_t, no __attribute__((destructor)), no __attribute__((visibility)). All guarded by #ifdef _MSC_VER / _WIN32.
5. **macOS quarantine**: Must run `xattr -dr com.apple.quarantine` on EVERY installed file. Use /bin/bash and /usr/bin/python3 (full paths) in shebangs, not /usr/bin/env.
6. **PowerShell**: No non-ASCII characters (em-dashes break parsing). No here-strings (@"..."@) — use string concatenation with backtick-r-n. Use single-quotes for regex patterns.
7. **bind=0.0.0.0**: Treat as "all interfaces" (same as empty), not a specific IPv4 address.
8. **rpc_simple returns int32_t**, not cudaError_t. All callers in gpushare_client.cpp must cast: `(cudaError_t)rpc_simple(...)`.
9. **goto over initialization**: C++ forbids goto jumping over variable declarations. Generated dispatch uses `do{...}while(0)` + `break` pattern instead.
10. **Struct packing Python**: gs_device_props_t is 352 bytes packed. Python format: `"<QQ" + "i"*14 + "QQQ"` at offset 256 after the name field.
11. **DllMain loader lock deadlock**: On Windows, calling `join()` on the recv thread inside `DllMain(DLL_PROCESS_DETACH)` deadlocks because the loader lock prevents the thread from exiting. Use `detach()` instead and let the OS clean up.
12. **Client pthread linking**: The pipelined recv thread requires `-lpthread` on Linux/macOS. CMakeLists.txt must link pthread for the client on non-Windows platforms.
13. **Windows CUDA_VISIBLE_DEVICES**: NEVER set `CUDA_VISIBLE_DEVICES` to a remote GPU index — it makes CUDA see 0 devices. Use `torch.cuda.set_device()` for remote GPUs instead.
14. **cuGetProcAddress is critical**: CUDA 12+ uses `cuGetProcAddress` for runtime symbol resolution. PyTorch's `c10_cuda.dll` calls it to resolve ALL CUDA functions. Must be implemented properly (using `GetProcAddress(self)` on Windows / `dlsym(RTLD_DEFAULT)` on Linux), NOT a weak stub. Without it: WinError 127 on Windows.
15. **Windows torch\lib: only replace nvcuda.dll**: Do NOT replace `cudart64_*.dll` in `torch\lib`. PyTorch's bundled CUDA runtime has internal functions (`__cudaRegisterFatBinary`, `__cudaRegisterFunction`) that `c10_cuda.dll` needs. Our DLL doesn't export these. Only replace `nvcuda.dll` — the bundled cudart internally calls nvcuda for GPU ops, so intercepting nvcuda is sufficient.
16. **cudaDeviceGetAttribute must forward to cuDeviceGetAttribute**: The runtime API `cudaDeviceGetAttribute` must NOT be a stub returning 0. PyTorch queries 40+ device attributes (canMapHostMemory, concurrentKernels, sharedMemPerMultiprocessor, etc.). Returning 0 for these makes PyTorch think the GPU is broken. Forward to the driver API implementation.
17. **cudaGetDeviceProperties must fill all fields**: PyTorch reads ~50 fields from `cudaDeviceProp`. Missing fields like `canMapHostMemory`, `concurrentKernels`, `unifiedAddressing`, `cooperativeLaunch` cause silent failures. Fill with server values or sensible modern-GPU defaults.
18. **Python hook circular imports**: The hook MUST NOT call `import torch` during `_do_patch()` — use `sys.modules.get('torch')` instead. Also must not patch during the import chain. Use import-depth tracking: only patch when depth returns to 0 (all imports complete).
19. **Windows DLL search order**: System32 beats PATH. On Windows, `ctypes.CDLL('nvcuda.dll')` loads from System32 first. Must load our DLL by FULL PATH (`C:\Program Files\gpushare\nvcuda.dll`) to preload it before PyTorch. Windows reuses already-loaded DLLs by base name.
20. **MS Store Python paths**: MS Store Python uses `%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.*\LocalCache\local-packages\` for site-packages, NOT `Lib\site-packages` under the exe directory. Use `python -c "import site; print(site.getsitepackages())"` to discover. WindowsApps directory is read-only sandbox.
21. **MinGW weak symbol conflicts**: MinGW on Windows doesn't support weak symbol override like GCC on Linux. If a function has both a weak stub in `generated_all_stubs.cpp` and a strong definition in `gpushare_client.cpp`, MinGW linker errors with "multiple definition". Remove the weak stub when adding a strong implementation.

## How to test

```bash
# Build
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# Quick test cycle
./build/gpushare-server &
sleep 2
GPUSHARE_SERVER=localhost:9847 ./build/test_basic       # C++ test
PYTHONPATH=python python examples/test_python.py localhost  # Python test
PYTHONPATH=python python examples/stress_test.py localhost --duration 10  # Stress
kill %1

# Check symbols
nm -D build/libgpushare_client.so.1.0.0 | grep " T " | wc -l  # should be ~2600+

# Validate scripts
for f in scripts/install-*.sh scripts/uninstall.sh; do bash -n "$f" && echo "$f: OK"; done

# Check PS scripts for non-ASCII (causes parse errors on Windows)
grep -Pn '[^\x00-\x7F]' scripts/install-client-windows.ps1 scripts/uninstall-windows.ps1
```

## How to add a new full-RPC function

1. Find signature in CUDA header (e.g., /opt/cuda/include/cublas_api.h)
2. Add to LIBS dict in `codegen/generate_stubs.py` with arg definitions
3. Each arg: (name, type, kind, byte_size) — kinds: SCALAR, DEV_PTR, HANDLE, HANDLE_OUT, HOST_IN, HOST_OUT
4. Run `python codegen/generate_stubs.py`
5. Rebuild: `cd build && make -j$(nproc)`
6. Test with ctypes or an application that uses that function

## Key design decisions

- **One library for all APIs**: Instead of separate libs for each NVIDIA library, one .so exports everything. Simpler installation, fewer symlinks to manage.
- **Weak symbols for coverage**: parse_headers.py generates weak stubs so apps LINK successfully even for functions without full RPC. Better than crashing with "symbol not found".
- **Config file over env vars**: Env vars don't persist across sessions and break in venvs. Config files at well-known paths work everywhere.
- **LAN/WAN auto-detect**: Server checks client IP against local interface subnets. No user configuration needed.
- **Per-client thread model**: Simple, each client gets full CUDA context. Trade-off: more memory per client, but simpler than async multiplexing.
- **Local GPU passthrough**: When a local NVIDIA GPU exists, the client detects it via dlopen of real CUDA libraries from `/usr/local/lib/gpushare/real/` (Linux/macOS) or `C:\Program Files\gpushare\real\` (Windows, using LoadLibraryExA/GetProcAddress). Both local and remote GPUs are presented to applications. Device 0..N-1 are local, device N+ are remote. `cudaSetDevice()` routes all subsequent calls. Config: `gpu_mode=all|remote|local` in client.conf, or `GPUSHARE_GPU_MODE` env var. Requires `RTLD_DEEPBIND` on Linux to prevent symbol conflicts. On Windows, DLL replacement is ALWAYS used (nvcuda.dll, cudart64_*.dll copied into Python directories). The C library handles dual-GPU routing: local device ops go through backed-up real DLLs in `C:\Program Files\gpushare\real\`, remote device ops go through the gpushare server.
- **Python startup hook**: `python/gpushare_hook.py` + `python/gpushare.pth` patches `torch.cuda` after import completes (using import-depth tracking to avoid circular imports). Two-phase GPU discovery: Phase 1 via ctypes to the loaded CUDA library, Phase 2 via TCP to gpushare server (fallback for when the real NVIDIA driver is loaded instead of ours). Patches 16 functions: `device_count`, `get_device_properties`, `get_device_name`, `get_device_capability`, `set_device`, `current_device`, `memory_allocated`, `mem_get_info`, `is_available`, `get_arch_list`, `_check_capability`, `_check_cubins`, `_lazy_init`, `torch.backends.cudnn`, `torch.backends.cuda.is_built`, `torch.version.cuda`. Suppresses SM compatibility warnings. No code changes needed in user applications.

## File layout

```
server/server.cpp           — GPU server (TCP + CUDA)
server/generated_dispatch.cpp — Auto-gen: 133 cuBLAS/cuDNN dispatch handlers
server/generated_all_dispatch.cpp — Auto-gen: dlsym resolver for all libraries
client/gpushare_client.cpp  — Client lib (runtime + driver + NVML, 112 hand-written functions)
client/generated_stubs.cpp  — Auto-gen: 133 cuBLAS/cuDNN client stubs
client/generated_all_stubs.cpp — Auto-gen: 2432 weak symbol exports
include/gpushare/protocol.h — Wire protocol + packed structs
include/gpushare/cuda_defs.h — CUDA/NVML/Driver types (no CUDA toolkit needed)
codegen/generate_stubs.py   — Codegen: function defs → full RPC stubs
codegen/parse_headers.py    — Codegen: CUDA headers → weak stubs
dashboard/app.py            — Web dashboard (1147 lines, zero dependencies)
tui/monitor.py              — Terminal UI (891 lines, curses only)
client/gpu_tray_windows.pyw — Windows tray widget (pystray + pillow)
python/gpushare/__init__.py — Python client library
python/gpushare_hook.py     — Python startup hook: monkey-patches torch.cuda for remote GPUs
python/gpushare.pth         — Python path config to auto-load gpushare_hook at startup
scripts/nvidia-smi          — nvidia-smi shim (Python, ctypes → NVML)
examples/stress_test.py     — Stress test + connection validator
examples/gpu.py             — GPU detection (4 methods)
examples/fooocus_gpu_patch.py — Drop-in GPU detection for PyTorch launchers (e.g., Fooocus)
CMakeLists.txt              — Build: server (needs CUDA) + client (no CUDA)
```

## Current state

- Server: Arch Linux, RTX 5070, CUDA 13.1, systemd service running
- Tested clients: macOS (M-series MacBook), Windows 11 (VS 2026)
- API exports: 2,600+ functions (112 hand-written + 133 generated RPC + 2,432 weak stubs)
- Verified working: GPU detection (NVML + CUDA), memory ops, PTX kernel execution, data transfer round-trip, stress test at ~2 GB/s local / ~44 MB/s over 1GbE LAN
- Performance: pipelined RPC with async/chunked memcpy, pinned buffer pool on server for DMA transfers
