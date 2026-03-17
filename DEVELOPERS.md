# gpushare Developer Documentation

**Comprehensive technical guide for contributors**

This document reflects the real development history of gpushare, including actual bugs encountered, their root causes, and fixes. If you are contributing code, read this entire document before submitting your first PR.

---

## Table of Contents

1. [Architecture Deep Dive](#1-architecture-deep-dive)
2. [Common Mistakes and Gotchas](#2-common-mistakes-and-gotchas)
3. [Testing Commands](#3-testing-commands)
4. [Adding a New CUDA Function](#4-adding-a-new-cuda-function)
5. [Protocol Reference](#5-protocol-reference)
6. [Troubleshooting Guide](#6-troubleshooting-guide)
7. [File Reference](#7-file-reference)

---

## 1. Architecture Deep Dive

gpushare is a GPU-over-network system that lets machines without a physical GPU run CUDA applications transparently. The architecture has seven major components.

### 1.1 Server (`server/server.cpp`)

The server runs on the machine with the physical GPU (e.g., Arch Linux with an RTX 5070). It is a single-process, multi-threaded TCP server.

**Key design decisions:**

- **Per-client threads.** Each accepted connection spawns a `client_thread` via pthreads. The thread owns the socket and processes messages in a loop until disconnect.
- **CUDA context management.** Every client thread calls `cudaSetDevice(g_device)` on entry to establish a CUDA context on that thread. Without this, Driver API calls (like `cuModuleLoadData`) fail with error 201 (`CUDA_ERROR_INVALID_CONTEXT`).
- **Resource tracking.** Each `ClientSession` tracks all allocated device pointers, loaded modules, created streams, and created events. On disconnect (or crash), `cleanup_resources()` frees everything so one bad client cannot leak GPU memory.
- **LAN/WAN detection.** On startup, `detect_local_networks()` enumerates all local interfaces via `getifaddrs()`. When a client connects, `classify_client()` checks whether the client IP falls within any local subnet. This is used for logging and could be used for bandwidth throttling.
- **nvidia-smi parsing.** The `GS_OP_GET_GPU_STATUS` handler parses nvidia-smi output to return live GPU metrics (temperature, utilization, memory, power, clocks) without linking against NVML directly.
- **Stats tracking.** Global atomics (`g_total_ops`, `g_total_bytes_in`, `g_total_bytes_out`) and per-client atomics track all operations. The `GS_OP_GET_STATS` handler aggregates these into a response for the dashboard.
- **Pinned buffer pool.** `PinnedBufferPool` pre-allocates 4 x 4 MB `cudaMallocHost` buffers at startup. Each buffer has an `is_pinned` flag tracking whether it is in use. Async transfer handlers (`handle_memcpy_h2d_async`, `handle_memcpy_d2h_async`) acquire a pinned buffer, initiate the transfer on a stream, and attach a `PendingTransfer` (cudaEvent + buffer index) so the buffer is released back to the pool once the event completes.
- **Chunked transfers.** `handle_memcpy_h2d_chunked` and `handle_memcpy_d2h_chunked` break large transfers into protocol-sized chunks, enabling progress reporting and avoiding the 256 MB max message limit for very large allocations.
- **Capability reporting.** The `handle_init` handler now returns a `capabilities` field in `gs_init_resp_t`, advertising `GS_CAP_ASYNC | GS_CAP_CHUNKED` so clients can discover and use optimized transfer paths.
- **Configuration.** Reads from `/etc/gpushare/server.conf` or a path given via `--config`. Supports `device`, `port`, `max_clients`, `log_level`, and `bind` (bind address).

**Server startup sequence:**

1. Parse CLI args and config file
2. Initialize CUDA (`cudaSetDevice`, `cuInit`)
3. Detect local networks for LAN/WAN classification
4. Create TCP socket (IPv6 dual-stack with `IPV6_V6ONLY=0`)
5. Bind and listen
6. Accept loop: spawn `client_thread` for each connection

### 1.2 Client Library (`client/gpushare_client.cpp`)

The client is a shared library (`libgpushare_client.so` / `.dylib` / `.dll`) that exports the same symbols as the real CUDA libraries. Applications link against it (or it replaces the real libraries via install scripts) and all CUDA calls are transparently forwarded over the network.

**Three API layers in one library:**

| Layer | Prefix | Examples |
|-------|--------|---------|
| CUDA Runtime | `cuda*` | `cudaMalloc`, `cudaMemcpy`, `cudaDeviceSynchronize` |
| CUDA Driver | `cu*` | `cuInit`, `cuModuleLoadData`, `cuLaunchKernel` |
| NVML | `nvml*` | `nvmlInit_v2`, `nvmlDeviceGetCount_v2`, `nvmlShutdown` |

**Server address resolution (priority order):**

1. `GPUSHARE_SERVER` environment variable
2. `~/.config/gpushare/client.conf`
3. `/etc/gpushare/client.conf` (Linux) or `C:\ProgramData\gpushare\client.conf` (Windows)
4. Default: `localhost:9847`

**Connection management (pipelined):**

- Single TCP connection per process with pipelined request/response handling
- `g_send_mtx` protects the send path only (not the full socket)
- A dedicated `recv_thread` reads all incoming responses and dispatches them by `req_id` via `std::promise`/`std::future` pairs stored in the `g_pending` map
- Request IDs are monotonically increasing (`g_req_id` atomic)
- Multiple threads can have in-flight requests simultaneously; each blocks on its own future
- Connection is established lazily on first CUDA call (via `ensure_connected()`)
- `g_server_caps` is read from the `gs_init_resp_t` during the init handshake and used to select async/chunked transfer paths with automatic fallback to synchronous transfers if the server does not advertise the corresponding capability
- On Windows, `DllMain` with `DLL_PROCESS_DETACH` calls `detach()` on the recv thread instead of `join()` to avoid a loader lock deadlock during process exit

**Handle mapping:**

Server-side GPU pointers are opaque `uint64_t` values. On 64-bit clients, these are cast directly to `void*`. The client maintains no mapping table on 64-bit systems. On 32-bit systems (rare for GPU work), a mapping table would be needed.

**Platform abstractions:**

- `sock_t` type: `int` on POSIX, `SOCKET` on Windows
- `sock_send`/`sock_recv` wrappers handle Windows cast requirements
- `PACKED_STRUCT_BEGIN`/`PACKED_STRUCT_END` macros handle MSVC vs GCC struct packing
- `ssize_t` is typedef'd to `SSIZE_T` on Windows
- Library cleanup uses `__attribute__((destructor))` on POSIX, `DllMain` on Windows

### 1.3 Generated Stubs (codegen/)

The code generation system has two tiers:

**Tier 1: Full-RPC stubs (`codegen/generate_stubs.py`) -- 133 functions**

These are hand-defined functions with full argument-level serialization. Each function entry specifies every argument with:
- Name
- C type (`i32`, `u64`, `ptr`, etc.)
- Kind (`SCALAR`, `DEV_PTR`, `HANDLE`, `HANDLE_OUT`, `HOST_IN`, `HOST_OUT`, `HOST_INOUT`)
- Size in bytes

The generator produces:
- `client/generated_stubs.cpp` -- client-side exported functions that serialize args and call `rpc_lib_call()`
- `server/generated_dispatch.cpp` -- server-side dispatch that deserializes args, calls the real CUDA function via `dlsym()`, and serializes results

Currently covers cuBLAS (Level 1/2/3 ops, GemmEx, strided batched) and cuDNN (convolution, activation, batch norm, pooling, softmax, dropout).

**Tier 2: Weak symbol stubs (`codegen/parse_headers.py`) -- 2432 functions**

For functions that do not need deep argument understanding, this script:
1. Reads actual CUDA header files from `/opt/cuda/include`
2. Extracts all function signatures via regex
3. Generates client stubs that forward raw bytes via `GS_OP_LIB_CALL`
4. Generates server dispatch that uses `dlsym()` to find the real function

These stubs are declared with `__attribute__((weak))` so Tier 1 definitions take precedence. Libraries covered: cuda_runtime, cuda_driver, cublas, cublasLt, cudnn, cufft, cusparse, cusolverDn, curand, nvrtc, nvjpeg.

Output files:
- `client/generated_all_stubs.cpp`
- `server/generated_all_dispatch.cpp`

### 1.4 Protocol (`include/gpushare/protocol.h`)

Binary protocol over TCP. All integers are little-endian. All structs use `#pragma pack(1)` (no padding).

**Header format (16 bytes):**

```
Offset  Size  Field    Description
0       4     magic    0x47505553 ("GPUS")
4       4     length   Total message size including header
8       4     req_id   Client-assigned, echoed in response
12      2     opcode   Operation code
14      2     flags    0x0001 = response, 0x0002 = error
```

Max message size: 256 MB (`GPUSHARE_MAX_MSG_SIZE`).

See [Section 5](#5-protocol-reference) for the full opcode and payload reference.

### 1.5 Dashboard (`dashboard/app.py`)

Zero-dependency Python HTTP server for web-based monitoring. Uses only the Python standard library (`http.server`, `json`, `subprocess`, `threading`, `struct`, `socket`).

**Modes:**
- **Server mode** (default, port 9848): Queries GPU status via `GS_OP_GET_GPU_STATUS` and client info via `GS_OP_GET_STATS` from the gpushare server. No local nvidia-smi dependency.
- **Client mode** (`--client`, port 9849): Fetches GPU status from the remote gpushare server (no local nvidia-smi needed). Fixed: `--config` with `server.conf` no longer forces client mode.

**Implementation:**
- `Poller` thread connects to the gpushare server every 2 seconds, sends `GS_OP_GET_STATS` and `GS_OP_GET_GPU_STATUS`, and updates shared state.
- `DashboardHandler` serves a single HTML page with all CSS and JS embedded inline.
- API endpoint `/api/stats` returns JSON for AJAX polling. Stats now include: server uptime, active clients, total connections, and allocated VRAM.
- No WebSocket or SSE -- uses simple polling from the frontend.

### 1.6 TUI (`tui/monitor.py`)

Curses-based terminal dashboard. Works on macOS and Linux with Python 3.7+, using only the standard library.

**Features:**
- Real-time GPU stats (memory, utilization, temperature, power)
- Client connection list with per-client bandwidth
- Service management (start/stop/restart via systemctl)
- Inline IP editing for server address
- Auto-stop detection on client disconnect
- Unicode box-drawing and sparkline charts

**Protocol interaction:** Connects to the gpushare server via raw sockets, sends `GS_OP_INIT` and `GS_OP_GET_STATS`, parses binary responses.

### 1.7 Python Client (`python/gpushare/`)

Pure-Python client library with socket-based communication. No compiled dependencies (except numpy for array operations).

**Key class: `RemoteGPU`**

Provides a Pythonic API: `connect()`, `malloc()`, `free()`, `memcpy_h2d()`, `memcpy_d2h()`, `load_module()`, `launch_kernel()`, `synchronize()`, etc.

**Auto-config:** `gpushare.connect()` with no arguments reads from `GPUSHARE_SERVER` env var, then `~/.config/gpushare/client.conf`, then `/etc/gpushare/client.conf`, then falls back to `localhost:9847`.

**Numpy integration:** `memcpy_h2d()` accepts numpy arrays directly (calls `.tobytes()`). `memcpy_d2h()` writes directly into a numpy array via `np.frombuffer().reshape()`.

### 1.8 Windows Tray (`client/gpu_tray_windows.pyw`)

System tray widget for Windows using `pystray` and `pillow`. Shows remote GPU stats (temperature, utilization, VRAM) in the system tray with hover tooltip and right-click menu. Communicates with the gpushare server via raw sockets using the binary protocol (no DLL dependency).

Run with `pythonw gpu_tray_windows.pyw` for no console window.

---

## 2. Common Mistakes and Gotchas

These are real bugs encountered during development. Each entry describes the symptom, root cause, and fix.

### Bug 1: CUDA context not current in client threads

**Symptom:** `cuModuleLoadData` returned error 201 (`CUDA_ERROR_INVALID_CONTEXT`).

**Root cause:** The CUDA Driver API requires a current context on each thread. The main thread initialized CUDA, but `client_thread` ran on a new pthread with no CUDA context.

**Fix:** Added `cudaSetDevice(g_device)` at the start of `client_thread()`. The Runtime API implicitly creates/pushes a context for the calling thread.

### Bug 2: PTX version mismatch

**Symptom:** `cuModuleLoadData` returned error 218 (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`).

**Root cause:** CUDA 13.1 uses PTX ISA 9.1. The client was generating PTX with ISA 8.7 (for older architectures). SM 12.0 (Blackwell/RTX 5070) requires a specific PTX version.

**Fix:** Use nvcc to generate correct PTX for the target architecture:

```bash
nvcc -ptx -arch=sm_120 kernel.cu
```

The generated PTX will use `.version 9.1` and `.reg .b32` (not `.reg .f32`). Always match the PTX version to the server's CUDA toolkit.

### Bug 3: PTX null termination

**Symptom:** `cuModuleLoadData` returned garbage errors or crashed.

**Root cause:** `cuModuleLoadData` expects a null-terminated PTX string. The network payload sent the raw PTX bytes without guaranteeing a `\0` terminator.

**Fix:** Server copies PTX data to a new buffer and appends `\0` before passing to `cuModuleLoadData`.

### Bug 4: MSVC compilation failures

**Symptom:** Multiple compile errors on Windows: `__attribute__((packed))` unknown, `ssize_t` undefined, `__attribute__((destructor))` unknown, `__attribute__((visibility))` unknown.

**Root cause:** These are GCC/Clang extensions that MSVC does not support.

**Fix:**

| GCC/Clang | MSVC Replacement |
|-----------|-----------------|
| `__attribute__((packed))` | `PACKED_STRUCT_BEGIN` / `PACKED_STRUCT_END` macros using `#pragma pack(push, 1)` / `#pragma pack(pop)` |
| `ssize_t` | `#include <BaseTsd.h>` + `typedef SSIZE_T ssize_t` |
| `__attribute__((destructor))` | `DllMain` with `DLL_PROCESS_DETACH` |
| `__attribute__((visibility("default")))` | `__declspec(dllexport)` |

### Bug 5: macOS quarantine

**Symptom:** Installed binary/library refused to execute with "operation not permitted" or similar, even with correct `chmod +x`.

**Root cause:** macOS adds the `com.apple.quarantine` extended attribute to downloaded files. Gatekeeper blocks execution of quarantined files.

**Fix:** Run `xattr -dr com.apple.quarantine` on ALL installed files: libraries, scripts, symlinks, and share directories.

### Bug 6: macOS SIP (System Integrity Protection)

**Symptom:** Scripts with `#!/usr/bin/env bash` shebang failed when run with `sudo`.

**Root cause:** SIP restricts `/usr/bin/env` behavior under sudo, stripping certain environment variables and blocking execution paths.

**Fix:** Use full paths in shebangs: `#!/bin/bash`, `#!/usr/bin/python3` instead of `#!/usr/bin/env bash`.

### Bug 7: macOS bash 3

**Symptom:** Install script failed with syntax error on `${var,,}`.

**Root cause:** macOS ships bash 3.x by default. The `${var,,}` lowercase syntax requires bash 4+. Also, `grep -P` (Perl regex) is not available on macOS.

**Fix:** Use explicit `case` statements for case-insensitive comparisons. Use `sed` instead of `grep -P`.

### Bug 8: PowerShell here-strings

**Symptom:** PowerShell install script threw parse errors with `@"..."@` here-strings.

**Root cause:** PowerShell here-strings with `@"` (expandable) break when indented or when the content contains unmatched parentheses. The `@"` must be on its own line, and `"@` must be at column 0.

**Fix:** Use string concatenation with backtick-escaped newlines (`` `r`n ``) instead of here-strings.

### Bug 9: PowerShell non-ASCII

**Symptom:** PowerShell scripts threw parse errors on certain lines.

**Root cause:** Em-dashes, box-drawing characters, and other non-ASCII characters in `.ps1` string literals cause parse errors depending on file encoding and PowerShell version.

**Fix:** Use only ASCII characters in `.ps1` files. Replace em-dashes with `--`, box-drawing with `-` and `|`.

### Bug 10: Stale CMake cache

**Symptom:** Building on a different machine produced "source directory does not match" CMake errors.

**Root cause:** `CMakeCache.txt` stores the absolute path in `CMAKE_HOME_DIRECTORY`. When the repo is cloned to a different path or machine, CMake detects the mismatch and refuses to build.

**Fix:** Check `CMAKE_HOME_DIRECTORY` in `CMakeCache.txt`. If it differs from the current source directory, delete the build directory and re-run CMake:

```bash
if grep -q "CMAKE_HOME_DIRECTORY" build/CMakeCache.txt; then
    cached=$(grep CMAKE_HOME_DIRECTORY build/CMakeCache.txt | cut -d= -f2)
    if [ "$cached" != "$(pwd)" ]; then
        rm -rf build && mkdir build
    fi
fi
```

### Bug 11: Visual Studio version detection

**Symptom:** CMake configuration failed because it could not find the correct Visual Studio generator.

**Root cause:** VS 2026 reports major version 18 via `vswhere`, but CMake needs "Visual Studio 18 2026" as the generator name. The `catalog_productLineVersion` field returns "18", not "2026".

**Fix:** Map major version to year using a lookup table:

```powershell
$vsYearMap = @{ "15"="2017"; "16"="2019"; "17"="2022"; "18"="2026" }
```

### Bug 12: systemd service holding port

**Symptom:** Running `./build/gpushare-server` manually failed with "Address already in use" after installing the systemd service.

**Root cause:** The systemd unit has `Restart=on-failure`, so the service auto-restarts and holds port 9847.

**Fix:**

```bash
# Stop the service before manual runs
sudo systemctl stop gpushare-server

# Or, update the binary and restart the service
sudo cp build/gpushare-server /usr/local/bin/
sudo systemctl restart gpushare-server
```

### Bug 13: rpc_simple return type

**Symptom:** Compilation errors after changing `rpc_simple()` return type from `cudaError_t` to `int32_t`.

**Root cause:** Generated stubs and hand-written code both called `rpc_simple()` and assigned the result to `cudaError_t` variables without a cast.

**Fix:** Cast at all call sites: `(cudaError_t)rpc_simple(...)`.

### Bug 14: Arch Linux nvcc not in PATH

**Symptom:** `nvcc` not found when building with `sudo make install` on Arch Linux.

**Root cause:** On Arch, CUDA installs to `/opt/cuda/bin/` which is in the user's PATH but not in root's PATH (sudo resets PATH).

**Fix:** In install scripts, auto-add `/opt/cuda/bin` to PATH and pass `-DCUDAToolkit_ROOT=/opt/cuda` to cmake:

```bash
export PATH="/opt/cuda/bin:$PATH"
cmake .. -DCUDAToolkit_ROOT=/opt/cuda
```

### Bug 15: goto over variable initialization

**Symptom:** C++ compile error: "jump to label crosses initialization of variable".

**Root cause:** Generated server dispatch code used `goto cleanup` labels that jumped over variable declarations within the same scope. C++ forbids jumping over non-trivial variable initialization.

**Fix:** Wrap each dispatch case in `do { ... } while(0)` and use `break` instead of `goto`.

### Bug 16: Python client connect() defaulting to localhost

**Symptom:** `gpushare.connect()` with no arguments always connected to localhost, ignoring config files.

**Root cause:** The initial implementation did not read config files when no server argument was given.

**Fix:** Added `_read_server_from_config()` that checks `~/.config/gpushare/client.conf` and `/etc/gpushare/client.conf` for a `server=` line.

### Bug 17: Struct packing mismatch in Python

**Symptom:** `device_properties()` raised "buffer too small" or returned garbage values.

**Root cause:** `gs_device_props_t` was parsed with the wrong struct format string. The byte count did not match the actual struct layout.

**Fix:** Count actual bytes: 256 (name) + 16 (2 x uint64) + 56 (14 x int32) + 24 (3 x uint64) = 352 bytes. Use format string `"<QQ" + "i"*14 + "QQQ"` offset by 256.

### Bug 19: DllMain thread join deadlock on Windows

**Symptom:** Client process hangs on exit (never terminates) on Windows.

**Root cause:** `DllMain` with `DLL_PROCESS_DETACH` runs under the Windows loader lock. Calling `std::thread::join()` on the recv thread from `DllMain` deadlocks because the recv thread itself needs the loader lock to terminate.

**Fix:** Call `detach()` instead of `join()` on the recv thread during `DLL_PROCESS_DETACH`. The process is exiting anyway, so the detached thread will be cleaned up by the OS. Only use `join()` for explicit `close()` calls outside of `DllMain`.

### Bug 18: bind=0.0.0.0 treated as specific address

**Symptom:** Server bound to 0.0.0.0 as a specific interface instead of all interfaces, causing connectivity issues on some systems.

**Root cause:** The config parser treated `bind_address=0.0.0.0` as a specific IPv4 address and bound to it explicitly, rather than interpreting it as "listen on all interfaces".

**Fix:** Treat `"0.0.0.0"` and `"::"` the same as an empty bind address (use IPv6 `in6addr_any` with dual-stack).

---

## 3. Testing Commands

These are the exact commands used during development. Run them from the project root (`~/gpushare`).

### Build

```bash
cd ~/gpushare/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

On Arch Linux with CUDA in /opt/cuda:

```bash
cd ~/gpushare/build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/opt/cuda && make -j$(nproc)
```

### Start Server

```bash
./build/gpushare-server --log-level debug
```

### C++ Client Test

```bash
GPUSHARE_SERVER=localhost:9847 ./build/test_basic
```

### Python Client Test

```bash
PYTHONPATH=python python examples/test_python.py localhost
```

### Stress Test

```bash
PYTHONPATH=python python examples/stress_test.py --duration 10
```

### GPU Detection

```bash
python examples/gpu.py
```

### Dashboard Test

```bash
python -c "
import sys, threading, time, json, urllib.request
sys.argv = ['app.py']
sys.path.insert(0, 'dashboard')
import app as d
d._state['mode'] = 'server'
d._gs_server_addr = 'localhost:9847'
p = d.Poller('server', 'localhost', 9847)
p.start()
time.sleep(3)
from http.server import HTTPServer
s = HTTPServer(('0.0.0.0', 9850), d.DashboardHandler)
t = threading.Thread(target=s.serve_forever, daemon=True)
t.start()
time.sleep(1)
r = urllib.request.urlopen('http://localhost:9850/api/stats')
print(json.loads(r.read()))
s.shutdown()
"
```

### TUI Syntax Check

```bash
python -c "import tui.monitor; print('OK')"
```

### Check Exported Symbols

```bash
# Total exported symbols
nm -D build/libgpushare_client.so.1.0.0 | grep " T " | wc -l

# CUDA Runtime symbols
nm -D build/libgpushare_client.so.1.0.0 | grep " T cuda" | wc -l

# CUDA Driver symbols
nm -D build/libgpushare_client.so.1.0.0 | grep " T cu[A-Z]" | wc -l

# NVML symbols
nm -D build/libgpushare_client.so.1.0.0 | grep " T nvml" | wc -l

# cuBLAS symbols
nm -D build/libgpushare_client.so.1.0.0 | grep " T cublas" | wc -l
```

### Validate Install Scripts

```bash
for f in scripts/install-*.sh scripts/uninstall.sh; do bash -n "$f" && echo "$f: OK"; done
```

### Validate Python Files

```bash
for f in dashboard/app.py tui/monitor.py scripts/nvidia-smi scripts/gpushare-patch-frameworks.py; do
    python -c "import py_compile; py_compile.compile('$f', doraise=True)" && echo "$f: OK"
done
```

### Check Non-ASCII in PowerShell Scripts

```bash
grep -Pn '[^\x00-\x7F]' scripts/install-client-windows.ps1 scripts/uninstall-windows.ps1
```

### Test NVML Detection via ctypes

```bash
python -c "
import ctypes
lib = ctypes.CDLL('build/libgpushare_client.so.1.0.0')
lib.nvmlInit_v2()
count = ctypes.c_uint(0)
lib.nvmlDeviceGetCount_v2(ctypes.byref(count))
print(f'GPUs: {count.value}')
lib.nvmlShutdown()
"
```

### Test CUDA Driver API via ctypes

```bash
python -c "
import ctypes
lib = ctypes.CDLL('build/libgpushare_client.so.1.0.0')
lib.cuInit(0)
count = ctypes.c_int(0)
lib.cuDeviceGetCount(ctypes.byref(count))
print(f'CUDA devices: {count.value}')
"
```

### Run Code Generation

```bash
python codegen/generate_stubs.py    # 133 full-RPC stubs
python codegen/parse_headers.py     # 2432 weak stubs from headers
```

### Kill Server on Specific Port

```bash
fuser -k 9847/tcp
```

### Check What Is Listening on a Port

```bash
ss -tlnp | grep 9847
```

---

## 4. Adding a New CUDA Function

This walkthrough shows how to add full RPC support for a new function. We will use `cublasSsymm_v2` as the example.

### Step 1: Find the signature in the CUDA header

Look up the function in the CUDA headers (typically at `/opt/cuda/include/cublas_v2.h`):

```c
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsymm_v2(
    cublasHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    int m, int n,
    const float *alpha,
    const float *A, int lda,
    const float *B, int ldb,
    const float *beta,
    float *C, int ldc);
```

### Step 2: Add it to codegen/generate_stubs.py

Open `codegen/generate_stubs.py` and add the function to the `LIBS["cublas"]["functions"]` list:

```python
("cublasSsymm_v2", "i32", [
    ("handle", "ptr", "HANDLE", 8),
    ("side", "i32", "SCALAR", 4),
    ("uplo", "i32", "SCALAR", 4),
    ("m", "i32", "SCALAR", 4),
    ("n", "i32", "SCALAR", 4),
    ("alpha", "ptr", "HOST_IN", 4),     # const float* = 4 bytes read from host
    ("A", "ptr", "DEV_PTR", 8),
    ("lda", "i32", "SCALAR", 4),
    ("B", "ptr", "DEV_PTR", 8),
    ("ldb", "i32", "SCALAR", 4),
    ("beta", "ptr", "HOST_IN", 4),      # const float* = 4 bytes read from host
    ("C", "ptr", "DEV_PTR", 8),
    ("ldc", "i32", "SCALAR", 4)]),
```

### Step 3: Understand the argument kinds

Each argument has a "kind" that tells the codegen how to serialize/deserialize it:

| Kind | Description | Serialization |
|------|-------------|---------------|
| `SCALAR` | Small value sent inline (int, float, enum) | Raw bytes, `size` bytes |
| `DEV_PTR` | Device pointer (already on GPU memory) | Sent as `uint64_t` handle |
| `HANDLE` | Library handle (opaque, mapped client-to-server) | Sent as `uint64_t`, server looks up real handle |
| `HANDLE_OUT` | Output handle pointer (created on server) | Server creates handle, returns `uint64_t` to client |
| `HOST_IN` | Host pointer to read-only input data | Client sends `size` bytes of data pointed to |
| `HOST_OUT` | Host pointer to output buffer | Server sends `size` bytes back to client |
| `HOST_INOUT` | Host pointer to input/output data | Client sends data, server returns modified data |

**Size rules:**
- `HOST_IN`/`HOST_OUT`: The `size` field is the number of bytes to transfer. For `const float *alpha`, that is 4 bytes. For `const double *alpha`, that is 8 bytes.
- `DEV_PTR` and `HANDLE`: Always 8 bytes (uint64_t handle).
- `SCALAR`: Use the actual size of the type (4 for int32/float, 8 for int64/double).

### Step 4: Run the code generator

```bash
python codegen/generate_stubs.py
```

This regenerates:
- `client/generated_stubs.cpp` -- the client stub for `cublasSsymm_v2`
- `server/generated_dispatch.cpp` -- the server dispatch case for `cublasSsymm_v2`

### Step 5: Rebuild

```bash
cd ~/gpushare/build && make -j$(nproc)
```

### Step 6: Test

Write a small test that calls `cublasSsymm_v2` via the client library. Verify the function:
- Appears in the exported symbols: `nm -D build/libgpushare_client.so.1.0.0 | grep cublasSsymm`
- Returns correct results when called through the network

### Important notes for adding functions

- If the function has variable-size host data (e.g., an array whose length depends on another argument), you need to reference the size argument by name in the size field. For example: `("workspace", "ptr", "HOST_IN", "workspaceSizeInBytes")`.
- The return type `"i32"` covers `cublasStatus_t`, `cudnnStatus_t`, `CUresult`, etc. -- they are all 32-bit integer enums.
- If a function does not exist in the Tier 2 weak stubs (unlikely for standard CUDA libraries), Tier 1 is the only definition. If it does exist in Tier 2, your Tier 1 definition will override it because Tier 2 uses `__attribute__((weak))`.

---

## 5. Protocol Reference

### Header Format

Every message (request and response) starts with a 16-byte header:

```
+--------+--------+--------+--------+
|         magic (0x47505553)         |  4 bytes, little-endian
+--------+--------+--------+--------+
|      length (total msg size)       |  4 bytes, little-endian
+--------+--------+--------+--------+
|           req_id                   |  4 bytes, little-endian
+--------+--------+--------+--------+
|   opcode   |      flags           |  2 + 2 bytes, little-endian
+--------+--------+--------+--------+
```

**Flags:**
- `0x0001` (`GS_FLAG_RESPONSE`): Set on responses
- `0x0002` (`GS_FLAG_ERROR`): Set on error responses
- `0x0004` (`GS_FLAG_CHUNKED`): Message is part of a chunked transfer
- `0x0008` (`GS_FLAG_LAST_CHUNK`): Final chunk in a chunked transfer sequence

### Opcodes

| Opcode | Name | Category |
|--------|------|----------|
| `0x0001` | `GS_OP_INIT` | Session |
| `0x0002` | `GS_OP_CLOSE` | Session |
| `0x0003` | `GS_OP_PING` | Session |
| `0x0010` | `GS_OP_GET_DEVICE_COUNT` | Device |
| `0x0011` | `GS_OP_GET_DEVICE_PROPS` | Device |
| `0x0012` | `GS_OP_SET_DEVICE` | Device |
| `0x0020` | `GS_OP_MALLOC` | Memory |
| `0x0021` | `GS_OP_FREE` | Memory |
| `0x0022` | `GS_OP_MEMCPY_H2D` | Memory |
| `0x0023` | `GS_OP_MEMCPY_D2H` | Memory |
| `0x0024` | `GS_OP_MEMCPY_D2D` | Memory |
| `0x0025` | `GS_OP_MEMSET` | Memory |
| `0x0026` | `GS_OP_MEMCPY_H2D_ASYNC` | Memory (async) |
| `0x0027` | `GS_OP_MEMCPY_D2H_ASYNC` | Memory (async) |
| `0x0030` | `GS_OP_MODULE_LOAD` | Kernel |
| `0x0031` | `GS_OP_MODULE_UNLOAD` | Kernel |
| `0x0032` | `GS_OP_GET_FUNCTION` | Kernel |
| `0x0033` | `GS_OP_LAUNCH_KERNEL` | Kernel |
| `0x0040` | `GS_OP_DEVICE_SYNC` | Sync |
| `0x0041` | `GS_OP_STREAM_CREATE` | Sync |
| `0x0042` | `GS_OP_STREAM_DESTROY` | Sync |
| `0x0043` | `GS_OP_STREAM_SYNC` | Sync |
| `0x0044` | `GS_OP_EVENT_CREATE` | Sync |
| `0x0045` | `GS_OP_EVENT_DESTROY` | Sync |
| `0x0046` | `GS_OP_EVENT_RECORD` | Sync |
| `0x0047` | `GS_OP_EVENT_SYNC` | Sync |
| `0x0048` | `GS_OP_EVENT_ELAPSED` | Sync |
| `0x0050` | `GS_OP_REGISTER_FAT_BIN` | Fat Binary |
| `0x0051` | `GS_OP_REGISTER_FUNCTION` | Fat Binary |
| `0x0052` | `GS_OP_UNREGISTER_FAT_BIN` | Fat Binary |
| `0x0060` | `GS_OP_GET_STATS` | Monitoring |
| `0x0070` | `GS_OP_GET_GPU_STATUS` | Monitoring |
| `0x0080` | `GS_OP_LIB_CALL` | Generic library call |
| `0x0081` | `GS_OP_HANDLE_CREATE` | Handle management |
| `0x0082` | `GS_OP_HANDLE_DESTROY` | Handle management |

### Payload Formats for Key Operations

#### GS_OP_INIT

**Request payload (8 bytes):**

```
+--------+--------+
|  version (u32)  |
+--------+--------+
| client_type(u32)|   0=native, 1=python
+--------+--------+
```

**Response payload (16 bytes):**

```
+--------+--------+--------+--------+
|  version (u32)  | session_id(u32) | max_transfer(u32) | capabilities(u32) |
+--------+--------+--------+--------+
```

The `capabilities` field is a bitmask: `GS_CAP_ASYNC` (0x01) indicates support for async pinned transfers, `GS_CAP_CHUNKED` (0x02) indicates support for chunked transfers.

#### GS_OP_MALLOC

**Request payload (8 bytes):**

```
+--------+--------+
|    size (u64)    |
+--------+---------+
```

**Response payload (12 bytes):**

```
+--------+--------+--------+
| device_ptr (u64)          | cuda_error(i32) |
+--------+--------+---------+
```

#### GS_OP_MEMCPY_H2D (Host to Device)

**Request payload (16 + N bytes):**

```
+--------+--------+
| device_ptr (u64) |
+--------+---------+
|    size (u64)    |
+--------+---------+
|   data[size]...  |
+------------------+
```

**Response payload (4 bytes):**

```
+--------+
| cuda_error(i32) |
+--------+
```

#### GS_OP_MEMCPY_D2H (Device to Host)

**Request payload (16 bytes):**

```
+--------+--------+
| device_ptr (u64) |
+--------+---------+
|    size (u64)    |
+--------+---------+
```

**Response payload (4 + N bytes):**

```
+--------+
| cuda_error(i32) |
+--------+
|   data[size]...  |
+------------------+
```

#### GS_OP_MODULE_LOAD

**Request payload (8 + N bytes):**

```
+--------+---------+
|  data_size (u64) |
+--------+---------+
|  PTX/cubin data  |  (data_size bytes; server null-terminates for PTX)
+------------------+
```

**Response payload (12 bytes):**

```
+--------+--------+--------+
| module_handle (u64)       | cuda_error(i32) |
+--------+--------+---------+
```

#### GS_OP_GET_FUNCTION

**Request payload (264 bytes):**

```
+--------+---------+
| module_handle(u64)|
+--------+---------+
| func_name[256]   |  (null-terminated, padded to 256 bytes)
+------------------+
```

**Response payload (12 bytes):**

```
+--------+--------+--------+
| func_handle (u64)         | cuda_error(i32) |
+--------+--------+---------+
```

#### GS_OP_LAUNCH_KERNEL

**Request payload (variable):**

```
+--------+---------+
| func_handle (u64) |
+--------+---------+
| grid_x  | grid_y  | grid_z  |  3 x u32
+---------+---------+---------+
| block_x | block_y | block_z |  3 x u32
+---------+---------+---------+
| shared_mem (u32)  |
+--------+---------+
| stream_handle(u64)|  (0 = default stream)
+--------+---------+
| num_args (u32)    | args_size (u32) |
+--------+---------+---------+
| serialized args:                    |
|   [size_0 (u32)][data_0 (size_0B)] |
|   [size_1 (u32)][data_1 (size_1B)] |
|   ...                               |
+-------------------------------------+
```

**Response payload (4 bytes):**

```
+--------+
| cuda_error(i32) |
+--------+
```

Device pointer arguments are sent as `uint64_t` handles (8 bytes each).

#### GS_OP_GET_STATS

**Request:** No payload (header only).

**Response payload (variable):**

```
+----------------------------------+
| uptime_secs (u64)               |
| total_ops (u64)                 |
| total_bytes_in (u64)            |
| total_bytes_out (u64)           |
| total_alloc_bytes (u64)         |
| active_clients (u32)            |
| total_connections (u32)         |
| num_clients (u32)               |   -- gs_stats_header_t (44 bytes)
+----------------------------------+
| For each client (num_clients):   |   -- gs_stats_client_t (112 bytes each)
|   session_id (u32)              |
|   addr[64] (char)               |
|   mem_allocated (u64)           |
|   ops_count (u64)               |
|   bytes_in (u64)                |
|   bytes_out (u64)               |
|   connected_secs (u64)          |
+----------------------------------+
```

#### GS_OP_GET_GPU_STATUS

**Request:** No payload.

**Response payload (48 bytes):**

```
+----------------------------------+
| mem_total (u64)                 |
| mem_used (u64)                  |
| mem_free (u64)                  |
| gpu_utilization (u32)  0-100    |
| mem_utilization (u32)  0-100    |
| temperature (u32)      Celsius  |
| power_draw_mw (u32)            |
| power_limit_mw (u32)           |
| fan_speed (u32)        0-100    |
| clock_sm_mhz (u32)             |
| clock_mem_mhz (u32)            |
+----------------------------------+
```

#### GS_OP_LIB_CALL (Generic Library Call)

**Request payload (variable):**

```
+---------+
| lib_id (u8)      |  1=cublas, 2=cudnn, etc.
+---------+
| func_id (u16)    |  function index within library
+---------+
| args_size (u32)  |  total size of serialized args
+---------+
| args_data[args_size] |
+----------------------+
```

**Response payload (variable):**

```
+---------+
| return_code (i32)  |
+---------+
| out_size (u32)     |
+---------+
| out_data[out_size] |
+--------------------+
```

---

## 6. Troubleshooting Guide

### "Connection refused"

**Symptom:** Client reports "Connection refused" when trying to connect.

**Likely cause:** Server is not running, or listening on a different port/interface.

**Fix:**

```bash
# Check if server is running
ss -tlnp | grep 9847

# Start server
./build/gpushare-server --log-level debug

# If using systemd
sudo systemctl start gpushare-server
sudo systemctl status gpushare-server
```

### "Address already in use"

**Symptom:** Starting the server fails with "bind: Address already in use".

**Likely cause:** Another instance is running (possibly the systemd service).

**Fix:**

```bash
# Check what holds the port
ss -tlnp | grep 9847

# Kill the process on that port
fuser -k 9847/tcp

# Or stop the systemd service
sudo systemctl stop gpushare-server
```

### "Module load failed: error 201"

**Symptom:** `cuModuleLoadData` returns CUDA error 201 (`CUDA_ERROR_INVALID_CONTEXT`).

**Likely cause:** CUDA context was not established on the server thread handling the request.

**Fix:** Ensure `cudaSetDevice(g_device)` is called at the beginning of `client_thread()`. If you are modifying the server, any new thread that calls Driver API functions must first establish a CUDA context.

### "Module load failed: error 218"

**Symptom:** `cuModuleLoadData` returns CUDA error 218 (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`).

**Likely cause:** PTX was compiled for the wrong architecture or CUDA version. The server's GPU requires a specific PTX ISA version.

**Fix:**

```bash
# Check server GPU architecture
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# For RTX 5070 (Blackwell): 12.0

# Recompile PTX for the correct architecture
nvcc -ptx -arch=sm_120 kernel.cu
# For RTX 4090: -arch=sm_89
# For RTX 3090: -arch=sm_86

# Verify PTX version in the generated file
head -5 kernel.ptx
# Should show: .version 9.1 (for CUDA 13.x)
```

### "Operation not permitted" (macOS)

**Symptom:** Running gpushare client scripts or library loading fails with permission errors on macOS.

**Likely cause:** macOS quarantine or SIP restrictions.

**Fix:**

```bash
# Remove quarantine from all gpushare files
sudo xattr -dr com.apple.quarantine /usr/local/lib/libgpushare*
sudo xattr -dr com.apple.quarantine /usr/local/bin/gpushare*
sudo xattr -dr com.apple.quarantine /usr/local/share/gpushare/

# If scripts fail under sudo, check the shebang
# BAD:  #!/usr/bin/env bash
# GOOD: #!/bin/bash
```

### "Parse error" (PowerShell)

**Symptom:** PowerShell install script fails to parse.

**Likely cause:** Non-ASCII characters or improperly formatted here-strings in the `.ps1` file.

**Fix:**

```bash
# Check for non-ASCII characters
grep -Pn '[^\x00-\x7F]' scripts/install-client-windows.ps1

# Replace any em-dashes, smart quotes, or box-drawing characters with ASCII equivalents
```

### "ssize_t not found" (MSVC)

**Symptom:** MSVC compilation fails with undefined `ssize_t`.

**Likely cause:** `ssize_t` is a POSIX type that does not exist in MSVC.

**Fix:** Ensure the Windows-specific block in `gpushare_client.cpp` includes:

```cpp
#ifdef _WIN32
  #include <BaseTsd.h>
  typedef SSIZE_T ssize_t;
#endif
```

### Server segfault on connect

**Symptom:** Server crashes when a client connects.

**Likely cause:** CUDA not initialized, or accessing GPU from a thread without a CUDA context.

**Fix:**

```bash
# Run with debug logging
./build/gpushare-server --log-level debug

# Check CUDA availability
nvidia-smi

# Check if CUDA runtime is accessible
python -c "import ctypes; ctypes.CDLL('libcudart.so')"
```

### Library not found / wrong version loaded

**Symptom:** Application loads the real CUDA library instead of gpushare, or fails to find the library entirely.

**Fix:**

```bash
# Check which library is loaded
LD_DEBUG=libs ./your_app 2>&1 | grep cuda

# Verify gpushare client library
ls -la /usr/local/lib/libcudart*
ls -la /usr/local/lib/libgpushare*

# Force library path (debugging)
LD_LIBRARY_PATH=/usr/local/lib ./your_app

# On macOS
DYLD_LIBRARY_PATH=/usr/local/lib ./your_app
```

---

## 7. File Reference

### Root

| File | Description |
|------|-------------|
| `CMakeLists.txt` | Top-level CMake build file. Defines two targets: `gpushare-server` (requires CUDA toolkit) and `gpushare_client` (shared library, no CUDA required). Uses C++17. |
| `README.md` | User-facing documentation and setup instructions. |
| `COMPARISON.md` | Comparison with other GPU sharing solutions. |
| `DEVELOPERS.md` | This file. |

### `include/gpushare/`

| File | Description |
|------|-------------|
| `protocol.h` | Wire protocol definitions: header struct, opcodes, all request/response payload structs, utility functions. Uses `PACKED_STRUCT_BEGIN`/`END` macros for MSVC compatibility. |
| `cuda_defs.h` | Minimal CUDA type definitions (cudaError_t, CUresult, nvmlReturn_t, cudaDeviceProp, CUdevice_attribute, etc.) for building the client without the CUDA toolkit installed. |

### `server/`

| File | Description |
|------|-------------|
| `server.cpp` | Main server implementation. TCP accept loop, per-client threads, CUDA context management, all opcode handlers, config file parsing, LAN/WAN detection, nvidia-smi parsing, resource cleanup. ~1000+ lines. |
| `generated_dispatch.cpp` | Auto-generated by `generate_stubs.py`. Server-side dispatch for Tier 1 library functions (cuBLAS, cuDNN). Deserializes args, calls real functions via `dlsym()`. |
| `generated_all_dispatch.cpp` | Auto-generated by `parse_headers.py`. Server-side dispatch for Tier 2 (all 2432 CUDA functions). Uses raw byte forwarding. |

### `client/`

| File | Description |
|------|-------------|
| `gpushare_client.cpp` | Main client library. Exports CUDA Runtime, Driver, and NVML symbols. Handles connection management, config file reading, request/response serialization. Cross-platform (Linux, macOS, Windows). |
| `generated_stubs.cpp` | Auto-generated by `generate_stubs.py`. Client-side stubs for Tier 1 functions (133 functions with full argument serialization). |
| `generated_all_stubs.cpp` | Auto-generated by `parse_headers.py`. Client-side stubs for Tier 2 functions (2432 weak symbol stubs). |
| `gpu_tray_windows.pyw` | Windows system tray widget. Uses pystray + pillow. Shows GPU stats (temp, utilization, VRAM) via hover tooltip. Communicates with server via raw socket protocol. |

### `codegen/`

| File | Description |
|------|-------------|
| `generate_stubs.py` | Tier 1 code generator. Hand-defined function signatures for cuBLAS and cuDNN with full argument metadata. Generates `client/generated_stubs.cpp` and `server/generated_dispatch.cpp`. |
| `parse_headers.py` | Tier 2 code generator. Parses actual CUDA headers from `/opt/cuda/include`, extracts all function signatures, generates weak-symbol forwarding stubs. Produces `client/generated_all_stubs.cpp` and `server/generated_all_dispatch.cpp`. |

### `python/`

| File | Description |
|------|-------------|
| `setup.py` | pip-installable package configuration. |
| `gpushare/__init__.py` | Python client library. `RemoteGPU` class with `connect()`, `malloc()`, `free()`, `memcpy_h2d()`, `memcpy_d2h()`, `load_module()`, `launch_kernel()`, etc. Auto-reads config files. Numpy integration. |

### `dashboard/`

| File | Description |
|------|-------------|
| `app.py` | Zero-dependency web dashboard. Python HTTP server with embedded HTML/CSS/JS. Server mode shows GPU stats and client list. Client mode shows connection status. Polls gpushare server via binary protocol for stats. |

### `tui/`

| File | Description |
|------|-------------|
| `monitor.py` | Curses-based terminal UI. Real-time GPU stats, client list, service management, inline IP editing. Uses box-drawing characters and sparkline charts. |

### `examples/`

| File | Description |
|------|-------------|
| `test_basic.cpp` | C++ client test: device query, malloc/free, memcpy round-trip (H2D, D2H, D2D), memset, synchronize. |
| `test_python.py` | Python client test (connects to server given as CLI arg). |
| `stress_test.py` | Python stress test with configurable duration. |
| `gpu.py` | GPU detection test. |
| `verify_windows.py` | Windows-specific verification test. |

### `scripts/`

| File | Description |
|------|-------------|
| `gpushare.service` | systemd unit file. Runs `gpushare-server` as a service with auto-restart, nice=-10, and unlimited memlock. |
| `install-server-arch.sh` | Server install script for Arch Linux. Handles CUDA in `/opt/cuda`, systemd service setup. |
| `install-client-linux.sh` | Client install script for Linux. Installs library, creates symlinks for libcudart/libcuda/libnvml, writes config file. |
| `install-client-macos.sh` | Client install script for macOS. Handles quarantine removal, SIP-compatible shebangs, framework patching. |
| `install-client-windows.ps1` | Client install script for Windows. Visual Studio detection with version-to-year mapping. ASCII-only. |
| `uninstall.sh` | Uninstall script for Linux/macOS. |
| `uninstall-windows.ps1` | Uninstall script for Windows. |
| `setup-server.sh` | Interactive server setup (port, device, config). |
| `setup-client.sh` | Interactive client setup (server address, config). |
| `gpushare-patch` | Binary patcher for CUDA framework paths (macOS). |
| `gpushare-patch-frameworks.py` | Python script to patch macOS framework search paths. |
| `nvidia-smi` | Python shim that mimics nvidia-smi output using gpushare's remote GPU data. Allows tools like `nvtop` to work on client machines. |

### `config/`

| File | Description |
|------|-------------|
| `gpushare-server.conf` | Example server configuration: device index, port, max clients, log level, bind address. |
| `gpushare-client.conf` | Example client configuration: server address, dashboard port, auto-connect, log level, timeout, retry count, auth token. |

### `build/` (generated)

| File | Description |
|------|-------------|
| `gpushare-server` | Compiled server binary. |
| `libgpushare_client.so.1.0.0` | Compiled client shared library. |
| `test_basic` | Compiled C++ test binary. |
| `CMakeCache.txt` | CMake cache (do NOT copy between machines -- see Bug 10). |

---

## Appendix: Configuration Reference

### Server Configuration (`gpushare-server.conf`)

```ini
# GPU device index (nvidia-smi index)
device=0

# Port for CUDA RPC traffic
port=9847

# Maximum concurrent client connections
max_clients=16

# Logging level: error, info, debug
log_level=info

# Bind address:
#   (empty)     = listen on ALL interfaces
#   192.168.x.x = LAN only
#   0.0.0.0     = all IPv4 (treated same as empty)
bind=
```

### Client Configuration (`gpushare-client.conf`)

```ini
# Server address (hostname:port)
server=192.168.1.100:9847

# Local dashboard port
dashboard_port=9849

# Auto-connect on library load
auto_connect=true

# Logging: debug, info, warn, error
log_level=info

# Connection timeout in milliseconds
timeout_ms=5000

# Retry attempts on connection failure
retry_count=3

# Authentication token (must match server)
auth_token=
```

### Config file search paths

**Client (C++ library):**
1. `GPUSHARE_SERVER` environment variable
2. `~/.config/gpushare/client.conf`
3. `/etc/gpushare/client.conf` (Linux)
4. `C:\ProgramData\gpushare\client.conf` (Windows)
5. Default: `localhost:9847`

**Client (Python library):**
1. `GPUSHARE_SERVER` environment variable
2. `~/.config/gpushare/client.conf`
3. `/etc/gpushare/client.conf` (Linux)
4. `%LOCALAPPDATA%\gpushare\client.conf` (Windows)
5. `C:\ProgramData\gpushare\client.conf` (Windows)
6. Default: `localhost:9847`

---

## Appendix: Build Matrix

| Platform | Server | Client Library | Python Client | Dashboard | TUI | Tray Widget |
|----------|--------|---------------|---------------|-----------|-----|-------------|
| Linux (Arch) | Yes (requires CUDA toolkit) | Yes | Yes | Yes | Yes | No |
| Linux (Ubuntu/Debian) | Yes (requires CUDA toolkit) | Yes | Yes | Yes | Yes | No |
| macOS | No (no CUDA toolkit) | Yes (cross-compiled or from release) | Yes | Yes | Yes | No |
| Windows | No | Yes (.dll via MSVC) | Yes | Yes | No (no curses) | Yes |

**Build dependencies:**

| Component | Dependencies |
|-----------|-------------|
| Server | CMake 3.18+, C++17 compiler, CUDA Toolkit, pthreads |
| Client (Linux) | CMake 3.18+, C++17 compiler |
| Client (Windows) | CMake 3.18+, MSVC (Visual Studio 2017+), ws2_32.lib |
| Python client | Python 3.7+, numpy |
| Dashboard | Python 3.7+ (stdlib only) |
| TUI | Python 3.7+ (stdlib only, curses) |
| Windows tray | Python 3.7+, pystray, pillow |
