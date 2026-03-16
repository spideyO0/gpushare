# gpushare — GPU over IP

Share your NVIDIA GPU with any machine on your network. Applications on client machines see the remote GPU as if it were local — no code changes, no wrappers, no VMs.

```
┌──────────────┐     1 Gbps LAN      ┌───────────────┐
│  Client (Mac) │ ◄─────────────────► │  Server (Arch) │
│  No GPU       │    TCP/9847         │  RTX 5070 12GB │
│  python train.py ──► remote GPU     │                │
└──────────────┘                      └───────────────┘
```

## What it does

One library (`libgpushare_client`) installs as every CUDA library on the client. Applications load it thinking it's the real NVIDIA stack, and all GPU operations are transparently forwarded to the server over TCP.

- **2,565 exported API functions** across 12 CUDA libraries
- **Zero code changes** — any CUDA application works as-is
- **Cross-platform** — Linux, macOS, Windows clients
- **No LD_PRELOAD** — system-wide interception via library symlinks
- **Works in Python venvs** — `libcuda.so.1` is always loaded from the system

## Quick start

### Server (machine with the GPU)

```bash
# Arch Linux
git clone <repo> && cd gpushare
sudo bash scripts/install-server-arch.sh
```

The server starts automatically via systemd. Check it with:
```bash
systemctl status gpushare-server
```

### Client (machine that needs GPU access)

**Linux:**
```bash
sudo bash scripts/install-client-linux.sh
```

**macOS:**
```bash
bash scripts/install-client-macos.sh
```

**Windows (PowerShell as Administrator):**
```powershell
.\scripts\install-client-windows.ps1
```

### Verify

```bash
nvidia-smi                    # shows remote GPU stats
python3 examples/gpu.py       # full detection test
python3 examples/stress_test.py   # bandwidth + compute stress test
```

## How it works

### Architecture

```
Client Machine                          Server Machine (Arch + RTX 5070)
┌────────────────────────┐              ┌────────────────────────────┐
│ Application (PyTorch)  │              │ gpushare-server            │
│         │               │              │       │                    │
│  loads libcuda.so.1    │   TCP/9847   │  executes real CUDA calls  │
│  loads libcudart.so.13 │◄────────────►│  on physical GPU           │
│  loads libcublas.so.12 │   binary     │       │                    │
│  loads libnvidia-ml.so │   protocol   │  libcudart ─► RTX 5070    │
│         │               │              │  libcublas ─► RTX 5070    │
│  libgpushare_client.so │              │  libcudnn  ─► RTX 5070    │
│  (one library = all)   │              │                            │
└────────────────────────┘              └────────────────────────────┘
```

### Interception layers

A single shared library provides all three NVIDIA API layers:

| Symlink on client | API | What uses it |
|---|---|---|
| `libcudart.so` | CUDA Runtime | C/C++ CUDA programs |
| `libcuda.so.1` | CUDA Driver | PyTorch, TensorFlow (from any venv) |
| `libnvidia-ml.so.1` | NVML | nvidia-smi, GPU monitoring tools |
| `libcublas.so.12` | cuBLAS | Matrix operations (PyTorch linear layers) |
| `libcudnn.so.9` | cuDNN | Convolutions, batch norm, activations |
| `libcufft.so.11` | cuFFT | FFT operations |
| `libcusparse.so.12` | cuSPARSE | Sparse matrix operations |
| `libcusolver.so.11` | cuSOLVER | Linear algebra solvers |
| `libcurand.so.10` | cuRAND | Random number generation |
| `libnvrtc.so.12` | NVRTC | Runtime CUDA compilation |
| `libnvjpeg.so.12` | nvJPEG | GPU-accelerated JPEG decode |

### Network

- **LAN clients** (same subnet): auto-detected, 8 MB TCP buffers, `TCP_QUICKACK` — full 1 Gbps throughput. Traffic stays on the switch, never touches the internet.
- **WAN/WiFi clients** (different subnet): auto-detected, 2 MB buffers — conservative settings for slower links.
- Both are served simultaneously from the same server.

### Binary protocol

Minimal 16-byte header per message:
```
[4B magic "GPUS"] [4B length] [4B request_id] [2B opcode] [2B flags]
```

133 functions have full argument-aware RPC serialization (cuBLAS GEMM, cuDNN convolutions, etc.). The remaining 2,432 functions are exported as symbols for link compatibility.

## API coverage

| Library | Functions | Full RPC | Notes |
|---|---|---|---|
| CUDA Runtime | 334 | 61 | cudaMalloc, cudaMemcpy, streams, events, graphs |
| CUDA Driver | 521 | 56 | cuInit, cuMemAlloc, cuLaunchKernel, contexts |
| NVML | 29 | 29 | Full GPU monitoring (temp, power, VRAM, util) |
| cuBLAS | 598 | 33 | SGEMM, DGEMM, GemmEx, strided batched |
| cublasLt | 58 | — | Lightweight BLAS |
| cuDNN | 66 | 66 | Convolution, batch norm, pooling, softmax, activation |
| cuFFT | 35 | — | FFT operations |
| cuSPARSE | 467 | — | Sparse matrix operations |
| cuSOLVER | 377 | — | Linear algebra solvers |
| cuRAND | 29 | — | Random number generation |
| NVRTC | 25 | — | Runtime compilation |
| nvJPEG | 81 | — | JPEG decode |
| **Total** | **2,565** | **133** | |

"Full RPC" = proper argument serialization, data forwarded to server, results returned.
"—" = exported symbol for link compatibility, returns success. Add full RPC support by adding the function to `codegen/generate_stubs.py`.

## Monitoring & management

### Web dashboard
```bash
python3 dashboard/app.py                    # server mode (port 9848)
python3 dashboard/app.py --client           # client mode (port 9849)
```
Real-time dark-theme dashboard with GPU stats, client table, bandwidth/memory charts. Client mode includes server IP change and service start/stop controls.

### TUI monitor (macOS/Linux)
```bash
gpushare-monitor
```
Curses-based terminal UI with keyboard controls:
- `[e]` Edit server IP
- `[s]` Start service
- `[x]` Stop service
- `[a]` Toggle auto-stop (stops service if server unreachable for 30s)
- `[r]` Reconnect
- `[q]` Quit

### Windows system tray widget
```
pythonw client/gpu_tray_windows.pyw
```
System tray icon showing GPU temperature and utilization. Right-click menu for status, nvidia-smi, dashboard, server change, and service control.

### nvidia-smi
```bash
nvidia-smi          # full output (matches real nvidia-smi format)
nvidia-smi -l 2     # live refresh every 2 seconds
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu --format=csv
```

## Configuration

### Server config (`/etc/gpushare/server.conf`)
```ini
port=9847
device=0
max_clients=16
log_level=info
# Set to your LAN IP to keep traffic off the internet:
bind=
```

### Client config (`~/.config/gpushare/client.conf` or `/etc/gpushare/client.conf`)
```ini
server=192.168.1.100:9847
```

The client library reads this config automatically. No environment variables needed. Server address can also be changed at runtime via the dashboard, TUI, or tray widget.

## Building from source

### Requirements

**Server:** CUDA toolkit, NVIDIA driver, cmake, gcc/g++
**Client:** cmake, gcc/g++ (or Visual Studio on Windows). No CUDA needed.

### Build

```bash
# Server + client
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Client only (no CUDA required)
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SERVER=OFF
make -j$(nproc)
```

### Code generation

The cuBLAS/cuDNN/etc stubs are auto-generated from CUDA headers:

```bash
python3 codegen/generate_stubs.py    # 133 full-RPC stubs
python3 codegen/parse_headers.py     # 2,432 weak symbol stubs
```

To add full RPC support for a new function, add it to `codegen/generate_stubs.py` and rebuild.

## Uninstalling

```bash
# Linux/macOS
bash scripts/uninstall.sh
bash scripts/uninstall.sh --purge    # also remove config

# Windows (PowerShell as Administrator)
.\scripts\uninstall-windows.ps1
.\scripts\uninstall-windows.ps1 -Purge
```

## Project structure

```
gpushare/
├── server/server.cpp                 # GPU server daemon
├── client/gpushare_client.cpp        # Client library (runtime + driver + NVML)
├── client/generated_stubs.cpp        # Auto-generated cuBLAS/cuDNN RPC stubs
├── client/generated_all_stubs.cpp    # Auto-generated weak stubs (all libraries)
├── include/gpushare/
│   ├── protocol.h                    # Wire protocol definitions
│   └── cuda_defs.h                   # CUDA/NVML types (no CUDA toolkit needed)
├── codegen/
│   ├── generate_stubs.py             # Full-RPC stub generator
│   └── parse_headers.py             # Header parser for all CUDA libs
├── python/gpushare/                  # Python client library
├── dashboard/app.py                  # Web dashboard (zero dependencies)
├── tui/monitor.py                    # Terminal UI monitor
├── client/gpu_tray_windows.pyw       # Windows system tray widget
├── examples/
│   ├── gpu.py                        # GPU detection script
│   ├── stress_test.py                # Connection + bandwidth + compute test
│   ├── verify_windows.py             # Windows verification script
│   └── test_basic.cpp                # C++ test
├── scripts/
│   ├── install-server-arch.sh        # Server installer (Arch Linux)
│   ├── install-client-linux.sh       # Client installer (all distros)
│   ├── install-client-macos.sh       # Client installer (macOS)
│   ├── install-client-windows.ps1    # Client installer (Windows)
│   ├── uninstall.sh                  # Uninstaller (Linux/macOS)
│   ├── uninstall-windows.ps1         # Uninstaller (Windows)
│   └── nvidia-smi                    # nvidia-smi shim (Python)
└── config/
    ├── gpushare-server.conf          # Server config template
    └── gpushare-client.conf          # Client config template
```

## Limitations

- **CUDA compute only** — graphics APIs (DirectX, Vulkan, OpenGL) are not supported. Blender rendering, games, and Photoshop GPU acceleration require a local GPU or a streaming solution (Sunshine/Moonlight).
- **1 Gbps bandwidth limit** — large data transfers (>100 MB) are bottlenecked by the network. For training, keep data on the server side.
- **PyTorch on macOS** — Apple's PyTorch builds are CPU-only (CUDA compiled out). Use the gpushare Python API directly for ML on Mac.
- **Windows Task Manager** — cannot show remote GPUs (requires kernel-mode WDDM driver). Use nvidia-smi or the tray widget instead.

## License

MIT
