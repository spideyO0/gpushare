# GEMINI.md - gpushare (GPU over IP)

## Project Overview
`gpushare` is a high-performance GPU virtualization solution that enables sharing NVIDIA GPUs over a network. It allows applications on client machines (Linux, macOS, Windows) to access a remote GPU as if it were local, requiring zero code changes or wrappers.

### Core Architecture
- **Server (`gpushare-server`):** A C++ daemon that runs on the machine with the physical GPU. it executes real CUDA operations on behalf of clients.
- **Client (`libgpushare_client`):** A transparent shared library that replaces the standard NVIDIA stack (`libcuda`, `libcudart`, `libnvidia-ml`, `libcublas`, `libcudnn`, etc.). It intercepts API calls and forwards them to the server via a custom binary protocol.
- **Codegen:** Python scripts in `codegen/` auto-generate C++ stubs for over 2,600 exported API functions across 12 CUDA libraries.
- **Transport:** Supports high-speed TCP with optimizations (tiered pinned buffers, chunked pipelining) and RDMA/InfiniBand for ultra-low latency.
- **Python Integration:** Includes a PyTorch startup hook and framework patcher to ensure seamless remote GPU detection in ML environments.

## Technologies
- **Languages:** C++17 (Core), Python 3.8+ (Tooling/Hooks), CUDA (Server-side).
- **Build System:** CMake (3.18+).
- **Dependencies:** 
  - **Server:** CUDA Toolkit, NVIDIA Driver.
  - **Client:** None (optional `rdma-core`, `lz4`, `zstd` for optimizations).
- **Platform Support:** 
  - **Server:** Linux (Arch preferred).
  - **Client:** Linux (all distros), macOS, Windows.

## Key Directories
- `server/`: Main server implementation and generated dispatch logic.
- `client/`: Client library implementation and generated RPC stubs.
- `include/gpushare/`: Protocol definitions, transport abstractions, and CUDA types.
- `codegen/`: Python scripts for parsing CUDA headers and generating C++ stubs.
- `python/`: Python client package and PyTorch integration hooks.
- `scripts/`: Cross-platform installation and uninstallation scripts.
- `dashboard/`: Web-based monitoring dashboard (Python/Flask).
- `tui/`: Curses-based terminal monitor.

## Building and Running

### Build All (Server + Client)
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Build Client Only (No CUDA required)
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SERVER=OFF
make -j$(nproc)
```

### Running the Server
```bash
sudo ./build/gpushare-server
```

### Installing the Client (Linux)
```bash
sudo bash scripts/install-client-linux.sh
```

## Development Conventions
- **Code Generation:** Do NOT manually edit `client/generated_stubs.cpp` or `server/generated_dispatch.cpp`. Instead, update `codegen/generate_stubs.py` and re-run the script.
- **Protocol:** Any changes to the wire format must be updated in `include/gpushare/protocol.h` and coordinated across both client and server.
- **Dual-GPU Support:** The client library supports "local GPU passthrough." It looks for real CUDA libraries in `/usr/local/lib/gpushare/real/` to expose local devices alongside remote ones.
- **Logging:** Use the `LOG_INFO`, `LOG_WARN`, `LOG_ERR`, and `LOG_DBG` macros defined in the source files.

## Windows Dual-GPU Support
If a local GPU is present, `gpushare` handles blending by default:
- **Default Indexing**: Device 0 = Local, Device 1+ = Remote.
- **Remote Priority**: Set `GPUSHARE_REMOTE_FIRST=1` to make Remote = Device 0, Local = Device 1+.
- **Configuration**: Use `remote_first=true` in `client.conf`.

## Testing and Verification
- `examples/gpu.py`: Basic GPU detection test.
- `examples/stress_test.py`: Bandwidth and compute stress test.
- `examples/test_basic.cpp`: C++ API verification.
- `nvidia-smi`: A provided shim script that matches the real `nvidia-smi` output format for remote GPUs.
