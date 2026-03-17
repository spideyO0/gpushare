# gpushare Performance Roadmap — rCUDA Parity & Beyond

## Status Key

| Symbol | Meaning |
|--------|---------|
| done | Implemented and merged |
| next | Next to implement |
| planned | Designed, not started |
| research | Needs investigation |

---

## Phase 1-4: Foundation (done)

These phases were implemented as the initial rCUDA-parity push.

### Phase 1: Server-Side Pinned Memory Staging (done)

**What:** `PinnedBufferPool` in `ClientSession` — 4 x 4MB buffers via `cudaMallocHost()` with `malloc()` fallback.

**Where:** `server/server.cpp`

**Result:** D2H transfers <= 4MB use pinned buffers instead of `std::vector<uint8_t>`, eliminating the pageable-to-pinned copy penalty that NVIDIA documents as ~2x slower.

**Verification:** Server log shows pinned buffer acquire/release during D2H. Measurable throughput increase on D2H transfers.

---

### Phase 2: Server-Side Async Memcpy (done)

**What:** New opcodes `GS_OP_MEMCPY_H2D_ASYNC` (0x0026) and `GS_OP_MEMCPY_D2H_ASYNC` (0x0027). Server queues GPU DMA via `cudaMemcpyAsync` and returns immediately. Pinned buffer release is deferred via `cudaEvent` tracking.

**Where:** `include/gpushare/protocol.h`, `server/server.cpp`, `client/gpushare_client.cpp`

**Protocol additions:**
- `gs_init_resp_t` gains `uint32_t capabilities` field (backward-compatible)
- `GS_CAP_ASYNC` (0x01), `GS_CAP_CHUNKED` (0x02)
- Async request structs with `stream_handle` field

**Result:** `cudaMemcpyAsync` H2D returns to the application before the GPU finishes the copy. Server tracks `PendingTransfer{pinned_buf, pool_index, event}` and reaps completed transfers before the next acquire.

---

### Phase 3: Chunked Transfer Pipelining (done)

**What:** Transfers > 4MB are split into 4MB chunks. Each chunk is sent as a separate message with `GS_FLAG_CHUNKED`. Server uses `cudaMemcpyAsync` + event tracking per chunk so it can ACK immediately while DMA continues.

**Where:** `include/gpushare/protocol.h`, `server/server.cpp`, `client/gpushare_client.cpp`

**Protocol additions:**
- `GS_FLAG_CHUNKED` (0x0004), `GS_FLAG_LAST_CHUNK` (0x0008)
- `gs_memcpy_h2d_chunk_t`, `gs_memcpy_d2h_chunk_t` structs

**Result:** A 256MB transfer completes in 64 chunks. The server pipelines GPU DMA with network I/O. Client falls back to single-message path if server doesn't advertise `GS_CAP_CHUNKED`.

---

### Phase 4: Client-Side Request Pipelining (done)

**What:** Replaced `g_sock_mtx` (global mutex serializing all client operations) with:
- `g_send_mtx` — protects `send()` calls only
- Dedicated `recv_thread` — reads responses, dispatches by `req_id` via `promise/future`
- `g_pending` map — `unordered_map<uint32_t, shared_ptr<PendingRequest>>`

**Where:** `client/gpushare_client.cpp`, `CMakeLists.txt` (added `pthread` to client link)

**Result:** Multiple application threads can have concurrent in-flight RPCs. Two threads calling `cudaMemcpy` concurrently both complete (previously deadlocked). Windows `DllMain` uses `detach()` instead of `join()` during process exit to avoid loader lock deadlock.

---

## Phase 5: Client-Side Pinned Buffer Staging (done)

**Gap:** rCUDA uses a three-stage pipeline: app memory -> client pinned buffer -> network -> server pinned buffer -> GPU. gpushare currently only has the server half. The client builds a `std::vector` for H2D and sends pageable data directly into the TCP stack.

**What:** Add a `ClientPinnedPool` on the client side. For H2D transfers, copy app data into a client-side pinned buffer before calling `send()`. This allows the OS to DMA directly from the pinned buffer into the NIC, bypassing the kernel's pageable copy.

**Design:**
- Pool: 4 x 4MB buffers allocated via platform-specific pinned memory:
  - Linux: `mlock()` on `mmap(MAP_ANONYMOUS)` pages
  - macOS: `mlock()` on `mmap()` pages
  - Windows: `VirtualAlloc(MEM_COMMIT)` + `VirtualLock()`
  - Note: Can't use `cudaMallocHost` on client (no CUDA toolkit)
- Acquire/release with mutex, `malloc()` fallback
- Used in `cudaMemcpy` H2D path and `cudaMemcpy_h2d_chunked`

**Files:**
- `client/gpushare_client.cpp` — `ClientPinnedPool` class, modify H2D paths

**Verification:** Benchmark H2D throughput before/after on 1GbE. Expect ~10-20% improvement for large transfers due to eliminating the kernel pageable staging copy.

---

## Phase 6: D2H Speculative Prefetch (done)

**Gap:** Current D2H chunked handler does: GPU copy chunk -> wait -> send -> next chunk. No overlap between GPU DMA and network send.

**What:** True double-buffering in `handle_memcpy_d2h_chunked`:
1. Acquire `buf_A` and `buf_B` from pinned pool
2. Start GPU copy of chunk 0 into `buf_A`
3. Wait for chunk 0 to complete
4. Start GPU copy of chunk 1 into `buf_B`
5. While GPU copies chunk 1: send chunk 0 from `buf_A` over network
6. Wait for chunk 1 to complete
7. Start GPU copy of chunk 2 into `buf_A` (recycled)
8. While GPU copies chunk 2: send chunk 1 from `buf_B` over network
9. ...alternate until done

**Files:**
- `server/server.cpp` — rewrite `handle_memcpy_d2h_chunked`

**Verification:** 256MB D2H transfer should be faster. Measure time from request to last byte received. On 1GbE (~110 MB/s), the GPU DMA (~12 GB/s) is negligible so the gain is modest. On 10GbE+ the overlap becomes significant.

---

## Phase 7: Adaptive Buffer Pool Sizing (done)

**Gap:** rCUDA dynamically adapts buffer count and size based on transfer size ranges. gpushare uses fixed 4 x 4MB. A 1KB transfer wastes a 4MB pool slot. A 100MB transfer gets one 4MB pinned buffer and the rest is `malloc`.

**What:** Tiered buffer pool with configurable size ranges:

| Tier | Buffer size | Count | Transfer range |
|------|------------|-------|----------------|
| Small | 64 KB | 8 | 0 - 256 KB |
| Medium | 1 MB | 4 | 256 KB - 4 MB |
| Large | 4 MB | 4 | 4 MB+ |

**Design:**
- `PinnedBufferPool` becomes `TieredBufferPool` with 3 sub-pools
- `acquire(size)` picks the smallest tier that fits
- Server config: `pinned_pool_small=8x64K`, `pinned_pool_medium=4x1M`, `pinned_pool_large=4x4M`
- Client pool (Phase 5) uses the same tiered design

**Files:**
- `server/server.cpp` — `TieredBufferPool` replaces `PinnedBufferPool`
- `client/gpushare_client.cpp` — same tiered design for client pool

**Verification:** Mixed workload benchmark — interleave 1KB control messages with 64MB bulk transfers. Pool utilization should be higher (no 4MB slots wasted on small transfers).

---

## Phase 8: Network Transport Abstraction (planned)

**Gap:** TCP is hardcoded in both client and server. rCUDA has a modular architecture with runtime-loadable network modules. If gpushare ever targets 10GbE+ or InfiniBand, the entire networking layer needs to be swapped.

**What:** Abstract the transport behind a `NetworkModule` interface:

```cpp
class NetworkModule {
public:
    virtual ~NetworkModule() = default;
    virtual bool connect(const char *host, int port) = 0;
    virtual bool send(const void *buf, size_t len) = 0;
    virtual bool recv(void *buf, size_t len) = 0;
    virtual void shutdown_read() = 0;
    virtual void close() = 0;
};
```

**Implementations:**
- `TcpModule` — current TCP code, refactored
- `RdmaModule` (future) — InfiniBand Verbs / RoCE
- `SharedMemModule` (future) — for local same-machine GPU sharing

**Design decisions:**
- Server side: `accept()` returns a `NetworkModule` instance per client
- Client side: `ensure_connected()` creates the module based on config (`transport=tcp|rdma|shm`)
- Modules loaded at runtime via `dlopen()` (like rCUDA) or compiled in

**Files:**
- `include/gpushare/network.h` — `NetworkModule` interface
- `client/tcp_module.cpp`, `server/tcp_module.cpp` — TCP implementations
- `client/gpushare_client.cpp` — replace raw socket calls with module calls
- `server/server.cpp` — replace raw socket calls with module calls

**Verification:** All existing tests pass with `TcpModule`. Throughput unchanged (same code path, just indirected).

---

## Phase 9: InfiniBand / RDMA Transport (planned)

**Gap:** rCUDA achieves 97.7% wire speed on InfiniBand via native Verbs API. gpushare on TCP over InfiniBand gets maybe 30-40% due to kernel overhead.

**Depends on:** Phase 8 (Network Transport Abstraction)

**What:** `RdmaModule` using `libibverbs`:
- Connection setup via RDMA CM (Connection Manager)
- Data transfer via RDMA Write with Immediate or Send/Recv verbs
- Pre-registered memory regions for zero-copy DMA
- GPUDirect RDMA support (future): NIC reads/writes GPU memory directly, bypassing CPU entirely

**Design:**
- Pre-register pinned pool buffers as RDMA memory regions at startup
- Use RDMA Write for large bulk data (H2D/D2H payloads)
- Use Send/Recv for small control messages (headers, ACKs)
- Separate completion queue per client thread

**Files:**
- `client/rdma_module.cpp` — client RDMA transport
- `server/rdma_module.cpp` — server RDMA transport
- `CMakeLists.txt` — optional `libibverbs` dependency

**Verification:** RDMA bandwidth test on InfiniBand FDR (56 Gbps). Target: >90% wire speed for large transfers. Compare with `ib_write_bw` baseline.

---

## Phase 10: Transfer Compression (planned)

**Gap:** rCUDA (2022 paper) adds pipelined compression as an additional pipeline stage. This reduces data volume over the network for compressible data (sparse tensors, zero-heavy buffers).

**What:** Optional compression stage between client and server:
- Client: compress before send (if enabled and data is compressible)
- Server: decompress after receive, before GPU copy
- Algorithm: LZ4 (fast, low overhead) or zstd (better ratio, higher CPU cost)
- Heuristic: sample first 4KB of transfer. If compression ratio < 0.8, send uncompressed.

**Protocol additions:**
- `GS_FLAG_COMPRESSED` (0x0010)
- Compressed message format: `[original_size: u64][compressed_data]`
- `GS_CAP_COMPRESS` capability bit

**Design:**
- Compression runs in the pipeline: app data -> compress -> pinned buffer -> send
- Server: recv -> pinned buffer -> decompress -> GPU copy
- Configurable: `compression=none|lz4|zstd` in client/server config
- Only applied to H2D/D2H data payloads, not control messages

**Files:**
- `include/gpushare/protocol.h` — new flag and capability
- `client/gpushare_client.cpp` — compress before send
- `server/server.cpp` — decompress after receive
- `CMakeLists.txt` — optional lz4/zstd dependency

**Verification:** Transfer 256MB of sparse float32 data (90% zeros). Compressed should transfer faster on 1GbE. Dense random data should detect incompressibility and skip compression.

---

## Phase 11: Multi-Server GPU Pooling (planned)

**Gap:** rCUDA supports multiple clients sharing multiple GPUs across multiple servers. gpushare is single-server, single-GPU.

**What:** Client can connect to multiple servers, each providing one or more GPUs:
- Config: `server=host1:port,host2:port`
- `ServerConnection` class per server (socket, recv thread, pending map)
- Device routing: server 0 owns devices 0..N0-1, server 1 owns N0..N0+N1-1
- `cudaSetDevice` routes to the correct `ServerConnection`

**Design:**
- On `cuInit`/`cudaGetDeviceCount`: connect to all servers, sum device counts
- `g_connections[]` array indexed by server
- `g_device_to_server[]` map: device index -> (server_index, local_device)
- Each `ServerConnection` has its own `send_mtx`, recv thread, pending map
- Stream handles are scoped per-server (tagged with server index)

**Files:**
- `client/gpushare_client.cpp` — `ServerConnection` class, device routing
- Config files — multi-server syntax

**Verification:** Two servers with one GPU each. `cudaGetDeviceCount` returns 2. `cudaSetDevice(0)` routes to server 1, `cudaSetDevice(1)` routes to server 2. Run training on device 0 and inference on device 1 simultaneously.

---

## Phase 12: GPU Context Scheduling (research)

**Gap:** rCUDA has server-side GPU scheduling for multi-tenant access. gpushare uses one thread per client with no coordination — clients compete for GPU time.

**What:** Server-side scheduler that manages GPU context switching between clients:
- Fair-share scheduling: each client gets proportional GPU time
- Priority levels: configurable per-client (e.g., training > inference)
- Memory limits: prevent one client from OOM'ing others
- Preemption: interrupt long-running kernels if higher-priority client needs GPU

**Research needed:**
- CUDA MPS (Multi-Process Service) integration — can we use MPS instead of rolling our own scheduler?
- CUDA context switching overhead — how expensive is `cudaSetDevice` + context switch?
- Memory isolation — can we enforce per-client VRAM limits without kernel module?

**Files:** TBD after research

---

## Phase 13: GPUDirect RDMA (research)

**Gap:** rCUDA supports GPUDirect RDMA — the NIC reads/writes GPU memory directly without involving the CPU. This eliminates the server-side pinned buffer copy entirely.

**Depends on:** Phase 9 (InfiniBand/RDMA Transport)

**What:** For RDMA transfers, register GPU memory directly with the NIC:
- `ibv_reg_mr()` on CUDA device pointer (requires `nvidia-peermem` kernel module)
- NIC DMAs data directly to/from GPU memory
- Eliminates: network recv -> pinned buffer -> `cudaMemcpy` to GPU (2 copies become 0)

**Research needed:**
- Requires NVIDIA driver + MOFED driver + `nvidia-peermem` module
- Only works with ConnectX-4+ NICs and Tesla/datacenter GPUs
- May not work with consumer GPUs (RTX 5070) — needs testing
- Latency vs throughput tradeoff for small transfers

**Files:** TBD after research

---

## Implementation Order

```
done     Phase 1: Server Pinned Buffers
done     Phase 2: Server Async Memcpy
done     Phase 3: Chunked Transfer Pipelining
done     Phase 4: Client Request Pipelining
           │
done     Phase 5: Client-Side Pinned Staging ──────── completes rCUDA three-stage pipeline
done     Phase 6: D2H Speculative Prefetch ────────── GPU prefetches next chunk while sending current
           │
done     Phase 7: Adaptive Buffer Sizing ──────────── rCUDA's configurable tiered pools
planned  Phase 8: Network Transport Abstraction ───── prerequisite for RDMA
planned  Phase 9: InfiniBand / RDMA Transport ─────── rCUDA's key advantage (97.7% wire speed)
planned  Phase 10: Transfer Compression ────────────── rCUDA 2022 pipelined compression
planned  Phase 11: Multi-Server GPU Pooling ────────── rCUDA's multi-node GPU sharing
           │
research Phase 12: GPU Context Scheduling ──────────── multi-tenant fairness
research Phase 13: GPUDirect RDMA ──────────────────── zero-copy NIC-to-GPU transfers
```

---

## Comparison Matrix: gpushare vs rCUDA

| Feature | rCUDA | gpushare (current) | gpushare (after roadmap) |
|---------|-------|--------------------|--------------------------|
| Client pinned staging | Yes (three-stage) | Yes (Phase 5) | Yes |
| Server pinned staging | Yes | Yes (Phase 1) | Yes |
| Async memcpy | Yes | Yes (Phase 2) | Yes |
| Chunked pipelining | Yes | Yes (Phase 3) | Yes |
| Client request pipelining | Yes | Yes (Phase 4) | Yes |
| D2H prefetch/overlap | Yes | Yes (Phase 6) | Yes |
| Adaptive buffer sizing | Yes (per size range) | Yes (Phase 7) | Yes |
| InfiniBand Verbs | Yes (97.7% wire) | No | Phase 9 |
| GPUDirect RDMA | Yes | No | Phase 13 |
| Transfer compression | Yes (2022) | No | Phase 10 |
| Multi-server pooling | Yes | No | Phase 11 |
| GPU scheduling | Yes | No | Phase 12 |
| Cross-platform client | No (Linux only) | Yes (Linux/macOS/Win) | Yes |
| Driver API + NVML | No | Yes | Yes |
| cuBLAS/cuDNN RPC | No | Yes (133 functions) | Yes |
| No LD_PRELOAD | No | Yes | Yes |
| Monitoring ecosystem | No | Yes | Yes |
| Capability negotiation | No | Yes | Yes |
| Config auto-discovery | No | Yes | Yes |

**Current rCUDA parity: ~60%. After full roadmap: ~95%+ (with cross-platform and API advantages rCUDA doesn't have).**

---

## Files Modified Per Phase

| File | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|------|---|---|---|---|---|---|---|---|---|----|----|
| `include/gpushare/protocol.h` | | X | X | | | | | | | X | |
| `include/gpushare/network.h` | | | | | | | | X | X | | |
| `server/server.cpp` | X | X | X | | | X | X | X | | X | |
| `client/gpushare_client.cpp` | | X | X | X | X | | X | X | | X | X |
| `CMakeLists.txt` | | | | X | | | | X | X | X | |
| `client/tcp_module.cpp` | | | | | | | | X | | | |
| `server/tcp_module.cpp` | | | | | | | | X | | | |
| `client/rdma_module.cpp` | | | | | | | | | X | | |
| `server/rdma_module.cpp` | | | | | | | | | X | | |

---

## Backward Compatibility Rules

Every phase must maintain:

1. **Old client + new server:** Works. Server still handles legacy non-chunked, non-async messages. New capability bits in init response are ignored by old clients (they only read the first 12 bytes).

2. **New client + old server:** Works. Client checks `g_server_caps` before using new features. Falls back to existing synchronous single-message path if server doesn't report capabilities.

3. **No breaking protocol changes.** New opcodes are additive. New flags are additive. New struct fields are appended (never inserted).

---

## Verification After Each Phase

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# Symbol count must remain >= 2600
nm -D build/libgpushare_client.so.1.0.0 | grep -E " [TW] " | wc -l

# Existing tests must pass
./build/gpushare-server &
GPUSHARE_SERVER=localhost:9847 ./build/test_basic
PYTHONPATH=python python examples/stress_test.py localhost --duration 10
kill %1
```
