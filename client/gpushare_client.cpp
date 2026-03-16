/*
 * gpushare client library — transparent CUDA runtime replacement
 *
 * Once installed, applications find this library as libcudart and
 * automatically use the remote GPU. No LD_PRELOAD or wrappers needed.
 *
 * Server address is read from (in priority order):
 *   1. GPUSHARE_SERVER environment variable
 *   2. Config file: ~/.config/gpushare/client.conf
 *   3. Config file: /etc/gpushare/client.conf  (Linux)
 *   4. Config file: C:\ProgramData\gpushare\client.conf  (Windows)
 *   5. Default: localhost:9847
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>

/* Platform-specific networking */
#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "ws2_32.lib")
  /* MSVC lacks ssize_t */
  #ifndef ssize_t
    #include <BaseTsd.h>
    typedef SSIZE_T ssize_t;
  #endif
  typedef SOCKET sock_t;
  #define SOCK_INVALID INVALID_SOCKET
  static void sock_init() {
      WSADATA wsa; WSAStartup(MAKEWORD(2,2), &wsa);
  }
  static void sock_close(sock_t s) { closesocket(s); }
  static ssize_t sock_send(sock_t s, const void *b, size_t n) {
      return send(s, (const char*)b, (int)n, 0);
  }
  static ssize_t sock_recv(sock_t s, void *b, size_t n) {
      return recv(s, (char*)b, (int)n, 0);
  }
#else
  #include <unistd.h>
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <netinet/tcp.h>
  #include <arpa/inet.h>
  #include <netdb.h>
  typedef int sock_t;
  #define SOCK_INVALID (-1)
  static void sock_init() {}
  static void sock_close(sock_t s) { close(s); }
  static ssize_t sock_send(sock_t s, const void *b, size_t n) {
      return send(s, b, n, 0);
  }
  static ssize_t sock_recv(sock_t s, void *b, size_t n) {
      return recv(s, b, n, 0);
  }
#endif

#include "gpushare/protocol.h"
#include "gpushare/cuda_defs.h"

/* ── Logging ─────────────────────────────────────────────── */
static int g_verbose = 0;
#define TRACE(fmt, ...) do { if (g_verbose) fprintf(stderr, "[gpushare] " fmt "\n", ##__VA_ARGS__); } while(0)
#define ERR(fmt, ...)   fprintf(stderr, "[gpushare] ERROR: " fmt "\n", ##__VA_ARGS__)

/* ── Connection state ────────────────────────────────────── */
static sock_t       g_sock = SOCK_INVALID;
static std::mutex   g_sock_mtx;
static std::atomic<uint32_t> g_req_id{1};
static std::string  g_server_host = "localhost";
static int          g_server_port = GPUSHARE_DEFAULT_PORT;
static cudaError_t  g_last_error  = cudaSuccess;

/* ── Handle mapping ──────────────────────────────────────── */
/*
 * The server sends back uint64_t handles (actual GPU pointers on server).
 * The client needs to present void* to the application.
 * On 64-bit systems we can just cast directly.
 * On 32-bit systems (rare for GPU work) we'd need a mapping table.
 */
static inline uint64_t ptr_to_handle(const void *p) {
    return (uint64_t)(uintptr_t)p;
}
static inline void *handle_to_ptr(uint64_t h) {
    return (void*)(uintptr_t)h;
}

/* ── Network helpers ─────────────────────────────────────── */

static bool send_all(sock_t fd, const void *buf, size_t len) {
    const uint8_t *p = (const uint8_t*)buf;
    while (len > 0) {
        ssize_t n = sock_send(fd, p, len);
        if (n <= 0) return false;
        p   += n;
        len -= n;
    }
    return true;
}

static bool recv_all(sock_t fd, void *buf, size_t len) {
    uint8_t *p = (uint8_t*)buf;
    while (len > 0) {
        ssize_t n = sock_recv(fd, p, len);
        if (n <= 0) return false;
        p   += n;
        len -= n;
    }
    return true;
}

static bool g_config_loaded = false;

static void parse_server_addr(const std::string &s) {
    auto colon = s.rfind(':');
    if (colon != std::string::npos) {
        g_server_host = s.substr(0, colon);
        g_server_port = atoi(s.substr(colon + 1).c_str());
    } else {
        g_server_host = s;
    }
}

static void load_config_file(const char *path) {
    std::ifstream f(path);
    if (!f.is_open()) return;
    std::string line;
    while (std::getline(f, line)) {
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos || line[start] == '#') continue;
        line = line.substr(start);
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        key.erase(key.find_last_not_of(" \t") + 1);
        val.erase(0, val.find_first_not_of(" \t"));
        val.erase(val.find_last_not_of(" \t") + 1);

        if (key == "server") parse_server_addr(val);
        else if (key == "log_level" && val == "debug") g_verbose = 1;
    }
    TRACE("Loaded config from %s", path);
}

static void load_config() {
    if (g_config_loaded) return;
    g_config_loaded = true;

    /* Try config files (lowest priority first, higher overrides) */
#ifdef _WIN32
    load_config_file("C:\\ProgramData\\gpushare\\client.conf");
    /* User-specific */
    const char *appdata = getenv("LOCALAPPDATA");
    if (appdata) {
        std::string path = std::string(appdata) + "\\gpushare\\client.conf";
        load_config_file(path.c_str());
    }
#else
    load_config_file("/etc/gpushare/client.conf");
    const char *home = getenv("HOME");
    if (home) {
        std::string path = std::string(home) + "/.config/gpushare/client.conf";
        load_config_file(path.c_str());
    }
#endif

    /* Environment variable overrides everything */
    const char *env = getenv("GPUSHARE_SERVER");
    if (env) parse_server_addr(env);

    if (getenv("GPUSHARE_VERBOSE")) g_verbose = 1;
}

static bool ensure_connected() {
    if (g_sock != SOCK_INVALID) return true;

    load_config();

    sock_init();

    TRACE("Connecting to %s:%d", g_server_host.c_str(), g_server_port);

    struct addrinfo hints = {}, *res = nullptr;
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", g_server_port);

    int rc = getaddrinfo(g_server_host.c_str(), port_str, &hints, &res);
    if (rc != 0) {
        ERR("DNS resolution failed for %s: %s", g_server_host.c_str(), gai_strerror(rc));
        return false;
    }

    g_sock = SOCK_INVALID;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        sock_t s = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (s == SOCK_INVALID) continue;
        if (connect(s, ai->ai_addr, ai->ai_addrlen) == 0) {
            g_sock = s;
            break;
        }
        sock_close(s);
    }
    freeaddrinfo(res);

    if (g_sock == SOCK_INVALID) {
        ERR("Cannot connect to gpushare server at %s:%d", g_server_host.c_str(), g_server_port);
        return false;
    }

    /* LAN-optimized TCP — maximize throughput on 1GbE, minimize latency */
    int flag = 1;
    setsockopt(g_sock, IPPROTO_TCP, TCP_NODELAY, (const char*)&flag, sizeof(flag));
#if !defined(_WIN32) && defined(TCP_QUICKACK)
    setsockopt(g_sock, IPPROTO_TCP, TCP_QUICKACK, (const char*)&flag, sizeof(flag));
#endif
    /* 8 MB buffers — saturate 1GbE without stalling on large transfers */
    int bufsize = 8 * 1024 * 1024;
    setsockopt(g_sock, SOL_SOCKET, SO_SNDBUF, (const char*)&bufsize, sizeof(bufsize));
    setsockopt(g_sock, SOL_SOCKET, SO_RCVBUF, (const char*)&bufsize, sizeof(bufsize));
    /* Keep-alive for connection health */
    setsockopt(g_sock, SOL_SOCKET, SO_KEEPALIVE, (const char*)&flag, sizeof(flag));

    /* Send init */
    gs_header_t hdr;
    gs_init_req_t req;
    gs_header_init(&hdr, GS_OP_INIT, g_req_id++, GPUSHARE_HEADER_SIZE + sizeof(req));
    req.version     = GPUSHARE_VERSION;
    req.client_type = 0;
    if (!send_all(g_sock, &hdr, sizeof(hdr)) || !send_all(g_sock, &req, sizeof(req))) {
        ERR("Failed to send init");
        sock_close(g_sock);
        g_sock = SOCK_INVALID;
        return false;
    }

    /* Read init response */
    gs_header_t resp_hdr;
    if (!recv_all(g_sock, &resp_hdr, sizeof(resp_hdr))) {
        ERR("Failed to read init response");
        sock_close(g_sock);
        g_sock = SOCK_INVALID;
        return false;
    }

    gs_init_resp_t resp;
    if (!recv_all(g_sock, &resp, sizeof(resp))) {
        ERR("Failed to read init payload");
        sock_close(g_sock);
        g_sock = SOCK_INVALID;
        return false;
    }

    TRACE("Connected! Session %u, server version %u", resp.session_id, resp.version);
    return true;
}

/* Send request and receive response. Returns payload (caller owns the data). */
/* Non-static so generated_stubs.cpp can use them */
bool rpc_call(uint16_t opcode, const void *req_payload, uint32_t req_len,
                     std::vector<uint8_t> &resp_payload, uint16_t *resp_flags = nullptr) {
    std::lock_guard<std::mutex> lock(g_sock_mtx);
    if (!ensure_connected()) return false;

    uint32_t rid = g_req_id++;
    gs_header_t hdr;
    gs_header_init(&hdr, opcode, rid, GPUSHARE_HEADER_SIZE + req_len);

    if (!send_all(g_sock, &hdr, sizeof(hdr))) goto fail;
    if (req_len > 0 && !send_all(g_sock, req_payload, req_len)) goto fail;

    /* Read response header */
    {
        gs_header_t resp_hdr;
        if (!recv_all(g_sock, &resp_hdr, sizeof(resp_hdr))) goto fail;
        if (!gs_header_validate(&resp_hdr)) goto fail;

        uint32_t pl = resp_hdr.length - GPUSHARE_HEADER_SIZE;
        resp_payload.resize(pl);
        if (pl > 0 && !recv_all(g_sock, resp_payload.data(), pl)) goto fail;

        if (resp_flags) *resp_flags = resp_hdr.flags;
    }
    return true;

fail:
    ERR("RPC call failed (opcode=0x%04x), disconnecting", opcode);
    sock_close(g_sock);
    g_sock = SOCK_INVALID;
    return false;
}

/* Convenience: RPC that returns just a cuda_error */
/* Accessible from generated_stubs.cpp */
int32_t rpc_simple(uint16_t opcode, const void *req_payload, uint32_t req_len) {
    std::vector<uint8_t> resp;
    if (!rpc_call(opcode, req_payload, req_len, resp)) return cudaErrorUnknown;
    if (resp.size() < sizeof(gs_generic_resp_t)) return cudaErrorUnknown;
    auto *r = (const gs_generic_resp_t*)resp.data();
    return (cudaError_t)r->cuda_error;
}

/* ── CUDA Runtime API Implementation ─────────────────────── */

#ifdef __cplusplus
extern "C" {
#endif

/* We use visibility attributes to export symbols */
#ifdef _WIN32
  #define GPUSHARE_EXPORT __declspec(dllexport)
  #define GPUSHARE_DESTRUCTOR  /* handled via DllMain */
#else
  #define GPUSHARE_EXPORT __attribute__((visibility("default")))
  #define GPUSHARE_DESTRUCTOR __attribute__((destructor))
#endif

GPUSHARE_EXPORT cudaError_t cudaGetDeviceCount(int *count) {
    TRACE("cudaGetDeviceCount");
    std::vector<uint8_t> resp;
    if (!rpc_call(GS_OP_GET_DEVICE_COUNT, nullptr, 0, resp)) return cudaErrorUnknown;
    if (resp.size() < sizeof(gs_device_count_resp_t)) return cudaErrorUnknown;
    auto *r = (const gs_device_count_resp_t*)resp.data();
    if (count) *count = r->count;
    return cudaSuccess;
}

GPUSHARE_EXPORT cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
    TRACE("cudaGetDeviceProperties(%d)", device);
    gs_device_props_req_t req;
    req.device = device;
    std::vector<uint8_t> resp;
    if (!rpc_call(GS_OP_GET_DEVICE_PROPS, &req, sizeof(req), resp)) return cudaErrorUnknown;
    if (resp.size() < sizeof(gs_device_props_t)) return cudaErrorUnknown;

    auto *r = (const gs_device_props_t*)resp.data();
    if (prop) {
        memset(prop, 0, sizeof(*prop));
        strncpy(prop->name, r->name, sizeof(prop->name) - 1);
        prop->totalGlobalMem     = r->total_global_mem;
        prop->sharedMemPerBlock  = r->shared_mem_per_block;
        prop->regsPerBlock       = r->regs_per_block;
        prop->warpSize           = r->warp_size;
        prop->maxThreadsPerBlock = r->max_threads_per_block;
        prop->maxThreadsDim[0]   = r->max_threads_dim[0];
        prop->maxThreadsDim[1]   = r->max_threads_dim[1];
        prop->maxThreadsDim[2]   = r->max_threads_dim[2];
        prop->maxGridSize[0]     = r->max_grid_size[0];
        prop->maxGridSize[1]     = r->max_grid_size[1];
        prop->maxGridSize[2]     = r->max_grid_size[2];
        prop->clockRate          = r->clock_rate;
        prop->major              = r->major;
        prop->minor              = r->minor;
        prop->multiProcessorCount= r->multi_processor_count;
        prop->maxThreadsPerMultiProcessor = r->max_threads_per_mp;
        prop->totalConstMem      = r->total_const_mem;
        prop->memoryBusWidth     = r->mem_bus_width;
        prop->l2CacheSize        = r->l2_cache_size;
    }
    return cudaSuccess;
}

GPUSHARE_EXPORT cudaError_t cudaSetDevice(int device) {
    TRACE("cudaSetDevice(%d)", device);
    gs_set_device_req_t req;
    req.device = device;
    return (cudaError_t)rpc_simple(GS_OP_SET_DEVICE, &req, sizeof(req));
}

GPUSHARE_EXPORT cudaError_t cudaMalloc(void **devPtr, size_t size) {
    TRACE("cudaMalloc(%zu bytes)", size);
    gs_malloc_req_t req;
    req.size = size;
    std::vector<uint8_t> resp;
    if (!rpc_call(GS_OP_MALLOC, &req, sizeof(req), resp)) return cudaErrorUnknown;
    if (resp.size() < sizeof(gs_malloc_resp_t)) return cudaErrorUnknown;
    auto *r = (const gs_malloc_resp_t*)resp.data();
    if (devPtr) *devPtr = handle_to_ptr(r->device_ptr);
    g_last_error = (cudaError_t)r->cuda_error;
    return g_last_error;
}

GPUSHARE_EXPORT cudaError_t cudaFree(void *devPtr) {
    TRACE("cudaFree(%p)", devPtr);
    if (!devPtr) return cudaSuccess;
    gs_free_req_t req;
    req.device_ptr = ptr_to_handle(devPtr);
    return (cudaError_t)rpc_simple(GS_OP_FREE, &req, sizeof(req));
}

GPUSHARE_EXPORT cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
    TRACE("cudaMemcpy(%zu bytes, kind=%d)", count, kind);

    if (kind == cudaMemcpyHostToDevice) {
        /* Send host data to server */
        std::vector<uint8_t> payload(sizeof(gs_memcpy_h2d_req_t) + count);
        auto *req = (gs_memcpy_h2d_req_t*)payload.data();
        req->device_ptr = ptr_to_handle(dst);
        req->size = count;
        memcpy(payload.data() + sizeof(gs_memcpy_h2d_req_t), src, count);
        g_last_error = (cudaError_t)rpc_simple(GS_OP_MEMCPY_H2D, payload.data(), payload.size());
        return g_last_error;

    } else if (kind == cudaMemcpyDeviceToHost) {
        /* Request data from server */
        gs_memcpy_d2h_req_t req;
        req.device_ptr = ptr_to_handle(src);
        req.size = count;
        std::vector<uint8_t> resp;
        if (!rpc_call(GS_OP_MEMCPY_D2H, &req, sizeof(req), resp)) return cudaErrorUnknown;
        if (resp.size() < sizeof(int32_t)) return cudaErrorUnknown;
        int32_t err;
        memcpy(&err, resp.data(), sizeof(err));
        if (err != 0) return (cudaError_t)err;
        if (resp.size() >= sizeof(int32_t) + count) {
            memcpy(dst, resp.data() + sizeof(int32_t), count);
        }
        return cudaSuccess;

    } else if (kind == cudaMemcpyDeviceToDevice) {
        gs_memcpy_d2d_req_t req;
        req.dst_ptr = ptr_to_handle(dst);
        req.src_ptr = ptr_to_handle(src);
        req.size = count;
        return (cudaError_t)rpc_simple(GS_OP_MEMCPY_D2D, &req, sizeof(req));

    } else if (kind == cudaMemcpyHostToHost) {
        memcpy(dst, src, count);
        return cudaSuccess;
    }

    return cudaErrorInvalidMemcpyDirection;
}

GPUSHARE_EXPORT cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                                             cudaMemcpyKind kind, cudaStream_t stream) {
    /* For now, implement as synchronous. Async optimization can be added later
       with a command queue that batches operations. */
    (void)stream;
    return cudaMemcpy(dst, src, count, kind);
}

GPUSHARE_EXPORT cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    TRACE("cudaMemset(%p, %d, %zu)", devPtr, value, count);
    gs_memset_req_t req;
    req.device_ptr = ptr_to_handle(devPtr);
    req.value = value;
    req.size  = count;
    return (cudaError_t)rpc_simple(GS_OP_MEMSET, &req, sizeof(req));
}

GPUSHARE_EXPORT cudaError_t cudaDeviceSynchronize(void) {
    TRACE("cudaDeviceSynchronize");
    return (cudaError_t)rpc_simple(GS_OP_DEVICE_SYNC, nullptr, 0);
}

GPUSHARE_EXPORT cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
    TRACE("cudaStreamCreate");
    std::vector<uint8_t> resp;
    if (!rpc_call(GS_OP_STREAM_CREATE, nullptr, 0, resp)) return cudaErrorUnknown;
    if (resp.size() < sizeof(gs_stream_create_resp_t)) return cudaErrorUnknown;
    auto *r = (const gs_stream_create_resp_t*)resp.data();
    if (pStream) *pStream = handle_to_ptr(r->stream_handle);
    return cudaSuccess;
}

GPUSHARE_EXPORT cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    TRACE("cudaStreamDestroy");
    gs_stream_req_t req;
    req.stream_handle = ptr_to_handle(stream);
    return (cudaError_t)rpc_simple(GS_OP_STREAM_DESTROY, &req, sizeof(req));
}

GPUSHARE_EXPORT cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    TRACE("cudaStreamSynchronize");
    gs_stream_req_t req;
    req.stream_handle = ptr_to_handle(stream);
    return (cudaError_t)rpc_simple(GS_OP_STREAM_SYNC, &req, sizeof(req));
}

GPUSHARE_EXPORT cudaError_t cudaEventCreate(cudaEvent_t *event) {
    TRACE("cudaEventCreate");
    std::vector<uint8_t> resp;
    if (!rpc_call(GS_OP_EVENT_CREATE, nullptr, 0, resp)) return cudaErrorUnknown;
    if (resp.size() < sizeof(gs_event_create_resp_t)) return cudaErrorUnknown;
    auto *r = (const gs_event_create_resp_t*)resp.data();
    if (event) *event = handle_to_ptr(r->event_handle);
    return cudaSuccess;
}

GPUSHARE_EXPORT cudaError_t cudaEventDestroy(cudaEvent_t event) {
    TRACE("cudaEventDestroy");
    uint64_t handle = ptr_to_handle(event);
    return (cudaError_t)rpc_simple(GS_OP_EVENT_DESTROY, &handle, sizeof(handle));
}

GPUSHARE_EXPORT cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    TRACE("cudaEventRecord");
    gs_event_record_req_t req;
    req.event_handle  = ptr_to_handle(event);
    req.stream_handle = ptr_to_handle(stream);
    return (cudaError_t)rpc_simple(GS_OP_EVENT_RECORD, &req, sizeof(req));
}

GPUSHARE_EXPORT cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    TRACE("cudaEventSynchronize");
    uint64_t handle = ptr_to_handle(event);
    return (cudaError_t)rpc_simple(GS_OP_EVENT_SYNC, &handle, sizeof(handle));
}

GPUSHARE_EXPORT cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
    TRACE("cudaEventElapsedTime");
    gs_event_elapsed_req_t req;
    req.start_event = ptr_to_handle(start);
    req.end_event   = ptr_to_handle(end);
    std::vector<uint8_t> resp;
    if (!rpc_call(GS_OP_EVENT_ELAPSED, &req, sizeof(req), resp)) return cudaErrorUnknown;
    if (resp.size() < sizeof(gs_event_elapsed_resp_t)) return cudaErrorUnknown;
    auto *r = (const gs_event_elapsed_resp_t*)resp.data();
    if (ms) *ms = r->milliseconds;
    return cudaSuccess;
}

GPUSHARE_EXPORT const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess:                     return "no error";
        case cudaErrorInvalidValue:           return "invalid argument";
        case cudaErrorMemoryAllocation:       return "out of memory";
        case cudaErrorInitializationError:    return "initialization error";
        case cudaErrorInvalidDevice:          return "invalid device ordinal";
        case cudaErrorInvalidMemcpyDirection: return "invalid memcpy direction";
        case cudaErrorNoDevice:               return "no CUDA-capable device";
        case cudaErrorUnknown:                return "unknown error";
        default:                              return "unrecognized error code";
    }
}

GPUSHARE_EXPORT const char* cudaGetErrorName(cudaError_t error) {
    switch (error) {
        case cudaSuccess:                     return "cudaSuccess";
        case cudaErrorInvalidValue:           return "cudaErrorInvalidValue";
        case cudaErrorMemoryAllocation:       return "cudaErrorMemoryAllocation";
        case cudaErrorInitializationError:    return "cudaErrorInitializationError";
        case cudaErrorInvalidDevice:          return "cudaErrorInvalidDevice";
        case cudaErrorNoDevice:               return "cudaErrorNoDevice";
        case cudaErrorUnknown:                return "cudaErrorUnknown";
        default:                              return "cudaErrorUnknown";
    }
}

GPUSHARE_EXPORT cudaError_t cudaGetLastError(void) {
    cudaError_t err = g_last_error;
    g_last_error = cudaSuccess;
    return err;
}

GPUSHARE_EXPORT cudaError_t cudaPeekAtLastError(void) {
    return g_last_error;
}

GPUSHARE_EXPORT cudaError_t cudaDeviceReset(void) {
    TRACE("cudaDeviceReset");
    /* Don't actually reset the server's device - just clean up our connection */
    return cudaSuccess;
}

GPUSHARE_EXPORT cudaError_t cudaDeviceGetAttribute(int *value, int attr, int device) {
    /* Stub - forward as device props query */
    (void)attr;
    (void)device;
    if (value) *value = 0;
    return cudaSuccess;
}

GPUSHARE_EXPORT cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
    (void)flags;
    return cudaEventCreate(event);
}

GPUSHARE_EXPORT cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags) {
    (void)flags;
    return cudaStreamCreate(pStream);
}

GPUSHARE_EXPORT cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority) {
    (void)flags;
    (void)priority;
    return cudaStreamCreate(pStream);
}

/* Module / kernel management via driver API forwarding */
GPUSHARE_EXPORT cudaError_t cudaFuncGetAttributes(void *attr, const void *func) {
    (void)attr;
    (void)func;
    return cudaSuccess;
}

/* ══════════════════════════════════════════════════════════════════════════
 * CUDA Driver API exports — makes this library also work as libcuda.so.1
 *
 * PyTorch, TensorFlow, and all CUDA programs load libcuda.so.1 from the
 * SYSTEM (never bundled by pip). By exporting these symbols, our library
 * provides the "NVIDIA driver" on machines without a real GPU.
 *
 * Any venv, any framework — they all dlopen("libcuda.so.1") and find us.
 * ══════════════════════════════════════════════════════════════════════════ */

/* Cached device properties — fetched once, reused for all attribute queries */
static bool g_props_cached = false;
static gs_device_props_t g_cached_props;

static CUresult cache_device_props() {
    if (g_props_cached) return CUDA_SUCCESS;
    gs_device_props_req_t req;
    req.device = 0;
    std::vector<uint8_t> resp;
    if (!rpc_call(GS_OP_GET_DEVICE_PROPS, &req, sizeof(req), resp)) return CUDA_ERROR_UNKNOWN;
    if (resp.size() < sizeof(gs_device_props_t)) return CUDA_ERROR_UNKNOWN;
    memcpy(&g_cached_props, resp.data(), sizeof(g_cached_props));
    g_props_cached = true;
    return CUDA_SUCCESS;
}

/* ── Initialization ──────────────────────────────────────── */

GPUSHARE_EXPORT CUresult cuInit(unsigned int flags) {
    TRACE("cuInit(%u)", flags);
    (void)flags;
    std::lock_guard<std::mutex> lock(g_sock_mtx);
    if (!ensure_connected()) return CUDA_ERROR_NO_DEVICE;
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuDriverGetVersion(int *version) {
    if (version) *version = 13010;  /* Report CUDA 13.1 */
    return CUDA_SUCCESS;
}

/* ── Device management ───────────────────────────────────── */

GPUSHARE_EXPORT CUresult cuDeviceGetCount(int *count) {
    TRACE("cuDeviceGetCount");
    int c = 0;
    cudaError_t err = cudaGetDeviceCount(&c);
    if (count) *count = c;
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_NO_DEVICE;
}

GPUSHARE_EXPORT CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    TRACE("cuDeviceGet(%d)", ordinal);
    if (device) *device = ordinal;
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    TRACE("cuDeviceGetName(dev=%d)", dev);
    CUresult err = cache_device_props();
    if (err != CUDA_SUCCESS) return err;
    if (name && len > 0) {
        strncpy(name, g_cached_props.name, len - 1);
        name[len - 1] = '\0';
    }
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
    TRACE("cuDeviceTotalMem(dev=%d)", dev);
    CUresult err = cache_device_props();
    if (err != CUDA_SUCCESS) return err;
    if (bytes) *bytes = g_cached_props.total_global_mem;
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
    return cuDeviceTotalMem_v2(bytes, dev);
}

GPUSHARE_EXPORT CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    TRACE("cuDeviceGetAttribute(attr=%d, dev=%d)", (int)attrib, dev);
    CUresult err = cache_device_props();
    if (err != CUDA_SUCCESS) return err;
    if (!pi) return CUDA_ERROR_INVALID_VALUE;

    switch (attrib) {
        case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:          *pi = g_cached_props.max_threads_per_block; break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:                *pi = g_cached_props.max_threads_dim[0]; break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y:                *pi = g_cached_props.max_threads_dim[1]; break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:                *pi = g_cached_props.max_threads_dim[2]; break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:                 *pi = g_cached_props.max_grid_size[0]; break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:                 *pi = g_cached_props.max_grid_size[1]; break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:                 *pi = g_cached_props.max_grid_size[2]; break;
        case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:    *pi = (int)g_cached_props.shared_mem_per_block; break;
        case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY:          *pi = (int)g_cached_props.total_const_mem; break;
        case CU_DEVICE_ATTRIBUTE_WARP_SIZE:                      *pi = g_cached_props.warp_size; break;
        case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK:        *pi = g_cached_props.regs_per_block; break;
        case CU_DEVICE_ATTRIBUTE_CLOCK_RATE:                     *pi = g_cached_props.clock_rate; break;
        case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:           *pi = g_cached_props.multi_processor_count; break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:       *pi = g_cached_props.major; break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:       *pi = g_cached_props.minor; break;
        case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: *pi = g_cached_props.max_threads_per_mp; break;
        case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING:             *pi = 1; break;
        case CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY:                 *pi = 1; break;
        case CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS:      *pi = 1; break;
        case CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED:         *pi = 0; break;
        default: *pi = 0; break;  /* Unknown attrs return 0 — safe default */
    }
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
    TRACE("cuDeviceGetUuid");
    if (uuid) {
        /* Generate a deterministic UUID from device index */
        memset(uuid, 0, sizeof(*uuid));
        uuid->bytes[0] = 'G'; uuid->bytes[1] = 'S';  /* "GS" prefix */
        uuid->bytes[15] = (char)dev;
    }
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev) {
    TRACE("cuDeviceComputeCapability");
    CUresult err = cache_device_props();
    if (err != CUDA_SUCCESS) return err;
    if (major) *major = g_cached_props.major;
    if (minor) *minor = g_cached_props.minor;
    return CUDA_SUCCESS;
}

/* ── Context management ──────────────────────────────────── */
/* We maintain a single context per connection — the server handles multi-tenancy */

static CUcontext g_fake_ctx = (CUcontext)(uintptr_t)0xBADC0DE1;

GPUSHARE_EXPORT CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    TRACE("cuCtxCreate_v2");
    (void)flags; (void)dev;
    if (pctx) *pctx = g_fake_ctx;
    return CUDA_SUCCESS;
}

/* CUDA 13.x v4 signature */
GPUSHARE_EXPORT CUresult cuCtxCreate_v4(CUcontext *pctx, void *params, unsigned int flags, CUdevice dev) {
    (void)params;
    return cuCtxCreate_v2(pctx, flags, dev);
}

GPUSHARE_EXPORT CUresult cuCtxDestroy_v2(CUcontext ctx) {
    TRACE("cuCtxDestroy_v2");
    (void)ctx;
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuCtxSetCurrent(CUcontext ctx) {
    (void)ctx;
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuCtxGetCurrent(CUcontext *pctx) {
    if (pctx) *pctx = g_fake_ctx;
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuCtxGetDevice(CUdevice *device) {
    if (device) *device = 0;
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuCtxSynchronize(void) {
    return (CUresult)cudaDeviceSynchronize();
}

GPUSHARE_EXPORT CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
    (void)ctx;
    if (version) *version = 13010;
    return CUDA_SUCCESS;
}

/* ── Memory (driver API) ─────────────────────────────────── */

GPUSHARE_EXPORT CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    TRACE("cuMemAlloc_v2(%zu)", bytesize);
    void *p = nullptr;
    cudaError_t err = cudaMalloc(&p, bytesize);
    if (dptr) *dptr = (CUdeviceptr)(uintptr_t)p;
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}

GPUSHARE_EXPORT CUresult cuMemFree_v2(CUdeviceptr dptr) {
    TRACE("cuMemFree_v2");
    void *p = (void*)(uintptr_t)dptr;
    cudaError_t err = cudaFree(p);
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

GPUSHARE_EXPORT CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t byteCount) {
    void *d = (void*)(uintptr_t)dst;
    cudaError_t err = cudaMemcpy(d, src, byteCount, cudaMemcpyHostToDevice);
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

GPUSHARE_EXPORT CUresult cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t byteCount) {
    void *s = (void*)(uintptr_t)src;
    cudaError_t err = cudaMemcpy(dst, s, byteCount, cudaMemcpyDeviceToHost);
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

GPUSHARE_EXPORT CUresult cuMemcpyDtoD_v2(CUdeviceptr dst, CUdeviceptr src, size_t byteCount) {
    void *d = (void*)(uintptr_t)dst;
    void *s = (void*)(uintptr_t)src;
    cudaError_t err = cudaMemcpy(d, s, byteCount, cudaMemcpyDeviceToDevice);
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

GPUSHARE_EXPORT CUresult cuMemsetD8_v2(CUdeviceptr dptr, unsigned char uc, size_t N) {
    void *p = (void*)(uintptr_t)dptr;
    cudaError_t err = cudaMemset(p, (int)uc, N);
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

GPUSHARE_EXPORT CUresult cuMemsetD32_v2(CUdeviceptr dptr, unsigned int ui, size_t N) {
    /* cudaMemset only sets bytes; for 32-bit we'd need a kernel.
       Approximate: if value fits in a byte, use memset. */
    void *p = (void*)(uintptr_t)dptr;
    cudaError_t err = cudaMemset(p, (int)(ui & 0xFF), N * 4);
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

GPUSHARE_EXPORT CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
    CUresult err = cache_device_props();
    if (err != CUDA_SUCCESS) return err;
    if (total) *total = g_cached_props.total_global_mem;
    if (free)  *free  = g_cached_props.total_global_mem;  /* approximate */
    return CUDA_SUCCESS;
}

/* Unversioned aliases for older code */
GPUSHARE_EXPORT CUresult cuMemAlloc(CUdeviceptr *d, size_t s) { return cuMemAlloc_v2(d, s); }
GPUSHARE_EXPORT CUresult cuMemFree(CUdeviceptr d) { return cuMemFree_v2(d); }
GPUSHARE_EXPORT CUresult cuMemcpyHtoD(CUdeviceptr d, const void *s, size_t n) { return cuMemcpyHtoD_v2(d, s, n); }
GPUSHARE_EXPORT CUresult cuMemcpyDtoH(void *d, CUdeviceptr s, size_t n) { return cuMemcpyDtoH_v2(d, s, n); }
GPUSHARE_EXPORT CUresult cuMemcpyDtoD(CUdeviceptr d, CUdeviceptr s, size_t n) { return cuMemcpyDtoD_v2(d, s, n); }
GPUSHARE_EXPORT CUresult cuMemsetD8(CUdeviceptr d, unsigned char v, size_t n) { return cuMemsetD8_v2(d, v, n); }
GPUSHARE_EXPORT CUresult cuMemsetD32(CUdeviceptr d, unsigned int v, size_t n) { return cuMemsetD32_v2(d, v, n); }
GPUSHARE_EXPORT CUresult cuMemGetInfo(size_t *f, size_t *t) { return cuMemGetInfo_v2(f, t); }

/* ── Module / Kernel (driver API) ────────────────────────── */
/* These are called by both the runtime and frameworks that use driver API */

GPUSHARE_EXPORT CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    TRACE("cuModuleLoadData");
    std::lock_guard<std::mutex> lock(g_sock_mtx);
    if (!ensure_connected()) return CUDA_ERROR_NOT_INITIALIZED;

    /* Determine data size (PTX is null-terminated) */
    const char *p = (const char*)image;
    size_t sz = 0;
    while (p[sz] != '\0' && sz < 100*1024*1024) sz++;
    sz++;  /* include the null */

    gs_module_load_req_t req;
    req.data_size = sz;
    std::vector<uint8_t> payload(sizeof(req) + sz);
    memcpy(payload.data(), &req, sizeof(req));
    memcpy(payload.data() + sizeof(req), image, sz);

    std::vector<uint8_t> resp;
    uint32_t rid = g_req_id++;
    gs_header_t hdr;
    gs_header_init(&hdr, GS_OP_MODULE_LOAD, rid, GPUSHARE_HEADER_SIZE + payload.size());
    if (!send_all(g_sock, &hdr, sizeof(hdr))) return CUDA_ERROR_UNKNOWN;
    if (!send_all(g_sock, payload.data(), payload.size())) return CUDA_ERROR_UNKNOWN;

    gs_header_t resp_hdr;
    if (!recv_all(g_sock, &resp_hdr, sizeof(resp_hdr))) return CUDA_ERROR_UNKNOWN;
    gs_module_load_resp_t mresp;
    uint32_t pl = resp_hdr.length - GPUSHARE_HEADER_SIZE;
    if (pl >= sizeof(mresp)) {
        if (!recv_all(g_sock, &mresp, sizeof(mresp))) return CUDA_ERROR_UNKNOWN;
        if (pl > sizeof(mresp)) {
            std::vector<uint8_t> drain(pl - sizeof(mresp));
            recv_all(g_sock, drain.data(), drain.size());
        }
    } else {
        return CUDA_ERROR_UNKNOWN;
    }

    if (module) *module = (CUmodule)(uintptr_t)mresp.module_handle;
    return (CUresult)mresp.cuda_error;
}

GPUSHARE_EXPORT CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,
                                             unsigned int numOptions, void *options, void **optionValues) {
    (void)numOptions; (void)options; (void)optionValues;
    return cuModuleLoadData(module, image);
}

GPUSHARE_EXPORT CUresult cuModuleUnload(CUmodule hmod) {
    TRACE("cuModuleUnload");
    uint64_t handle = (uint64_t)(uintptr_t)hmod;
    gs_generic_resp_t resp;
    resp.cuda_error = (int32_t)rpc_simple(GS_OP_MODULE_UNLOAD, &handle, sizeof(handle));
    return (CUresult)resp.cuda_error;
}

GPUSHARE_EXPORT CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    TRACE("cuModuleGetFunction(%s)", name ? name : "null");
    gs_get_function_req_t req;
    req.module_handle = (uint64_t)(uintptr_t)hmod;
    memset(req.func_name, 0, sizeof(req.func_name));
    if (name) strncpy(req.func_name, name, sizeof(req.func_name) - 1);

    std::vector<uint8_t> resp;
    if (!rpc_call(GS_OP_GET_FUNCTION, &req, sizeof(req), resp)) return CUDA_ERROR_UNKNOWN;
    if (resp.size() < sizeof(gs_get_function_resp_t)) return CUDA_ERROR_UNKNOWN;
    auto *r = (const gs_get_function_resp_t*)resp.data();
    if (hfunc) *hfunc = (CUfunction)(uintptr_t)r->func_handle;
    return (CUresult)r->cuda_error;
}

GPUSHARE_EXPORT CUresult cuLaunchKernel(CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra) {
    TRACE("cuLaunchKernel");
    (void)extra;
    /* For now, forward without args. Full arg serialization would require
       knowing the kernel's parameter sizes, which we don't have here.
       This is sufficient for the burn kernel and simple cases. */
    gs_launch_kernel_req_t req;
    memset(&req, 0, sizeof(req));
    req.func_handle = (uint64_t)(uintptr_t)f;
    req.grid_x = gridDimX; req.grid_y = gridDimY; req.grid_z = gridDimZ;
    req.block_x = blockDimX; req.block_y = blockDimY; req.block_z = blockDimZ;
    req.shared_mem = sharedMemBytes;
    req.stream_handle = (uint64_t)(uintptr_t)hStream;
    req.num_args = 0;
    req.args_size = 0;
    return (CUresult)rpc_simple(GS_OP_LAUNCH_KERNEL, &req, sizeof(req));
}

/* ── Stream / Event (driver API) ─────────────────────────── */

GPUSHARE_EXPORT CUresult cuStreamCreate(CUstream *phStream, unsigned int flags) {
    (void)flags;
    cudaStream_t s;
    cudaError_t err = cudaStreamCreate(&s);
    if (phStream) *phStream = (CUstream)s;
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

GPUSHARE_EXPORT CUresult cuStreamDestroy_v2(CUstream hStream) {
    cudaError_t err = cudaStreamDestroy((cudaStream_t)hStream);
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

GPUSHARE_EXPORT CUresult cuStreamSynchronize(CUstream hStream) {
    cudaError_t err = cudaStreamSynchronize((cudaStream_t)hStream);
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

GPUSHARE_EXPORT CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int flags) {
    (void)hStream; (void)hEvent; (void)flags;
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuEventCreate(CUevent *phEvent, unsigned int flags) {
    (void)flags;
    cudaEvent_t e;
    cudaError_t err = cudaEventCreate(&e);
    if (phEvent) *phEvent = (CUevent)e;
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

GPUSHARE_EXPORT CUresult cuEventDestroy_v2(CUevent hEvent) {
    cudaError_t err = cudaEventDestroy((cudaEvent_t)hEvent);
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

GPUSHARE_EXPORT CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    cudaError_t err = cudaEventRecord((cudaEvent_t)hEvent, (cudaStream_t)hStream);
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

GPUSHARE_EXPORT CUresult cuEventSynchronize(CUevent hEvent) {
    cudaError_t err = cudaEventSynchronize((cudaEvent_t)hEvent);
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

GPUSHARE_EXPORT CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
    cudaError_t err = cudaEventElapsedTime(pMilliseconds, (cudaEvent_t)hStart, (cudaEvent_t)hEnd);
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

/* ── Misc driver API stubs ───────────────────────────────── */

GPUSHARE_EXPORT CUresult cuDeviceGetProperties(void *prop, CUdevice dev) {
    (void)prop; (void)dev;
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    (void)dev;
    if (pctx) *pctx = g_fake_ctx;
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
    (void)dev;
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active) {
    (void)dev;
    if (flags) *flags = 0;
    if (active) *active = 1;
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuFuncGetAttribute(int *pi, int attrib, CUfunction func) {
    (void)attrib; (void)func;
    if (pi) *pi = 0;
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuGetErrorName(CUresult error, const char **pStr) {
    if (pStr) *pStr = "CUDA_SUCCESS";
    if (error != CUDA_SUCCESS && pStr) *pStr = "CUDA_ERROR";
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuGetErrorString(CUresult error, const char **pStr) {
    if (pStr) *pStr = (error == CUDA_SUCCESS) ? "no error" : "unknown error";
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuPointerGetAttribute(void *data, int attribute, CUdeviceptr ptr) {
    (void)data; (void)attribute; (void)ptr;
    return CUDA_SUCCESS;
}

/* ══════════════════════════════════════════════════════════════════════════
 * NVML API exports — makes this library also work as libnvidia-ml.so.1
 *
 * GPU applications, nvidia-smi, and ML frameworks use NVML to DETECT GPUs.
 * Without NVML, apps won't "see" any GPU even if CUDA is available.
 * By exporting NVML symbols, GPU applications natively discover the remote GPU.
 * ══════════════════════════════════════════════════════════════════════════ */

static bool g_nvml_initialized = false;
static gs_gpu_status_t g_gpu_status;
static time_t g_status_last_query = 0;

static nvmlReturn_t refresh_gpu_status() {
    /* Cache for 1 second to avoid hammering the server */
    time_t now = time(nullptr);
    if (now - g_status_last_query < 1 && g_status_last_query != 0)
        return NVML_SUCCESS;

    std::vector<uint8_t> resp;
    if (!rpc_call(GS_OP_GET_GPU_STATUS, nullptr, 0, resp)) return NVML_ERROR_UNKNOWN;
    if (resp.size() < sizeof(gs_gpu_status_t)) return NVML_ERROR_UNKNOWN;
    memcpy(&g_gpu_status, resp.data(), sizeof(g_gpu_status));
    g_status_last_query = now;
    return NVML_SUCCESS;
}

/* Fake device handle — just a non-null pointer */
static char g_nvml_dev_sentinel = 0;
static nvmlDevice_t g_nvml_device = (nvmlDevice_t)&g_nvml_dev_sentinel;

GPUSHARE_EXPORT nvmlReturn_t nvmlInit(void) {
    TRACE("nvmlInit");
    std::lock_guard<std::mutex> lock(g_sock_mtx);
    if (!ensure_connected()) return NVML_ERROR_NOT_FOUND;
    g_nvml_initialized = true;
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlInit_v2(void) { return nvmlInit(); }
GPUSHARE_EXPORT nvmlReturn_t nvmlInitWithFlags(unsigned int flags) { (void)flags; return nvmlInit(); }

GPUSHARE_EXPORT nvmlReturn_t nvmlShutdown(void) {
    g_nvml_initialized = false;
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlSystemGetDriverVersion(char *version, unsigned int length) {
    if (version && length > 0) strncpy(version, "560.35.03", length - 1);
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlSystemGetCudaDriverVersion(int *version) {
    if (version) *version = 13010;
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int *version) {
    return nvmlSystemGetCudaDriverVersion(version);
}

GPUSHARE_EXPORT nvmlReturn_t nvmlSystemGetNVMLVersion(char *version, unsigned int length) {
    if (version && length > 0) strncpy(version, "12.560.35.03", length - 1);
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount) {
    TRACE("nvmlDeviceGetCount_v2");
    if (deviceCount) *deviceCount = 1;
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetCount(unsigned int *deviceCount) {
    return nvmlDeviceGetCount_v2(deviceCount);
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device) {
    TRACE("nvmlDeviceGetHandleByIndex(%u)", index);
    if (index > 0) return NVML_ERROR_INVALID_ARGUMENT;
    if (device) *device = g_nvml_device;
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device) {
    return nvmlDeviceGetHandleByIndex_v2(index, device);
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length) {
    TRACE("nvmlDeviceGetName");
    (void)device;
    CUresult err = cache_device_props();
    if (err != CUDA_SUCCESS) return NVML_ERROR_UNKNOWN;
    if (name && length > 0) {
        strncpy(name, g_cached_props.name, length - 1);
        name[length - 1] = '\0';
    }
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory) {
    TRACE("nvmlDeviceGetMemoryInfo");
    (void)device;
    nvmlReturn_t err = refresh_gpu_status();
    if (err != NVML_SUCCESS) return err;
    if (memory) {
        memory->total = g_gpu_status.mem_total;
        memory->used  = g_gpu_status.mem_used;
        memory->free  = g_gpu_status.mem_free;
    }
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t *memory) {
    TRACE("nvmlDeviceGetMemoryInfo_v2");
    (void)device;
    nvmlReturn_t err = refresh_gpu_status();
    if (err != NVML_SUCCESS) return err;
    if (memory) {
        memory->version = 2;
        memory->total = g_gpu_status.mem_total;
        memory->used  = g_gpu_status.mem_used;
        memory->free  = g_gpu_status.mem_free;
        memory->reserved = 0;
    }
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t *utilization) {
    TRACE("nvmlDeviceGetUtilizationRates");
    (void)device;
    nvmlReturn_t err = refresh_gpu_status();
    if (err != NVML_SUCCESS) return err;
    if (utilization) {
        utilization->gpu    = g_gpu_status.gpu_utilization;
        utilization->memory = g_gpu_status.mem_utilization;
    }
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device,
                                                       nvmlTemperatureSensors_t sensor,
                                                       unsigned int *temp) {
    (void)device; (void)sensor;
    nvmlReturn_t err = refresh_gpu_status();
    if (err != NVML_SUCCESS) return err;
    if (temp) *temp = g_gpu_status.temperature;
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int *power) {
    (void)device;
    nvmlReturn_t err = refresh_gpu_status();
    if (err != NVML_SUCCESS) return err;
    if (power) *power = g_gpu_status.power_draw_mw;
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int *limit) {
    (void)device;
    nvmlReturn_t err = refresh_gpu_status();
    if (err != NVML_SUCCESS) return err;
    if (limit) *limit = g_gpu_status.power_limit_mw;
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int *speed) {
    (void)device;
    nvmlReturn_t err = refresh_gpu_status();
    if (err != NVML_SUCCESS) return err;
    if (speed) *speed = g_gpu_status.fan_speed;
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid, unsigned int length) {
    (void)device;
    if (uuid && length > 0) strncpy(uuid, "GPU-gpushare-remote-00000000", length - 1);
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci) {
    (void)device;
    if (pci) {
        memset(pci, 0, sizeof(*pci));
        strncpy(pci->busIdLegacy, "0000:00:00.0", sizeof(pci->busIdLegacy) - 1);
        strncpy(pci->busId, "00000000:00:00.0", sizeof(pci->busId) - 1);
    }
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetPciInfo(nvmlDevice_t d, nvmlPciInfo_t *p) {
    return nvmlDeviceGetPciInfo_v3(d, p);
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device,
                                                                  int *major, int *minor) {
    (void)device;
    return (nvmlReturn_t)cuDeviceComputeCapability(major, minor, 0);
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, int type, unsigned int *clock) {
    (void)device;
    nvmlReturn_t err = refresh_gpu_status();
    if (err != NVML_SUCCESS) return err;
    if (clock) *clock = (type == 0) ? g_gpu_status.clock_sm_mhz : g_gpu_status.clock_mem_mhz;
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index) {
    (void)device;
    if (index) *index = 0;
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device,
    unsigned int *infoCount, void *infos) {
    (void)device; (void)infos;
    if (infoCount) *infoCount = 0;
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v3(nvmlDevice_t device,
    unsigned int *infoCount, void *infos) {
    (void)device; (void)infos;
    if (infoCount) *infoCount = 0;
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT const char* nvmlErrorString(nvmlReturn_t result) {
    switch (result) {
        case NVML_SUCCESS: return "Success";
        case NVML_ERROR_NOT_FOUND: return "Not Found";
        default: return "Unknown Error";
    }
}

/* ── Cleanup ─────────────────────────────────────────────── */

static void GPUSHARE_DESTRUCTOR gpushare_cleanup(void) {
    std::lock_guard<std::mutex> lock(g_sock_mtx);
    if (g_sock != SOCK_INVALID) {
        gs_header_t hdr;
        gs_header_init(&hdr, GS_OP_CLOSE, 0, GPUSHARE_HEADER_SIZE);
        send_all(g_sock, &hdr, sizeof(hdr));
        sock_close(g_sock);
        g_sock = SOCK_INVALID;
        TRACE("Disconnected from server");
    }
}

#ifdef _WIN32
/* On Windows, use DllMain for cleanup instead of __attribute__((destructor)) */
BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved) {
    (void)hinstDLL; (void)lpvReserved;
    if (fdwReason == DLL_PROCESS_DETACH) {
        gpushare_cleanup();
    }
    return TRUE;
}
#endif

#ifdef __cplusplus
}
#endif
