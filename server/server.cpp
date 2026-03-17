/*
 * gpushare server — GPU sharing daemon
 * Runs on the machine with the physical GPU (Arch Linux + RTX 5070).
 * Accepts client connections and executes CUDA operations on their behalf.
 *
 * Build: cmake --build build (from project root)
 * Run:   ./build/gpushare-server [--port 9847] [--device 0] [--config /etc/gpushare/server.conf]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <csignal>
#include <ctime>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <ifaddrs.h>
#include <pthread.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <mutex>
#include <atomic>
#include <memory>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include "gpushare/protocol.h"

/* ── Logging ─────────────────────────────────────────────── */
static int g_log_level = 1; /* 0=error, 1=info, 2=debug */
#define LOG_INFO(fmt, ...)  do { if (g_log_level >= 1) fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__); fflush(stdout); } while(0)
#define LOG_WARN(fmt, ...)  do { if (g_log_level >= 1) fprintf(stdout, "[WARN]  " fmt "\n", ##__VA_ARGS__); fflush(stdout); } while(0)
#define LOG_ERR(fmt, ...)   do { fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__); fflush(stderr); } while(0)
#define LOG_DBG(fmt, ...)   do { if (g_log_level >= 2) fprintf(stdout, "[DEBUG] " fmt "\n", ##__VA_ARGS__); fflush(stdout); } while(0)

/* ── Config ──────────────────────────────────────────────── */
static int          g_port      = GPUSHARE_DEFAULT_PORT;
static int          g_device    = 0;
static int          g_max_clients = 16;
static std::string  g_config_path;
static std::string  g_bind_addr;   /* empty = all interfaces, or "192.168.x.x" for LAN-only */
static std::atomic<bool> g_running{true};

/* ── Global statistics ───────────────────────────────────── */
static time_t g_start_time = 0;
static std::atomic<uint64_t> g_total_ops{0};
static std::atomic<uint64_t> g_total_bytes_in{0};
static std::atomic<uint64_t> g_total_bytes_out{0};
static std::atomic<uint32_t> g_total_connections{0};

/* ── LAN/WAN detection ───────────────────────────────────── */
enum NetType { NET_LAN, NET_WAN };

/* Get all local IPv4 addresses + subnet masks */
struct LocalNet { uint32_t addr; uint32_t mask; char ifname[32]; };
static std::vector<LocalNet> g_local_nets;

static void detect_local_networks() {
    struct ifaddrs *ifap = nullptr;
    if (getifaddrs(&ifap) != 0) return;
    for (struct ifaddrs *ifa = ifap; ifa; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET) continue;
        if (!ifa->ifa_netmask) continue;
        /* Skip loopback */
        if (ifa->ifa_flags & IFF_LOOPBACK) continue;

        LocalNet ln;
        ln.addr = ((struct sockaddr_in*)ifa->ifa_addr)->sin_addr.s_addr;
        ln.mask = ((struct sockaddr_in*)ifa->ifa_netmask)->sin_addr.s_addr;
        strncpy(ln.ifname, ifa->ifa_name, sizeof(ln.ifname) - 1);
        ln.ifname[sizeof(ln.ifname) - 1] = '\0';
        g_local_nets.push_back(ln);

        char addr_s[INET_ADDRSTRLEN], mask_s[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &ln.addr, addr_s, sizeof(addr_s));
        inet_ntop(AF_INET, &ln.mask, mask_s, sizeof(mask_s));
        LOG_DBG("Local network: %s/%s on %s", addr_s, mask_s, ln.ifname);
    }
    freeifaddrs(ifap);
}

/* Check if a client IPv4 address is on the same LAN subnet */
static NetType classify_client(const struct sockaddr_storage *client_addr) {
    uint32_t client_ip = 0;

    if (client_addr->ss_family == AF_INET) {
        client_ip = ((const struct sockaddr_in*)client_addr)->sin_addr.s_addr;
    } else if (client_addr->ss_family == AF_INET6) {
        const struct sockaddr_in6 *a6 = (const struct sockaddr_in6*)client_addr;
        /* Check for IPv4-mapped IPv6 (::ffff:x.x.x.x) */
        const uint8_t *b = a6->sin6_addr.s6_addr;
        if (b[0]==0 && b[1]==0 && b[2]==0 && b[3]==0 &&
            b[4]==0 && b[5]==0 && b[6]==0 && b[7]==0 &&
            b[8]==0 && b[9]==0 && b[10]==0xff && b[11]==0xff) {
            memcpy(&client_ip, &b[12], 4);
        } else {
            /* Pure IPv6 — treat as WAN for now */
            return NET_WAN;
        }
    }

    if (client_ip == 0) return NET_WAN;

    /* Compare against each local interface's subnet */
    for (const auto &ln : g_local_nets) {
        if ((client_ip & ln.mask) == (ln.addr & ln.mask)) {
            return NET_LAN;
        }
    }
    return NET_WAN;
}

/* ── Pinned buffer pool ──────────────────────────────────── */
/* Phase 1: Pinned (page-locked) host memory for faster DMA transfers.
 * cudaMallocHost memory allows the GPU to use DMA directly, bypassing
 * an extra copy through a pageable staging buffer. */
#define PINNED_BUF_COUNT 4
#define PINNED_BUF_SIZE  (4 * 1024 * 1024)  /* 4 MB each */

struct PinnedBufferPool {
    void *buffers[PINNED_BUF_COUNT] = {};
    bool  in_use[PINNED_BUF_COUNT]  = {};
    bool  is_pinned[PINNED_BUF_COUNT] = {};
    std::mutex mtx;
    bool initialized = false;

    void init() {
        for (int i = 0; i < PINNED_BUF_COUNT; i++) {
            cudaError_t err = cudaMallocHost(&buffers[i], PINNED_BUF_SIZE);
            is_pinned[i] = (err == cudaSuccess);
            if (!is_pinned[i]) {
                LOG_WARN("cudaMallocHost failed for pinned buffer %d, using malloc fallback", i);
                buffers[i] = malloc(PINNED_BUF_SIZE);
            }
        }
        initialized = true;
        LOG_DBG("Pinned buffer pool: %d x %d MB allocated", PINNED_BUF_COUNT,
                PINNED_BUF_SIZE / (1024 * 1024));
    }

    void destroy() {
        for (int i = 0; i < PINNED_BUF_COUNT; i++) {
            if (buffers[i]) {
                if (is_pinned[i]) cudaFreeHost(buffers[i]);
                else free(buffers[i]);
                buffers[i] = nullptr;
            }
        }
        initialized = false;
    }

    /* Returns a pinned buffer, or malloc fallback if all busy.
     * index is set to the pool slot (-1 if fallback). */
    void *acquire(size_t size, int &index) {
        std::lock_guard<std::mutex> lock(mtx);
        if (size <= PINNED_BUF_SIZE) {
            for (int i = 0; i < PINNED_BUF_COUNT; i++) {
                if (!in_use[i] && buffers[i]) {
                    in_use[i] = true;
                    index = i;
                    return buffers[i];
                }
            }
        }
        /* Fallback: pageable allocation */
        index = -1;
        return malloc(size);
    }

    void release(void *buf, int index) {
        if (index >= 0 && index < PINNED_BUF_COUNT) {
            std::lock_guard<std::mutex> lock(mtx);
            in_use[index] = false;
        } else {
            free(buf);
        }
    }
};

/* Phase 2: Track pending async transfers so we can release pinned buffers
 * after the GPU finishes the async copy. */
struct PendingTransfer {
    void *pinned_buf;
    int pool_index;
    cudaEvent_t completion_event;
};

/* ── Per-client session ──────────────────────────────────── */
struct ClientSession {
    int fd;
    uint32_t session_id;
    char addr_str[INET6_ADDRSTRLEN];
    time_t connected_at;
    NetType net_type;

    /* Per-client stats */
    std::atomic<uint64_t> ops_count{0};
    std::atomic<uint64_t> bytes_in{0};
    std::atomic<uint64_t> bytes_out{0};
    std::atomic<uint64_t> mem_allocated{0};

    /* Receive buffer */
    std::vector<uint8_t> recv_buf;
    size_t recv_offset = 0;

    /* Resource tracking for cleanup */
    std::unordered_set<void*>      allocated_ptrs;
    std::unordered_map<void*, size_t> alloc_sizes; /* track sizes for stats */
    std::unordered_set<CUmodule>   loaded_modules;
    std::unordered_set<cudaStream_t> created_streams;
    std::unordered_set<cudaEvent_t>  created_events;

    /* Map client function pointers to server CUfunction handles */
    std::unordered_map<uint64_t, CUfunction> func_map;
    /* Map fatbin handles */
    std::unordered_map<uint64_t, CUmodule>   fatbin_map;

    /* Pinned memory pool (Phase 1) */
    PinnedBufferPool pinned_pool;

    /* Pending async transfers (Phase 2) */
    std::vector<PendingTransfer> pending_transfers;

    /* Phase 6: D2H prefetch — when processing chunk N, speculatively start
     * GPU DMA for chunk N+1 so it's ready when the next request arrives. */
    struct D2hPrefetch {
        uint64_t device_ptr = 0;   /* base device pointer of the transfer */
        uint64_t next_offset = 0;  /* offset of the prefetched chunk */
        uint32_t next_size = 0;    /* size of prefetched chunk */
        void *buf = nullptr;       /* pinned buffer holding prefetched data */
        int pool_idx = -1;
        cudaEvent_t event = nullptr;
        bool valid = false;
    } d2h_prefetch;

    ClientSession(int fd_, uint32_t sid, const char *addr, NetType nt)
        : fd(fd_), session_id(sid), net_type(nt) {
        strncpy(addr_str, addr, sizeof(addr_str) - 1);
        addr_str[sizeof(addr_str) - 1] = '\0';
        connected_at = time(nullptr);
        recv_buf.resize(4096);
        pinned_pool.init();
    }

    ~ClientSession() {
        cleanup_resources();
        close(fd);
    }

    /* Reap completed async transfers, releasing their pinned buffers */
    void reap_pending_transfers() {
        auto it = pending_transfers.begin();
        while (it != pending_transfers.end()) {
            cudaError_t err = cudaEventQuery(it->completion_event);
            if (err == cudaSuccess) {
                pinned_pool.release(it->pinned_buf, it->pool_index);
                cudaEventDestroy(it->completion_event);
                it = pending_transfers.erase(it);
            } else {
                ++it;
            }
        }
    }

    void cleanup_resources() {
        /* Release D2H prefetch if active */
        if (d2h_prefetch.valid) {
            if (d2h_prefetch.event) {
                cudaEventSynchronize(d2h_prefetch.event);
                cudaEventDestroy(d2h_prefetch.event);
            }
            pinned_pool.release(d2h_prefetch.buf, d2h_prefetch.pool_idx);
            d2h_prefetch.valid = false;
        }

        /* Wait for and release all pending async transfers */
        for (auto &pt : pending_transfers) {
            cudaEventSynchronize(pt.completion_event);
            pinned_pool.release(pt.pinned_buf, pt.pool_index);
            cudaEventDestroy(pt.completion_event);
        }
        pending_transfers.clear();

        for (void *p : allocated_ptrs) cudaFree(p);
        for (auto s : created_streams) cudaStreamDestroy(s);
        for (auto e : created_events)  cudaEventDestroy(e);
        for (auto m : loaded_modules)  cuModuleUnload(m);
        allocated_ptrs.clear();
        alloc_sizes.clear();
        created_streams.clear();
        created_events.clear();
        loaded_modules.clear();
        func_map.clear();
        fatbin_map.clear();
        mem_allocated = 0;
        pinned_pool.destroy();
        LOG_INFO("Session %u: resources cleaned up", session_id);
    }

    void track_op(uint64_t in_bytes = 0, uint64_t out_bytes = 0) {
        ops_count++;
        bytes_in += in_bytes;
        bytes_out += out_bytes;
        g_total_ops++;
        g_total_bytes_in += in_bytes;
        g_total_bytes_out += out_bytes;
    }
};

static std::unordered_map<int, std::unique_ptr<ClientSession>> g_sessions;
static uint32_t g_next_session = 1;
static std::mutex g_sessions_mtx;

/* ── Config file parser ──────────────────────────────────── */
static void load_config(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open()) return;

    std::string line;
    while (std::getline(f, line)) {
        /* Skip comments and empty lines */
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos || line[start] == '#') continue;
        line = line.substr(start);

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        /* Trim */
        key.erase(key.find_last_not_of(" \t") + 1);
        val.erase(0, val.find_first_not_of(" \t"));
        val.erase(val.find_last_not_of(" \t") + 1);

        if (key == "port") g_port = atoi(val.c_str());
        else if (key == "device") g_device = atoi(val.c_str());
        else if (key == "max_clients") g_max_clients = atoi(val.c_str());
        else if (key == "bind" || key == "bind_address") g_bind_addr = val;
        else if (key == "log_level") {
            if (val == "error") g_log_level = 0;
            else if (val == "info") g_log_level = 1;
            else if (val == "debug") g_log_level = 2;
        }
    }
    LOG_INFO("Loaded config from %s", path.c_str());
}

/* ── Network helpers ─────────────────────────────────────── */
static bool send_all(int fd, const void *buf, size_t len) {
    const uint8_t *p = (const uint8_t*)buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
        if (n <= 0) return false;
        p   += n;
        len -= n;
    }
    return true;
}

static bool send_response(int fd, uint16_t opcode, uint32_t req_id,
                           const void *payload, uint32_t payload_len,
                           uint16_t flags = GS_FLAG_RESPONSE) {
    uint32_t total = GPUSHARE_HEADER_SIZE + payload_len;
    gs_header_t hdr;
    gs_header_init(&hdr, opcode, req_id, total);
    hdr.flags = flags;
    if (!send_all(fd, &hdr, sizeof(hdr))) return false;
    if (payload_len > 0 && !send_all(fd, payload, payload_len)) return false;
    return true;
}

static bool send_error(int fd, uint16_t opcode, uint32_t req_id, int32_t cuda_err) {
    gs_generic_resp_t resp;
    resp.cuda_error = cuda_err;
    return send_response(fd, opcode, req_id, &resp, sizeof(resp),
                         GS_FLAG_RESPONSE | GS_FLAG_ERROR);
}

static bool send_cuda_result(int fd, uint16_t opcode, uint32_t req_id, cudaError_t err) {
    gs_generic_resp_t resp;
    resp.cuda_error = (int32_t)err;
    uint16_t flags = GS_FLAG_RESPONSE;
    if (err != cudaSuccess) flags |= GS_FLAG_ERROR;
    return send_response(fd, opcode, req_id, &resp, sizeof(resp), flags);
}

/* ── Message handlers ────────────────────────────────────── */

static bool handle_init(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    gs_init_resp_t resp;
    resp.version = GPUSHARE_VERSION;
    resp.session_id = s->session_id;
    resp.max_transfer_size = GPUSHARE_MAX_MSG_SIZE - GPUSHARE_HEADER_SIZE;
    resp.capabilities = GS_CAP_ASYNC | GS_CAP_CHUNKED;
    LOG_INFO("Session %u: client connected from %s (caps=0x%x)",
             s->session_id, s->addr_str, resp.capabilities);
    s->track_op();
    return send_response(s->fd, hdr->opcode, hdr->req_id, &resp, sizeof(resp));
}

static bool handle_ping(ClientSession *s, const gs_header_t *hdr) {
    s->track_op();
    return send_response(s->fd, hdr->opcode, hdr->req_id, nullptr, 0);
}

static bool handle_get_device_count(ClientSession *s, const gs_header_t *hdr) {
    int count = 0;
    cudaGetDeviceCount(&count);
    gs_device_count_resp_t resp;
    resp.count = count;
    s->track_op(0, sizeof(resp));
    return send_response(s->fd, hdr->opcode, hdr->req_id, &resp, sizeof(resp));
}

static bool handle_get_device_props(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    auto *req = (const gs_device_props_req_t*)payload;
    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, req->device);
    if (err != cudaSuccess) return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, err);

    gs_device_props_t resp;
    memset(&resp, 0, sizeof(resp));
    strncpy(resp.name, props.name, sizeof(resp.name) - 1);
    resp.total_global_mem     = props.totalGlobalMem;
    resp.shared_mem_per_block = props.sharedMemPerBlock;
    resp.regs_per_block       = props.regsPerBlock;
    resp.warp_size            = props.warpSize;
    resp.max_threads_per_block= props.maxThreadsPerBlock;
    resp.max_threads_dim[0]   = props.maxThreadsDim[0];
    resp.max_threads_dim[1]   = props.maxThreadsDim[1];
    resp.max_threads_dim[2]   = props.maxThreadsDim[2];
    resp.max_grid_size[0]     = props.maxGridSize[0];
    resp.max_grid_size[1]     = props.maxGridSize[1];
    resp.max_grid_size[2]     = props.maxGridSize[2];
    resp.clock_rate           = 0;  /* removed in CUDA 13.x */
    resp.major                = props.major;
    resp.minor                = props.minor;
    resp.multi_processor_count= props.multiProcessorCount;
    resp.max_threads_per_mp   = props.maxThreadsPerMultiProcessor;
    resp.total_const_mem      = props.totalConstMem;
    resp.mem_bus_width        = props.memoryBusWidth;
    resp.l2_cache_size        = props.l2CacheSize;
    s->track_op(sizeof(*req), sizeof(resp));
    return send_response(s->fd, hdr->opcode, hdr->req_id, &resp, sizeof(resp));
}

static bool handle_set_device(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    auto *req = (const gs_set_device_req_t*)payload;
    cudaError_t err = cudaSetDevice(req->device);
    s->track_op();
    return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, err);
}

static bool handle_malloc(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    auto *req = (const gs_malloc_req_t*)payload;
    void *ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, req->size);

    gs_malloc_resp_t resp;
    resp.device_ptr = (uint64_t)(uintptr_t)ptr;
    resp.cuda_error = (int32_t)err;

    if (err == cudaSuccess && ptr) {
        s->allocated_ptrs.insert(ptr);
        s->alloc_sizes[ptr] = req->size;
        s->mem_allocated += req->size;
    }
    s->track_op();
    return send_response(s->fd, hdr->opcode, hdr->req_id, &resp, sizeof(resp));
}

static bool handle_free(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    auto *req = (const gs_free_req_t*)payload;
    void *ptr = (void*)(uintptr_t)req->device_ptr;
    cudaError_t err = cudaFree(ptr);
    auto it = s->alloc_sizes.find(ptr);
    if (it != s->alloc_sizes.end()) {
        uint64_t cur = s->mem_allocated.load();
        if (cur >= it->second) s->mem_allocated -= it->second;
        s->alloc_sizes.erase(it);
    }
    s->allocated_ptrs.erase(ptr);
    s->track_op();
    return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, err);
}

static bool handle_memcpy_h2d(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    auto *req = (const gs_memcpy_h2d_req_t*)payload;
    void *dst = (void*)(uintptr_t)req->device_ptr;
    const void *src = payload + sizeof(gs_memcpy_h2d_req_t);
    cudaError_t err = cudaMemcpy(dst, src, req->size, cudaMemcpyHostToDevice);
    s->track_op(req->size, 0);
    return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, err);
}

static bool handle_memcpy_d2h(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    auto *req = (const gs_memcpy_d2h_req_t*)payload;
    void *src = (void*)(uintptr_t)req->device_ptr;

    /* Phase 1: Use pinned buffer for transfers <= 4MB for faster DMA */
    int pool_idx = -1;
    void *host_buf = nullptr;
    if (req->size <= PINNED_BUF_SIZE) {
        s->reap_pending_transfers();
        host_buf = s->pinned_pool.acquire(req->size, pool_idx);
    } else {
        host_buf = malloc(req->size);
    }

    if (!host_buf) return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, cudaErrorMemoryAllocation);

    cudaError_t err = cudaMemcpy(host_buf, src, req->size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        s->pinned_pool.release(host_buf, pool_idx);
        return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, err);
    }

    uint32_t payload_len = sizeof(int32_t) + req->size;
    uint32_t total = GPUSHARE_HEADER_SIZE + payload_len;
    gs_header_t resp_hdr;
    gs_header_init(&resp_hdr, hdr->opcode, hdr->req_id, total);
    resp_hdr.flags = GS_FLAG_RESPONSE;

    int32_t cuda_err = 0;
    bool ok = send_all(s->fd, &resp_hdr, sizeof(resp_hdr))
           && send_all(s->fd, &cuda_err, sizeof(cuda_err))
           && send_all(s->fd, host_buf, req->size);

    s->pinned_pool.release(host_buf, pool_idx);
    if (!ok) return false;
    s->track_op(0, req->size);
    return true;
}

/* ── Phase 2: Async memcpy handlers ──────────────────────── */

static bool handle_memcpy_h2d_async(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    auto *req = (const gs_memcpy_h2d_async_req_t*)payload;
    void *dst = (void*)(uintptr_t)req->device_ptr;
    const void *src_data = payload + sizeof(gs_memcpy_h2d_async_req_t);
    cudaStream_t stream = (cudaStream_t)(uintptr_t)req->stream_handle;

    /* Reap any completed transfers first */
    s->reap_pending_transfers();

    /* Copy network data into pinned buffer for async DMA */
    int pool_idx = -1;
    void *pinned_buf = s->pinned_pool.acquire(req->size, pool_idx);
    if (!pinned_buf) return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, cudaErrorMemoryAllocation);

    memcpy(pinned_buf, src_data, req->size);

    cudaError_t err = cudaMemcpyAsync(dst, pinned_buf, req->size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        s->pinned_pool.release(pinned_buf, pool_idx);
        return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, err);
    }

    /* Track buffer release via event — buffer freed when GPU finishes */
    cudaEvent_t ev = nullptr;
    cudaError_t ev_err = cudaEventCreate(&ev);
    if (ev_err != cudaSuccess) {
        s->pinned_pool.release(pinned_buf, pool_idx);
        return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, ev_err);
    }
    cudaEventRecord(ev, stream);
    s->pending_transfers.push_back({pinned_buf, pool_idx, ev});

    /* Send success immediately (before GPU completes) */
    s->track_op(req->size, 0);
    return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, cudaSuccess);
}

static bool handle_memcpy_d2h_async(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    /* Phase 2: D2H async still synchronous but uses pinned buffer.
     * True async D2H requires streaming response chunks back, deferred to Phase 3. */
    auto *req = (const gs_memcpy_d2h_async_req_t*)payload;
    void *src = (void*)(uintptr_t)req->device_ptr;
    cudaStream_t stream = (cudaStream_t)(uintptr_t)req->stream_handle;

    s->reap_pending_transfers();

    int pool_idx = -1;
    void *host_buf = nullptr;
    if (req->size <= PINNED_BUF_SIZE) {
        host_buf = s->pinned_pool.acquire(req->size, pool_idx);
    } else {
        host_buf = malloc(req->size);
    }
    if (!host_buf) return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, cudaErrorMemoryAllocation);

    /* Use the stream for the copy, then synchronize to get data */
    cudaError_t err = cudaMemcpyAsync(host_buf, src, req->size, cudaMemcpyDeviceToHost, stream);
    if (err == cudaSuccess) err = cudaStreamSynchronize(stream);

    if (err != cudaSuccess) {
        s->pinned_pool.release(host_buf, pool_idx);
        return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, err);
    }

    uint32_t payload_len = sizeof(int32_t) + req->size;
    uint32_t total = GPUSHARE_HEADER_SIZE + payload_len;
    gs_header_t resp_hdr;
    gs_header_init(&resp_hdr, hdr->opcode, hdr->req_id, total);
    resp_hdr.flags = GS_FLAG_RESPONSE;

    int32_t cuda_err = 0;
    bool ok = send_all(s->fd, &resp_hdr, sizeof(resp_hdr))
           && send_all(s->fd, &cuda_err, sizeof(cuda_err))
           && send_all(s->fd, host_buf, req->size);

    s->pinned_pool.release(host_buf, pool_idx);
    if (!ok) return false;
    s->track_op(0, req->size);
    return true;
}

/* ── Phase 3: Chunked transfer handlers ──────────────────── */

static bool handle_memcpy_h2d_chunked(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    auto *chunk = (const gs_memcpy_h2d_chunk_t*)payload;
    void *dst = (void*)(uintptr_t)(chunk->device_ptr + chunk->chunk_offset);
    const void *src_data = payload + sizeof(gs_memcpy_h2d_chunk_t);
    cudaStream_t stream = (cudaStream_t)(uintptr_t)chunk->stream_handle;

    s->reap_pending_transfers();

    /* Copy chunk data into pinned buffer and async DMA to device */
    int pool_idx = -1;
    void *pinned_buf = s->pinned_pool.acquire(chunk->chunk_size, pool_idx);
    if (!pinned_buf) return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, cudaErrorMemoryAllocation);

    memcpy(pinned_buf, src_data, chunk->chunk_size);

    cudaError_t err = cudaMemcpyAsync(dst, pinned_buf, chunk->chunk_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        s->pinned_pool.release(pinned_buf, pool_idx);
        return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, err);
    }

    /* Track for deferred release */
    cudaEvent_t ev = nullptr;
    cudaError_t ev_err = cudaEventCreate(&ev);
    if (ev_err != cudaSuccess) {
        s->pinned_pool.release(pinned_buf, pool_idx);
        return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, ev_err);
    }
    cudaEventRecord(ev, stream);
    s->pending_transfers.push_back({pinned_buf, pool_idx, ev});

    /* ACK immediately so client can send next chunk */
    s->track_op(chunk->chunk_size, 0);
    LOG_DBG("Session %u: H2D chunk offset=%lu size=%u (total=%lu)",
            s->session_id, (unsigned long)chunk->chunk_offset,
            chunk->chunk_size, (unsigned long)chunk->total_size);
    return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, cudaSuccess);
}

/* Phase 6: Per-chunk D2H with speculative prefetch.
 * When processing chunk N, if there's a next chunk, start GPU DMA for it
 * into a second pinned buffer. When the client sends the request for chunk N+1,
 * the data is already in host memory — zero GPU wait. */
static bool handle_memcpy_d2h_chunked(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    auto *chunk = (const gs_memcpy_d2h_chunk_t*)payload;
    uint64_t dev_base = chunk->device_ptr;
    uint64_t cur_offset = chunk->chunk_offset;
    cudaStream_t stream = (cudaStream_t)(uintptr_t)chunk->stream_handle;
    uint32_t csize = chunk->chunk_size;

    s->reap_pending_transfers();

    void *send_buf = nullptr;
    int send_idx = -1;

    /* Check if we have a prefetch hit for this exact chunk */
    if (s->d2h_prefetch.valid &&
        s->d2h_prefetch.device_ptr == dev_base &&
        s->d2h_prefetch.next_offset == cur_offset &&
        s->d2h_prefetch.next_size == csize) {
        /* Prefetch hit! Wait for GPU copy to finish, then use the buffer */
        cudaEventSynchronize(s->d2h_prefetch.event);
        cudaEventDestroy(s->d2h_prefetch.event);
        send_buf = s->d2h_prefetch.buf;
        send_idx = s->d2h_prefetch.pool_idx;
        s->d2h_prefetch.valid = false;
        LOG_DBG("Session %u: D2H prefetch HIT offset=%lu", s->session_id, (unsigned long)cur_offset);
    } else {
        /* Prefetch miss — release stale prefetch if any */
        if (s->d2h_prefetch.valid) {
            cudaEventSynchronize(s->d2h_prefetch.event);
            cudaEventDestroy(s->d2h_prefetch.event);
            s->pinned_pool.release(s->d2h_prefetch.buf, s->d2h_prefetch.pool_idx);
            s->d2h_prefetch.valid = false;
        }
        /* Copy this chunk synchronously */
        send_buf = s->pinned_pool.acquire(csize, send_idx);
        if (!send_buf) return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, cudaErrorMemoryAllocation);

        void *src = (void*)(uintptr_t)(dev_base + cur_offset);
        cudaError_t err = cudaMemcpyAsync(send_buf, src, csize, cudaMemcpyDeviceToHost, stream);
        if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            s->pinned_pool.release(send_buf, send_idx);
            return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, err);
        }
    }

    /* Speculatively prefetch the next chunk while we send this one */
    uint64_t next_offset = cur_offset + csize;
    if (next_offset < chunk->total_size) {
        uint32_t next_size = (uint32_t)std::min((uint64_t)PINNED_BUF_SIZE,
                                                 chunk->total_size - next_offset);
        int pf_idx = -1;
        void *pf_buf = s->pinned_pool.acquire(next_size, pf_idx);
        if (pf_buf) {
            cudaEvent_t pf_ev = nullptr;
            if (cudaEventCreate(&pf_ev) == cudaSuccess) {
                void *pf_src = (void*)(uintptr_t)(dev_base + next_offset);
                cudaMemcpyAsync(pf_buf, pf_src, next_size, cudaMemcpyDeviceToHost, stream);
                cudaEventRecord(pf_ev, stream);
                s->d2h_prefetch.device_ptr = dev_base;
                s->d2h_prefetch.next_offset = next_offset;
                s->d2h_prefetch.next_size = next_size;
                s->d2h_prefetch.buf = pf_buf;
                s->d2h_prefetch.pool_idx = pf_idx;
                s->d2h_prefetch.event = pf_ev;
                s->d2h_prefetch.valid = true;
            } else {
                s->pinned_pool.release(pf_buf, pf_idx);
            }
        }
    }

    /* Send current chunk over network (while GPU prefetches next chunk) */
    uint32_t payload_len = sizeof(int32_t) + csize;
    uint32_t total = GPUSHARE_HEADER_SIZE + payload_len;
    gs_header_t resp_hdr;
    gs_header_init(&resp_hdr, hdr->opcode, hdr->req_id, total);
    resp_hdr.flags = GS_FLAG_RESPONSE | GS_FLAG_CHUNKED;
    if (next_offset >= chunk->total_size)
        resp_hdr.flags |= GS_FLAG_LAST_CHUNK;

    int32_t cuda_err = 0;
    bool ok = send_all(s->fd, &resp_hdr, sizeof(resp_hdr))
           && send_all(s->fd, &cuda_err, sizeof(cuda_err))
           && send_all(s->fd, send_buf, csize);

    s->pinned_pool.release(send_buf, send_idx);
    if (!ok) return false;
    s->track_op(0, csize);
    return true;
}

static bool handle_memcpy_d2d(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    auto *req = (const gs_memcpy_d2d_req_t*)payload;
    void *dst = (void*)(uintptr_t)req->dst_ptr;
    void *src = (void*)(uintptr_t)req->src_ptr;
    cudaError_t err = cudaMemcpy(dst, src, req->size, cudaMemcpyDeviceToDevice);
    s->track_op();
    return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, err);
}

static bool handle_memset(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    auto *req = (const gs_memset_req_t*)payload;
    void *ptr = (void*)(uintptr_t)req->device_ptr;
    cudaError_t err = cudaMemset(ptr, req->value, req->size);
    s->track_op();
    return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, err);
}

static bool handle_module_load(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    auto *req = (const gs_module_load_req_t*)payload;
    const uint8_t *data = payload + sizeof(gs_module_load_req_t);

    /* cuModuleLoadData requires null-terminated PTX.
       Copy into a buffer and ensure null termination. */
    std::vector<uint8_t> ptx_buf(data, data + req->data_size);
    if (ptx_buf.empty() || ptx_buf.back() != 0) ptx_buf.push_back(0);

    CUmodule mod = nullptr;
    CUresult err = cuModuleLoadData(&mod, ptx_buf.data());

    gs_module_load_resp_t resp;
    resp.module_handle = (uint64_t)(uintptr_t)mod;
    resp.cuda_error = (int32_t)err;

    if (err == CUDA_SUCCESS && mod) {
        s->loaded_modules.insert(mod);
    }
    s->track_op(req->data_size, 0);
    return send_response(s->fd, hdr->opcode, hdr->req_id, &resp, sizeof(resp));
}

static bool handle_module_unload(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    uint64_t handle;
    memcpy(&handle, payload, sizeof(handle));
    CUmodule mod = (CUmodule)(uintptr_t)handle;
    CUresult err = cuModuleUnload(mod);
    s->loaded_modules.erase(mod);
    gs_generic_resp_t resp;
    resp.cuda_error = (int32_t)err;
    s->track_op();
    return send_response(s->fd, hdr->opcode, hdr->req_id, &resp, sizeof(resp));
}

static bool handle_get_function(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    auto *req = (const gs_get_function_req_t*)payload;
    CUmodule mod = (CUmodule)(uintptr_t)req->module_handle;
    CUfunction func = nullptr;
    CUresult err = cuModuleGetFunction(&func, mod, req->func_name);

    gs_get_function_resp_t resp;
    resp.func_handle = (uint64_t)(uintptr_t)func;
    resp.cuda_error = (int32_t)err;
    s->track_op();
    return send_response(s->fd, hdr->opcode, hdr->req_id, &resp, sizeof(resp));
}

static bool handle_launch_kernel(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    auto *req = (const gs_launch_kernel_req_t*)payload;
    CUfunction func = (CUfunction)(uintptr_t)req->func_handle;
    cudaStream_t stream = (cudaStream_t)(uintptr_t)req->stream_handle;

    const uint8_t *arg_data = payload + sizeof(gs_launch_kernel_req_t);
    std::vector<void*> arg_ptrs;
    std::vector<std::vector<uint8_t>> arg_storage;

    uint32_t offset = 0;
    for (uint32_t i = 0; i < req->num_args && offset < req->args_size; i++) {
        uint32_t arg_size;
        memcpy(&arg_size, arg_data + offset, sizeof(arg_size));
        offset += sizeof(arg_size);
        arg_storage.emplace_back(arg_data + offset, arg_data + offset + arg_size);
        offset += arg_size;
    }

    arg_ptrs.resize(arg_storage.size());
    for (size_t i = 0; i < arg_storage.size(); i++) {
        arg_ptrs[i] = arg_storage[i].data();
    }

    CUresult err = cuLaunchKernel(
        func,
        req->grid_x, req->grid_y, req->grid_z,
        req->block_x, req->block_y, req->block_z,
        req->shared_mem, (CUstream)stream,
        arg_ptrs.data(), nullptr
    );

    gs_generic_resp_t resp;
    resp.cuda_error = (int32_t)err;
    s->track_op(req->args_size, 0);
    return send_response(s->fd, hdr->opcode, hdr->req_id, &resp, sizeof(resp));
}

static bool handle_device_sync(ClientSession *s, const gs_header_t *hdr) {
    cudaError_t err = cudaDeviceSynchronize();
    s->track_op();
    return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, err);
}

static bool handle_stream_create(ClientSession *s, const gs_header_t *hdr) {
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    gs_stream_create_resp_t resp;
    resp.stream_handle = (uint64_t)(uintptr_t)stream;
    if (err == cudaSuccess) s->created_streams.insert(stream);
    s->track_op();
    return send_response(s->fd, hdr->opcode, hdr->req_id, &resp, sizeof(resp));
}

static bool handle_stream_destroy(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    auto *req = (const gs_stream_req_t*)payload;
    cudaStream_t stream = (cudaStream_t)(uintptr_t)req->stream_handle;
    cudaError_t err = cudaStreamDestroy(stream);
    s->created_streams.erase(stream);
    s->track_op();
    return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, err);
}

static bool handle_stream_sync(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    auto *req = (const gs_stream_req_t*)payload;
    cudaStream_t stream = (cudaStream_t)(uintptr_t)req->stream_handle;
    cudaError_t err = cudaStreamSynchronize(stream);
    s->track_op();
    return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, err);
}

static bool handle_event_create(ClientSession *s, const gs_header_t *hdr) {
    cudaEvent_t event;
    cudaError_t err = cudaEventCreate(&event);
    gs_event_create_resp_t resp;
    resp.event_handle = (uint64_t)(uintptr_t)event;
    if (err == cudaSuccess) s->created_events.insert(event);
    s->track_op();
    return send_response(s->fd, hdr->opcode, hdr->req_id, &resp, sizeof(resp));
}

static bool handle_event_destroy(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    uint64_t handle;
    memcpy(&handle, payload, sizeof(handle));
    cudaEvent_t event = (cudaEvent_t)(uintptr_t)handle;
    cudaError_t err = cudaEventDestroy(event);
    s->created_events.erase(event);
    s->track_op();
    return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, err);
}

static bool handle_event_record(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    auto *req = (const gs_event_record_req_t*)payload;
    cudaEvent_t event = (cudaEvent_t)(uintptr_t)req->event_handle;
    cudaStream_t stream = (cudaStream_t)(uintptr_t)req->stream_handle;
    cudaError_t err = cudaEventRecord(event, stream);
    s->track_op();
    return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, err);
}

static bool handle_event_sync(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    uint64_t handle;
    memcpy(&handle, payload, sizeof(handle));
    cudaEvent_t event = (cudaEvent_t)(uintptr_t)handle;
    cudaError_t err = cudaEventSynchronize(event);
    s->track_op();
    return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, err);
}

static bool handle_event_elapsed(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    auto *req = (const gs_event_elapsed_req_t*)payload;
    cudaEvent_t start = (cudaEvent_t)(uintptr_t)req->start_event;
    cudaEvent_t end   = (cudaEvent_t)(uintptr_t)req->end_event;
    float ms = 0;
    cudaError_t err = cudaEventElapsedTime(&ms, start, end);
    if (err != cudaSuccess) return send_cuda_result(s->fd, hdr->opcode, hdr->req_id, err);
    gs_event_elapsed_resp_t resp;
    resp.milliseconds = ms;
    s->track_op();
    return send_response(s->fd, hdr->opcode, hdr->req_id, &resp, sizeof(resp));
}

static bool handle_register_fatbin(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    auto *req = (const gs_register_fatbin_req_t*)payload;
    const void *data = payload + sizeof(gs_register_fatbin_req_t);

    CUmodule mod = nullptr;
    CUresult err = cuModuleLoadData(&mod, data);

    uint64_t handle = (uint64_t)(uintptr_t)mod;
    if (err == CUDA_SUCCESS && mod) {
        s->loaded_modules.insert(mod);
        s->fatbin_map[handle] = mod;
    }

    gs_register_fatbin_resp_t resp;
    resp.fatbin_handle = handle;
    s->track_op(req->data_size, 0);
    return send_response(s->fd, hdr->opcode, hdr->req_id, &resp, sizeof(resp));
}

static bool handle_register_function(ClientSession *s, const gs_header_t *hdr, const uint8_t *payload) {
    auto *req = (const gs_register_function_req_t*)payload;
    auto it = s->fatbin_map.find(req->fatbin_handle);
    if (it == s->fatbin_map.end()) {
        return send_error(s->fd, hdr->opcode, hdr->req_id, CUDA_ERROR_NOT_FOUND);
    }

    CUfunction func = nullptr;
    CUresult err = cuModuleGetFunction(&func, it->second, req->device_name);
    if (err == CUDA_SUCCESS) {
        s->func_map[req->host_func] = func;
    }

    gs_generic_resp_t resp;
    resp.cuda_error = (int32_t)err;
    s->track_op();
    return send_response(s->fd, hdr->opcode, hdr->req_id, &resp, sizeof(resp));
}

/* ── Live GPU status handler ──────────────────────────────── */
static bool handle_get_gpu_status(ClientSession *s, const gs_header_t *hdr) {
    gs_gpu_status_t status;
    memset(&status, 0, sizeof(status));

    /* Memory info from CUDA */
    size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
        status.mem_total = total_mem;
        status.mem_free  = free_mem;
        status.mem_used  = total_mem - free_mem;
    }

    /* Try NVML for utilization, temp, power (best-effort) */
    /* We parse nvidia-smi since it's always available and avoids linking NVML */
    FILE *fp = popen("nvidia-smi --query-gpu=utilization.gpu,utilization.memory,"
                     "temperature.gpu,power.draw,power.limit,fan.speed,"
                     "clocks.current.sm,clocks.current.memory "
                     "--format=csv,noheader,nounits 2>/dev/null", "r");
    if (fp) {
        char buf[512];
        if (fgets(buf, sizeof(buf), fp)) {
            unsigned u_gpu=0, u_mem=0, temp=0, fan=0, clk_sm=0, clk_mem=0;
            float pwr_draw=0, pwr_limit=0;
            if (sscanf(buf, " %u , %u , %u , %f , %f , %u , %u , %u",
                       &u_gpu, &u_mem, &temp, &pwr_draw, &pwr_limit,
                       &fan, &clk_sm, &clk_mem) >= 5) {
                status.gpu_utilization = u_gpu;
                status.mem_utilization = u_mem;
                status.temperature     = temp;
                status.power_draw_mw   = (uint32_t)(pwr_draw * 1000);
                status.power_limit_mw  = (uint32_t)(pwr_limit * 1000);
                status.fan_speed       = fan;
                status.clock_sm_mhz    = clk_sm;
                status.clock_mem_mhz   = clk_mem;
            }
        }
        pclose(fp);
    }

    s->track_op(0, sizeof(status));
    return send_response(s->fd, hdr->opcode, hdr->req_id, &status, sizeof(status));
}

/* ── Stats handler ───────────────────────────────────────── */
static bool handle_get_stats(ClientSession *s, const gs_header_t *hdr) {
    std::lock_guard<std::mutex> lock(g_sessions_mtx);

    gs_stats_header_t stats_hdr;
    memset(&stats_hdr, 0, sizeof(stats_hdr));
    stats_hdr.uptime_secs      = (uint64_t)(time(nullptr) - g_start_time);
    stats_hdr.total_ops        = g_total_ops.load();
    stats_hdr.total_bytes_in   = g_total_bytes_in.load();
    stats_hdr.total_bytes_out  = g_total_bytes_out.load();
    stats_hdr.active_clients   = (uint32_t)g_sessions.size();
    stats_hdr.total_connections= g_total_connections.load();

    /* Calculate total allocated memory */
    uint64_t total_alloc = 0;
    for (auto &kv : g_sessions) total_alloc += kv.second->mem_allocated.load();
    stats_hdr.total_alloc_bytes = total_alloc;

    uint32_t n = std::min((uint32_t)g_sessions.size(), (uint32_t)GS_MAX_STAT_CLIENTS);
    stats_hdr.num_clients = n;

    /* Build response */
    uint32_t payload_len = sizeof(gs_stats_header_t) + n * sizeof(gs_stats_client_t);
    std::vector<uint8_t> payload(payload_len);
    memcpy(payload.data(), &stats_hdr, sizeof(stats_hdr));

    uint32_t idx = 0;
    time_t now = time(nullptr);
    for (auto &kv : g_sessions) {
        if (idx >= n) break;
        auto &cs = kv.second;
        gs_stats_client_t cl;
        memset(&cl, 0, sizeof(cl));
        cl.session_id     = cs->session_id;
        strncpy(cl.addr, cs->addr_str, sizeof(cl.addr) - 1);
        cl.mem_allocated  = cs->mem_allocated.load();
        cl.ops_count      = cs->ops_count.load();
        cl.bytes_in       = cs->bytes_in.load();
        cl.bytes_out      = cs->bytes_out.load();
        cl.connected_secs = (uint64_t)(now - cs->connected_at);

        memcpy(payload.data() + sizeof(gs_stats_header_t) + idx * sizeof(gs_stats_client_t),
               &cl, sizeof(cl));
        idx++;
    }

    return send_response(s->fd, hdr->opcode, hdr->req_id, payload.data(), payload_len);
}

/* ── Generic library call (cuBLAS, cuDNN, etc.) ──────────── */
/* Defined in generated_dispatch.cpp */
extern bool dispatch_lib_call(const uint8_t *data, uint32_t len,
                              std::vector<uint8_t> &response);

static bool handle_lib_call(ClientSession *s, const gs_header_t *hdr,
                            const uint8_t *payload, uint32_t payload_len) {
    std::vector<uint8_t> response;
    if (!dispatch_lib_call(payload, payload_len, response)) {
        return send_error(s->fd, hdr->opcode, hdr->req_id, -1);
    }
    s->track_op(payload_len, response.size());
    return send_response(s->fd, hdr->opcode, hdr->req_id,
                         response.data(), response.size());
}

/* ── Message dispatch ────────────────────────────────────── */

static bool handle_message(ClientSession *s, const uint8_t *msg, uint32_t len) {
    if (len < GPUSHARE_HEADER_SIZE) return false;

    const gs_header_t *hdr = (const gs_header_t*)msg;
    if (!gs_header_validate(hdr)) {
        LOG_WARN("Session %u: invalid header", s->session_id);
        return false;
    }

    const uint8_t *payload = msg + GPUSHARE_HEADER_SIZE;

    switch (hdr->opcode) {
        case GS_OP_INIT:             return handle_init(s, hdr, payload);
        case GS_OP_PING:             return handle_ping(s, hdr);
        case GS_OP_CLOSE:            return false;
        case GS_OP_GET_DEVICE_COUNT: return handle_get_device_count(s, hdr);
        case GS_OP_GET_DEVICE_PROPS: return handle_get_device_props(s, hdr, payload);
        case GS_OP_SET_DEVICE:       return handle_set_device(s, hdr, payload);
        case GS_OP_MALLOC:           return handle_malloc(s, hdr, payload);
        case GS_OP_FREE:             return handle_free(s, hdr, payload);
        case GS_OP_MEMCPY_H2D:
            if (hdr->flags & GS_FLAG_CHUNKED)
                return handle_memcpy_h2d_chunked(s, hdr, payload);
            return handle_memcpy_h2d(s, hdr, payload);
        case GS_OP_MEMCPY_D2H:
            if (hdr->flags & GS_FLAG_CHUNKED)
                return handle_memcpy_d2h_chunked(s, hdr, payload);
            return handle_memcpy_d2h(s, hdr, payload);
        case GS_OP_MEMCPY_D2D:      return handle_memcpy_d2d(s, hdr, payload);
        case GS_OP_MEMSET:           return handle_memset(s, hdr, payload);
        case GS_OP_MEMCPY_H2D_ASYNC:return handle_memcpy_h2d_async(s, hdr, payload);
        case GS_OP_MEMCPY_D2H_ASYNC:return handle_memcpy_d2h_async(s, hdr, payload);
        case GS_OP_MODULE_LOAD:      return handle_module_load(s, hdr, payload);
        case GS_OP_MODULE_UNLOAD:    return handle_module_unload(s, hdr, payload);
        case GS_OP_GET_FUNCTION:     return handle_get_function(s, hdr, payload);
        case GS_OP_LAUNCH_KERNEL:    return handle_launch_kernel(s, hdr, payload);
        case GS_OP_DEVICE_SYNC:      return handle_device_sync(s, hdr);
        case GS_OP_STREAM_CREATE:    return handle_stream_create(s, hdr);
        case GS_OP_STREAM_DESTROY:   return handle_stream_destroy(s, hdr, payload);
        case GS_OP_STREAM_SYNC:      return handle_stream_sync(s, hdr, payload);
        case GS_OP_EVENT_CREATE:     return handle_event_create(s, hdr);
        case GS_OP_EVENT_DESTROY:    return handle_event_destroy(s, hdr, payload);
        case GS_OP_EVENT_RECORD:     return handle_event_record(s, hdr, payload);
        case GS_OP_EVENT_SYNC:       return handle_event_sync(s, hdr, payload);
        case GS_OP_EVENT_ELAPSED:    return handle_event_elapsed(s, hdr, payload);
        case GS_OP_REGISTER_FAT_BIN: return handle_register_fatbin(s, hdr, payload);
        case GS_OP_REGISTER_FUNCTION:return handle_register_function(s, hdr, payload);
        case GS_OP_GET_STATS:        return handle_get_stats(s, hdr);
        case GS_OP_GET_GPU_STATUS:   return handle_get_gpu_status(s, hdr);
        case GS_OP_LIB_CALL:        return handle_lib_call(s, hdr, payload, len - GPUSHARE_HEADER_SIZE);
        default:
            LOG_WARN("Session %u: unknown opcode 0x%04x", s->session_id, hdr->opcode);
            return send_error(s->fd, hdr->opcode, hdr->req_id, -1);
    }
}

/* ── Client thread ───────────────────────────────────────── */

static void *client_thread(void *arg) {
    struct ClientInfo { int fd; char addr[INET6_ADDRSTRLEN]; NetType net; };
    auto *info = (ClientInfo*)arg;
    int fd = info->fd;
    char addr[INET6_ADDRSTRLEN];
    strncpy(addr, info->addr, sizeof(addr));
    NetType net = info->net;
    delete info;

    uint32_t sid;
    {
        std::lock_guard<std::mutex> lock(g_sessions_mtx);
        sid = g_next_session++;
        g_sessions[fd] = std::make_unique<ClientSession>(fd, sid, addr, net);
        g_total_connections++;
    }

    ClientSession *s;
    {
        std::lock_guard<std::mutex> lock(g_sessions_mtx);
        s = g_sessions[fd].get();
    }

    const char *net_str = (net == NET_LAN) ? "LAN" : "WAN";
    LOG_INFO("Session %u: connected (fd=%d, addr=%s, %s)", sid, fd, addr, net_str);

    /* Ensure CUDA context is current on this thread */
    cudaSetDevice(g_device);

    /* Per-client TCP tuning based on network type */
    int flag = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
    setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &flag, sizeof(flag));

    int bufsize;
    int recv_chunk;
    if (net == NET_LAN) {
        /* LAN: maximize throughput — full 1 Gbps over switch
         * Large buffers + TCP_QUICKACK = saturate the link.
         * Traffic stays on the switch, never touches internet gateway. */
        bufsize = 8 * 1024 * 1024;   /* 8 MB — fills 1GbE pipe */
        recv_chunk = 256 * 1024;      /* 256 KB recv buffer */
#ifdef TCP_QUICKACK
        setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &flag, sizeof(flag));
#endif
    } else {
        /* WAN/WiFi: conservative buffers — avoid bloating slow links.
         * Still fast, but won't overwhelm WiFi or internet connections. */
        bufsize = 2 * 1024 * 1024;   /* 2 MB */
        recv_chunk = 64 * 1024;       /* 64 KB recv buffer */
    }
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));

    std::vector<uint8_t> buf(recv_chunk);
    std::vector<uint8_t> msg_buf;
    size_t msg_offset = 0;

    while (g_running) {
        ssize_t n = recv(fd, buf.data(), buf.size(), 0);
        if (n <= 0) break;

        msg_buf.insert(msg_buf.end(), buf.data(), buf.data() + n);

        while (msg_buf.size() - msg_offset >= GPUSHARE_HEADER_SIZE) {
            const gs_header_t *hdr = (const gs_header_t*)(msg_buf.data() + msg_offset);

            if (!gs_header_validate(hdr)) {
                LOG_ERR("Session %u: corrupt message, disconnecting", sid);
                goto disconnect;
            }

            if (msg_buf.size() - msg_offset < hdr->length) {
                if (msg_offset > 1024 * 1024) {
                    msg_buf.erase(msg_buf.begin(), msg_buf.begin() + msg_offset);
                    msg_offset = 0;
                }
                break;
            }

            if (!handle_message(s, msg_buf.data() + msg_offset, hdr->length)) {
                goto disconnect;
            }
            msg_offset += hdr->length;
        }

        if (msg_offset == msg_buf.size()) {
            msg_buf.clear();
            msg_offset = 0;
        }
    }

disconnect:
    LOG_INFO("Session %u: disconnected", sid);
    {
        std::lock_guard<std::mutex> lock(g_sessions_mtx);
        g_sessions.erase(fd);
    }
    return nullptr;
}

/* ── Signal handling ─────────────────────────────────────── */

static void sig_handler(int) {
    g_running = false;
}

/* ── Main ────────────────────────────────────────────────── */

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options]\n"
            "  --port PORT        Listen port (default: %d)\n"
            "  --bind ADDR        Bind to specific IP (default: all interfaces)\n"
            "                     Use LAN IP to keep traffic off the internet\n"
            "  --device DEV       CUDA device index (default: 0)\n"
            "  --config PATH      Config file path\n"
            "  --log-level LEVEL  error|info|debug (default: info)\n"
            "  --help             Show this help\n",
            prog, GPUSHARE_DEFAULT_PORT);
}

int main(int argc, char *argv[]) {
    g_start_time = time(nullptr);

    /* Try default config locations */
    const char *default_configs[] = {
        "/etc/gpushare/server.conf",
        nullptr
    };
    for (int i = 0; default_configs[i]; i++) {
        if (access(default_configs[i], R_OK) == 0) {
            load_config(default_configs[i]);
            break;
        }
    }

    /* Parse args (override config) */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--port") && i + 1 < argc)
            g_port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bind") && i + 1 < argc)
            g_bind_addr = argv[++i];
        else if (!strcmp(argv[i], "--device") && i + 1 < argc)
            g_device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--config") && i + 1 < argc)
            load_config(argv[++i]);
        else if (!strcmp(argv[i], "--log-level") && i + 1 < argc) {
            const char *lvl = argv[++i];
            if (!strcmp(lvl, "error")) g_log_level = 0;
            else if (!strcmp(lvl, "info")) g_log_level = 1;
            else if (!strcmp(lvl, "debug")) g_log_level = 2;
        }
        else if (!strcmp(argv[i], "--help")) {
            print_usage(argv[0]);
            return 0;
        }
    }

    /* Initialize CUDA */
    CUresult cu_err = cuInit(0);
    if (cu_err != CUDA_SUCCESS) {
        LOG_ERR("cuInit failed: %d", cu_err);
        return 1;
    }

    cudaError_t cuda_err = cudaSetDevice(g_device);
    if (cuda_err != cudaSuccess) {
        LOG_ERR("cudaSetDevice(%d) failed: %s", g_device, cudaGetErrorString(cuda_err));
        return 1;
    }

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, g_device);
    LOG_INFO("GPU: %s (%.0f MB VRAM, SM %d.%d, %d SMs)",
             props.name, props.totalGlobalMem / 1048576.0,
             props.major, props.minor, props.multiProcessorCount);

    /* Create listening socket.
     * If bind_address is set to a LAN IP (e.g. 192.168.x.x), traffic stays
     * on the local switch and never touches the internet gateway.
     * If empty, listens on all interfaces. */
    int listen_fd;
    if (!g_bind_addr.empty() && g_bind_addr != "0.0.0.0" && g_bind_addr != "::") {
        /* Bind to specific IPv4 address (LAN-only mode) */
        listen_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd < 0) {
            LOG_ERR("socket: %s", strerror(errno));
            return 1;
        }

        int opt = 1;
        setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr4 = {};
        addr4.sin_family = AF_INET;
        addr4.sin_port   = htons(g_port);
        if (inet_pton(AF_INET, g_bind_addr.c_str(), &addr4.sin_addr) != 1) {
            LOG_ERR("Invalid bind address: %s", g_bind_addr.c_str());
            close(listen_fd);
            return 1;
        }

        if (bind(listen_fd, (struct sockaddr*)&addr4, sizeof(addr4)) < 0) {
            LOG_ERR("bind(%s:%d): %s", g_bind_addr.c_str(), g_port, strerror(errno));
            close(listen_fd);
            return 1;
        }
        LOG_INFO("Bound to %s (LAN-only — traffic stays off the internet)", g_bind_addr.c_str());
    } else {
        /* Bind to all interfaces (IPv4 + IPv6) */
        listen_fd = socket(AF_INET6, SOCK_STREAM, 0);
        if (listen_fd < 0) {
            LOG_ERR("socket: %s", strerror(errno));
            return 1;
        }

        int opt = 1;
        setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        int off = 0;
        setsockopt(listen_fd, IPPROTO_IPV6, IPV6_V6ONLY, &off, sizeof(off));

        struct sockaddr_in6 addr = {};
        addr.sin6_family = AF_INET6;
        addr.sin6_port   = htons(g_port);
        addr.sin6_addr   = in6addr_any;

        if (bind(listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            LOG_ERR("bind: %s", strerror(errno));
            close(listen_fd);
            return 1;
        }
    }

    /* LAN-optimized TCP settings on the listening socket */
    {
        /* Large socket buffers — maximize throughput on 1GbE LAN */
        int bufsize = 8 * 1024 * 1024;  /* 8 MB */
        setsockopt(listen_fd, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));
        setsockopt(listen_fd, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));
    }

    if (listen(listen_fd, g_max_clients) < 0) {
        LOG_ERR("listen: %s", strerror(errno));
        close(listen_fd);
        return 1;
    }

    signal(SIGINT,  sig_handler);
    signal(SIGTERM, sig_handler);
    signal(SIGPIPE, SIG_IGN);

    /* Detect local network interfaces for LAN/WAN client classification */
    detect_local_networks();
    if (g_local_nets.empty()) {
        LOG_WARN("Could not detect local networks — all clients treated as LAN");
    } else {
        for (const auto &ln : g_local_nets) {
            char a[INET_ADDRSTRLEN], m[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &ln.addr, a, sizeof(a));
            inet_ntop(AF_INET, &ln.mask, m, sizeof(m));
            LOG_INFO("LAN subnet: %s/%s (%s)", a, m, ln.ifname);
        }
    }

    const char *bind_info = g_bind_addr.empty() ? "all interfaces" : g_bind_addr.c_str();
    LOG_INFO("gpushare server v%d listening on %s:%d (device %d, max %d clients)",
             GPUSHARE_VERSION, bind_info, g_port, g_device, g_max_clients);
    LOG_INFO("LAN clients → 8MB buffers (full 1Gbps) | WAN clients → 2MB buffers");

    while (g_running) {
        struct sockaddr_storage client_addr;
        socklen_t addr_len = sizeof(client_addr);
        int client_fd = accept(listen_fd, (struct sockaddr*)&client_addr, &addr_len);
        if (client_fd < 0) {
            if (g_running) LOG_WARN("accept: %s", strerror(errno));
            continue;
        }

        /* Extract printable address and classify LAN vs WAN */
        char addr_str[INET6_ADDRSTRLEN] = {};
        if (client_addr.ss_family == AF_INET6) {
            inet_ntop(AF_INET6, &((struct sockaddr_in6*)&client_addr)->sin6_addr,
                      addr_str, sizeof(addr_str));
        } else {
            inet_ntop(AF_INET, &((struct sockaddr_in*)&client_addr)->sin_addr,
                      addr_str, sizeof(addr_str));
        }

        NetType net = classify_client(&client_addr);
        const char *net_label = (net == NET_LAN) ? "LAN" : "WAN";
        LOG_INFO("New connection from %s [%s]", addr_str, net_label);

        /* Check client limit */
        {
            std::lock_guard<std::mutex> lock(g_sessions_mtx);
            if ((int)g_sessions.size() >= g_max_clients) {
                LOG_WARN("Max clients reached (%d), rejecting %s", g_max_clients, addr_str);
                close(client_fd);
                continue;
            }
        }

        struct ClientInfo { int fd; char addr[INET6_ADDRSTRLEN]; NetType net; };
        auto *info = new ClientInfo;
        info->fd = client_fd;
        strncpy(info->addr, addr_str, sizeof(info->addr));
        info->net = net;

        pthread_t tid;
        pthread_attr_t pattr;
        pthread_attr_init(&pattr);
        pthread_attr_setdetachstate(&pattr, PTHREAD_CREATE_DETACHED);
        if (pthread_create(&tid, &pattr, client_thread, info) != 0) {
            LOG_ERR("pthread_create: %s", strerror(errno));
            close(client_fd);
            delete info;
        }
        pthread_attr_destroy(&pattr);
    }

    LOG_INFO("Shutting down...");
    close(listen_fd);

    {
        std::lock_guard<std::mutex> lock(g_sessions_mtx);
        g_sessions.clear();
    }

    cudaDeviceReset();
    LOG_INFO("Server stopped. Stats: %lu ops, %lu bytes in, %lu bytes out, %u total connections",
             g_total_ops.load(), g_total_bytes_in.load(), g_total_bytes_out.load(),
             g_total_connections.load());
    return 0;
}
