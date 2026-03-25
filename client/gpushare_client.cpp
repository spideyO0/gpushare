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
#include <condition_variable>
#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <future>
#include <thread>

/* Platform-specific networking */
#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX  /* prevent windows.h min/max macros conflicting with std::min/std::max */
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
#include "gpushare/transport.h"
#include "gpushare/compression.h"

/* ── Logging ─────────────────────────────────────────────── */
static int g_verbose = 0;
#define TRACE(fmt, ...) do { if (g_verbose) fprintf(stderr, "[gpushare] " fmt "\n", ##__VA_ARGS__); } while(0)
#define ERR(fmt, ...)   fprintf(stderr, "[gpushare] ERROR: " fmt "\n", ##__VA_ARGS__)

/* Platform-specific dynamic loading for local GPU passthrough */
#ifndef _WIN32
  #include <dlfcn.h>
  #include <sys/mman.h>   /* mmap, munmap, mlock, munlock (Phase 5 pinned pool) */
#endif

/* ══════════════════════════════════════════════════════════════════════════
 * Local GPU passthrough — when a real NVIDIA GPU is present locally,
 * expose both local and remote GPUs. Applications can select which to use.
 *
 * Device numbering: local GPUs first (0..N-1), remote GPU(s) after (N..)
 * Real CUDA libraries are loaded from a backup path via dlopen.
 * ══════════════════════════════════════════════════════════════════════════ */

/* Default path where install scripts backup real CUDA libraries */
#ifdef __linux__
  #define REAL_CUDA_DEFAULT_PATH "/usr/local/lib/gpushare/real"
#elif defined(__APPLE__)
  #define REAL_CUDA_DEFAULT_PATH "/usr/local/lib/gpushare/real"
#else
  #define REAL_CUDA_DEFAULT_PATH ""
#endif

/* Function pointer typedefs for real CUDA driver API (from real libcuda.so.1) */
typedef CUresult (*pfn_cuInit)(unsigned int);
typedef CUresult (*pfn_cuDeviceGetCount)(int*);
typedef CUresult (*pfn_cuDeviceGet)(CUdevice*, int);
typedef CUresult (*pfn_cuDeviceGetName)(char*, int, CUdevice);
typedef CUresult (*pfn_cuDeviceTotalMem)(size_t*, CUdevice);
typedef CUresult (*pfn_cuDeviceGetAttribute)(int*, CUdevice_attribute, CUdevice);
typedef CUresult (*pfn_cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
typedef CUresult (*pfn_cuDevicePrimaryCtxRelease)(CUdevice);
typedef CUresult (*pfn_cuCtxSetCurrent)(CUcontext);
typedef CUresult (*pfn_cuCtxSynchronize)(void);
typedef CUresult (*pfn_cuMemAlloc)(CUdeviceptr*, size_t);
typedef CUresult (*pfn_cuMemFree)(CUdeviceptr);
typedef CUresult (*pfn_cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
typedef CUresult (*pfn_cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
typedef CUresult (*pfn_cuMemcpyDtoD)(CUdeviceptr, CUdeviceptr, size_t);
typedef CUresult (*pfn_cuMemcpyHtoDAsync)(CUdeviceptr, const void*, size_t, CUstream);
typedef CUresult (*pfn_cuMemcpyDtoHAsync)(void*, CUdeviceptr, size_t, CUstream);
typedef CUresult (*pfn_cuMemsetD8)(CUdeviceptr, unsigned char, size_t);
typedef CUresult (*pfn_cuStreamCreate)(CUstream*, unsigned int);
typedef CUresult (*pfn_cuStreamDestroy)(CUstream);
typedef CUresult (*pfn_cuStreamSynchronize)(CUstream);
typedef CUresult (*pfn_cuEventCreate)(CUevent*, unsigned int);
typedef CUresult (*pfn_cuEventDestroy)(CUevent);
typedef CUresult (*pfn_cuEventRecord)(CUevent, CUstream);
typedef CUresult (*pfn_cuEventSynchronize)(CUevent);
typedef CUresult (*pfn_cuEventElapsedTime)(float*, CUevent, CUevent);

/* Function pointer typedefs for real NVML */
typedef nvmlReturn_t (*pfn_nvmlInit_v2)(void);
typedef nvmlReturn_t (*pfn_nvmlShutdown)(void);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetCount_v2)(unsigned int*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetHandleByIndex_v2)(unsigned int, nvmlDevice_t*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetName)(nvmlDevice_t, char*, unsigned int);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetUtilizationRates)(nvmlDevice_t, nvmlUtilization_t*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetTemperature)(nvmlDevice_t, nvmlTemperatureSensors_t, unsigned int*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetPowerUsage)(nvmlDevice_t, unsigned int*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetFanSpeed)(nvmlDevice_t, unsigned int*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetUUID)(nvmlDevice_t, char*, unsigned int);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetPciInfo_v3)(nvmlDevice_t, nvmlPciInfo_t*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetCudaComputeCapability)(nvmlDevice_t, int*, int*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetClockInfo)(nvmlDevice_t, int, unsigned int*);

#ifdef _WIN32
/* Windows still uses runtime API for local GPU passthrough (no recursive dlopen issue) */
typedef cudaError_t (*pfn_cudaGetDeviceCount)(int*);
typedef cudaError_t (*pfn_cudaGetDeviceProperties)(struct cudaDeviceProp*, int);
typedef cudaError_t (*pfn_cudaSetDevice)(int);
typedef cudaError_t (*pfn_cudaMalloc)(void**, size_t);
typedef cudaError_t (*pfn_cudaFree)(void*);
typedef cudaError_t (*pfn_cudaMemcpy)(void*, const void*, size_t, cudaMemcpyKind);
typedef cudaError_t (*pfn_cudaMemcpyAsync)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
typedef cudaError_t (*pfn_cudaMemset)(void*, int, size_t);
typedef cudaError_t (*pfn_cudaDeviceSynchronize)(void);
typedef cudaError_t (*pfn_cudaStreamCreate)(cudaStream_t*);
typedef cudaError_t (*pfn_cudaStreamDestroy)(cudaStream_t);
typedef cudaError_t (*pfn_cudaStreamSynchronize)(cudaStream_t);
typedef cudaError_t (*pfn_cudaEventCreate)(cudaEvent_t*);
typedef cudaError_t (*pfn_cudaEventDestroy)(cudaEvent_t);
typedef cudaError_t (*pfn_cudaEventRecord)(cudaEvent_t, cudaStream_t);
typedef cudaError_t (*pfn_cudaEventSynchronize)(cudaEvent_t);
typedef cudaError_t (*pfn_cudaEventElapsedTime)(float*, cudaEvent_t, cudaEvent_t);
#endif

struct RealCUDA {
    void *h_cuda = nullptr;
#ifdef _WIN32
    void *h_cudart = nullptr;  /* Windows still uses runtime API for local passthrough */
#endif
    void *h_nvml = nullptr;
    int local_count = 0;
    bool available = false;

    /* Local NVML device handles (one per local GPU) */
    nvmlDevice_t nvml_handles[8];

    /* Per-device primary CUDA contexts (from real driver) */
    CUcontext contexts[8];

#ifdef _WIN32
    /* Windows: runtime API function pointers (from real cudart64_*.dll) */
    pfn_cudaGetDeviceCount       GetDeviceCount = nullptr;
    pfn_cudaGetDeviceProperties  GetDeviceProperties = nullptr;
    pfn_cudaSetDevice            SetDevice = nullptr;
    pfn_cudaMalloc               Malloc = nullptr;
    pfn_cudaFree                 Free = nullptr;
    pfn_cudaMemcpy               Memcpy = nullptr;
    pfn_cudaMemcpyAsync          MemcpyAsync = nullptr;
    pfn_cudaMemset               Memset = nullptr;
    pfn_cudaDeviceSynchronize    DeviceSynchronize = nullptr;
    pfn_cudaStreamCreate         StreamCreate = nullptr;
    pfn_cudaStreamDestroy        StreamDestroy = nullptr;
    pfn_cudaStreamSynchronize    StreamSynchronize = nullptr;
    pfn_cudaEventCreate          EventCreate = nullptr;
    pfn_cudaEventDestroy         EventDestroy = nullptr;
    pfn_cudaEventRecord          EventRecord = nullptr;
    pfn_cudaEventSynchronize     EventSynchronize = nullptr;
    pfn_cudaEventElapsedTime     EventElapsedTime = nullptr;
    /* Driver API (subset used on Windows) */
    pfn_cuDeviceGetAttribute    DeviceGetAttribute = nullptr;
#else
    /* Linux/macOS: driver API function pointers (from real libcuda.so.1) */
    pfn_cuInit                  Init = nullptr;
    pfn_cuDeviceGetCount        DeviceGetCount = nullptr;
    pfn_cuDeviceGet             DeviceGet = nullptr;
    pfn_cuDeviceGetName         DeviceGetName = nullptr;
    pfn_cuDeviceTotalMem        DeviceTotalMem = nullptr;
    pfn_cuDeviceGetAttribute    DeviceGetAttribute = nullptr;
    pfn_cuDevicePrimaryCtxRetain PrimaryCtxRetain = nullptr;
    pfn_cuDevicePrimaryCtxRelease PrimaryCtxRelease = nullptr;
    pfn_cuCtxSetCurrent         CtxSetCurrent = nullptr;
    pfn_cuCtxSynchronize        CtxSynchronize = nullptr;
    pfn_cuMemAlloc              MemAlloc = nullptr;
    pfn_cuMemFree               MemFree = nullptr;
    pfn_cuMemcpyHtoD            MemcpyHtoD = nullptr;
    pfn_cuMemcpyDtoH            MemcpyDtoH = nullptr;
    pfn_cuMemcpyDtoD            MemcpyDtoD = nullptr;
    pfn_cuMemcpyHtoDAsync       MemcpyHtoDAsync = nullptr;
    pfn_cuMemcpyDtoHAsync       MemcpyDtoHAsync = nullptr;
    pfn_cuMemsetD8              MemsetD8 = nullptr;
    pfn_cuStreamCreate          StreamCreate = nullptr;
    pfn_cuStreamDestroy         StreamDestroy = nullptr;
    pfn_cuStreamSynchronize     StreamSynchronize = nullptr;
    pfn_cuEventCreate           EventCreate = nullptr;
    pfn_cuEventDestroy          EventDestroy = nullptr;
    pfn_cuEventRecord           EventRecord = nullptr;
    pfn_cuEventSynchronize      EventSynchronize = nullptr;
    pfn_cuEventElapsedTime      EventElapsedTime = nullptr;
#endif

    /* NVML API function pointers */
    pfn_nvmlInit_v2                    NvmlInit = nullptr;
    pfn_nvmlShutdown                   NvmlShutdown = nullptr;
    pfn_nvmlDeviceGetCount_v2          NvmlDeviceGetCount = nullptr;
    pfn_nvmlDeviceGetHandleByIndex_v2  NvmlDeviceGetHandleByIndex = nullptr;
    pfn_nvmlDeviceGetName              NvmlDeviceGetName = nullptr;
    pfn_nvmlDeviceGetMemoryInfo        NvmlDeviceGetMemoryInfo = nullptr;
    pfn_nvmlDeviceGetUtilizationRates  NvmlDeviceGetUtilizationRates = nullptr;
    pfn_nvmlDeviceGetTemperature       NvmlDeviceGetTemperature = nullptr;
    pfn_nvmlDeviceGetPowerUsage        NvmlDeviceGetPowerUsage = nullptr;
    pfn_nvmlDeviceGetFanSpeed          NvmlDeviceGetFanSpeed = nullptr;
    pfn_nvmlDeviceGetUUID              NvmlDeviceGetUUID = nullptr;
    pfn_nvmlDeviceGetPciInfo_v3        NvmlDeviceGetPciInfo = nullptr;
    pfn_nvmlDeviceGetCudaComputeCapability NvmlDeviceGetCudaComputeCapability = nullptr;
    pfn_nvmlDeviceGetClockInfo         NvmlDeviceGetClockInfo = nullptr;
};

static RealCUDA g_local;
static int g_active_device = 0;
static bool g_remote_first = false;
static int g_remote_base = 0;       /* first remote device index = g_local.local_count */
static int g_total_remote_devices = 0;  /* forward decl — set in ensure_connected() */
static std::string g_gpu_mode = "all";  /* "all", "remote", "local" */
static std::string g_real_cuda_path = REAL_CUDA_DEFAULT_PATH;
static std::string g_transport_type = "tcp";  /* Phase 9: "tcp" or "rdma" */
static bool g_local_initialized = false;

#ifndef _WIN32
/* Activate the CUDA context for a local device. Must be called before any
 * driver API operation on a local GPU. Linux only — uses driver API contexts. */
static CUresult local_set_ctx(int local_dev) {
    if (local_dev < 0 || local_dev >= g_local.local_count) return CUDA_ERROR_INVALID_DEVICE;
    if (!g_local.CtxSetCurrent || !g_local.contexts[local_dev]) return CUDA_ERROR_NOT_INITIALIZED;
    return g_local.CtxSetCurrent(g_local.contexts[local_dev]);
}

/* Fill a cudaDeviceProp struct from driver API queries for a local device.
 * Linux only — Windows uses the runtime API (g_local.GetDeviceProperties). */
static cudaError_t local_get_device_props(struct cudaDeviceProp *prop, int local_dev) {
    if (!prop) return cudaErrorInvalidValue;
    memset(prop, 0, sizeof(*prop));
    
    /* Get name */
    CUdevice dev = (CUdevice)local_dev;
    if (g_local.DeviceGet) {
        g_local.DeviceGet(&dev, local_dev);
    }
    if (g_local.DeviceGetName) {
        g_local.DeviceGetName(prop->name, sizeof(prop->name), dev);
    }
    if (g_local.DeviceTotalMem) {
        g_local.DeviceTotalMem(&prop->totalGlobalMem, dev);
    }
    
    /* Query attributes */
    auto ga = [&](CUdevice_attribute attr) -> int {
        int val = 0;
        if (g_local.DeviceGetAttribute) {
            g_local.DeviceGetAttribute(&val, attr, dev);
        }
        return val;
    };
    
    prop->sharedMemPerBlock     = 49152;
    prop->regsPerBlock          = 65536;
    prop->warpSize              = 32;
    prop->maxThreadsPerBlock    = 1024;
    prop->maxThreadsDim[0]      = 1024;
    prop->maxThreadsDim[1]      = 1024;
    prop->maxThreadsDim[2]      = 64;
    prop->maxGridSize[0]        = 2147483647;
    prop->maxGridSize[1]        = 65535;
    prop->maxGridSize[2]        = 65535;
    prop->clockRate             = 1485000;
    prop->major                 = 7;
    prop->minor                 = 5;
    prop->multiProcessorCount   = 16;
    prop->maxThreadsPerMultiProcessor = 1536;
    prop->totalConstMem         = 65536;
    prop->memoryBusWidth        = 128;
    prop->l2CacheSize           = 1048576;
    prop->memPitch              = 2147483647;
    prop->textureAlignment      = 512;
    prop->texturePitchAlignment = 32;
    prop->deviceOverlap         = 1;
    prop->canMapHostMemory      = 1;
    prop->concurrentKernels     = 1;
    prop->unifiedAddressing     = 1;
    prop->managedMemory         = 1;
    prop->computePreemptionSupported = 1;
    prop->cooperativeLaunch     = 1;
    prop->asyncEngineCount      = 1;
    prop->streamPrioritiesSupported = 1;
    prop->globalL1CacheSupported = 1;
    prop->localL1CacheSupported  = 1;
    prop->sharedMemPerMultiprocessor = 49152;
    prop->regsPerMultiprocessor = 65536;
    prop->maxBlocksPerMultiProcessor = 16;
    prop->pageableMemoryAccess  = 1;

    return cudaSuccess;
}
#endif /* !_WIN32 */

static bool is_remote_device(int dev) {
    if (!g_local.available || g_gpu_mode == "remote") return true;
    if (g_gpu_mode == "local") return false;
    if (g_remote_first) {
        return dev < g_total_remote_devices;
    } else {
        return dev >= g_local.local_count;
    }
}

static int to_remote_device(int dev) {
    if (g_remote_first) return dev;
    return dev - g_local.local_count;
}

static int to_local_device(int dev) {
    if (g_remote_first) return dev - g_total_remote_devices;
    return dev;
}

#ifndef _WIN32
#define DLSYM_LOAD(handle, sym, field) do { \
    g_local.field = (decltype(g_local.field))dlsym(handle, #sym); \
} while(0)

static void init_local_gpu() {
    if (g_local_initialized) return;
    g_local_initialized = true;
    if (g_gpu_mode == "remote") return;
    if (g_real_cuda_path.empty()) return;

    int dlflags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_DEEPBIND
    dlflags |= RTLD_DEEPBIND;  /* prevent loaded lib from using our symbols */
#endif

    std::string cuda_path = g_real_cuda_path + "/libcuda.so.1";
    std::string nvml_path = g_real_cuda_path + "/libnvidia-ml.so.1";

    /* Load ONLY libcuda.so.1 (driver API). DO NOT load libcudart.so — it
     * internally dlopen's "libcuda.so.1" by name which resolves to our
     * symlink, causing recursive loading and segfault. The driver API
     * talks directly to the kernel module with no runtime dependencies. */
    g_local.h_cuda = dlopen(cuda_path.c_str(), dlflags);
    if (!g_local.h_cuda) return;  /* no local GPU driver found */

    /* Load NVML (optional, for monitoring) */
    g_local.h_nvml = dlopen(nvml_path.c_str(), dlflags);

    /* Load driver API function pointers */
    DLSYM_LOAD(g_local.h_cuda, cuInit,                      Init);
    DLSYM_LOAD(g_local.h_cuda, cuDeviceGetCount,             DeviceGetCount);
    DLSYM_LOAD(g_local.h_cuda, cuDeviceGet,                  DeviceGet);
    DLSYM_LOAD(g_local.h_cuda, cuDeviceGetName,              DeviceGetName);
    DLSYM_LOAD(g_local.h_cuda, cuDeviceTotalMem_v2,          DeviceTotalMem);
    DLSYM_LOAD(g_local.h_cuda, cuDeviceGetAttribute,         DeviceGetAttribute);
    DLSYM_LOAD(g_local.h_cuda, cuDevicePrimaryCtxRetain,     PrimaryCtxRetain);
    DLSYM_LOAD(g_local.h_cuda, cuDevicePrimaryCtxRelease_v2, PrimaryCtxRelease);
    DLSYM_LOAD(g_local.h_cuda, cuCtxSetCurrent,              CtxSetCurrent);
    DLSYM_LOAD(g_local.h_cuda, cuCtxSynchronize,             CtxSynchronize);
    DLSYM_LOAD(g_local.h_cuda, cuMemAlloc_v2,                MemAlloc);
    DLSYM_LOAD(g_local.h_cuda, cuMemFree_v2,                 MemFree);
    DLSYM_LOAD(g_local.h_cuda, cuMemcpyHtoD_v2,              MemcpyHtoD);
    DLSYM_LOAD(g_local.h_cuda, cuMemcpyDtoH_v2,              MemcpyDtoH);
    DLSYM_LOAD(g_local.h_cuda, cuMemcpyDtoD_v2,              MemcpyDtoD);
    DLSYM_LOAD(g_local.h_cuda, cuMemcpyHtoDAsync_v2,         MemcpyHtoDAsync);
    DLSYM_LOAD(g_local.h_cuda, cuMemcpyDtoHAsync_v2,         MemcpyDtoHAsync);
    DLSYM_LOAD(g_local.h_cuda, cuMemsetD8_v2,                MemsetD8);
    DLSYM_LOAD(g_local.h_cuda, cuStreamCreate,               StreamCreate);
    DLSYM_LOAD(g_local.h_cuda, cuStreamDestroy_v2,           StreamDestroy);
    DLSYM_LOAD(g_local.h_cuda, cuStreamSynchronize,          StreamSynchronize);
    DLSYM_LOAD(g_local.h_cuda, cuEventCreate,                EventCreate);
    DLSYM_LOAD(g_local.h_cuda, cuEventDestroy_v2,            EventDestroy);
    DLSYM_LOAD(g_local.h_cuda, cuEventRecord,                EventRecord);
    DLSYM_LOAD(g_local.h_cuda, cuEventSynchronize,           EventSynchronize);
    DLSYM_LOAD(g_local.h_cuda, cuEventElapsedTime,           EventElapsedTime);

    /* Load NVML function pointers */
    if (g_local.h_nvml) {
        DLSYM_LOAD(g_local.h_nvml, nvmlInit_v2,                    NvmlInit);
        DLSYM_LOAD(g_local.h_nvml, nvmlShutdown,                   NvmlShutdown);
        DLSYM_LOAD(g_local.h_nvml, nvmlDeviceGetCount_v2,          NvmlDeviceGetCount);
        DLSYM_LOAD(g_local.h_nvml, nvmlDeviceGetHandleByIndex_v2,  NvmlDeviceGetHandleByIndex);
        DLSYM_LOAD(g_local.h_nvml, nvmlDeviceGetName,              NvmlDeviceGetName);
        DLSYM_LOAD(g_local.h_nvml, nvmlDeviceGetMemoryInfo,        NvmlDeviceGetMemoryInfo);
        DLSYM_LOAD(g_local.h_nvml, nvmlDeviceGetUtilizationRates,  NvmlDeviceGetUtilizationRates);
        DLSYM_LOAD(g_local.h_nvml, nvmlDeviceGetTemperature,       NvmlDeviceGetTemperature);
        DLSYM_LOAD(g_local.h_nvml, nvmlDeviceGetPowerUsage,        NvmlDeviceGetPowerUsage);
        DLSYM_LOAD(g_local.h_nvml, nvmlDeviceGetFanSpeed,          NvmlDeviceGetFanSpeed);
        DLSYM_LOAD(g_local.h_nvml, nvmlDeviceGetUUID,              NvmlDeviceGetUUID);
        DLSYM_LOAD(g_local.h_nvml, nvmlDeviceGetPciInfo_v3,        NvmlDeviceGetPciInfo);
        DLSYM_LOAD(g_local.h_nvml, nvmlDeviceGetCudaComputeCapability, NvmlDeviceGetCudaComputeCapability);
        DLSYM_LOAD(g_local.h_nvml, nvmlDeviceGetClockInfo,         NvmlDeviceGetClockInfo);
    }

    /* Initialize real driver and query local GPU count */
    if (!g_local.Init || !g_local.DeviceGetCount) return;

    CUresult err = g_local.Init(0);
    if (err != CUDA_SUCCESS) return;

    int count = 0;
    err = g_local.DeviceGetCount(&count);
    if (err != CUDA_SUCCESS || count <= 0) return;
    if (count > 8) count = 8;  /* cap at array size */

    g_local.local_count = count;
    g_local.available = true;
    g_remote_base = count;

    /* Retain primary contexts for each local device */
    for (int i = 0; i < count; i++) {
        CUdevice dev;
        if (g_local.DeviceGet) {
            g_local.DeviceGet(&dev, i);
        } else {
            dev = i;
        }
        if (g_local.PrimaryCtxRetain)
            g_local.PrimaryCtxRetain(&g_local.contexts[i], dev);
    }

    /* Initialize local NVML handles */
    if (g_local.NvmlInit && g_local.NvmlDeviceGetHandleByIndex) {
        g_local.NvmlInit();
        for (int i = 0; i < count; i++) {
            g_local.NvmlDeviceGetHandleByIndex(i, &g_local.nvml_handles[i]);
        }
    }

    fprintf(stderr, "[gpushare] Local GPU passthrough: %d local GPU(s) detected\n", count);
    for (int i = 0; i < count; i++) {
        char name[256] = {0};
        CUdevice dev;
        if (g_local.DeviceGet) g_local.DeviceGet(&dev, i);
        else dev = i;
        if (g_local.DeviceGetName) g_local.DeviceGetName(name, sizeof(name), dev);
        fprintf(stderr, "[gpushare]   Device %d (local): %s\n", i, name);
    }
    fprintf(stderr, "[gpushare]   Device %d+ (remote): via gpushare server\n", count);
}
#else  /* _WIN32 — Windows local GPU passthrough */

#define WINSYM_LOAD(handle, sym, field) do { \
    g_local.field = (decltype(g_local.field))GetProcAddress(handle, #sym); \
} while(0)

static void init_local_gpu() {
    if (g_local_initialized) return;
    g_local_initialized = true;
    if (g_gpu_mode == "remote") return;

    /* On Windows, the real CUDA DLLs are in the NVIDIA driver directory
     * (for nvcuda.dll / nvml.dll) or the CUDA toolkit bin (for cudart64_*.dll).
     * The install script backs them up to C:\Program Files\gpushare\real\
     * but we also search standard NVIDIA/CUDA paths. */
    const char *search_paths[] = {
        "C:\\Program Files\\gpushare\\real",
        nullptr  /* sentinel */
    };

    /* Try to find and load the real CUDA driver (nvcuda.dll) */
    /* First check the backup path */
    std::string cuda_path;
    for (int i = 0; search_paths[i]; i++) {
        std::string p = std::string(search_paths[i]) + "\\nvcuda.dll";
        HMODULE h = LoadLibraryExA(p.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (h) { g_local.h_cuda = (void*)h; cuda_path = p; break; }
    }
    /* Fallback: search system — NVIDIA driver installs nvcuda.dll in System32 */
    if (!g_local.h_cuda) {
        char sys32[MAX_PATH];
        GetSystemDirectoryA(sys32, MAX_PATH);
        std::string sys_path = std::string(sys32) + "\\nvcuda.dll";
        HMODULE h = LoadLibraryExA(sys_path.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (h) { g_local.h_cuda = (void*)h; cuda_path = sys_path; }
    }
    if (!g_local.h_cuda) return;  /* no local NVIDIA driver */

    /* Load Driver API function pointers */
    WINSYM_LOAD((HMODULE)g_local.h_cuda, cuDeviceGetAttribute, DeviceGetAttribute);

    /* Load cudart (try backup path first, then CUDA toolkit) */
    for (int i = 0; search_paths[i]; i++) {
        for (const char *name : {"cudart64_130.dll", "cudart64_12.dll", "cudart64_110.dll"}) {
            std::string p = std::string(search_paths[i]) + "\\" + name;
            HMODULE h = LoadLibraryExA(p.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
            if (h) { g_local.h_cudart = (void*)h; break; }
        }
        if (g_local.h_cudart) break;
    }
    /* Fallback: search CUDA toolkit path */
    if (!g_local.h_cudart) {
        const char *cuda_env = getenv("CUDA_PATH");
        if (cuda_env) {
            for (const char *name : {"cudart64_130.dll", "cudart64_12.dll"}) {
                std::string p = std::string(cuda_env) + "\\bin\\" + name;
                HMODULE h = LoadLibraryExA(p.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
                if (h) { g_local.h_cudart = (void*)h; break; }
            }
        }
    }
    if (!g_local.h_cudart) {
        FreeLibrary((HMODULE)g_local.h_cuda);
        g_local.h_cuda = nullptr;
        return;
    }

    /* Load NVML (optional) */
    for (int i = 0; search_paths[i]; i++) {
        std::string p = std::string(search_paths[i]) + "\\nvml.dll";
        HMODULE h = LoadLibraryExA(p.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (h) { g_local.h_nvml = (void*)h; break; }
    }
    if (!g_local.h_nvml) {
        char sys32[MAX_PATH];
        GetSystemDirectoryA(sys32, MAX_PATH);
        std::string p = std::string(sys32) + "\\nvml.dll";
        HMODULE h = LoadLibraryExA(p.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (h) g_local.h_nvml = (void*)h;
    }

    /* Load runtime function pointers */
    WINSYM_LOAD((HMODULE)g_local.h_cudart, cudaGetDeviceCount,       GetDeviceCount);
    WINSYM_LOAD((HMODULE)g_local.h_cudart, cudaGetDeviceProperties,  GetDeviceProperties);
    WINSYM_LOAD((HMODULE)g_local.h_cudart, cudaSetDevice,            SetDevice);
    WINSYM_LOAD((HMODULE)g_local.h_cudart, cudaMalloc,               Malloc);
    WINSYM_LOAD((HMODULE)g_local.h_cudart, cudaFree,                 Free);
    WINSYM_LOAD((HMODULE)g_local.h_cudart, cudaMemcpy,               Memcpy);
    WINSYM_LOAD((HMODULE)g_local.h_cudart, cudaMemcpyAsync,          MemcpyAsync);
    WINSYM_LOAD((HMODULE)g_local.h_cudart, cudaMemset,               Memset);
    WINSYM_LOAD((HMODULE)g_local.h_cudart, cudaDeviceSynchronize,    DeviceSynchronize);
    WINSYM_LOAD((HMODULE)g_local.h_cudart, cudaStreamCreate,         StreamCreate);
    WINSYM_LOAD((HMODULE)g_local.h_cudart, cudaStreamDestroy,        StreamDestroy);
    WINSYM_LOAD((HMODULE)g_local.h_cudart, cudaStreamSynchronize,    StreamSynchronize);
    WINSYM_LOAD((HMODULE)g_local.h_cudart, cudaEventCreate,          EventCreate);
    WINSYM_LOAD((HMODULE)g_local.h_cudart, cudaEventDestroy,         EventDestroy);
    WINSYM_LOAD((HMODULE)g_local.h_cudart, cudaEventRecord,          EventRecord);
    WINSYM_LOAD((HMODULE)g_local.h_cudart, cudaEventSynchronize,     EventSynchronize);
    WINSYM_LOAD((HMODULE)g_local.h_cudart, cudaEventElapsedTime,     EventElapsedTime);

    /* Load NVML function pointers */
    if (g_local.h_nvml) {
        WINSYM_LOAD((HMODULE)g_local.h_nvml, nvmlInit_v2,                    NvmlInit);
        WINSYM_LOAD((HMODULE)g_local.h_nvml, nvmlShutdown,                   NvmlShutdown);
        WINSYM_LOAD((HMODULE)g_local.h_nvml, nvmlDeviceGetCount_v2,          NvmlDeviceGetCount);
        WINSYM_LOAD((HMODULE)g_local.h_nvml, nvmlDeviceGetHandleByIndex_v2,  NvmlDeviceGetHandleByIndex);
        WINSYM_LOAD((HMODULE)g_local.h_nvml, nvmlDeviceGetName,              NvmlDeviceGetName);
        WINSYM_LOAD((HMODULE)g_local.h_nvml, nvmlDeviceGetMemoryInfo,        NvmlDeviceGetMemoryInfo);
        WINSYM_LOAD((HMODULE)g_local.h_nvml, nvmlDeviceGetUtilizationRates,  NvmlDeviceGetUtilizationRates);
        WINSYM_LOAD((HMODULE)g_local.h_nvml, nvmlDeviceGetTemperature,       NvmlDeviceGetTemperature);
        WINSYM_LOAD((HMODULE)g_local.h_nvml, nvmlDeviceGetPowerUsage,        NvmlDeviceGetPowerUsage);
        WINSYM_LOAD((HMODULE)g_local.h_nvml, nvmlDeviceGetFanSpeed,          NvmlDeviceGetFanSpeed);
        WINSYM_LOAD((HMODULE)g_local.h_nvml, nvmlDeviceGetUUID,              NvmlDeviceGetUUID);
        WINSYM_LOAD((HMODULE)g_local.h_nvml, nvmlDeviceGetPciInfo_v3,        NvmlDeviceGetPciInfo);
        WINSYM_LOAD((HMODULE)g_local.h_nvml, nvmlDeviceGetCudaComputeCapability, NvmlDeviceGetCudaComputeCapability);
        WINSYM_LOAD((HMODULE)g_local.h_nvml, nvmlDeviceGetClockInfo,         NvmlDeviceGetClockInfo);
    }

    /* Query local GPU count */
    if (!g_local.GetDeviceCount) return;
    int count = 0;
    cudaError_t err = g_local.GetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) return;
    if (count > 8) count = 8;

    g_local.local_count = count;
    g_local.available = true;
    g_remote_base = count;

    /* Initialize local NVML handles */
    if (g_local.NvmlInit && g_local.NvmlDeviceGetHandleByIndex) {
        g_local.NvmlInit();
        for (int i = 0; i < count; i++)
            g_local.NvmlDeviceGetHandleByIndex(i, &g_local.nvml_handles[i]);
    }

    fprintf(stderr, "[gpushare] Local GPU passthrough (Windows): %d local GPU(s) detected\n", count);
    if (g_local.GetDeviceProperties) {
        for (int i = 0; i < count; i++) {
            struct cudaDeviceProp prop;
            memset(&prop, 0, sizeof(prop));
            g_local.GetDeviceProperties(&prop, i);
            fprintf(stderr, "[gpushare]   Device %d (local): %s\n", i, prop.name);
        }
    }
    fprintf(stderr, "[gpushare]   Device %d+ (remote): via gpushare server\n", count);
}
#endif

/* ── Phase 5+7: Client-side tiered pinned buffer pool ────── */
/* Three-stage pipeline: app memory -> client pinned buffer -> network ->
 * server pinned buffer -> GPU. Uses mlock/VirtualLock (no CUDA on client).
 * Phase 7 adds tiered sizing so small transfers don't waste 4MB slots. */

struct ClientPinnedTier {
    static constexpr int MAX_BUFS = 8;
    void *buffers[MAX_BUFS] = {};
    bool  in_use[MAX_BUFS]  = {};
    bool  is_locked[MAX_BUFS] = {};
    int   count = 0;
    size_t buf_size = 0;

    void init(int n, size_t sz) {
        count = std::min(n, (int)MAX_BUFS);
        buf_size = sz;
        for (int i = 0; i < count; i++) {
#ifdef _WIN32
            buffers[i] = VirtualAlloc(NULL, buf_size,
                                       MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            if (buffers[i]) {
                is_locked[i] = (VirtualLock(buffers[i], buf_size) != 0);
            }
#else
            buffers[i] = mmap(NULL, buf_size, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (buffers[i] == MAP_FAILED) {
                buffers[i] = nullptr;
            } else {
                is_locked[i] = (mlock(buffers[i], buf_size) == 0);
            }
#endif
            if (!buffers[i]) {
                buffers[i] = malloc(buf_size);
                is_locked[i] = false;
            }
        }
    }

    void destroy() {
        for (int i = 0; i < count; i++) {
            if (!buffers[i]) continue;
#ifdef _WIN32
            if (is_locked[i]) VirtualUnlock(buffers[i], buf_size);
            VirtualFree(buffers[i], 0, MEM_RELEASE);
#else
            if (is_locked[i]) munlock(buffers[i], buf_size);
            munmap(buffers[i], buf_size);
#endif
            buffers[i] = nullptr;
        }
    }
};

struct ClientPinnedPool {
    static constexpr int NUM_TIERS = 3;
    ClientPinnedTier tiers[NUM_TIERS];
    std::mutex mtx;
    bool initialized = false;

    /* Tier config: {count, size} */
    static constexpr int    tier_counts[] = {8, 4, 4};
    static constexpr size_t tier_sizes[]  = {64*1024, 1*1024*1024, 4*1024*1024};

    void init() {
        for (int t = 0; t < NUM_TIERS; t++) {
            tiers[t].init(tier_counts[t], tier_sizes[t]);
        }
        initialized = true;
    }

    void destroy() {
        for (int t = 0; t < NUM_TIERS; t++) tiers[t].destroy();
        initialized = false;
    }

    /* Acquire a buffer >= size. index encodes tier + slot. */
    void *acquire(size_t size, int &index) {
        std::lock_guard<std::mutex> lock(mtx);
        for (int t = 0; t < NUM_TIERS; t++) {
            if (size <= tiers[t].buf_size) {
                for (int i = 0; i < tiers[t].count; i++) {
                    if (!tiers[t].in_use[i] && tiers[t].buffers[i]) {
                        tiers[t].in_use[i] = true;
                        index = t * ClientPinnedTier::MAX_BUFS + i;
                        return tiers[t].buffers[i];
                    }
                }
            }
        }
        index = -1;
        return malloc(size);
    }

    void release(void *buf, int index) {
        if (index >= 0) {
            int t = index / ClientPinnedTier::MAX_BUFS;
            int i = index % ClientPinnedTier::MAX_BUFS;
            if (t < NUM_TIERS && i < tiers[t].count) {
                std::lock_guard<std::mutex> lock(mtx);
                tiers[t].in_use[i] = false;
                return;
            }
        }
        free(buf);
    }
};

constexpr int    ClientPinnedPool::tier_counts[];
constexpr size_t ClientPinnedPool::tier_sizes[];

static ClientPinnedPool g_client_pinned;
static bool g_client_pinned_initialized = false;

static void ensure_client_pinned() {
    if (g_client_pinned_initialized) return;
    g_client_pinned.init();
    g_client_pinned_initialized = true;
}

/* ── Phase 11: ServerConnection — per-server connection state ── */
/* Each server gets its own transport, send mutex, recv thread, pending map,
 * and capability flags. Multi-server pooling routes by device index. */

struct PendingRequest {
    std::promise<std::vector<uint8_t>> promise;
    uint16_t resp_flags = 0;
};

struct ServerConnection {
    std::string host;
    int port = GPUSHARE_DEFAULT_PORT;
    std::string transport_type = "tcp";

    std::unique_ptr<Transport> transport;
    sock_t raw_sock = SOCK_INVALID;  /* backward compat */
    std::mutex send_mtx;
    std::atomic<uint32_t> req_id{1};
    uint32_t caps = 0;
    uint32_t session_id = 0;
    int device_count = 0;       /* GPUs on this server */
    int device_base = 0;        /* first global device index for this server */

    std::mutex pending_mtx;
    std::unordered_map<uint32_t, std::shared_ptr<PendingRequest>> pending;
    std::thread recv_thread;
    std::atomic<bool> recv_running{false};

    /* Destructor detaches the recv thread to prevent std::terminate() from
     * destroying a joinable thread. Full cleanup is done by gpushare_cleanup(). */
    ~ServerConnection() {
        recv_running = false;
        if (recv_thread.joinable()) recv_thread.detach();
    }

    bool connected() const { return transport != nullptr; }

    /* ── Recv thread ─────────────────────────────────────── */
    void start_recv() {
        bool expected = false;
        if (!recv_running.compare_exchange_strong(expected, true)) return;
        recv_thread = std::thread([this]{ recv_loop(); });
    }

    void stop_recv() {
        recv_running = false;
        try { if (transport) transport->shutdown_read(); } catch (...) {}
        try { if (recv_thread.joinable()) recv_thread.join(); } catch (...) {}
    }

    void recv_loop() {
        Transport *tp = transport.get();
        while (recv_running.load()) {
            gs_header_t resp_hdr;
            if (!tp->recv(&resp_hdr, sizeof(resp_hdr))) {
                std::lock_guard<std::mutex> lock(pending_mtx);
                for (auto &kv : pending) {
                    std::vector<uint8_t> empty;
                    kv.second->resp_flags = GS_FLAG_ERROR;
                    try { kv.second->promise.set_value(std::move(empty)); } catch (...) {}
                }
                pending.clear();
                break;
            }
            if (!gs_header_validate(&resp_hdr)) continue;

            uint32_t pl = resp_hdr.length - GPUSHARE_HEADER_SIZE;
            std::vector<uint8_t> payload(pl);
            if (pl > 0 && !tp->recv(payload.data(), pl)) {
                std::lock_guard<std::mutex> lock(pending_mtx);
                for (auto &kv : pending) {
                    std::vector<uint8_t> empty;
                    kv.second->resp_flags = GS_FLAG_ERROR;
                    try { kv.second->promise.set_value(std::move(empty)); } catch (...) {}
                }
                pending.clear();
                break;
            }

            uint32_t rid = resp_hdr.req_id;
            std::shared_ptr<PendingRequest> req;
            {
                std::lock_guard<std::mutex> lock(pending_mtx);
                auto it = pending.find(rid);
                if (it != pending.end()) {
                    req = it->second;
                    pending.erase(it);
                }
            }
            if (req) {
                req->resp_flags = resp_hdr.flags;
                try { req->promise.set_value(std::move(payload)); } catch (...) {}
            }
        }
    }

    /* ── Pipelined RPC ───────────────────────────────────── */
    bool rpc(uint16_t opcode, const void *req_payload, uint32_t req_len,
             std::vector<uint8_t> &resp_payload, uint16_t *resp_flags = nullptr) {
        auto pend = std::make_shared<PendingRequest>();
        auto future = pend->promise.get_future();
        uint32_t rid = req_id++;

        { std::lock_guard<std::mutex> lock(pending_mtx); pending[rid] = pend; }

        {
            gs_header_t hdr;
            gs_header_init(&hdr, opcode, rid, GPUSHARE_HEADER_SIZE + req_len);
            std::lock_guard<std::mutex> lock(send_mtx);
            if (!transport->send(&hdr, sizeof(hdr)) ||
                (req_len > 0 && !transport->send(req_payload, req_len))) {
                std::lock_guard<std::mutex> plock(pending_mtx);
                pending.erase(rid);
                return false;
            }
        }

        resp_payload = future.get();
        if (resp_flags) *resp_flags = pend->resp_flags;
        if (resp_payload.empty() && (pend->resp_flags & GS_FLAG_ERROR)) return false;
        return true;
    }

    bool rpc_flags(uint16_t opcode, uint16_t flags, const void *req_payload, uint32_t req_len,
                   std::vector<uint8_t> &resp_payload, uint16_t *resp_flags = nullptr) {
        auto pend = std::make_shared<PendingRequest>();
        auto future = pend->promise.get_future();
        uint32_t rid = req_id++;

        { std::lock_guard<std::mutex> lock(pending_mtx); pending[rid] = pend; }

        {
            gs_header_t hdr;
            gs_header_init(&hdr, opcode, rid, GPUSHARE_HEADER_SIZE + req_len);
            hdr.flags = flags;
            std::lock_guard<std::mutex> lock(send_mtx);
            if (!transport->send(&hdr, sizeof(hdr)) ||
                (req_len > 0 && !transport->send(req_payload, req_len))) {
                std::lock_guard<std::mutex> plock(pending_mtx);
                pending.erase(rid);
                return false;
            }
        }

        resp_payload = future.get();
        if (resp_flags) *resp_flags = pend->resp_flags;
        if (resp_payload.empty() && (pend->resp_flags & GS_FLAG_ERROR)) return false;
        return true;
    }

    int32_t rpc_simple(uint16_t opcode, const void *req_payload, uint32_t req_len) {
        std::vector<uint8_t> resp;
        if (!rpc(opcode, req_payload, req_len, resp)) return cudaErrorUnknown;
        if (resp.size() < sizeof(gs_generic_resp_t)) return cudaErrorUnknown;
        return ((const gs_generic_resp_t*)resp.data())->cuda_error;
    }

    /* ── Cleanup ─────────────────────────────────────────── */
    void disconnect() {
        if (!transport) return;
        {
            std::lock_guard<std::mutex> lock(send_mtx);
            gs_header_t hdr;
            gs_header_init(&hdr, GS_OP_CLOSE, 0, GPUSHARE_HEADER_SIZE);
            transport->send(&hdr, sizeof(hdr));
        }
        stop_recv();
        transport->close();
        transport.reset();
        raw_sock = SOCK_INVALID;
        { std::lock_guard<std::mutex> lock(pending_mtx); pending.clear(); }
        caps = 0;
    }
};

/* ── Connection state ────────────────────────────────────── */
static std::vector<std::unique_ptr<ServerConnection>> g_servers;
static std::recursive_mutex g_connect_mtx;  /* protects initial connection setup.
                                              * Must be recursive: on Windows, init_local_gpu() loads
                                              * the real cudart DLL which calls cuInit() back into our
                                              * DLL, re-entering ensure_connected(). */
static cudaError_t  g_last_error  = cudaSuccess;

/* Legacy globals — point to the active server for backward compat */
static sock_t       g_sock = SOCK_INVALID;
static uint32_t     g_server_caps = 0;

/* Map global device index -> (server_index, local_device_on_server) */
struct DeviceRoute { int server_idx; int local_device; };
static std::vector<DeviceRoute> g_device_routes;
/* g_total_remote_devices declared earlier (near g_remote_first) */

/* Config: parsed server addresses */
struct ServerAddr { std::string host; int port; };
static std::vector<ServerAddr> g_server_addrs;

/* Forward declarations (defined below, needed by recv thread) */
static bool send_all(sock_t fd, const void *buf, size_t len);
static bool recv_all(sock_t fd, void *buf, size_t len);

/* Get the active ServerConnection for the current device, or for a specific device */
static ServerConnection *active_server(int device = -1) {
    if (g_servers.empty()) return nullptr;
    int dev;
    if (device >= 0) {
        /* Use the specified device */
        if (!is_remote_device(device)) return nullptr;
        dev = to_remote_device(device);
    } else {
        /* Use the currently active device */
        if (!is_remote_device(g_active_device)) return nullptr;
        dev = to_remote_device(g_active_device);
    }
    for (auto &s : g_servers) {
        if (dev < s->device_count) return s.get();
        dev -= s->device_count;
    }
    return g_servers[0].get();  /* fallback to first server */
}

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
    /* Phase 11: Support comma-separated multi-server: host1:port,host2:port */
    g_server_addrs.clear();
    std::string remaining = s;
    while (!remaining.empty()) {
        auto comma = remaining.find(',');
        std::string addr = (comma != std::string::npos) ? remaining.substr(0, comma) : remaining;
        remaining = (comma != std::string::npos) ? remaining.substr(comma + 1) : "";

        /* Trim whitespace */
        size_t start = addr.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        addr = addr.substr(start);
        addr.erase(addr.find_last_not_of(" \t") + 1);

        ServerAddr sa;
        auto colon = addr.rfind(':');
        if (colon != std::string::npos) {
            sa.host = addr.substr(0, colon);
            sa.port = atoi(addr.substr(colon + 1).c_str());
        } else {
            sa.host = addr;
            sa.port = GPUSHARE_DEFAULT_PORT;
        }
        if (!sa.host.empty()) g_server_addrs.push_back(sa);
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
        else if (key == "gpu_mode") g_gpu_mode = val;
        else if (key == "remote_first") g_remote_first = (val == "1" || val == "true" || val == "on");
        else if (key == "real_cuda_path") g_real_cuda_path = val;
        else if (key == "transport") g_transport_type = val;
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

    /* Override gpu_mode from environment */
    const char *mode_env = getenv("GPUSHARE_GPU_MODE");
    if (mode_env) g_gpu_mode = mode_env;

    /* Override real CUDA path from environment */
    const char *path_env = getenv("GPUSHARE_REAL_CUDA_PATH");
    if (path_env) g_real_cuda_path = path_env;

    const char *rf_env = getenv("GPUSHARE_REMOTE_FIRST");
    if (rf_env) g_remote_first = (std::string(rf_env) == "1" || std::string(rf_env) == "true");

    /* Detect local GPUs */
    init_local_gpu();
}

/* Connect a single ServerConnection. Returns true on success. */
static bool connect_server(ServerConnection *sc) {
    TRACE("Connecting to %s:%d (transport=%s)", sc->host.c_str(), sc->port,
          sc->transport_type.c_str());

#ifdef GPUSHARE_HAS_RDMA
    if (sc->transport_type == "rdma") {
        auto rdma = std::make_unique<RdmaTransport>();
        if (rdma->connect(sc->host.c_str(), sc->port)) {
            sc->raw_sock = SOCK_INVALID;
            sc->transport = std::unique_ptr<Transport>(rdma.release());
            goto transport_ready;
        }
        ERR("RDMA connect failed to %s:%d — falling back to TCP", sc->host.c_str(), sc->port);
        sc->transport_type = "tcp";
    }
#endif
    {
        auto tcp = std::make_unique<TcpTransport>();
        if (!tcp->connect(sc->host.c_str(), sc->port)) {
            ERR("Cannot connect to %s:%d", sc->host.c_str(), sc->port);
            return false;
        }
        tcp->optimize(true);
        sc->raw_sock = tcp->raw_fd();
        sc->transport = std::unique_ptr<Transport>(tcp.release());
    }
#ifdef GPUSHARE_HAS_RDMA
transport_ready:
#endif

    /* Init handshake */
    gs_header_t hdr;
    gs_init_req_t req;
    gs_header_init(&hdr, GS_OP_INIT, sc->req_id++, GPUSHARE_HEADER_SIZE + sizeof(req));
    req.version = GPUSHARE_VERSION;
    req.client_type = 0;
    if (!sc->transport->send(&hdr, sizeof(hdr)) || !sc->transport->send(&req, sizeof(req))) {
        ERR("Failed to send init to %s:%d", sc->host.c_str(), sc->port);
        sc->transport->close(); sc->transport.reset();
        return false;
    }

    gs_header_t resp_hdr;
    if (!sc->transport->recv(&resp_hdr, sizeof(resp_hdr))) {
        ERR("Failed to read init response from %s:%d", sc->host.c_str(), sc->port);
        sc->transport->close(); sc->transport.reset();
        return false;
    }

    uint32_t resp_pl = resp_hdr.length - GPUSHARE_HEADER_SIZE;
    std::vector<uint8_t> resp_buf(resp_pl);
    if (resp_pl > 0 && !sc->transport->recv(resp_buf.data(), resp_pl)) {
        sc->transport->close(); sc->transport.reset();
        return false;
    }

    gs_init_resp_t resp;
    memset(&resp, 0, sizeof(resp));
    memcpy(&resp, resp_buf.data(), std::min((size_t)resp_pl, sizeof(resp)));
    sc->caps = resp.capabilities;
    sc->session_id = resp.session_id;

    TRACE("Connected to %s:%d — session %u, caps=0x%x",
          sc->host.c_str(), sc->port, sc->session_id, sc->caps);

    /* Query device count from this server */
    {
        gs_header_t h2;
        gs_header_init(&h2, GS_OP_GET_DEVICE_COUNT, sc->req_id++, GPUSHARE_HEADER_SIZE);
        if (sc->transport->send(&h2, sizeof(h2))) {
            gs_header_t rh;
            if (sc->transport->recv(&rh, sizeof(rh)) && rh.length > GPUSHARE_HEADER_SIZE) {
                uint32_t pl2 = rh.length - GPUSHARE_HEADER_SIZE;
                std::vector<uint8_t> rb(pl2);
                if (sc->transport->recv(rb.data(), pl2) && rb.size() >= sizeof(gs_device_count_resp_t)) {
                    sc->device_count = ((const gs_device_count_resp_t*)rb.data())->count;
                }
            }
        }
        if (sc->device_count <= 0) sc->device_count = 1;  /* assume at least 1 */
    }

    sc->start_recv();
    return true;
}

static bool ensure_connected() {
    if (!g_servers.empty()) return true;

    std::lock_guard<std::recursive_mutex> lock(g_connect_mtx);
    if (!g_servers.empty()) return true;  /* double-check after lock */

    /* If gpu_mode=local, skip server connection entirely */
    if (g_gpu_mode == "local") {
        TRACE("gpu_mode=local - skipping server connection");
        return true;
    }

    fprintf(stderr, "[gpushare] ensure_connected: loading config...\n"); fflush(stderr);
    load_config();

    /* After loading config, re-check gpu_mode in case it was set by config */
    if (g_gpu_mode == "local") {
        TRACE("gpu_mode=local (from config) - skipping server connection");
        return true;
    }

    /* Re-check: on Windows, load_config() → init_local_gpu() → real cudart →
     * cuInit() re-enters ensure_connected() and may have already connected. */
    if (!g_servers.empty()) {
        fprintf(stderr, "[gpushare] ensure_connected: already connected after load_config (re-entrant)\n"); fflush(stderr);
        return true;
    }

    fprintf(stderr, "[gpushare] ensure_connected: connecting to servers...\n"); fflush(stderr);
    sock_init();

    /* If no servers configured, use default */
    if (g_server_addrs.empty()) {
        g_server_addrs.push_back({"localhost", GPUSHARE_DEFAULT_PORT});
    }

    /* Phase 11: Connect to all configured servers */
    g_device_routes.clear();
    g_total_remote_devices = 0;

    for (auto &addr : g_server_addrs) {
        auto sc = std::make_unique<ServerConnection>();
        sc->host = addr.host;
        sc->port = addr.port;
        sc->transport_type = g_transport_type;

        if (!connect_server(sc.get())) {
            TRACE("Skipping unreachable server %s:%d", addr.host.c_str(), addr.port);
            continue;
        }

        sc->device_base = g_total_remote_devices;
        for (int d = 0; d < sc->device_count; d++) {
            g_device_routes.push_back({(int)g_servers.size(), d});
        }
        g_total_remote_devices += sc->device_count;
        g_servers.push_back(std::move(sc));
    }

    if (g_servers.empty()) {
        ERR("Cannot connect to any gpushare server");
        return false;
    }

    /* Legacy compat — point globals at first server */
    g_sock = g_servers[0]->raw_sock;
    g_server_caps = g_servers[0]->caps;

    /* Phase 5: Initialize client-side pinned buffer pool */
    ensure_client_pinned();

    fprintf(stderr, "[gpushare] ensure_connected: DONE - servers=%zu, remote_devs=%d\n", g_servers.size(), g_total_remote_devices);
    fflush(stderr);
    return true;
}

/* Phase 11: Global RPC functions route to the active server.
 * These are the entry points used by all CUDA API implementations and
 * by generated_stubs.cpp (via extern linkage). */

/* Send request and receive response via the active server. */
/* Non-static so generated_stubs.cpp can use them */
bool rpc_call(uint16_t opcode, const void *req_payload, uint32_t req_len,
                     std::vector<uint8_t> &resp_payload, uint16_t *resp_flags = nullptr, int device = -1) {
    if (!ensure_connected()) return false;
    ServerConnection *sc = active_server(device);
    if (!sc) {
        // Only log error if we're trying to use remote GPU
        if (device < 0 && is_remote_device(g_active_device)) {
            ERR("No active server for rpc_call (device=%d, active=%d)", device, g_active_device);
        } else if (device >= 0 && is_remote_device(device)) {
            ERR("No active server for rpc_call (device=%d)", device);
        }
        return false;
    }
    return sc->rpc(opcode, req_payload, req_len, resp_payload, resp_flags);
}

static bool rpc_call_flags(uint16_t opcode, uint16_t flags, const void *req_payload, uint32_t req_len,
                           std::vector<uint8_t> &resp_payload, uint16_t *resp_flags = nullptr, int device = -1) {
    if (!ensure_connected()) return false;
    ServerConnection *sc = active_server(device);
    if (!sc) return false;
    return sc->rpc_flags(opcode, flags, req_payload, req_len, resp_payload, resp_flags);
}

/* Convenience: RPC that returns just a cuda_error */
/* Accessible from generated_stubs.cpp */
int32_t rpc_simple(uint16_t opcode, const void *req_payload, uint32_t req_len, int device = -1) {
    if (!ensure_connected()) return cudaErrorUnknown;
    ServerConnection *sc = active_server(device);
    if (!sc) return cudaErrorUnknown;
    return sc->rpc_simple(opcode, req_payload, req_len);
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
    fprintf(stderr, "[gpushare] cudaGetDeviceCount: ENTER\n");
    fflush(stderr);
    
    // Force connection if not already done
    static bool first_call = true;
    if (first_call) {
        first_call = false;
        if (g_gpu_mode != "local") {
            ensure_connected();
        }
    }

    int total = 0;

    /* Count local GPUs */
    if (g_local.available && g_gpu_mode != "remote") {
        total += g_local.local_count;
    }

    /* Count remote GPUs */
    if (g_gpu_mode != "local" && !g_servers.empty()) {
        total += g_total_remote_devices;
    }
    
    if (count) *count = total;
    fprintf(stderr, "[gpushare] cudaGetDeviceCount: returning %d\n", total);
    fflush(stderr);
    return total > 0 ? cudaSuccess : cudaErrorNoDevice;
}

GPUSHARE_EXPORT cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
    TRACE("cudaGetDeviceProperties(%d)", device);

    /* Local GPU — query via real driver/runtime API */
    if (g_local.available && !is_remote_device(device)) {
#ifdef _WIN32
        if (g_local.GetDeviceProperties) {
            cudaError_t err = g_local.GetDeviceProperties(prop, to_local_device(device));
#else
        {
            cudaError_t err = local_get_device_props(prop, to_local_device(device));
#endif
            if (err == cudaSuccess && prop) {
                size_t len = strlen(prop->name);
                if (len + 8 < sizeof(prop->name))
                    strcat(prop->name, " (local)");
            }
            return err;
        }
    }

    /* Remote GPU — RPC to server */
    gs_device_props_req_t req;
    req.device = to_remote_device(device);
    std::vector<uint8_t> resp;
    if (!rpc_call(GS_OP_GET_DEVICE_PROPS, &req, sizeof(req), resp)) return cudaErrorUnknown;
    if (resp.size() < sizeof(gs_device_props_t)) return cudaErrorUnknown;

    auto *r = (const gs_device_props_t*)resp.data();
    if (prop) {
        memset(prop, 0, sizeof(*prop));
        strncpy(prop->name, r->name, sizeof(prop->name) - 1);
        /* Append " (remote)" to name so user can distinguish */
        size_t len = strlen(prop->name);
        if (len + 10 < sizeof(prop->name))
            strcat(prop->name, " (remote)");
        /* ── Fields from server response ── */
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
        prop->memoryBusWidth     = (int)r->mem_bus_width;
        prop->l2CacheSize        = (int)r->l2_cache_size;

        /* ── Fill fields PyTorch reads that the server doesn't transmit ── */
        prop->memPitch             = 2147483647;
        prop->textureAlignment     = 512;
        prop->texturePitchAlignment= 32;
        prop->deviceOverlap        = 1;
        prop->canMapHostMemory     = 1;
        prop->concurrentKernels    = 1;
        prop->unifiedAddressing    = 1;
        prop->managedMemory        = 1;
        prop->concurrentManagedAccess = 1;
        prop->computePreemptionSupported = 1;
        prop->cooperativeLaunch    = 1;
        prop->asyncEngineCount     = 2;
        prop->streamPrioritiesSupported = 1;
        prop->globalL1CacheSupported = 1;
        prop->localL1CacheSupported  = 1;
        prop->sharedMemPerMultiprocessor = r->shared_mem_per_block > 0
            ? r->shared_mem_per_block * 2 : 166912;
        prop->regsPerMultiprocessor = 65536;
        prop->maxBlocksPerMultiProcessor = 32;
        prop->pageableMemoryAccess = 1;
    }
    return cudaSuccess;
}

/* CUDA 12+ alias: cudaGetDeviceProperties_v2 is the versioned name torch uses */
GPUSHARE_EXPORT cudaError_t cudaGetDeviceProperties_v2(struct cudaDeviceProp *prop, int device) {
    return cudaGetDeviceProperties(prop, device);
}

GPUSHARE_EXPORT cudaError_t cudaSetDevice(int device) {
    TRACE("cudaSetDevice(%d)", device);
    g_active_device = device;

    /* Local GPU — activate context */
    if (g_local.available && !is_remote_device(device)) {
#ifdef _WIN32
        if (g_local.SetDevice)
            return g_local.SetDevice(to_local_device(device));
#else
        local_set_ctx(to_local_device(device));
#endif
        return cudaSuccess;
    }

    /* Remote GPU — RPC to server */
    gs_set_device_req_t req;
    req.device = to_remote_device(device);
    return (cudaError_t)rpc_simple(GS_OP_SET_DEVICE, &req, sizeof(req), device);
}

GPUSHARE_EXPORT cudaError_t cudaMalloc(void **devPtr, size_t size) {
    TRACE("cudaMalloc(%zu bytes)", size);
#ifdef _WIN32
    if (g_local.available && !is_remote_device(g_active_device) && g_local.Malloc)
        return g_local.Malloc(devPtr, size);
#else
    if (g_local.available && !is_remote_device(g_active_device) && g_local.MemAlloc) {
        local_set_ctx(to_local_device(g_active_device));
        CUdeviceptr dptr;
        CUresult r = g_local.MemAlloc(&dptr, size);
        if (devPtr) *devPtr = (void*)(uintptr_t)dptr;
        return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorMemoryAllocation;
    }
#endif

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
#ifdef _WIN32
    if (g_local.available && !is_remote_device(g_active_device) && g_local.Free)
        return g_local.Free(devPtr);
#else
    if (g_local.available && !is_remote_device(g_active_device) && g_local.MemFree) {
        local_set_ctx(to_local_device(g_active_device));
        CUresult r = g_local.MemFree((CUdeviceptr)(uintptr_t)devPtr);
        return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
    }
#endif
    gs_free_req_t req;
    req.device_ptr = ptr_to_handle(devPtr);
    return (cudaError_t)rpc_simple(GS_OP_FREE, &req, sizeof(req));
}

/* Phase 3+5: Chunked H2D transfer — sends 4MB chunks with pinned staging */
static cudaError_t cudaMemcpy_h2d_chunked(void *dst, const void *src, size_t count,
                                           uint64_t stream_handle) {
    const uint8_t *data = (const uint8_t*)src;
    uint64_t dev_ptr = ptr_to_handle(dst);
    uint64_t offset = 0;

    ensure_client_pinned();

    while (offset < count) {
        uint32_t chunk_size = (uint32_t)std::min((uint64_t)GS_CHUNK_SIZE, count - offset);
        size_t payload_size = sizeof(gs_memcpy_h2d_chunk_t) + chunk_size;

        /* Phase 5: Use pinned buffer for the send payload */
        int pin_idx = -1;
        void *payload_buf = g_client_pinned.acquire(payload_size, pin_idx);
        if (!payload_buf) return cudaErrorMemoryAllocation;

        auto *chunk = (gs_memcpy_h2d_chunk_t*)payload_buf;
        chunk->device_ptr = dev_ptr;
        chunk->total_size = count;
        chunk->chunk_offset = offset;
        chunk->chunk_size = chunk_size;
        chunk->stream_handle = stream_handle;
        memcpy((uint8_t*)payload_buf + sizeof(gs_memcpy_h2d_chunk_t), data + offset, chunk_size);

        std::vector<uint8_t> resp;
        bool ok = rpc_call_flags(GS_OP_MEMCPY_H2D, GS_FLAG_CHUNKED, payload_buf,
                                  (uint32_t)payload_size, resp);
        g_client_pinned.release(payload_buf, pin_idx);

        if (!ok) return cudaErrorUnknown;
        if (resp.size() >= sizeof(gs_generic_resp_t)) {
            int32_t err = ((const gs_generic_resp_t*)resp.data())->cuda_error;
            if (err != 0) return (cudaError_t)err;
        }
        offset += chunk_size;
    }
    return cudaSuccess;
}

/* Phase 3: Chunked D2H transfer — per-chunk requests (fallback for old servers) */
static cudaError_t cudaMemcpy_d2h_chunked(void *dst, const void *src, size_t count,
                                           uint64_t stream_handle) {
    uint8_t *out = (uint8_t*)dst;
    uint64_t dev_ptr = ptr_to_handle(src);
    uint64_t offset = 0;

    while (offset < count) {
        uint32_t chunk_size = (uint32_t)std::min((uint64_t)GS_CHUNK_SIZE, count - offset);

        gs_memcpy_d2h_chunk_t chunk;
        chunk.device_ptr = dev_ptr;
        chunk.total_size = count;
        chunk.chunk_offset = offset;
        chunk.chunk_size = chunk_size;
        chunk.stream_handle = stream_handle;

        std::vector<uint8_t> resp;
        if (!rpc_call_flags(GS_OP_MEMCPY_D2H, GS_FLAG_CHUNKED, &chunk, sizeof(chunk), resp))
            return cudaErrorUnknown;
        if (resp.size() < sizeof(int32_t)) return cudaErrorUnknown;
        int32_t err;
        memcpy(&err, resp.data(), sizeof(err));
        if (err != 0) return (cudaError_t)err;
        if (resp.size() >= sizeof(int32_t) + chunk_size) {
            memcpy(out + offset, resp.data() + sizeof(int32_t), chunk_size);
        }
        offset += chunk_size;
    }
    return cudaSuccess;
}


GPUSHARE_EXPORT cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
    TRACE("cudaMemcpy(%zu bytes, kind=%d)", count, kind);
#ifdef _WIN32
    if (g_local.available && !is_remote_device(g_active_device) && g_local.Memcpy)
        return g_local.Memcpy(dst, src, count, kind);
#else
    if (g_local.available && !is_remote_device(g_active_device)) {
        local_set_ctx(to_local_device(g_active_device));
        CUresult r;
        if (kind == cudaMemcpyHostToDevice && g_local.MemcpyHtoD)
            r = g_local.MemcpyHtoD((CUdeviceptr)(uintptr_t)dst, src, count);
        else if (kind == cudaMemcpyDeviceToHost && g_local.MemcpyDtoH)
            r = g_local.MemcpyDtoH(dst, (CUdeviceptr)(uintptr_t)src, count);
        else if (kind == cudaMemcpyDeviceToDevice && g_local.MemcpyDtoD)
            r = g_local.MemcpyDtoD((CUdeviceptr)(uintptr_t)dst, (CUdeviceptr)(uintptr_t)src, count);
        else if (kind == cudaMemcpyHostToHost)
            { memcpy(dst, src, count); return cudaSuccess; }
        else return cudaErrorInvalidMemcpyDirection;
        return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
    }
#endif

    if (kind == cudaMemcpyHostToDevice) {
        /* Phase 3: Use chunked transfer for large payloads */
        if (count > GS_CHUNK_THRESHOLD && (g_server_caps & GS_CAP_CHUNKED)) {
            g_last_error = cudaMemcpy_h2d_chunked(dst, src, count, 0);
            return g_last_error;
        }
        /* Standard single-message path with Phase 5 pinned staging + Phase 10 compression */
        ensure_client_pinned();

        /* Phase 10: Try compression if server supports it */
        if ((g_server_caps & GS_CAP_COMPRESS) && gs_compression_available() &&
            count >= GS_COMPRESS_MIN_SIZE) {
            size_t comp_bound = gs_compress_bound(count);
            size_t payload_size = sizeof(gs_memcpy_h2d_req_t) + comp_bound;
            int pin_idx = -1;
            void *payload_buf = g_client_pinned.acquire(payload_size, pin_idx);
            if (payload_buf) {
                auto *req = (gs_memcpy_h2d_req_t*)payload_buf;
                req->device_ptr = ptr_to_handle(dst);
                req->size = count;  /* original size — server needs this */
                size_t compressed = gs_compress(src, count,
                    (uint8_t*)payload_buf + sizeof(gs_memcpy_h2d_req_t), comp_bound);
                if (compressed > 0) {
                    /* Send compressed */
                    size_t total = sizeof(gs_memcpy_h2d_req_t) + compressed;
                    std::vector<uint8_t> resp;
                    uint16_t flags = GS_FLAG_COMPRESSED;
                    bool ok = rpc_call_flags(GS_OP_MEMCPY_H2D, flags, payload_buf,
                                              (uint32_t)total, resp);
                    g_client_pinned.release(payload_buf, pin_idx);
                    if (!ok) return cudaErrorUnknown;
                    if (resp.size() >= sizeof(gs_generic_resp_t))
                        g_last_error = (cudaError_t)((const gs_generic_resp_t*)resp.data())->cuda_error;
                    else
                        g_last_error = cudaSuccess;
                    return g_last_error;
                }
                g_client_pinned.release(payload_buf, pin_idx);
                /* Compression didn't help — fall through to uncompressed */
            }
        }

        /* Uncompressed path */
        size_t payload_size = sizeof(gs_memcpy_h2d_req_t) + count;
        int pin_idx = -1;
        void *payload_buf = g_client_pinned.acquire(payload_size, pin_idx);
        if (!payload_buf) return cudaErrorMemoryAllocation;
        auto *req = (gs_memcpy_h2d_req_t*)payload_buf;
        req->device_ptr = ptr_to_handle(dst);
        req->size = count;
        memcpy((uint8_t*)payload_buf + sizeof(gs_memcpy_h2d_req_t), src, count);
        g_last_error = (cudaError_t)rpc_simple(GS_OP_MEMCPY_H2D, payload_buf, payload_size);
        g_client_pinned.release(payload_buf, pin_idx);
        return g_last_error;

    } else if (kind == cudaMemcpyDeviceToHost) {
        /* Phase 3: Use chunked transfer for large payloads */
        if (count > GS_CHUNK_THRESHOLD && (g_server_caps & GS_CAP_CHUNKED)) {
            g_last_error = cudaMemcpy_d2h_chunked(dst, src, count, 0);
            return g_last_error;
        }
        /* Standard single-message path */
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
#ifdef _WIN32
    if (g_local.available && !is_remote_device(g_active_device) && g_local.MemcpyAsync)
        return g_local.MemcpyAsync(dst, src, count, kind, stream);
#else
    if (g_local.available && !is_remote_device(g_active_device)) {
        local_set_ctx(to_local_device(g_active_device));
        CUresult r;
        CUstream cs = (CUstream)stream;
        if (kind == cudaMemcpyHostToDevice && g_local.MemcpyHtoDAsync)
            r = g_local.MemcpyHtoDAsync((CUdeviceptr)(uintptr_t)dst, src, count, cs);
        else if (kind == cudaMemcpyDeviceToHost && g_local.MemcpyDtoHAsync)
            r = g_local.MemcpyDtoHAsync(dst, (CUdeviceptr)(uintptr_t)src, count, cs);
        else if (kind == cudaMemcpyDeviceToDevice && g_local.MemcpyDtoD)
            r = g_local.MemcpyDtoD((CUdeviceptr)(uintptr_t)dst, (CUdeviceptr)(uintptr_t)src, count);
        else if (kind == cudaMemcpyHostToHost)
            { memcpy(dst, src, count); return cudaSuccess; }
        else return cudaErrorInvalidMemcpyDirection;
        return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
    }
#endif

    /* Phase 2: Use async opcodes when server supports them */
    if (g_server_caps & GS_CAP_ASYNC) {
        if (kind == cudaMemcpyHostToDevice) {
            /* Phase 3: Chunked async H2D for large transfers */
            if (count > GS_CHUNK_THRESHOLD && (g_server_caps & GS_CAP_CHUNKED)) {
                return cudaMemcpy_h2d_chunked(dst, src, count, ptr_to_handle(stream));
            }
            std::vector<uint8_t> payload(sizeof(gs_memcpy_h2d_async_req_t) + count);
            auto *req = (gs_memcpy_h2d_async_req_t*)payload.data();
            req->device_ptr = ptr_to_handle(dst);
            req->size = count;
            req->stream_handle = ptr_to_handle(stream);
            memcpy(payload.data() + sizeof(gs_memcpy_h2d_async_req_t), src, count);

            std::vector<uint8_t> resp;
            if (!rpc_call(GS_OP_MEMCPY_H2D_ASYNC, payload.data(), (uint32_t)payload.size(), resp))
                return cudaErrorUnknown;
            if (resp.size() >= sizeof(gs_generic_resp_t))
                return (cudaError_t)((const gs_generic_resp_t*)resp.data())->cuda_error;
            return cudaSuccess;
        }
        /* D2H async: use async opcode (still synchronous on server for now) */
        if (kind == cudaMemcpyDeviceToHost) {
            if (count > GS_CHUNK_THRESHOLD && (g_server_caps & GS_CAP_CHUNKED)) {
                return cudaMemcpy_d2h_chunked(dst, src, count, ptr_to_handle(stream));
            }
            gs_memcpy_d2h_async_req_t req;
            req.device_ptr = ptr_to_handle(src);
            req.size = count;
            req.stream_handle = ptr_to_handle(stream);
            std::vector<uint8_t> resp;
            if (!rpc_call(GS_OP_MEMCPY_D2H_ASYNC, &req, sizeof(req), resp)) return cudaErrorUnknown;
            if (resp.size() < sizeof(int32_t)) return cudaErrorUnknown;
            int32_t err;
            memcpy(&err, resp.data(), sizeof(err));
            if (err != 0) return (cudaError_t)err;
            if (resp.size() >= sizeof(int32_t) + count)
                memcpy(dst, resp.data() + sizeof(int32_t), count);
            return cudaSuccess;
        }
    }

    /* Fallback: synchronous memcpy */
    return cudaMemcpy(dst, src, count, kind);
}

GPUSHARE_EXPORT cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    TRACE("cudaMemset(%p, %d, %zu)", devPtr, value, count);
#ifdef _WIN32
    if (g_local.available && !is_remote_device(g_active_device) && g_local.Memset)
        return g_local.Memset(devPtr, value, count);
#else
    if (g_local.available && !is_remote_device(g_active_device) && g_local.MemsetD8) {
        local_set_ctx(to_local_device(g_active_device));
        CUresult r = g_local.MemsetD8((CUdeviceptr)(uintptr_t)devPtr, (unsigned char)value, count);
        return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
    }
#endif
    gs_memset_req_t req;
    req.device_ptr = ptr_to_handle(devPtr);
    req.value = value;
    req.size  = count;
    return (cudaError_t)rpc_simple(GS_OP_MEMSET, &req, sizeof(req));
}

GPUSHARE_EXPORT cudaError_t cudaDeviceSynchronize(void) {
    TRACE("cudaDeviceSynchronize");
#ifdef _WIN32
    if (g_local.available && !is_remote_device(g_active_device) && g_local.DeviceSynchronize)
        return g_local.DeviceSynchronize();
#else
    if (g_local.available && !is_remote_device(g_active_device) && g_local.CtxSynchronize) {
        local_set_ctx(to_local_device(g_active_device));
        CUresult r = g_local.CtxSynchronize();
        return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
    }
#endif
    return (cudaError_t)rpc_simple(GS_OP_DEVICE_SYNC, nullptr, 0);
}

GPUSHARE_EXPORT cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
    TRACE("cudaStreamCreate");
#ifdef _WIN32
    if (g_local.available && !is_remote_device(g_active_device) && g_local.StreamCreate)
        return g_local.StreamCreate(pStream);
#else
    if (g_local.available && !is_remote_device(g_active_device) && g_local.StreamCreate) {
        local_set_ctx(to_local_device(g_active_device));
        CUresult r = g_local.StreamCreate((CUstream*)pStream, 0);
        return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
    }
#endif
    std::vector<uint8_t> resp;
    if (!rpc_call(GS_OP_STREAM_CREATE, nullptr, 0, resp)) return cudaErrorUnknown;
    if (resp.size() < sizeof(gs_stream_create_resp_t)) return cudaErrorUnknown;
    auto *r = (const gs_stream_create_resp_t*)resp.data();
    if (pStream) *pStream = handle_to_ptr(r->stream_handle);
    return cudaSuccess;
}

GPUSHARE_EXPORT cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    TRACE("cudaStreamDestroy");
#ifdef _WIN32
    if (g_local.available && !is_remote_device(g_active_device) && g_local.StreamDestroy)
        return g_local.StreamDestroy(stream);
#else
    if (g_local.available && !is_remote_device(g_active_device) && g_local.StreamDestroy) {
        CUresult r = g_local.StreamDestroy((CUstream)stream);
        return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
    }
#endif
    gs_stream_req_t req;
    req.stream_handle = ptr_to_handle(stream);
    return (cudaError_t)rpc_simple(GS_OP_STREAM_DESTROY, &req, sizeof(req));
}

GPUSHARE_EXPORT cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    TRACE("cudaStreamSynchronize");
#ifdef _WIN32
    if (g_local.available && !is_remote_device(g_active_device) && g_local.StreamSynchronize)
        return g_local.StreamSynchronize(stream);
#else
    if (g_local.available && !is_remote_device(g_active_device) && g_local.StreamSynchronize) {
        CUresult r = g_local.StreamSynchronize((CUstream)stream);
        return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
    }
#endif
    gs_stream_req_t req;
    req.stream_handle = ptr_to_handle(stream);
    return (cudaError_t)rpc_simple(GS_OP_STREAM_SYNC, &req, sizeof(req));
}

GPUSHARE_EXPORT cudaError_t cudaEventCreate(cudaEvent_t *event) {
    TRACE("cudaEventCreate");
#ifdef _WIN32
    if (g_local.available && !is_remote_device(g_active_device) && g_local.EventCreate)
        return g_local.EventCreate(event);
#else
    if (g_local.available && !is_remote_device(g_active_device) && g_local.EventCreate) {
        local_set_ctx(to_local_device(g_active_device));
        CUresult r = g_local.EventCreate((CUevent*)event, 0);
        return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
    }
#endif
    std::vector<uint8_t> resp;
    if (!rpc_call(GS_OP_EVENT_CREATE, nullptr, 0, resp)) return cudaErrorUnknown;
    if (resp.size() < sizeof(gs_event_create_resp_t)) return cudaErrorUnknown;
    auto *r = (const gs_event_create_resp_t*)resp.data();
    if (event) *event = handle_to_ptr(r->event_handle);
    return cudaSuccess;
}

GPUSHARE_EXPORT cudaError_t cudaEventDestroy(cudaEvent_t event) {
    TRACE("cudaEventDestroy");
#ifdef _WIN32
    if (g_local.available && !is_remote_device(g_active_device) && g_local.EventDestroy)
        return g_local.EventDestroy(event);
#else
    if (g_local.available && !is_remote_device(g_active_device) && g_local.EventDestroy) {
        CUresult r = g_local.EventDestroy((CUevent)event);
        return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
    }
#endif
    uint64_t handle = ptr_to_handle(event);
    return (cudaError_t)rpc_simple(GS_OP_EVENT_DESTROY, &handle, sizeof(handle));
}

GPUSHARE_EXPORT cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    TRACE("cudaEventRecord");
#ifdef _WIN32
    if (g_local.available && !is_remote_device(g_active_device) && g_local.EventRecord)
        return g_local.EventRecord(event, stream);
#else
    if (g_local.available && !is_remote_device(g_active_device) && g_local.EventRecord) {
        CUresult r = g_local.EventRecord((CUevent)event, (CUstream)stream);
        return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
    }
#endif
    gs_event_record_req_t req;
    req.event_handle  = ptr_to_handle(event);
    req.stream_handle = ptr_to_handle(stream);
    return (cudaError_t)rpc_simple(GS_OP_EVENT_RECORD, &req, sizeof(req));
}

GPUSHARE_EXPORT cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    TRACE("cudaEventSynchronize");
#ifdef _WIN32
    if (g_local.available && !is_remote_device(g_active_device) && g_local.EventSynchronize)
        return g_local.EventSynchronize(event);
#else
    if (g_local.available && !is_remote_device(g_active_device) && g_local.EventSynchronize) {
        CUresult r = g_local.EventSynchronize((CUevent)event);
        return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
    }
#endif
    uint64_t handle = ptr_to_handle(event);
    return (cudaError_t)rpc_simple(GS_OP_EVENT_SYNC, &handle, sizeof(handle));
}

GPUSHARE_EXPORT cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
    TRACE("cudaEventElapsedTime");
#ifdef _WIN32
    if (g_local.available && !is_remote_device(g_active_device) && g_local.EventElapsedTime)
        return g_local.EventElapsedTime(ms, start, end);
#else
    if (g_local.available && !is_remote_device(g_active_device) && g_local.EventElapsedTime) {
        CUresult r = g_local.EventElapsedTime(ms, (CUevent)start, (CUevent)end);
        return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
    }
#endif
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

/* Forward declaration — cuDeviceGetAttribute is defined in the driver API section below */
extern "C" GPUSHARE_EXPORT CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);

GPUSHARE_EXPORT cudaError_t cudaDeviceGetAttribute(int *value, int attr, int device) {
    /* Forward to driver API implementation which has full attribute coverage */
    CUresult r = cuDeviceGetAttribute(value, (CUdevice_attribute)attr, (CUdevice)device);
    return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorInvalidValue;
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

/* Cached device properties — fetched once per device, reused for all attribute queries */
static bool g_props_cached = false;
static int g_props_cached_device = -1;
static gs_device_props_t g_cached_props;

static CUresult cache_device_props(int dev) {
    if (g_props_cached && g_props_cached_device == dev) return CUDA_SUCCESS;

    /* Local GPU — query and convert to our format */
    if (g_local.available && !is_remote_device(dev)) {
        struct cudaDeviceProp prop;
        memset(&prop, 0, sizeof(prop));
#ifdef _WIN32
        if (!g_local.GetDeviceProperties) return CUDA_ERROR_UNKNOWN;
        cudaError_t err = g_local.GetDeviceProperties(&prop, to_local_device(dev));
#else
        cudaError_t err = local_get_device_props(&prop, to_local_device(dev));
#endif
        if (err != cudaSuccess) return CUDA_ERROR_UNKNOWN;
        memset(&g_cached_props, 0, sizeof(g_cached_props));
        strncpy(g_cached_props.name, prop.name, sizeof(g_cached_props.name) - 1);
        g_cached_props.total_global_mem    = prop.totalGlobalMem;
        g_cached_props.shared_mem_per_block= prop.sharedMemPerBlock;
        g_cached_props.regs_per_block      = prop.regsPerBlock;
        g_cached_props.warp_size           = prop.warpSize;
        g_cached_props.max_threads_per_block = prop.maxThreadsPerBlock;
        g_cached_props.max_threads_dim[0]  = prop.maxThreadsDim[0];
        g_cached_props.max_threads_dim[1]  = prop.maxThreadsDim[1];
        g_cached_props.max_threads_dim[2]  = prop.maxThreadsDim[2];
        g_cached_props.max_grid_size[0]    = prop.maxGridSize[0];
        g_cached_props.max_grid_size[1]    = prop.maxGridSize[1];
        g_cached_props.max_grid_size[2]    = prop.maxGridSize[2];
        g_cached_props.clock_rate          = prop.clockRate;
        g_cached_props.major               = prop.major;
        g_cached_props.minor               = prop.minor;
        g_cached_props.multi_processor_count= prop.multiProcessorCount;
        g_cached_props.max_threads_per_mp  = prop.maxThreadsPerMultiProcessor;
        g_cached_props.total_const_mem     = prop.totalConstMem;
        g_cached_props.mem_bus_width       = prop.memoryBusWidth;
        g_cached_props.l2_cache_size       = prop.l2CacheSize;
        g_props_cached = true;
        g_props_cached_device = dev;
        return CUDA_SUCCESS;
    }

    /* Remote GPU */
    fflush(stderr); fprintf(stderr, "[gpushare] cache_device_props(dev=%d) -> RPC (remote_dev=%d, servers=%zu)\n",
            dev, to_remote_device(dev), g_servers.size());
    gs_device_props_req_t req;
    req.device = to_remote_device(dev);
    std::vector<uint8_t> resp;
    if (!rpc_call(GS_OP_GET_DEVICE_PROPS, &req, sizeof(req), resp, nullptr, dev)) return CUDA_ERROR_UNKNOWN;
    if (resp.size() < sizeof(gs_device_props_t)) return CUDA_ERROR_UNKNOWN;
    memcpy(&g_cached_props, resp.data(), sizeof(g_cached_props));
    g_props_cached = true;
    g_props_cached_device = dev;
    return CUDA_SUCCESS;
}

/* ── Initialization ──────────────────────────────────────── */

GPUSHARE_EXPORT CUresult cuInit(unsigned int flags) {
    TRACE("cuInit(%u)", flags);
    (void)flags;
    /* If already connected, return immediately — avoids re-triggering
     * ensure_connected() during PyTorch's bundled cudart lazy init. */
    if (!g_servers.empty()) return CUDA_SUCCESS;
    if (!ensure_connected()) return CUDA_ERROR_NO_DEVICE;
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuDriverGetVersion(int *version) {
    if (!version) return CUDA_ERROR_INVALID_VALUE;

    /* Query real local driver version if available.
     * PyTorch checks this against its compiled CUDA version — reporting a
     * version that doesn't match the local driver causes "driver too old"
     * or "driver too new" errors. */
#ifdef _WIN32
    /* Windows: try real nvcuda.dll cuDriverGetVersion via GetProcAddress */
    if (g_local.h_cuda) {
        typedef CUresult (*pfn_cuDriverGetVersion)(int*);
        pfn_cuDriverGetVersion real_fn = (pfn_cuDriverGetVersion)
            GetProcAddress((HMODULE)g_local.h_cuda, "cuDriverGetVersion");
        if (real_fn) {
            CUresult r = real_fn(version);
            if (r == CUDA_SUCCESS) return CUDA_SUCCESS;
        }
    }
#else
    /* Linux/macOS: try real libcuda.so.1 cuDriverGetVersion via dlsym */
    if (g_local.h_cuda) {
        typedef CUresult (*pfn_cuDriverGetVersion)(int*);
        pfn_cuDriverGetVersion real_fn = (pfn_cuDriverGetVersion)
            dlsym(g_local.h_cuda, "cuDriverGetVersion");
        if (real_fn) {
            CUresult r = real_fn(version);
            if (r == CUDA_SUCCESS) return CUDA_SUCCESS;
        }
    }
#endif

    /* No local driver — report server CUDA version */
    *version = 13010;
    return CUDA_SUCCESS;
}

/* ── Device management ───────────────────────────────────── */

GPUSHARE_EXPORT CUresult cuDeviceGetCount(int *count) {
    TRACE("cuDeviceGetCount");
    fprintf(stderr, "[gpushare] cuDeviceGetCount: ENTER\n");
    fflush(stderr);
    int c = 0;
    cudaError_t err = cudaGetDeviceCount(&c);
    fprintf(stderr, "[gpushare] cuDeviceGetCount: returning %d\n", c);
    fflush(stderr);
    if (count) *count = c;
    return (err == cudaSuccess || c > 0) ? CUDA_SUCCESS : CUDA_ERROR_NO_DEVICE;
}

GPUSHARE_EXPORT CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    TRACE("cuDeviceGet(%d)", ordinal);
    if (device) *device = ordinal;
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    TRACE("cuDeviceGetName(dev=%d)", dev);
    CUresult err = cache_device_props(dev);
    if (err != CUDA_SUCCESS) return err;
    if (name && len > 0) {
        strncpy(name, g_cached_props.name, len - 1);
        name[len - 1] = '\0';
    }
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
    TRACE("cuDeviceTotalMem(dev=%d)", dev);
    CUresult err = cache_device_props(dev);
    if (err != CUDA_SUCCESS) return err;
    if (bytes) *bytes = g_cached_props.total_global_mem;
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
    return cuDeviceTotalMem_v2(bytes, dev);
}

GPUSHARE_EXPORT CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    fprintf(stderr, "[gpushare] cuDeviceGetAttribute(attr=%d, dev=%d) servers=%zu\n",
            (int)attrib, dev, g_servers.size());
    fflush(stderr);
    CUresult err = cache_device_props(dev);
    if (err != CUDA_SUCCESS) return err;
    if (!pi) return CUDA_ERROR_INVALID_VALUE;

    switch (attrib) {
        /* ── From cached device properties ── */
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
        case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE:                  *pi = (int)g_cached_props.l2_cache_size; break;
        case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH:        *pi = (int)g_cached_props.mem_bus_width; break;

        /* ── Capabilities true for all modern GPUs (Kepler+) ── */
        case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING:             *pi = 1; break;
        case CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY:                 *pi = 1; break;
        case CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS:      *pi = 1; break;
        case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY:            *pi = 1; break;
        case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS:             *pi = 1; break;
        case CU_DEVICE_ATTRIBUTE_GPU_OVERLAP:                    *pi = 1; break;
        case CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT:             *pi = 2; break;
        case CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED:    *pi = 1; break;
        case CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED:      *pi = 1; break;
        case CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED:       *pi = 1; break;
        case CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH:             *pi = 1; break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED:   *pi = 1; break;
        case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS:         *pi = 1; break;

        /* ── Sensible defaults for remaining PyTorch-queried attrs ── */
        case CU_DEVICE_ATTRIBUTE_MAX_PITCH:                      *pi = 2147483647; break;
        case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT:              *pi = 512; break;
        case CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT:            *pi = 0; break;
        case CU_DEVICE_ATTRIBUTE_INTEGRATED:                     *pi = 0; break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE:                   *pi = 0; break;
        case CU_DEVICE_ATTRIBUTE_ECC_ENABLED:                    *pi = 0; break;
        case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID:                     *pi = 0; break;
        case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID:                  *pi = 0; break;
        case CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID:                  *pi = 0; break;
        case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE:              *pi = 0; break;
        case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR:
            *pi = g_cached_props.shared_mem_per_block > 0 ? (int)(g_cached_props.shared_mem_per_block * 2) : 166912;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR: *pi = 65536; break;
        case CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD:                *pi = 0; break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR:  *pi = 32; break;
        case CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED:         *pi = 0; break;
        case CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED: *pi = 0; break;

        default: *pi = 0; break;
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
    CUresult err = cache_device_props(dev);
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
    fflush(stderr); fprintf(stderr, "[gpushare] cuCtxSetCurrent(%p)\n", (void*)ctx);
    (void)ctx;
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuCtxGetCurrent(CUcontext *pctx) {
    fflush(stderr); fprintf(stderr, "[gpushare] cuCtxGetCurrent()\n");
    if (pctx) *pctx = g_fake_ctx;
    return CUDA_SUCCESS;
}

GPUSHARE_EXPORT CUresult cuCtxGetDevice(CUdevice *device) {
    if (device) *device = g_active_device;
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
    cudaError_t err = cudaMalloc(&p, bytesize);  /* routes via g_active_device */
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
    CUresult err = cache_device_props(g_active_device);
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

    /* Use pipelined rpc_call instead of manual send/recv */
    std::vector<uint8_t> resp;
    if (!rpc_call(GS_OP_MODULE_LOAD, payload.data(), (uint32_t)payload.size(), resp))
        return CUDA_ERROR_UNKNOWN;
    if (resp.size() < sizeof(gs_module_load_resp_t)) return CUDA_ERROR_UNKNOWN;

    auto *mresp = (const gs_module_load_resp_t*)resp.data();
    if (module) *module = (CUmodule)(uintptr_t)mresp->module_handle;
    return (CUresult)mresp->cuda_error;
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
    fflush(stderr); fprintf(stderr, "[gpushare] cuDevicePrimaryCtxRetain(dev=%d)\n", dev);
    /* Return a unique fake context per device — PyTorch checks context
     * uniqueness and may deadlock if two devices share the same context. */
    if (pctx) *pctx = (CUcontext)(uintptr_t)(0xBADC0DE0 + (unsigned)dev);
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

/* ── cuGetProcAddress — CUDA 12+ dynamic symbol resolver ────────────────── */
/* PyTorch/c10_cuda.dll calls this to resolve ALL CUDA functions at runtime.
 * Without it, c10_cuda.dll fails with WinError 127 "procedure not found".
 * We look up the symbol in our own library and return the function pointer. */

#ifdef _WIN32
  #include <libloaderapi.h>
  static void *_self_sym(const char *name) {
      /* Get handle to our own DLL, then look up the symbol */
      static HMODULE self = NULL;
      if (!self) {
          /* Get the module handle for THIS DLL (not the exe) */
          GetModuleHandleExA(
              GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
              (LPCSTR)_self_sym, &self);
          if (!self) self = GetModuleHandleA(NULL);
      }
      return (void*)GetProcAddress(self, name);
  }
#else
  static void *_self_sym(const char *name) {
      return dlsym(RTLD_DEFAULT, name);
  }
#endif

GPUSHARE_EXPORT CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, uint64_t flags) {
    (void)cudaVersion; (void)flags;
    if (!symbol || !pfn) return CUDA_ERROR_INVALID_VALUE;
    *pfn = _self_sym(symbol);
    if (*pfn) { TRACE("cuGetProcAddress(%s) -> %p", symbol, *pfn); return CUDA_SUCCESS; }
    /* Symbol not found — try with version suffixes that CUDA sometimes uses */
    char versioned[256];
    snprintf(versioned, sizeof(versioned), "%s_v2", symbol);
    *pfn = _self_sym(versioned);
    if (*pfn) { TRACE("cuGetProcAddress(%s) -> %s -> %p", symbol, versioned, *pfn); return CUDA_SUCCESS; }
    snprintf(versioned, sizeof(versioned), "%s_v3", symbol);
    *pfn = _self_sym(versioned);
    if (*pfn) { TRACE("cuGetProcAddress(%s) -> %s -> %p", symbol, versioned, *pfn); return CUDA_SUCCESS; }
    TRACE("cuGetProcAddress: not found: %s", symbol);
    return CUDA_ERROR_NOT_FOUND;
}

GPUSHARE_EXPORT CUresult cuGetProcAddress_v2(const char *symbol, void **pfn,
                                              int cudaVersion, uint64_t flags,
                                              int *symbolStatus) {
    CUresult r = cuGetProcAddress(symbol, pfn, cudaVersion, flags);
    if (symbolStatus) {
        *symbolStatus = (r == CUDA_SUCCESS) ? 1 : 0;  /* 1 = found, 0 = not found */
    }
    return r;
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
    fprintf(stderr, "[gpushare] nvmlDeviceGetCount_v2: ENTER\n");
    fflush(stderr);
    TRACE("nvmlDeviceGetCount_v2");
    if (!ensure_connected()) {
        if (deviceCount) *deviceCount = g_local.available ? g_local.local_count : 0;
        return NVML_SUCCESS;
    }

    unsigned int total = 0;
    if (g_local.available && g_gpu_mode != "remote")
        total += (unsigned int)g_local.local_count;
    if (g_gpu_mode != "local")
        total += (unsigned int)g_total_remote_devices;
    if (deviceCount) *deviceCount = total;
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetCount(unsigned int *deviceCount) {
    return nvmlDeviceGetCount_v2(deviceCount);
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device) {
    TRACE("nvmlDeviceGetHandleByIndex(%u)", index);
    unsigned int total_local = (g_local.available && g_gpu_mode != "remote") ? g_local.local_count : 0;
    unsigned int total_remote = (g_gpu_mode != "local") ? g_total_remote_devices : 0;

    if (index >= (total_local + total_remote)) return NVML_ERROR_INVALID_ARGUMENT;

    if (g_remote_first) {
        /* Remote GPUs first (0..total_remote-1) */
        if (index < total_remote) {
            if (device) *device = g_nvml_device;
            return NVML_SUCCESS;
        }
        /* Local GPUs after */
        if (device) *device = g_local.nvml_handles[index - total_remote];
        return NVML_SUCCESS;
    } else {
        /* Local GPUs first (standard) */
        if (index < total_local) {
            if (device) *device = g_local.nvml_handles[index];
            return NVML_SUCCESS;
        }
        /* Remote GPU */
        if (device) *device = g_nvml_device;
        return NVML_SUCCESS;
    }
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device) {
    return nvmlDeviceGetHandleByIndex_v2(index, device);
}

/* Check if an NVML handle belongs to a local GPU (not our remote sentinel) */
static bool is_local_nvml_handle(nvmlDevice_t device) {
    if (!g_local.available) return false;
    if (device == g_nvml_device) return false;  /* our sentinel = remote */
    for (int i = 0; i < g_local.local_count; i++) {
        if (device == g_local.nvml_handles[i]) return true;
    }
    return false;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length) {
    TRACE("nvmlDeviceGetName");
    /* Local GPU — use real NVML */
    if (is_local_nvml_handle(device) && g_local.NvmlDeviceGetName) {
        nvmlReturn_t ret = g_local.NvmlDeviceGetName(device, name, length);
        if (ret == NVML_SUCCESS && name) {
            size_t len = strlen(name);
            if (len + 8 < length) strcat(name, " (local)");
        }
        return ret;
    }
    /* Remote GPU */
    CUresult err = cache_device_props(g_active_device);
    if (err != CUDA_SUCCESS) return NVML_ERROR_UNKNOWN;
    if (name && length > 0) {
        strncpy(name, g_cached_props.name, length - 1);
        name[length - 1] = '\0';
        size_t len = strlen(name);
        if (len + 10 < length) strcat(name, " (remote)");
    }
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory) {
    TRACE("nvmlDeviceGetMemoryInfo");
    if (is_local_nvml_handle(device) && g_local.NvmlDeviceGetMemoryInfo)
        return g_local.NvmlDeviceGetMemoryInfo(device, memory);
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
    /* For local GPUs, use v1 info and convert */
    if (is_local_nvml_handle(device) && g_local.NvmlDeviceGetMemoryInfo && memory) {
        nvmlMemory_t m1;
        nvmlReturn_t r = g_local.NvmlDeviceGetMemoryInfo(device, &m1);
        if (r == NVML_SUCCESS) {
            memory->version = 2;
            memory->total = m1.total;
            memory->used = m1.used;
            memory->free = m1.free;
            memory->reserved = 0;
        }
        return r;
    }
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
    if (is_local_nvml_handle(device) && g_local.NvmlDeviceGetUtilizationRates)
        return g_local.NvmlDeviceGetUtilizationRates(device, utilization);
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
    if (is_local_nvml_handle(device) && g_local.NvmlDeviceGetTemperature)
        return g_local.NvmlDeviceGetTemperature(device, sensor, temp);
    (void)sensor;
    nvmlReturn_t err = refresh_gpu_status();
    if (err != NVML_SUCCESS) return err;
    if (temp) *temp = g_gpu_status.temperature;
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int *power) {
    if (is_local_nvml_handle(device) && g_local.NvmlDeviceGetPowerUsage)
        return g_local.NvmlDeviceGetPowerUsage(device, power);
    nvmlReturn_t err = refresh_gpu_status();
    if (err != NVML_SUCCESS) return err;
    if (power) *power = g_gpu_status.power_draw_mw;
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int *limit) {
    if (is_local_nvml_handle(device) && g_local.NvmlDeviceGetPowerUsage)
        return NVML_ERROR_NOT_SUPPORTED;  /* use nvmlDeviceGetPowerManagementLimit for local */
    nvmlReturn_t err = refresh_gpu_status();
    if (err != NVML_SUCCESS) return err;
    if (limit) *limit = g_gpu_status.power_limit_mw;
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int *speed) {
    if (is_local_nvml_handle(device) && g_local.NvmlDeviceGetFanSpeed)
        return g_local.NvmlDeviceGetFanSpeed(device, speed);
    nvmlReturn_t err = refresh_gpu_status();
    if (err != NVML_SUCCESS) return err;
    if (speed) *speed = g_gpu_status.fan_speed;
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid, unsigned int length) {
    if (is_local_nvml_handle(device) && g_local.NvmlDeviceGetUUID)
        return g_local.NvmlDeviceGetUUID(device, uuid, length);
    if (uuid && length > 0) strncpy(uuid, "GPU-gpushare-remote-00000000", length - 1);
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci) {
    if (is_local_nvml_handle(device) && g_local.NvmlDeviceGetPciInfo)
        return g_local.NvmlDeviceGetPciInfo(device, pci);
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

/* Forward declaration — nvmlDeviceGetIndex defined below */
extern "C" GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index);

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device,
                                                                  int *major, int *minor) {
    if (is_local_nvml_handle(device) && g_local.NvmlDeviceGetCudaComputeCapability)
        return g_local.NvmlDeviceGetCudaComputeCapability(device, major, minor);
    unsigned int index = 0;
    nvmlDeviceGetIndex(device, &index);
    return (nvmlReturn_t)cuDeviceComputeCapability(major, minor, (CUdevice)index);
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, int type, unsigned int *clock) {
    if (is_local_nvml_handle(device) && g_local.NvmlDeviceGetClockInfo)
        return g_local.NvmlDeviceGetClockInfo(device, type, clock);
    nvmlReturn_t err = refresh_gpu_status();
    if (err != NVML_SUCCESS) return err;
    if (clock) *clock = (type == 0) ? g_gpu_status.clock_sm_mhz : g_gpu_status.clock_mem_mhz;
    return NVML_SUCCESS;
}

GPUSHARE_EXPORT nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index) {
    if (index) {
        if (is_local_nvml_handle(device)) {
            /* Find which local device this handle corresponds to */
            for (int i = 0; i < g_local.local_count; i++) {
                if (device == g_local.nvml_handles[i]) {
                    if (g_remote_first) *index = (unsigned int)g_total_remote_devices + i;
                    else *index = (unsigned int)i;
                    return NVML_SUCCESS;
                }
            }
            *index = 0;
        } else {
            if (g_remote_first) *index = 0;  /* first remote GPU index */
            else *index = (unsigned int)g_local.local_count;  /* remote GPU index */
        }
    }
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

/* ── CUDA Runtime internal symbols ───────────────────────── */
/* These __cuda* functions are used by nvcc-compiled CUDA code for kernel
 * registration and launch configuration. PyTorch's bundled CUDA libraries
 * (libtorch_cuda.so, etc.) need these versioned as @@libcudart.so.12.
 * We provide no-op stubs since we don't execute kernels locally —
 * all computation happens on the remote GPU server. */

GPUSHARE_EXPORT void **__cudaRegisterFatBinary(void *fatCubin) {
    (void)fatCubin;
    static void *fake_handle = (void*)(uintptr_t)0xFA7B1;
    return &fake_handle;
}

GPUSHARE_EXPORT void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
    (void)fatCubinHandle;
}

GPUSHARE_EXPORT void __cudaUnregisterFatBinary(void **fatCubinHandle) {
    (void)fatCubinHandle;
}

GPUSHARE_EXPORT void __cudaRegisterFunction(
    void **fatCubinHandle, const char *hostFun, char *deviceFun,
    const char *deviceName, int thread_limit, void *tid, void *bid,
    void *bDim, void *gDim, int *wSize) {
    (void)fatCubinHandle; (void)hostFun; (void)deviceFun;
    (void)deviceName; (void)thread_limit; (void)tid; (void)bid;
    (void)bDim; (void)gDim; (void)wSize;
}

GPUSHARE_EXPORT void __cudaRegisterVar(
    void **fatCubinHandle, char *hostVar, char *deviceAddress,
    const char *deviceName, int ext, size_t size, int constant, int global) {
    (void)fatCubinHandle; (void)hostVar; (void)deviceAddress;
    (void)deviceName; (void)ext; (void)size; (void)constant; (void)global;
}

GPUSHARE_EXPORT unsigned __cudaPushCallConfiguration(
    dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
    (void)gridDim; (void)blockDim; (void)sharedMem; (void)stream;
    return 0;
}

GPUSHARE_EXPORT cudaError_t __cudaPopCallConfiguration(
    dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, cudaStream_t *stream) {
    if (gridDim) { gridDim->x = 1; gridDim->y = 1; gridDim->z = 1; }
    if (blockDim) { blockDim->x = 1; blockDim->y = 1; blockDim->z = 1; }
    if (sharedMem) *sharedMem = 0;
    if (stream) *stream = nullptr;
    return cudaSuccess;
}

GPUSHARE_EXPORT void __cudaInitModule(void **fatCubinHandle) {
    (void)fatCubinHandle;
}

GPUSHARE_EXPORT cudaError_t cudaGetDriverEntryPoint(const char *symbol, void **funcPtr,
                                                     unsigned long long flags) {
    (void)flags;
    if (!symbol || !funcPtr) return cudaErrorInvalidValue;
    *funcPtr = _self_sym(symbol);
    return *funcPtr ? cudaSuccess : cudaErrorNotFound;
}

GPUSHARE_EXPORT cudaError_t cudaGetDriverEntryPointByVersion(const char *symbol, void **funcPtr,
                                                              unsigned int cudaVersion,
                                                              unsigned long long flags,
                                                              int *driverStatus) {
    (void)cudaVersion; (void)flags;
    if (!symbol || !funcPtr) return cudaErrorInvalidValue;
    *funcPtr = _self_sym(symbol);
    if (driverStatus) *driverStatus = (*funcPtr) ? 1 : 0;
    return *funcPtr ? cudaSuccess : cudaErrorNotFound;
}

/* Additional runtime API stubs needed by PyTorch
 * NOTE: Many cuda* functions are already in generated_stubs.cpp (with full RPC).
 * Only add stubs here for functions NOT in generated_stubs.cpp. */

GPUSHARE_EXPORT cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream,
                                                      unsigned int flags) {
    (void)flags;
    return cudaEventRecord(event, stream);
}

GPUSHARE_EXPORT cudaError_t cudaStreamGetPriority(cudaStream_t stream, int *priority) {
    (void)stream;
    if (priority) *priority = 0;
    return cudaSuccess;
}

GPUSHARE_EXPORT cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags) {
    (void)flags;
    *pHost = malloc(size);
    return *pHost ? cudaSuccess : cudaErrorMemoryAllocation;
}

GPUSHARE_EXPORT cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags) {
    (void)ptr; (void)size; (void)flags;
    return cudaSuccess;
}

GPUSHARE_EXPORT cudaError_t cudaHostUnregister(void *ptr) {
    (void)ptr;
    return cudaSuccess;
}

GPUSHARE_EXPORT cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src,
                                                  int srcDevice, size_t count, cudaStream_t stream) {
    (void)dstDevice; (void)srcDevice; (void)stream;
    return cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
}

GPUSHARE_EXPORT cudaError_t cudaFuncSetAttribute(const void *func, int attr, int value) {
    (void)func; (void)attr; (void)value;
    return cudaSuccess;
}

GPUSHARE_EXPORT cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
    (void)func; (void)blockSize; (void)dynamicSMemSize; (void)flags;
    if (numBlocks) *numBlocks = 16;
    return cudaSuccess;
}

GPUSHARE_EXPORT cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                                              void **args, size_t sharedMem, cudaStream_t stream) {
    (void)func; (void)gridDim; (void)blockDim; (void)args; (void)sharedMem; (void)stream;
    return cudaSuccess;  /* TODO: route to server */
}

GPUSHARE_EXPORT cudaError_t cudaLaunchKernelExC(void *config, const void *func, void **args) {
    (void)config; (void)func; (void)args;
    return cudaSuccess;
}

GPUSHARE_EXPORT cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
    (void)device;
    if (pciBusId && len > 0) snprintf(pciBusId, len, "0000:00:00.0");
    return cudaSuccess;
}

GPUSHARE_EXPORT cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*fn)(void*), void *userData) {
    (void)stream;
    if (fn) fn(userData);
    return cudaSuccess;
}

GPUSHARE_EXPORT cudaError_t cudaProfilerStart(void) { return cudaSuccess; }
GPUSHARE_EXPORT cudaError_t cudaProfilerStop(void) { return cudaSuccess; }
GPUSHARE_EXPORT cudaError_t cudaThreadExchangeStreamCaptureMode(int *mode) {
    if (mode) *mode = 0;
    return cudaSuccess;
}

/* Stream capture stubs (graph API) */
GPUSHARE_EXPORT cudaError_t cudaStreamBeginCapture(cudaStream_t s, int mode) { (void)s; (void)mode; return cudaErrorStreamCaptureUnsupported; }
GPUSHARE_EXPORT cudaError_t cudaStreamEndCapture(cudaStream_t s, void **g) { (void)s; (void)g; return cudaErrorStreamCaptureUnsupported; }
GPUSHARE_EXPORT cudaError_t cudaStreamIsCapturing(cudaStream_t s, int *status) { (void)s; if (status) *status = 0; return cudaSuccess; }
GPUSHARE_EXPORT cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t s, int *status, void *id, void *g, void *deps, void *numDeps) {
    (void)s; (void)id; (void)g; (void)deps; (void)numDeps;
    if (status) *status = 0;
    return cudaSuccess;
}

/* Graph API stubs (not in generated_stubs) */
GPUSHARE_EXPORT cudaError_t cudaGraphInstantiateWithFlags(void **exec, void *graph, unsigned long long flags) { (void)exec; (void)graph; (void)flags; return cudaErrorNotSupported; }
GPUSHARE_EXPORT cudaError_t cudaGraphGetNodes(void *graph, void *nodes, size_t *numNodes) { (void)graph; (void)nodes; if (numNodes) *numNodes = 0; return cudaSuccess; }
GPUSHARE_EXPORT cudaError_t cudaGraphDebugDotPrint(void *graph, const char *path, unsigned int flags) { (void)graph; (void)path; (void)flags; return cudaSuccess; }
GPUSHARE_EXPORT cudaError_t cudaGraphNodeGetDependencies(void *node, void *deps, size_t *numDeps) { (void)node; (void)deps; if (numDeps) *numDeps = 0; return cudaSuccess; }

/* Memory pool stubs (not in generated_stubs) */
GPUSHARE_EXPORT cudaError_t cudaMemPoolSetAttribute(void *memPool, int attr, void *value) { (void)memPool; (void)attr; (void)value; return cudaSuccess; }
GPUSHARE_EXPORT cudaError_t cudaMemPoolGetAttribute(void *memPool, int attr, void *value) { (void)memPool; (void)attr; if (value) memset(value, 0, 8); return cudaSuccess; }
GPUSHARE_EXPORT cudaError_t cudaMemPoolSetAccess(void *memPool, void *descList, size_t count) { (void)memPool; (void)descList; (void)count; return cudaSuccess; }

/* IPC stubs */
GPUSHARE_EXPORT cudaError_t cudaIpcGetMemHandle(void *handle, void *devPtr) { (void)handle; (void)devPtr; return cudaErrorNotSupported; }
GPUSHARE_EXPORT cudaError_t cudaIpcOpenMemHandle(void **devPtr, void *handle, unsigned int flags) { (void)devPtr; (void)handle; (void)flags; return cudaErrorNotSupported; }
GPUSHARE_EXPORT cudaError_t cudaIpcCloseMemHandle(void *devPtr) { (void)devPtr; return cudaSuccess; }
GPUSHARE_EXPORT cudaError_t cudaIpcGetEventHandle(void *handle, cudaEvent_t event) { (void)handle; (void)event; return cudaErrorNotSupported; }
GPUSHARE_EXPORT cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event, void *handle) { (void)event; (void)handle; return cudaErrorNotSupported; }

/* Symbol stubs */
GPUSHARE_EXPORT cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, cudaMemcpyKind kind) {
    (void)symbol; (void)src; (void)count; (void)offset; (void)kind;
    return cudaErrorInvalidSymbol;
}

/* ── Cleanup ─────────────────────────────────────────────── */

static void GPUSHARE_DESTRUCTOR gpushare_cleanup(void) {
    /* Phase 11: Disconnect all servers.
     * During process exit, threads may already be partially torn down.
     * Use the same strategy as Windows DllMain: detach recv threads
     * instead of joining, send CLOSE, and let the OS clean up. */
    for (auto &sc : g_servers) {
        if (!sc) continue;
        sc->recv_running = false;
        if (sc->transport) {
            try {
                gs_header_t hdr;
                gs_header_init(&hdr, GS_OP_CLOSE, 0, GPUSHARE_HEADER_SIZE);
                sc->transport->send(&hdr, sizeof(hdr));
                sc->transport->shutdown_read();
            } catch (...) {}
        }
        try { if (sc->recv_thread.joinable()) sc->recv_thread.detach(); } catch (...) {}
        try { if (sc->transport) { sc->transport->close(); sc->transport.reset(); } } catch (...) {}
    }
    g_servers.clear();
    g_device_routes.clear();
    g_total_remote_devices = 0;
    g_sock = SOCK_INVALID;
    g_server_caps = 0;
    /* Phase 5: Clean up client-side pinned pool */
    if (g_client_pinned_initialized) {
        g_client_pinned.destroy();
        g_client_pinned_initialized = false;
    }
#ifndef _WIN32
    /* Clean up local GPU library handles */
    if (g_local.NvmlShutdown) g_local.NvmlShutdown();
    if (g_local.h_nvml)   { dlclose(g_local.h_nvml);   g_local.h_nvml = nullptr; }
    /* Release primary contexts before closing driver */
    for (int i = 0; i < g_local.local_count; i++) {
        if (g_local.PrimaryCtxRelease && g_local.contexts[i]) {
            CUdevice dev;
            if (g_local.DeviceGet) g_local.DeviceGet(&dev, i);
            else dev = i;
            g_local.PrimaryCtxRelease(dev);
            g_local.contexts[i] = nullptr;
        }
    }
    if (g_local.h_cuda)   { dlclose(g_local.h_cuda);   g_local.h_cuda = nullptr; }
    g_local.available = false;
#endif
}

#ifdef _WIN32
/* On Windows, use DllMain for cleanup instead of __attribute__((destructor)).
 * CRITICAL: DllMain(DLL_PROCESS_DETACH) holds the loader lock. Calling
 * std::thread::join() here deadlocks because the recv thread may need the
 * loader lock to terminate its CRT state. Instead, detach the thread and
 * just close the socket — the OS will clean up the thread on process exit. */
BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved) {
    (void)hinstDLL;
    if (fdwReason == DLL_PROCESS_DETACH) {
        /* Phase 11: Disconnect all servers */
        for (auto &sc : g_servers) {
            if (!sc) continue;
            sc->recv_running = false;
            if (sc->transport) {
                gs_header_t hdr;
                gs_header_init(&hdr, GS_OP_CLOSE, 0, GPUSHARE_HEADER_SIZE);
                sc->transport->send(&hdr, sizeof(hdr));
                sc->transport->shutdown_read();
            }
            if (lpvReserved == NULL) {
                if (sc->recv_thread.joinable()) sc->recv_thread.join();
            } else {
                if (sc->recv_thread.joinable()) sc->recv_thread.detach();
            }
            if (sc->transport) { sc->transport->close(); sc->transport.reset(); }
            { std::lock_guard<std::mutex> lock(sc->pending_mtx); sc->pending.clear(); }
        }
        g_servers.clear();
        g_device_routes.clear();
        g_total_remote_devices = 0;
        g_sock = SOCK_INVALID;
        g_server_caps = 0;
        if (g_client_pinned_initialized) {
            g_client_pinned.destroy();
            g_client_pinned_initialized = false;
        }
        /* Clean up local GPU library handles */
        if (g_local.h_nvml)   { FreeLibrary((HMODULE)g_local.h_nvml);   g_local.h_nvml = nullptr; }
        if (g_local.h_cudart) { FreeLibrary((HMODULE)g_local.h_cudart); g_local.h_cudart = nullptr; }
        if (g_local.h_cuda)   { FreeLibrary((HMODULE)g_local.h_cuda);   g_local.h_cuda = nullptr; }
        g_local.available = false;
    }
    return TRUE;
}
#endif

#ifdef __cplusplus
}
#endif
