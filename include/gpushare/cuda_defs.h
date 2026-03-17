/*
 * Minimal CUDA type definitions for the gpushare client library.
 * This allows building the client on machines WITHOUT the CUDA toolkit.
 * These types mirror the real CUDA definitions exactly.
 */

#ifndef GPUSHARE_CUDA_DEFS_H
#define GPUSHARE_CUDA_DEFS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Error types ─────────────────────────────────────────── */
typedef enum {
    cudaSuccess                    = 0,
    cudaErrorInvalidValue          = 1,
    cudaErrorMemoryAllocation      = 2,
    cudaErrorInitializationError   = 3,
    cudaErrorCudartUnloading       = 4,
    cudaErrorProfilerDisabled      = 5,
    cudaErrorInvalidDevice         = 10,
    cudaErrorInvalidMemcpyDirection= 21,
    cudaErrorInsufficientDriver    = 35,
    cudaErrorNoDevice              = 100,
    cudaErrorInvalidDevicePointer  = 700,
    cudaErrorNotReady              = 600,
    cudaErrorUnknown               = 999,
} cudaError_t;

typedef enum {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault        = 4,
} cudaMemcpyKind;

/* ── CUDA Driver API types ────────────────────────────────── */
typedef enum {
    CUDA_SUCCESS                    = 0,
    CUDA_ERROR_INVALID_VALUE        = 1,
    CUDA_ERROR_OUT_OF_MEMORY        = 2,
    CUDA_ERROR_NOT_INITIALIZED      = 3,
    CUDA_ERROR_DEINITIALIZED        = 4,
    CUDA_ERROR_NO_DEVICE            = 100,
    CUDA_ERROR_INVALID_DEVICE       = 101,
    CUDA_ERROR_INVALID_CONTEXT      = 201,
    CUDA_ERROR_NOT_FOUND            = 500,
    CUDA_ERROR_NOT_SUPPORTED        = 801,
    CUDA_ERROR_UNKNOWN              = 999,
} CUresult;

typedef int            CUdevice;
typedef void*          CUcontext;
typedef void*          CUmodule;
typedef void*          CUfunction;
typedef void*          CUstream;
typedef void*          CUevent;
typedef unsigned long long CUdeviceptr;

typedef struct { char bytes[16]; } CUuuid;

/* Driver API device attributes — covers all attributes PyTorch/TF query */
typedef enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK            = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X                  = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y                  = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z                  = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X                   = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y                   = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z                   = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK      = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY            = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE                        = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH                        = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK          = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE                       = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT                = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP                      = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT             = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT              = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED                       = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY              = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE                     = 20,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS               = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED                      = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID                       = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID                    = 34,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID                    = 50,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE                = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH          = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE                    = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR   = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT               = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING               = 41,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED      = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED        = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED         = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY                   = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD                  = 84,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS           = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS        = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED     = 90,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH               = 95,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR    = 106,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED           = 115,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR         = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR         = 76,
} CUdevice_attribute;

/* ── NVML types (GPU management library) ─────────────────── */
typedef enum {
    NVML_SUCCESS                   = 0,
    NVML_ERROR_UNINITIALIZED       = 1,
    NVML_ERROR_INVALID_ARGUMENT    = 2,
    NVML_ERROR_NOT_SUPPORTED       = 3,
    NVML_ERROR_NO_PERMISSION       = 4,
    NVML_ERROR_NOT_FOUND           = 6,
    NVML_ERROR_UNKNOWN             = 999,
} nvmlReturn_t;

typedef void* nvmlDevice_t;

typedef struct {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;

/* v2 adds reserved field */
typedef struct {
    unsigned int version;
    unsigned long long total;
    unsigned long long used;
    unsigned long long free;
    unsigned long long reserved;
} nvmlMemory_v2_t;

typedef struct {
    unsigned int gpu;
    unsigned int memory;
} nvmlUtilization_t;

typedef enum {
    NVML_TEMPERATURE_GPU = 0,
} nvmlTemperatureSensors_t;

typedef struct {
    char busIdLegacy[16];
    unsigned int domain;
    unsigned int bus;
    unsigned int device;
    unsigned int pciDeviceId;
    unsigned int pciSubSystemId;
    char busId[32];
} nvmlPciInfo_t;

/* ── Dim3 ────────────────────────────────────────────────── */
typedef struct {
    unsigned int x, y, z;
} dim3;

/* ── Stream and Event (opaque handles) ───────────────────── */
typedef void* cudaStream_t;
typedef void* cudaEvent_t;

/* ── Device properties ───────────────────────────────────── */
struct cudaDeviceProp {
    char     name[256];
    size_t   totalGlobalMem;
    size_t   sharedMemPerBlock;
    int      regsPerBlock;
    int      warpSize;
    size_t   memPitch;
    int      maxThreadsPerBlock;
    int      maxThreadsDim[3];
    int      maxGridSize[3];
    int      clockRate;
    size_t   totalConstMem;
    int      major;
    int      minor;
    size_t   textureAlignment;
    size_t   texturePitchAlignment;
    int      deviceOverlap;
    int      multiProcessorCount;
    int      kernelExecTimeoutEnabled;
    int      integrated;
    int      canMapHostMemory;
    int      computeMode;
    int      maxTexture1D;
    int      maxTexture1DMipmap;
    int      maxTexture1DLinear;
    int      maxTexture2D[2];
    int      maxTexture2DMipmap[2];
    int      maxTexture2DLinear[3];
    int      maxTexture2DGather[2];
    int      maxTexture3D[3];
    int      maxTexture3DAlt[3];
    int      maxTextureCubemap;
    int      maxTexture1DLayered[2];
    int      maxTexture2DLayered[3];
    int      maxTextureCubemapLayered[2];
    int      maxSurface1D;
    int      maxSurface2D[2];
    int      maxSurface3D[3];
    int      maxSurface1DLayered[2];
    int      maxSurface2DLayered[3];
    int      maxSurfaceCubemap;
    int      maxSurfaceCubemapLayered[2];
    size_t   surfaceAlignment;
    int      concurrentKernels;
    int      ECCEnabled;
    int      pciBusID;
    int      pciDeviceID;
    int      pciDomainID;
    int      tccDriver;
    int      asyncEngineCount;
    int      unifiedAddressing;
    int      memoryClockRate;
    int      memoryBusWidth;
    int      l2CacheSize;
    int      persistingL2CacheMaxSize;
    int      maxThreadsPerMultiProcessor;
    int      streamPrioritiesSupported;
    int      globalL1CacheSupported;
    int      localL1CacheSupported;
    size_t   sharedMemPerMultiprocessor;
    int      regsPerMultiprocessor;
    int      managedMemory;
    int      isMultiGpuBoard;
    int      multiGpuBoardGroupID;
    int      hostNativeAtomicSupported;
    int      singleToDoublePrecisionPerfRatio;
    int      pageableMemoryAccess;
    int      concurrentManagedAccess;
    int      computePreemptionSupported;
    int      canUseHostPointerForRegisteredMem;
    int      cooperativeLaunch;
    int      cooperativeMultiDeviceLaunch;
    size_t   sharedMemPerBlockOptin;
    int      pageableMemoryAccessUsesHostPageTables;
    int      directManagedMemAccessFromHost;
    int      maxBlocksPerMultiProcessor;
    int      accessPolicyMaxWindowSize;
    size_t   reservedSharedMemPerBlock;
    /* Pad to ensure struct is large enough */
    char     _padding[512];
};

#ifdef __cplusplus
}
#endif

#endif /* GPUSHARE_CUDA_DEFS_H */
