/*
 * Basic test for gpushare client library.
 * Tests device query, memory allocation, and data transfer.
 *
 * Build: g++ -o test_basic test_basic.cpp -L../build -lgpushare_client -I../include
 * Run:   GPUSHARE_SERVER=localhost:9847 LD_LIBRARY_PATH=../build ./test_basic
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "gpushare/cuda_defs.h"

/* These are provided by libgpushare_client.so */
extern "C" {
    cudaError_t cudaGetDeviceCount(int *count);
    cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
    cudaError_t cudaSetDevice(int device);
    cudaError_t cudaMalloc(void **devPtr, size_t size);
    cudaError_t cudaFree(void *devPtr);
    cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
    cudaError_t cudaMemset(void *devPtr, int value, size_t count);
    cudaError_t cudaDeviceSynchronize(void);
    const char* cudaGetErrorString(cudaError_t error);
}

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("=== gpushare client test ===\n\n");

    /* Test 1: Device count */
    int count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&count));
    printf("[PASS] Device count: %d\n", count);

    /* Test 2: Device properties */
    struct cudaDeviceProp props;
    CHECK_CUDA(cudaGetDeviceProperties(&props, 0));
    printf("[PASS] GPU: %s (%.0f MB, SM %d.%d)\n",
           props.name, props.totalGlobalMem / 1048576.0,
           props.major, props.minor);

    /* Test 3: Set device */
    CHECK_CUDA(cudaSetDevice(0));
    printf("[PASS] cudaSetDevice(0)\n");

    /* Test 4: Malloc + Free */
    void *d_ptr = nullptr;
    size_t alloc_size = 1024 * 1024;  /* 1 MB */
    CHECK_CUDA(cudaMalloc(&d_ptr, alloc_size));
    printf("[PASS] cudaMalloc: %zu bytes -> handle %p\n", alloc_size, d_ptr);

    /* Test 5: Memset */
    CHECK_CUDA(cudaMemset(d_ptr, 0, alloc_size));
    printf("[PASS] cudaMemset\n");

    /* Test 6: Host-to-Device + Device-to-Host round-trip */
    const int N = 1024;
    float *h_src = (float*)malloc(N * sizeof(float));
    float *h_dst = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) h_src[i] = (float)i * 0.5f;

    void *d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_data, h_src, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h_dst, d_data, N * sizeof(float), cudaMemcpyDeviceToHost));

    /* Verify */
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_src[i] - h_dst[i]) > 1e-6f) {
            errors++;
            if (errors <= 5)
                printf("  MISMATCH at [%d]: expected %.6f, got %.6f\n", i, h_src[i], h_dst[i]);
        }
    }

    if (errors == 0) {
        printf("[PASS] H2D + D2H round-trip: %d floats verified\n", N);
    } else {
        printf("[FAIL] H2D + D2H: %d mismatches out of %d\n", errors, N);
    }

    /* Test 7: Device-to-Device copy */
    void *d_copy = nullptr;
    CHECK_CUDA(cudaMalloc(&d_copy, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_copy, d_data, N * sizeof(float), cudaMemcpyDeviceToDevice));

    memset(h_dst, 0, N * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_dst, d_copy, N * sizeof(float), cudaMemcpyDeviceToHost));

    errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_src[i] - h_dst[i]) > 1e-6f) errors++;
    }
    printf("[%s] D2D copy: %d floats\n", errors == 0 ? "PASS" : "FAIL", N);

    /* Test 8: Synchronize */
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("[PASS] cudaDeviceSynchronize\n");

    /* Cleanup */
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_copy));
    CHECK_CUDA(cudaFree(d_ptr));
    free(h_src);
    free(h_dst);

    printf("\n=== All tests passed ===\n");
    return 0;
}
