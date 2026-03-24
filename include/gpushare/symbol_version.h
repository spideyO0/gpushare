/* Symbol version header for gpushare client library.
 * This header uses linker script style symbol versioning to ensure
 * CUDA Runtime API functions (cuda*) get the @@libcudart.so.12 version tag.
 */
#ifndef GPUSHARE_SYMBOL_VERSION_H
#define GPUSHARE_SYMBOL_VERSION_H

#ifdef __cplusplus
extern "C" {
#endif

/* Symbol version macros for libcudart.so.12 compatibility */
/* These tell the linker which version to assign to exported symbols */
#define CUDA_SYMBOL_VERSION __attribute__((symver("cuda*@@libcudart.so.12")))
#define DRIVER_SYMBOL_VERSION __attribute__((symver("cu*@@GPUSHARE_1.0")))
#define LIB_SYMBOL_VERSION __attribute__((symver("nvl*@@GPUSHARE_1.0")))

#ifdef __cplusplus
}
#endif

#endif /* GPUSHARE_SYMBOL_VERSION_H */
