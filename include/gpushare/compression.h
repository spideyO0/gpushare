#ifndef GPUSHARE_COMPRESSION_H
#define GPUSHARE_COMPRESSION_H

/*
 * Phase 10: Transfer compression.
 *
 * Optional LZ4/zstd compression for data payloads. Only compresses when
 * the data is compressible (ratio < 0.8 threshold) — dense random data
 * is sent uncompressed to avoid wasting CPU cycles.
 *
 * Compressed payload format (prepended when GS_FLAG_COMPRESSED is set):
 *   [uint64_t original_size][compressed_data...]
 *
 * Build: auto-detected. LZ4 preferred (faster), zstd as fallback (better ratio).
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef GPUSHARE_HAS_LZ4
  #include <lz4.h>
#endif

#ifdef GPUSHARE_HAS_ZSTD
  #include <zstd.h>
#endif

/* Compression algorithm IDs (stored in compressed header for future compat) */
#define GS_COMPRESS_NONE  0
#define GS_COMPRESS_LZ4   1
#define GS_COMPRESS_ZSTD  2

/* Compressed payload header: original size + algorithm ID */
struct gs_compress_header_t {
    uint64_t original_size;
    uint8_t  algorithm;      /* GS_COMPRESS_LZ4 or GS_COMPRESS_ZSTD */
    uint8_t  reserved[7];    /* pad to 16 bytes */
};
static_assert(sizeof(gs_compress_header_t) == 16, "compress header must be 16 bytes");

/* Minimum size to attempt compression (below this, overhead > benefit) */
#define GS_COMPRESS_MIN_SIZE  4096

/* Compression ratio threshold — skip if compressed/original > this */
#define GS_COMPRESS_RATIO_THRESHOLD  0.85

/* ── Compression functions ───────────────────────────────── */

/*
 * Try to compress `src` (size `src_len`) into `dst`.
 * Returns compressed size on success (including the 16-byte header),
 * or 0 if compression should be skipped (data incompressible or too small).
 * `dst` must be pre-allocated with at least gs_compress_bound(src_len) bytes.
 */
inline size_t gs_compress(const void *src, size_t src_len,
                          void *dst, size_t dst_cap) {
    if (src_len < GS_COMPRESS_MIN_SIZE) return 0;
    if (dst_cap < sizeof(gs_compress_header_t) + src_len) return 0;

    uint8_t *out = (uint8_t*)dst;
    gs_compress_header_t hdr = {};
    hdr.original_size = src_len;

    size_t compressed_size = 0;

#ifdef GPUSHARE_HAS_LZ4
    hdr.algorithm = GS_COMPRESS_LZ4;
    int max_dst = LZ4_compressBound((int)src_len);
    if (dst_cap >= sizeof(hdr) + (size_t)max_dst) {
        int ret = LZ4_compress_default(
            (const char*)src, (char*)(out + sizeof(hdr)),
            (int)src_len, max_dst);
        if (ret > 0) compressed_size = (size_t)ret;
    }
#elif defined(GPUSHARE_HAS_ZSTD)
    hdr.algorithm = GS_COMPRESS_ZSTD;
    size_t bound = ZSTD_compressBound(src_len);
    if (dst_cap >= sizeof(hdr) + bound) {
        size_t ret = ZSTD_compress(
            out + sizeof(hdr), bound, src, src_len, 1 /* fast level */);
        if (!ZSTD_isError(ret)) compressed_size = ret;
    }
#endif

    if (compressed_size == 0) return 0;

    /* Check ratio — skip if not worth it */
    double ratio = (double)compressed_size / (double)src_len;
    if (ratio > GS_COMPRESS_RATIO_THRESHOLD) return 0;

    memcpy(out, &hdr, sizeof(hdr));
    return sizeof(hdr) + compressed_size;
}

/*
 * Decompress `src` (size `src_len`, includes 16-byte header) into `dst`.
 * `dst` must be pre-allocated with at least `original_size` bytes
 * (read from the header). Returns original size on success, 0 on failure.
 */
inline size_t gs_decompress(const void *src, size_t src_len,
                            void *dst, size_t dst_cap) {
    if (src_len < sizeof(gs_compress_header_t)) return 0;

    gs_compress_header_t hdr;
    memcpy(&hdr, src, sizeof(hdr));

    if (hdr.original_size > dst_cap) return 0;

    const uint8_t *compressed = (const uint8_t*)src + sizeof(hdr);
    size_t compressed_len = src_len - sizeof(hdr);

#ifdef GPUSHARE_HAS_LZ4
    if (hdr.algorithm == GS_COMPRESS_LZ4) {
        int ret = LZ4_decompress_safe(
            (const char*)compressed, (char*)dst,
            (int)compressed_len, (int)hdr.original_size);
        return (ret > 0) ? (size_t)ret : 0;
    }
#endif

#ifdef GPUSHARE_HAS_ZSTD
    if (hdr.algorithm == GS_COMPRESS_ZSTD) {
        size_t ret = ZSTD_decompress(
            dst, hdr.original_size, compressed, compressed_len);
        return ZSTD_isError(ret) ? 0 : ret;
    }
#endif

    return 0;  /* unknown algorithm */
}

/* Maximum compressed output size for a given input size */
inline size_t gs_compress_bound(size_t src_len) {
#ifdef GPUSHARE_HAS_LZ4
    return sizeof(gs_compress_header_t) + (size_t)LZ4_compressBound((int)src_len);
#elif defined(GPUSHARE_HAS_ZSTD)
    return sizeof(gs_compress_header_t) + ZSTD_compressBound(src_len);
#else
    return 0;  /* no compression available */
#endif
}

/* Check if any compression algorithm is available */
inline bool gs_compression_available() {
#if defined(GPUSHARE_HAS_LZ4) || defined(GPUSHARE_HAS_ZSTD)
    return true;
#else
    return false;
#endif
}

/* Return the name of the active compression algorithm */
inline const char *gs_compression_name() {
#ifdef GPUSHARE_HAS_LZ4
    return "lz4";
#elif defined(GPUSHARE_HAS_ZSTD)
    return "zstd";
#else
    return "none";
#endif
}

#endif /* GPUSHARE_COMPRESSION_H */
