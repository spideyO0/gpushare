#ifndef GPUSHARE_PROTOCOL_H
#define GPUSHARE_PROTOCOL_H

#include <stdint.h>
#include <string.h>

/* MSVC does not support __attribute__((packed)) — use #pragma pack instead */
#ifdef _MSC_VER
  #define PACKED_STRUCT_BEGIN __pragma(pack(push, 1))
  #define PACKED_STRUCT_END   __pragma(pack(pop))
  #define ATTR_PACKED
#else
  #define PACKED_STRUCT_BEGIN
  #define PACKED_STRUCT_END
  #define ATTR_PACKED __attribute__((packed))
#endif

#define GPUSHARE_MAGIC       0x47505553  /* "GPUS" */
#define GPUSHARE_VERSION     1
#define GPUSHARE_DEFAULT_PORT 9847
#define GPUSHARE_MAX_MSG_SIZE (256 * 1024 * 1024)  /* 256 MB max single transfer */
#define GPUSHARE_HEADER_SIZE 16

/* ── Wire format ──────────────────────────────────────────────
 *  Bytes  Field
 *  0-3    magic   (0x47505553)
 *  4-7    length  (total message size including header)
 *  8-11   req_id  (client-assigned, echoed in response)
 *  12-13  opcode
 *  14-15  flags   (0x0001 = response, 0x0002 = error)
 *  16+    payload (opcode-specific)
 * ──────────────────────────────────────────────────────────── */

PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint32_t magic;
    uint32_t length;
    uint32_t req_id;
    uint16_t opcode;
    uint16_t flags;
} gs_header_t; PACKED_STRUCT_END

#define GS_FLAG_RESPONSE    0x0001
#define GS_FLAG_ERROR       0x0002
#define GS_FLAG_CHUNKED     0x0004
#define GS_FLAG_LAST_CHUNK  0x0008
#define GS_FLAG_COMPRESSED  0x0010

/* ── Opcodes ─────────────────────────────────────────────── */
enum gs_opcode {
    /* Session */
    GS_OP_INIT              = 0x0001,
    GS_OP_CLOSE             = 0x0002,
    GS_OP_PING              = 0x0003,

    /* Device management */
    GS_OP_GET_DEVICE_COUNT  = 0x0010,
    GS_OP_GET_DEVICE_PROPS  = 0x0011,
    GS_OP_SET_DEVICE        = 0x0012,

    /* Memory */
    GS_OP_MALLOC            = 0x0020,
    GS_OP_FREE              = 0x0021,
    GS_OP_MEMCPY_H2D       = 0x0022,  /* host→device: payload has data */
    GS_OP_MEMCPY_D2H       = 0x0023,  /* device→host: response has data */
    GS_OP_MEMCPY_D2D       = 0x0024,
    GS_OP_MEMSET            = 0x0025,
    GS_OP_MEMCPY_H2D_ASYNC = 0x0026,  /* async host→device with stream */
    GS_OP_MEMCPY_D2H_ASYNC = 0x0027,  /* async device→host with stream */
    GS_OP_MEMCPY_D2H_BULK  = 0x0028,  /* reserved for future bulk D2H streaming */

    /* Kernel execution */
    GS_OP_MODULE_LOAD       = 0x0030,  /* load PTX / cubin */
    GS_OP_MODULE_UNLOAD     = 0x0031,
    GS_OP_GET_FUNCTION      = 0x0032,
    GS_OP_LAUNCH_KERNEL     = 0x0033,

    /* Synchronization */
    GS_OP_DEVICE_SYNC       = 0x0040,
    GS_OP_STREAM_CREATE     = 0x0041,
    GS_OP_STREAM_DESTROY    = 0x0042,
    GS_OP_STREAM_SYNC       = 0x0043,
    GS_OP_EVENT_CREATE      = 0x0044,
    GS_OP_EVENT_DESTROY     = 0x0045,
    GS_OP_EVENT_RECORD      = 0x0046,
    GS_OP_EVENT_SYNC        = 0x0047,
    GS_OP_EVENT_ELAPSED     = 0x0048,

    /* Fat binary registration (for compiled CUDA apps) */
    GS_OP_REGISTER_FAT_BIN  = 0x0050,
    GS_OP_REGISTER_FUNCTION = 0x0051,
    GS_OP_UNREGISTER_FAT_BIN= 0x0052,

    /* Statistics / monitoring */
    GS_OP_GET_STATS         = 0x0060,

    /* Live GPU status (NVML-style: memory, utilization, temp, power) */
    GS_OP_GET_GPU_STATUS    = 0x0070,

    /* Generic library call (cuBLAS, cuDNN, cuFFT, cuSPARSE, etc.)
     * Payload: [u8 lib_id][u16 func_id][u32 args_size][args_data]
     * Response: [i32 return_code][u32 out_size][out_data] */
    GS_OP_LIB_CALL          = 0x0080,

    /* Library handle management */
    GS_OP_HANDLE_CREATE     = 0x0081,
    GS_OP_HANDLE_DESTROY    = 0x0082,
};

/* ── Payload structures ──────────────────────────────────── */

/* GS_OP_INIT request */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint32_t version;
    uint32_t client_type;  /* 0=native, 1=python */
} gs_init_req_t; PACKED_STRUCT_END

/* GS_OP_INIT response
 * Backward-compatible: old clients read only the first 12 bytes.
 * New clients check if payload >= 16 bytes to read capabilities. */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint32_t version;
    uint32_t session_id;
    uint32_t max_transfer_size;
    uint32_t capabilities;      /* bitmask of GS_CAP_* flags */
} gs_init_resp_t; PACKED_STRUCT_END

#define GS_CAP_ASYNC    0x01    /* server supports async memcpy opcodes */
#define GS_CAP_CHUNKED  0x02    /* server supports chunked transfers */
#define GS_CAP_COMPRESS 0x04    /* server supports compressed payloads */

/* GS_OP_GET_DEVICE_COUNT response */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    int32_t count;
} gs_device_count_resp_t; PACKED_STRUCT_END

/* GS_OP_GET_DEVICE_PROPS request */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    int32_t device;
} gs_device_props_req_t; PACKED_STRUCT_END

/* Simplified device properties (cross-platform compatible) */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    char     name[256];
    uint64_t total_global_mem;
    uint64_t shared_mem_per_block;
    int32_t  regs_per_block;
    int32_t  warp_size;
    int32_t  max_threads_per_block;
    int32_t  max_threads_dim[3];
    int32_t  max_grid_size[3];
    int32_t  clock_rate;
    int32_t  major;
    int32_t  minor;
    int32_t  multi_processor_count;
    int32_t  max_threads_per_mp;
    uint64_t total_const_mem;
    uint64_t mem_bus_width;
    uint64_t l2_cache_size;
} gs_device_props_t; PACKED_STRUCT_END

/* GS_OP_SET_DEVICE request */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    int32_t device;
} gs_set_device_req_t; PACKED_STRUCT_END

/* GS_OP_MALLOC request */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t size;
} gs_malloc_req_t; PACKED_STRUCT_END

/* GS_OP_MALLOC response */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t device_ptr;    /* server-side handle (opaque to client) */
    int32_t  cuda_error;
} gs_malloc_resp_t; PACKED_STRUCT_END

/* GS_OP_FREE request */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t device_ptr;
} gs_free_req_t; PACKED_STRUCT_END

/* GS_OP_MEMCPY_H2D request: header + data follows */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t device_ptr;
    uint64_t size;
    /* followed by `size` bytes of host data */
} gs_memcpy_h2d_req_t; PACKED_STRUCT_END

/* GS_OP_MEMCPY_D2H request */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t device_ptr;
    uint64_t size;
} gs_memcpy_d2h_req_t; PACKED_STRUCT_END
/* response: header + `size` bytes of data */

/* GS_OP_MEMCPY_D2D request */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t dst_ptr;
    uint64_t src_ptr;
    uint64_t size;
} gs_memcpy_d2d_req_t; PACKED_STRUCT_END

/* GS_OP_MEMCPY_H2D_ASYNC request: like H2D but with stream handle */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t device_ptr;
    uint64_t size;
    uint64_t stream_handle;
    /* followed by `size` bytes of host data */
} gs_memcpy_h2d_async_req_t; PACKED_STRUCT_END

/* GS_OP_MEMCPY_D2H_ASYNC request */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t device_ptr;
    uint64_t size;
    uint64_t stream_handle;
} gs_memcpy_d2h_async_req_t; PACKED_STRUCT_END

/* Chunked transfer headers (used with GS_FLAG_CHUNKED on H2D/D2H opcodes) */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t device_ptr;
    uint64_t total_size;
    uint64_t chunk_offset;
    uint32_t chunk_size;
    uint64_t stream_handle;
    /* followed by `chunk_size` bytes of host data */
} gs_memcpy_h2d_chunk_t; PACKED_STRUCT_END

PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t device_ptr;
    uint64_t total_size;
    uint64_t chunk_offset;
    uint32_t chunk_size;
    uint64_t stream_handle;
} gs_memcpy_d2h_chunk_t; PACKED_STRUCT_END

/* GS_OP_MEMCPY_D2H_BULK request — server streams back all chunks with
 * double-buffered GPU DMA overlapping network sends */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t device_ptr;
    uint64_t size;
    uint64_t stream_handle;
} gs_memcpy_d2h_bulk_req_t; PACKED_STRUCT_END

/* Chunk size for pipelined transfers */
#define GS_CHUNK_SIZE        (4 * 1024 * 1024)  /* 4 MB */
#define GS_CHUNK_THRESHOLD   GS_CHUNK_SIZE       /* use chunking above this */

/* GS_OP_MEMSET request */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t device_ptr;
    int32_t  value;
    uint64_t size;
} gs_memset_req_t; PACKED_STRUCT_END

/* GS_OP_MODULE_LOAD request: header + PTX/cubin data */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t data_size;
    /* followed by PTX or cubin data */
} gs_module_load_req_t; PACKED_STRUCT_END

/* GS_OP_MODULE_LOAD response */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t module_handle;
    int32_t  cuda_error;
} gs_module_load_resp_t; PACKED_STRUCT_END

/* GS_OP_GET_FUNCTION request */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t module_handle;
    char     func_name[256];
} gs_get_function_req_t; PACKED_STRUCT_END

/* GS_OP_GET_FUNCTION response */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t func_handle;
    int32_t  cuda_error;
} gs_get_function_resp_t; PACKED_STRUCT_END

/* GS_OP_LAUNCH_KERNEL request */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t func_handle;
    uint32_t grid_x, grid_y, grid_z;
    uint32_t block_x, block_y, block_z;
    uint32_t shared_mem;
    uint64_t stream_handle;   /* 0 = default stream */
    uint32_t num_args;
    uint32_t args_size;       /* total bytes of serialized args */
    /* followed by serialized kernel arguments:
       for each arg: [4B size][size bytes data]
       device pointers are sent as uint64_t handles */
} gs_launch_kernel_req_t; PACKED_STRUCT_END

/* GS_OP_STREAM_CREATE response */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t stream_handle;
} gs_stream_create_resp_t; PACKED_STRUCT_END

/* GS_OP_STREAM_DESTROY / STREAM_SYNC request */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t stream_handle;
} gs_stream_req_t; PACKED_STRUCT_END

/* GS_OP_EVENT_CREATE response */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t event_handle;
} gs_event_create_resp_t; PACKED_STRUCT_END

/* GS_OP_EVENT_RECORD request */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t event_handle;
    uint64_t stream_handle;
} gs_event_record_req_t; PACKED_STRUCT_END

/* GS_OP_EVENT_ELAPSED request */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t start_event;
    uint64_t end_event;
} gs_event_elapsed_req_t; PACKED_STRUCT_END

/* GS_OP_EVENT_ELAPSED response */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    float milliseconds;
} gs_event_elapsed_resp_t; PACKED_STRUCT_END

/* Generic response (for ops that only return cuda error) */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    int32_t cuda_error;
} gs_generic_resp_t; PACKED_STRUCT_END

/* GS_OP_REGISTER_FAT_BIN request */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t data_size;
    /* followed by fat binary data */
} gs_register_fatbin_req_t; PACKED_STRUCT_END

/* GS_OP_REGISTER_FAT_BIN response */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t fatbin_handle;
} gs_register_fatbin_resp_t; PACKED_STRUCT_END

/* GS_OP_REGISTER_FUNCTION request */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t fatbin_handle;
    uint64_t host_func;      /* client-side function pointer (used as key) */
    char     device_name[256];
} gs_register_function_req_t; PACKED_STRUCT_END

/* GS_OP_GET_STATS response */
#define GS_MAX_STAT_CLIENTS 32
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    /* Server uptime */
    uint64_t uptime_secs;
    /* Aggregate stats */
    uint64_t total_ops;
    uint64_t total_bytes_in;
    uint64_t total_bytes_out;
    uint64_t total_alloc_bytes;
    uint32_t active_clients;
    uint32_t total_connections;
    /* Per-client info (up to GS_MAX_STAT_CLIENTS) */
    uint32_t num_clients;  /* actual count in array below */
} gs_stats_header_t; PACKED_STRUCT_END

PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint32_t session_id;
    char     addr[64];
    uint64_t mem_allocated;
    uint64_t ops_count;
    uint64_t bytes_in;
    uint64_t bytes_out;
    uint64_t connected_secs;
} gs_stats_client_t; PACKED_STRUCT_END

/* GS_OP_GET_GPU_STATUS response — live GPU metrics */
PACKED_STRUCT_BEGIN typedef struct ATTR_PACKED {
    uint64_t mem_total;
    uint64_t mem_used;
    uint64_t mem_free;
    uint32_t gpu_utilization;      /* 0-100 */
    uint32_t mem_utilization;      /* 0-100 */
    uint32_t temperature;          /* Celsius */
    uint32_t power_draw_mw;        /* milliwatts */
    uint32_t power_limit_mw;       /* milliwatts */
    uint32_t fan_speed;            /* 0-100 */
    uint32_t clock_sm_mhz;
    uint32_t clock_mem_mhz;
} gs_gpu_status_t; PACKED_STRUCT_END

/* ── Utility functions ───────────────────────────────────── */

static inline void gs_header_init(gs_header_t *h, uint16_t opcode, uint32_t req_id, uint32_t total_len) {
    h->magic  = GPUSHARE_MAGIC;
    h->length = total_len;
    h->req_id = req_id;
    h->opcode = opcode;
    h->flags  = 0;
}

static inline int gs_header_validate(const gs_header_t *h) {
    return h->magic == GPUSHARE_MAGIC && h->length >= GPUSHARE_HEADER_SIZE
           && h->length <= GPUSHARE_MAX_MSG_SIZE;
}

#endif /* GPUSHARE_PROTOCOL_H */
