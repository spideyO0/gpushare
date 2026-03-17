cmake_minimum_required(VERSION 3.18)
project(gpushare LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Common include path
include_directories(${CMAKE_SOURCE_DIR}/include)

# ── RDMA / InfiniBand support (Phase 9) — auto-detected ────
# Automatically enables RDMA if rdma-core (libibverbs + librdmacm) is found.
# Use -DENABLE_RDMA=OFF to force-disable even when libraries are present.
option(ENABLE_RDMA "Enable RDMA/InfiniBand transport (auto-detected)" ON)
set(HAVE_RDMA OFF)
if(ENABLE_RDMA AND NOT WIN32)
    find_library(IBVERBS_LIB ibverbs)
    find_library(RDMACM_LIB rdmacm)
    find_path(IBVERBS_INCLUDE infiniband/verbs.h)
    find_path(RDMACM_INCLUDE rdma/rdma_cma.h)
    if(IBVERBS_LIB AND RDMACM_LIB AND IBVERBS_INCLUDE AND RDMACM_INCLUDE)
        set(HAVE_RDMA ON)
        message(STATUS "RDMA: ENABLED (libibverbs + librdmacm auto-detected)")
    else()
        message(STATUS "RDMA: not available (install rdma-core for InfiniBand support)")
    endif()
else()
    if(WIN32)
        message(STATUS "RDMA: not supported on Windows")
    elseif(NOT ENABLE_RDMA)
        message(STATUS "RDMA: disabled by -DENABLE_RDMA=OFF")
    endif()
endif()

# ── Server (requires CUDA toolkit) ──────────────────────────
option(BUILD_SERVER "Build the GPU server (requires CUDA)" ON)
if(BUILD_SERVER)
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        add_executable(gpushare-server
            server/server.cpp
            server/generated_dispatch.cpp
            server/generated_all_dispatch.cpp
        )
        target_link_libraries(gpushare-server PRIVATE
            CUDA::cudart
            CUDA::cuda_driver
            pthread
            dl
        )
        target_compile_definitions(gpushare-server PRIVATE
            _GNU_SOURCE
        )
        if(HAVE_RDMA)
            target_compile_definitions(gpushare-server PRIVATE GPUSHARE_HAS_RDMA)
            target_link_libraries(gpushare-server PRIVATE ${IBVERBS_LIB} ${RDMACM_LIB})
            target_include_directories(gpushare-server PRIVATE ${IBVERBS_INCLUDE} ${RDMACM_INCLUDE})
        endif()
        install(TARGETS gpushare-server RUNTIME DESTINATION bin)
        message(STATUS "Server: ENABLED (CUDA ${CUDAToolkit_VERSION})")
    else()
        message(WARNING "CUDA toolkit not found — server will NOT be built")
        set(BUILD_SERVER OFF)
    endif()
endif()

# ── Client library (no CUDA required) ───────────────────────
option(BUILD_CLIENT "Build the client shared library" ON)
if(BUILD_CLIENT)
    add_library(gpushare_client SHARED
        client/gpushare_client.cpp
        client/generated_stubs.cpp
        client/generated_all_stubs.cpp
    )

    # Don't link against CUDA — that's the whole point
    if(WIN32)
        target_link_libraries(gpushare_client PRIVATE ws2_32)
        # Statically link MinGW runtimes so the DLL has no external dependencies
        # (libstdc++-6.dll, libgcc_s_seh-1.dll, libwinpthread-1.dll)
        if(MINGW)
            target_link_options(gpushare_client PRIVATE
                -static-libgcc -static-libstdc++ -Wl,-Bstatic -lpthread -Wl,-Bdynamic
            )
        endif()
    else()
        # dl is needed for dlopen/dlsym (local GPU passthrough)
        # pthread is needed for the recv thread (Phase 4 pipelining)
        target_link_libraries(gpushare_client PRIVATE dl pthread)
        if(HAVE_RDMA)
            target_compile_definitions(gpushare_client PRIVATE GPUSHARE_HAS_RDMA)
            target_link_libraries(gpushare_client PRIVATE ${IBVERBS_LIB} ${RDMACM_LIB})
            target_include_directories(gpushare_client PRIVATE ${IBVERBS_INCLUDE} ${RDMACM_INCLUDE})
        endif()
    endif()

    # Export CUDA symbols so LD_PRELOAD / DYLD_INSERT_LIBRARIES works
    set_target_properties(gpushare_client PROPERTIES
        OUTPUT_NAME "gpushare_client"
        VERSION 1.0.0
        SOVERSION 1
        C_VISIBILITY_PRESET default
        CXX_VISIBILITY_PRESET default
    )

    # On Windows (MinGW), drop the "lib" prefix so the DLL is named gpushare_client.dll
    if(WIN32)
        set_target_properties(gpushare_client PROPERTIES PREFIX "")
    endif()

    install(TARGETS gpushare_client LIBRARY DESTINATION lib)
    message(STATUS "Client: ENABLED")
endif()

# ── Python module install hint ───────────────────────────────
message(STATUS "")
message(STATUS "Build targets:")
if(BUILD_SERVER)
    message(STATUS "  gpushare-server  — GPU server daemon")
endif()
if(BUILD_CLIENT)
    message(STATUS "  gpushare_client  — Client shared library (libgpushare_client.so)")
endif()
message(STATUS "")
message(STATUS "Python client: pip install -e python/  (no build needed)")
