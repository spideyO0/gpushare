cmake_minimum_required(VERSION 3.18)
project(gpushare LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Common include path
include_directories(${CMAKE_SOURCE_DIR}/include)

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
    else()
        # dl is needed for dlopen/dlsym (local GPU passthrough)
        target_link_libraries(gpushare_client PRIVATE dl)
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
