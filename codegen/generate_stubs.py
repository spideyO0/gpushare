#!/usr/bin/env python3
"""
gpushare code generator — auto-generates client stubs and server dispatch
for cuBLAS, cuDNN, and other CUDA libraries.

Usage: python generate_stubs.py

Outputs:
  ../client/generated_stubs.cpp   — client-side exported functions
  ../server/generated_dispatch.cpp — server-side dispatch handler
  ../include/gpushare/generated_types.h — types for all libraries
"""

import os
import json

# ── Argument types ──────────────────────────────────────────
# Each arg has a "kind" that tells the codegen how to serialize/deserialize:
#   SCALAR      — small value sent inline (int, float, enum)
#   DEV_PTR     — device pointer (uint64_t handle, already on GPU)
#   HANDLE      — library handle (opaque, mapped client↔server)
#   HANDLE_OUT  — output handle pointer (created on server, returned to client)
#   HOST_IN     — host pointer to read-only data (sent to server)
#   HOST_OUT    — host pointer to output data (returned from server)
#   HOST_INOUT  — host pointer to in/out data

# Size constants
SZ = {
    "i32": 4, "u32": 4, "i64": 8, "u64": 8,
    "f32": 4, "f64": 8, "f16": 2,
    "ptr": 8, "handle": 8, "size_t": 8,
    "enum": 4, "bool": 4,
}

# ── Library definitions ─────────────────────────────────────
# Each function: (name, return_type, [(arg_name, c_type, kind, size_or_ref)])
# size_or_ref: byte size for fixed, or arg name reference for variable

LIBS = {}

# ── cuBLAS ──────────────────────────────────────────────────
LIBS["cublas"] = {
    "id": 1,
    "header": "cublas_v2.h",
    "link": "cublas",
    "handle_type": "cublasHandle_t",
    "functions": [
        # Handle management
        ("cublasCreate_v2", "i32", [("handle", "ptr", "HANDLE_OUT", 8)]),
        ("cublasDestroy_v2", "i32", [("handle", "ptr", "HANDLE", 8)]),
        ("cublasSetStream_v2", "i32", [("handle", "ptr", "HANDLE", 8), ("stream", "ptr", "DEV_PTR", 8)]),
        ("cublasGetStream_v2", "i32", [("handle", "ptr", "HANDLE", 8), ("stream", "ptr", "HOST_OUT", 8)]),
        ("cublasSetMathMode", "i32", [("handle", "ptr", "HANDLE", 8), ("mode", "i32", "SCALAR", 4)]),
        ("cublasSetWorkspace_v2", "i32", [("handle", "ptr", "HANDLE", 8), ("workspace", "ptr", "DEV_PTR", 8), ("size", "u64", "SCALAR", 8)]),

        # Level 1: vector ops
        ("cublasSaxpy_v2", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("n", "i32", "SCALAR", 4),
            ("alpha", "ptr", "HOST_IN", 4), ("x", "ptr", "DEV_PTR", 8),
            ("incx", "i32", "SCALAR", 4), ("y", "ptr", "DEV_PTR", 8), ("incy", "i32", "SCALAR", 4)]),
        ("cublasDaxpy_v2", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("n", "i32", "SCALAR", 4),
            ("alpha", "ptr", "HOST_IN", 8), ("x", "ptr", "DEV_PTR", 8),
            ("incx", "i32", "SCALAR", 4), ("y", "ptr", "DEV_PTR", 8), ("incy", "i32", "SCALAR", 4)]),
        ("cublasSscal_v2", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("n", "i32", "SCALAR", 4),
            ("alpha", "ptr", "HOST_IN", 4), ("x", "ptr", "DEV_PTR", 8), ("incx", "i32", "SCALAR", 4)]),
        ("cublasDscal_v2", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("n", "i32", "SCALAR", 4),
            ("alpha", "ptr", "HOST_IN", 8), ("x", "ptr", "DEV_PTR", 8), ("incx", "i32", "SCALAR", 4)]),
        ("cublasScopy_v2", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("n", "i32", "SCALAR", 4),
            ("x", "ptr", "DEV_PTR", 8), ("incx", "i32", "SCALAR", 4),
            ("y", "ptr", "DEV_PTR", 8), ("incy", "i32", "SCALAR", 4)]),
        ("cublasDcopy_v2", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("n", "i32", "SCALAR", 4),
            ("x", "ptr", "DEV_PTR", 8), ("incx", "i32", "SCALAR", 4),
            ("y", "ptr", "DEV_PTR", 8), ("incy", "i32", "SCALAR", 4)]),
        ("cublasSdot_v2", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("n", "i32", "SCALAR", 4),
            ("x", "ptr", "DEV_PTR", 8), ("incx", "i32", "SCALAR", 4),
            ("y", "ptr", "DEV_PTR", 8), ("incy", "i32", "SCALAR", 4),
            ("result", "ptr", "HOST_OUT", 4)]),
        ("cublasDdot_v2", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("n", "i32", "SCALAR", 4),
            ("x", "ptr", "DEV_PTR", 8), ("incx", "i32", "SCALAR", 4),
            ("y", "ptr", "DEV_PTR", 8), ("incy", "i32", "SCALAR", 4),
            ("result", "ptr", "HOST_OUT", 8)]),
        ("cublasSnrm2_v2", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("n", "i32", "SCALAR", 4),
            ("x", "ptr", "DEV_PTR", 8), ("incx", "i32", "SCALAR", 4),
            ("result", "ptr", "HOST_OUT", 4)]),
        ("cublasSasum_v2", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("n", "i32", "SCALAR", 4),
            ("x", "ptr", "DEV_PTR", 8), ("incx", "i32", "SCALAR", 4),
            ("result", "ptr", "HOST_OUT", 4)]),
        ("cublasIsamax_v2", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("n", "i32", "SCALAR", 4),
            ("x", "ptr", "DEV_PTR", 8), ("incx", "i32", "SCALAR", 4),
            ("result", "ptr", "HOST_OUT", 4)]),

        # Level 2: matrix-vector
        ("cublasSgemv_v2", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("trans", "i32", "SCALAR", 4),
            ("m", "i32", "SCALAR", 4), ("n", "i32", "SCALAR", 4),
            ("alpha", "ptr", "HOST_IN", 4), ("A", "ptr", "DEV_PTR", 8),
            ("lda", "i32", "SCALAR", 4), ("x", "ptr", "DEV_PTR", 8),
            ("incx", "i32", "SCALAR", 4), ("beta", "ptr", "HOST_IN", 4),
            ("y", "ptr", "DEV_PTR", 8), ("incy", "i32", "SCALAR", 4)]),
        ("cublasDgemv_v2", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("trans", "i32", "SCALAR", 4),
            ("m", "i32", "SCALAR", 4), ("n", "i32", "SCALAR", 4),
            ("alpha", "ptr", "HOST_IN", 8), ("A", "ptr", "DEV_PTR", 8),
            ("lda", "i32", "SCALAR", 4), ("x", "ptr", "DEV_PTR", 8),
            ("incx", "i32", "SCALAR", 4), ("beta", "ptr", "HOST_IN", 8),
            ("y", "ptr", "DEV_PTR", 8), ("incy", "i32", "SCALAR", 4)]),

        # Level 3: matrix-matrix (THE critical ones for PyTorch)
        ("cublasSgemm_v2", "i32", [
            ("handle", "ptr", "HANDLE", 8),
            ("transa", "i32", "SCALAR", 4), ("transb", "i32", "SCALAR", 4),
            ("m", "i32", "SCALAR", 4), ("n", "i32", "SCALAR", 4), ("k", "i32", "SCALAR", 4),
            ("alpha", "ptr", "HOST_IN", 4), ("A", "ptr", "DEV_PTR", 8), ("lda", "i32", "SCALAR", 4),
            ("B", "ptr", "DEV_PTR", 8), ("ldb", "i32", "SCALAR", 4),
            ("beta", "ptr", "HOST_IN", 4), ("C", "ptr", "DEV_PTR", 8), ("ldc", "i32", "SCALAR", 4)]),
        ("cublasDgemm_v2", "i32", [
            ("handle", "ptr", "HANDLE", 8),
            ("transa", "i32", "SCALAR", 4), ("transb", "i32", "SCALAR", 4),
            ("m", "i32", "SCALAR", 4), ("n", "i32", "SCALAR", 4), ("k", "i32", "SCALAR", 4),
            ("alpha", "ptr", "HOST_IN", 8), ("A", "ptr", "DEV_PTR", 8), ("lda", "i32", "SCALAR", 4),
            ("B", "ptr", "DEV_PTR", 8), ("ldb", "i32", "SCALAR", 4),
            ("beta", "ptr", "HOST_IN", 8), ("C", "ptr", "DEV_PTR", 8), ("ldc", "i32", "SCALAR", 4)]),
        ("cublasHgemm", "i32", [
            ("handle", "ptr", "HANDLE", 8),
            ("transa", "i32", "SCALAR", 4), ("transb", "i32", "SCALAR", 4),
            ("m", "i32", "SCALAR", 4), ("n", "i32", "SCALAR", 4), ("k", "i32", "SCALAR", 4),
            ("alpha", "ptr", "HOST_IN", 2), ("A", "ptr", "DEV_PTR", 8), ("lda", "i32", "SCALAR", 4),
            ("B", "ptr", "DEV_PTR", 8), ("ldb", "i32", "SCALAR", 4),
            ("beta", "ptr", "HOST_IN", 2), ("C", "ptr", "DEV_PTR", 8), ("ldc", "i32", "SCALAR", 4)]),

        # GemmEx — the one PyTorch uses most (mixed precision)
        ("cublasGemmEx", "i32", [
            ("handle", "ptr", "HANDLE", 8),
            ("transa", "i32", "SCALAR", 4), ("transb", "i32", "SCALAR", 4),
            ("m", "i32", "SCALAR", 4), ("n", "i32", "SCALAR", 4), ("k", "i32", "SCALAR", 4),
            ("alpha", "ptr", "HOST_IN", 8),
            ("A", "ptr", "DEV_PTR", 8), ("Atype", "i32", "SCALAR", 4), ("lda", "i32", "SCALAR", 4),
            ("B", "ptr", "DEV_PTR", 8), ("Btype", "i32", "SCALAR", 4), ("ldb", "i32", "SCALAR", 4),
            ("beta", "ptr", "HOST_IN", 8),
            ("C", "ptr", "DEV_PTR", 8), ("Ctype", "i32", "SCALAR", 4), ("ldc", "i32", "SCALAR", 4),
            ("computeType", "i32", "SCALAR", 4), ("algo", "i32", "SCALAR", 4)]),

        # Strided batched (used for batch operations in transformers)
        ("cublasSgemmStridedBatched", "i32", [
            ("handle", "ptr", "HANDLE", 8),
            ("transa", "i32", "SCALAR", 4), ("transb", "i32", "SCALAR", 4),
            ("m", "i32", "SCALAR", 4), ("n", "i32", "SCALAR", 4), ("k", "i32", "SCALAR", 4),
            ("alpha", "ptr", "HOST_IN", 4),
            ("A", "ptr", "DEV_PTR", 8), ("lda", "i32", "SCALAR", 4), ("strideA", "i64", "SCALAR", 8),
            ("B", "ptr", "DEV_PTR", 8), ("ldb", "i32", "SCALAR", 4), ("strideB", "i64", "SCALAR", 8),
            ("beta", "ptr", "HOST_IN", 4),
            ("C", "ptr", "DEV_PTR", 8), ("ldc", "i32", "SCALAR", 4), ("strideC", "i64", "SCALAR", 8),
            ("batchCount", "i32", "SCALAR", 4)]),
        ("cublasGemmStridedBatchedEx", "i32", [
            ("handle", "ptr", "HANDLE", 8),
            ("transa", "i32", "SCALAR", 4), ("transb", "i32", "SCALAR", 4),
            ("m", "i32", "SCALAR", 4), ("n", "i32", "SCALAR", 4), ("k", "i32", "SCALAR", 4),
            ("alpha", "ptr", "HOST_IN", 8),
            ("A", "ptr", "DEV_PTR", 8), ("Atype", "i32", "SCALAR", 4), ("lda", "i32", "SCALAR", 4), ("strideA", "i64", "SCALAR", 8),
            ("B", "ptr", "DEV_PTR", 8), ("Btype", "i32", "SCALAR", 4), ("ldb", "i32", "SCALAR", 4), ("strideB", "i64", "SCALAR", 8),
            ("beta", "ptr", "HOST_IN", 8),
            ("C", "ptr", "DEV_PTR", 8), ("Ctype", "i32", "SCALAR", 4), ("ldc", "i32", "SCALAR", 4), ("strideC", "i64", "SCALAR", 8),
            ("batchCount", "i32", "SCALAR", 4),
            ("computeType", "i32", "SCALAR", 4), ("algo", "i32", "SCALAR", 4)]),

        # Triangular solve (used in some models)
        ("cublasStrsm_v2", "i32", [
            ("handle", "ptr", "HANDLE", 8),
            ("side", "i32", "SCALAR", 4), ("uplo", "i32", "SCALAR", 4),
            ("trans", "i32", "SCALAR", 4), ("diag", "i32", "SCALAR", 4),
            ("m", "i32", "SCALAR", 4), ("n", "i32", "SCALAR", 4),
            ("alpha", "ptr", "HOST_IN", 4), ("A", "ptr", "DEV_PTR", 8), ("lda", "i32", "SCALAR", 4),
            ("B", "ptr", "DEV_PTR", 8), ("ldb", "i32", "SCALAR", 4)]),
        ("cublasDtrsm_v2", "i32", [
            ("handle", "ptr", "HANDLE", 8),
            ("side", "i32", "SCALAR", 4), ("uplo", "i32", "SCALAR", 4),
            ("trans", "i32", "SCALAR", 4), ("diag", "i32", "SCALAR", 4),
            ("m", "i32", "SCALAR", 4), ("n", "i32", "SCALAR", 4),
            ("alpha", "ptr", "HOST_IN", 8), ("A", "ptr", "DEV_PTR", 8), ("lda", "i32", "SCALAR", 4),
            ("B", "ptr", "DEV_PTR", 8), ("ldb", "i32", "SCALAR", 4)]),

        # Syrkx (symmetric rank-k update)
        ("cublasSsyrk_v2", "i32", [
            ("handle", "ptr", "HANDLE", 8),
            ("uplo", "i32", "SCALAR", 4), ("trans", "i32", "SCALAR", 4),
            ("n", "i32", "SCALAR", 4), ("k", "i32", "SCALAR", 4),
            ("alpha", "ptr", "HOST_IN", 4), ("A", "ptr", "DEV_PTR", 8), ("lda", "i32", "SCALAR", 4),
            ("beta", "ptr", "HOST_IN", 4), ("C", "ptr", "DEV_PTR", 8), ("ldc", "i32", "SCALAR", 4)]),

        # Geam (matrix add/transpose)
        ("cublasSgeam", "i32", [
            ("handle", "ptr", "HANDLE", 8),
            ("transa", "i32", "SCALAR", 4), ("transb", "i32", "SCALAR", 4),
            ("m", "i32", "SCALAR", 4), ("n", "i32", "SCALAR", 4),
            ("alpha", "ptr", "HOST_IN", 4), ("A", "ptr", "DEV_PTR", 8), ("lda", "i32", "SCALAR", 4),
            ("beta", "ptr", "HOST_IN", 4), ("B", "ptr", "DEV_PTR", 8), ("ldb", "i32", "SCALAR", 4),
            ("C", "ptr", "DEV_PTR", 8), ("ldc", "i32", "SCALAR", 4)]),
        ("cublasDgeam", "i32", [
            ("handle", "ptr", "HANDLE", 8),
            ("transa", "i32", "SCALAR", 4), ("transb", "i32", "SCALAR", 4),
            ("m", "i32", "SCALAR", 4), ("n", "i32", "SCALAR", 4),
            ("alpha", "ptr", "HOST_IN", 8), ("A", "ptr", "DEV_PTR", 8), ("lda", "i32", "SCALAR", 4),
            ("beta", "ptr", "HOST_IN", 8), ("B", "ptr", "DEV_PTR", 8), ("ldb", "i32", "SCALAR", 4),
            ("C", "ptr", "DEV_PTR", 8), ("ldc", "i32", "SCALAR", 4)]),

        # Version/pointer mode
        ("cublasGetVersion_v2", "i32", [("handle", "ptr", "HANDLE", 8), ("version", "ptr", "HOST_OUT", 4)]),
        ("cublasSetPointerMode_v2", "i32", [("handle", "ptr", "HANDLE", 8), ("mode", "i32", "SCALAR", 4)]),
        ("cublasGetPointerMode_v2", "i32", [("handle", "ptr", "HANDLE", 8), ("mode", "ptr", "HOST_OUT", 4)]),
    ]
}

# ── cuDNN ───────────────────────────────────────────────────
LIBS["cudnn"] = {
    "id": 2,
    "header": "cudnn.h",
    "link": "cudnn",
    "handle_type": "cudnnHandle_t",
    "functions": [
        # Handle management
        ("cudnnCreate", "i32", [("handle", "ptr", "HANDLE_OUT", 8)]),
        ("cudnnDestroy", "i32", [("handle", "ptr", "HANDLE", 8)]),
        ("cudnnSetStream", "i32", [("handle", "ptr", "HANDLE", 8), ("stream", "ptr", "DEV_PTR", 8)]),
        ("cudnnGetStream", "i32", [("handle", "ptr", "HANDLE", 8), ("stream", "ptr", "HOST_OUT", 8)]),

        # Tensor descriptor
        ("cudnnCreateTensorDescriptor", "i32", [("desc", "ptr", "HANDLE_OUT", 8)]),
        ("cudnnDestroyTensorDescriptor", "i32", [("desc", "ptr", "HANDLE", 8)]),
        ("cudnnSetTensor4dDescriptor", "i32", [
            ("desc", "ptr", "HANDLE", 8), ("format", "i32", "SCALAR", 4), ("dataType", "i32", "SCALAR", 4),
            ("n", "i32", "SCALAR", 4), ("c", "i32", "SCALAR", 4), ("h", "i32", "SCALAR", 4), ("w", "i32", "SCALAR", 4)]),
        ("cudnnSetTensor4dDescriptorEx", "i32", [
            ("desc", "ptr", "HANDLE", 8), ("dataType", "i32", "SCALAR", 4),
            ("n", "i32", "SCALAR", 4), ("c", "i32", "SCALAR", 4), ("h", "i32", "SCALAR", 4), ("w", "i32", "SCALAR", 4),
            ("nStride", "i32", "SCALAR", 4), ("cStride", "i32", "SCALAR", 4),
            ("hStride", "i32", "SCALAR", 4), ("wStride", "i32", "SCALAR", 4)]),

        # Filter descriptor
        ("cudnnCreateFilterDescriptor", "i32", [("desc", "ptr", "HANDLE_OUT", 8)]),
        ("cudnnDestroyFilterDescriptor", "i32", [("desc", "ptr", "HANDLE", 8)]),
        ("cudnnSetFilter4dDescriptor", "i32", [
            ("desc", "ptr", "HANDLE", 8), ("dataType", "i32", "SCALAR", 4), ("format", "i32", "SCALAR", 4),
            ("k", "i32", "SCALAR", 4), ("c", "i32", "SCALAR", 4), ("h", "i32", "SCALAR", 4), ("w", "i32", "SCALAR", 4)]),

        # Convolution descriptor
        ("cudnnCreateConvolutionDescriptor", "i32", [("desc", "ptr", "HANDLE_OUT", 8)]),
        ("cudnnDestroyConvolutionDescriptor", "i32", [("desc", "ptr", "HANDLE", 8)]),
        ("cudnnSetConvolution2dDescriptor", "i32", [
            ("desc", "ptr", "HANDLE", 8),
            ("padH", "i32", "SCALAR", 4), ("padW", "i32", "SCALAR", 4),
            ("strideH", "i32", "SCALAR", 4), ("strideW", "i32", "SCALAR", 4),
            ("dilationH", "i32", "SCALAR", 4), ("dilationW", "i32", "SCALAR", 4),
            ("mode", "i32", "SCALAR", 4), ("computeType", "i32", "SCALAR", 4)]),
        ("cudnnSetConvolutionGroupCount", "i32", [("desc", "ptr", "HANDLE", 8), ("groupCount", "i32", "SCALAR", 4)]),
        ("cudnnSetConvolutionMathType", "i32", [("desc", "ptr", "HANDLE", 8), ("mathType", "i32", "SCALAR", 4)]),
        ("cudnnGetConvolution2dForwardOutputDim", "i32", [
            ("desc", "ptr", "HANDLE", 8), ("inputDesc", "ptr", "HANDLE", 8), ("filterDesc", "ptr", "HANDLE", 8),
            ("n", "ptr", "HOST_OUT", 4), ("c", "ptr", "HOST_OUT", 4), ("h", "ptr", "HOST_OUT", 4), ("w", "ptr", "HOST_OUT", 4)]),

        # Convolution forward
        ("cudnnGetConvolutionForwardWorkspaceSize", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("xDesc", "ptr", "HANDLE", 8), ("wDesc", "ptr", "HANDLE", 8),
            ("convDesc", "ptr", "HANDLE", 8), ("yDesc", "ptr", "HANDLE", 8),
            ("algo", "i32", "SCALAR", 4), ("sizeInBytes", "ptr", "HOST_OUT", 8)]),
        ("cudnnConvolutionForward", "i32", [
            ("handle", "ptr", "HANDLE", 8),
            ("alpha", "ptr", "HOST_IN", 8), ("xDesc", "ptr", "HANDLE", 8), ("x", "ptr", "DEV_PTR", 8),
            ("wDesc", "ptr", "HANDLE", 8), ("w", "ptr", "DEV_PTR", 8),
            ("convDesc", "ptr", "HANDLE", 8), ("algo", "i32", "SCALAR", 4),
            ("workspace", "ptr", "DEV_PTR", 8), ("workspaceSizeInBytes", "u64", "SCALAR", 8),
            ("beta", "ptr", "HOST_IN", 8), ("yDesc", "ptr", "HANDLE", 8), ("y", "ptr", "DEV_PTR", 8)]),

        # Convolution backward
        ("cudnnConvolutionBackwardData", "i32", [
            ("handle", "ptr", "HANDLE", 8),
            ("alpha", "ptr", "HOST_IN", 8), ("wDesc", "ptr", "HANDLE", 8), ("w", "ptr", "DEV_PTR", 8),
            ("dyDesc", "ptr", "HANDLE", 8), ("dy", "ptr", "DEV_PTR", 8),
            ("convDesc", "ptr", "HANDLE", 8), ("algo", "i32", "SCALAR", 4),
            ("workspace", "ptr", "DEV_PTR", 8), ("workspaceSizeInBytes", "u64", "SCALAR", 8),
            ("beta", "ptr", "HOST_IN", 8), ("dxDesc", "ptr", "HANDLE", 8), ("dx", "ptr", "DEV_PTR", 8)]),
        ("cudnnConvolutionBackwardFilter", "i32", [
            ("handle", "ptr", "HANDLE", 8),
            ("alpha", "ptr", "HOST_IN", 8), ("xDesc", "ptr", "HANDLE", 8), ("x", "ptr", "DEV_PTR", 8),
            ("dyDesc", "ptr", "HANDLE", 8), ("dy", "ptr", "DEV_PTR", 8),
            ("convDesc", "ptr", "HANDLE", 8), ("algo", "i32", "SCALAR", 4),
            ("workspace", "ptr", "DEV_PTR", 8), ("workspaceSizeInBytes", "u64", "SCALAR", 8),
            ("beta", "ptr", "HOST_IN", 8), ("dwDesc", "ptr", "HANDLE", 8), ("dw", "ptr", "DEV_PTR", 8)]),

        # Activation
        ("cudnnCreateActivationDescriptor", "i32", [("desc", "ptr", "HANDLE_OUT", 8)]),
        ("cudnnDestroyActivationDescriptor", "i32", [("desc", "ptr", "HANDLE", 8)]),
        ("cudnnSetActivationDescriptor", "i32", [
            ("desc", "ptr", "HANDLE", 8), ("mode", "i32", "SCALAR", 4),
            ("reluNanOpt", "i32", "SCALAR", 4), ("coef", "f64", "SCALAR", 8)]),
        ("cudnnActivationForward", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("activationDesc", "ptr", "HANDLE", 8),
            ("alpha", "ptr", "HOST_IN", 8), ("xDesc", "ptr", "HANDLE", 8), ("x", "ptr", "DEV_PTR", 8),
            ("beta", "ptr", "HOST_IN", 8), ("yDesc", "ptr", "HANDLE", 8), ("y", "ptr", "DEV_PTR", 8)]),
        ("cudnnActivationBackward", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("activationDesc", "ptr", "HANDLE", 8),
            ("alpha", "ptr", "HOST_IN", 8),
            ("yDesc", "ptr", "HANDLE", 8), ("y", "ptr", "DEV_PTR", 8),
            ("dyDesc", "ptr", "HANDLE", 8), ("dy", "ptr", "DEV_PTR", 8),
            ("xDesc", "ptr", "HANDLE", 8), ("x", "ptr", "DEV_PTR", 8),
            ("beta", "ptr", "HOST_IN", 8), ("dxDesc", "ptr", "HANDLE", 8), ("dx", "ptr", "DEV_PTR", 8)]),

        # Pooling
        ("cudnnCreatePoolingDescriptor", "i32", [("desc", "ptr", "HANDLE_OUT", 8)]),
        ("cudnnDestroyPoolingDescriptor", "i32", [("desc", "ptr", "HANDLE", 8)]),
        ("cudnnSetPooling2dDescriptor", "i32", [
            ("desc", "ptr", "HANDLE", 8), ("mode", "i32", "SCALAR", 4), ("maxpoolingNanOpt", "i32", "SCALAR", 4),
            ("windowH", "i32", "SCALAR", 4), ("windowW", "i32", "SCALAR", 4),
            ("padH", "i32", "SCALAR", 4), ("padW", "i32", "SCALAR", 4),
            ("strideH", "i32", "SCALAR", 4), ("strideW", "i32", "SCALAR", 4)]),
        ("cudnnPoolingForward", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("poolingDesc", "ptr", "HANDLE", 8),
            ("alpha", "ptr", "HOST_IN", 8), ("xDesc", "ptr", "HANDLE", 8), ("x", "ptr", "DEV_PTR", 8),
            ("beta", "ptr", "HOST_IN", 8), ("yDesc", "ptr", "HANDLE", 8), ("y", "ptr", "DEV_PTR", 8)]),

        # Softmax
        ("cudnnSoftmaxForward", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("algo", "i32", "SCALAR", 4), ("mode", "i32", "SCALAR", 4),
            ("alpha", "ptr", "HOST_IN", 8), ("xDesc", "ptr", "HANDLE", 8), ("x", "ptr", "DEV_PTR", 8),
            ("beta", "ptr", "HOST_IN", 8), ("yDesc", "ptr", "HANDLE", 8), ("y", "ptr", "DEV_PTR", 8)]),
        ("cudnnSoftmaxBackward", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("algo", "i32", "SCALAR", 4), ("mode", "i32", "SCALAR", 4),
            ("alpha", "ptr", "HOST_IN", 8),
            ("yDesc", "ptr", "HANDLE", 8), ("y", "ptr", "DEV_PTR", 8),
            ("dyDesc", "ptr", "HANDLE", 8), ("dy", "ptr", "DEV_PTR", 8),
            ("beta", "ptr", "HOST_IN", 8), ("dxDesc", "ptr", "HANDLE", 8), ("dx", "ptr", "DEV_PTR", 8)]),

        # Batch normalization
        ("cudnnBatchNormalizationForwardTraining", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("mode", "i32", "SCALAR", 4),
            ("alpha", "ptr", "HOST_IN", 8), ("beta", "ptr", "HOST_IN", 8),
            ("xDesc", "ptr", "HANDLE", 8), ("x", "ptr", "DEV_PTR", 8),
            ("yDesc", "ptr", "HANDLE", 8), ("y", "ptr", "DEV_PTR", 8),
            ("bnScaleBiasMeanVarDesc", "ptr", "HANDLE", 8),
            ("bnScale", "ptr", "DEV_PTR", 8), ("bnBias", "ptr", "DEV_PTR", 8),
            ("expAvgFactor", "f64", "SCALAR", 8),
            ("resultRunningMean", "ptr", "DEV_PTR", 8), ("resultRunningVariance", "ptr", "DEV_PTR", 8),
            ("epsilon", "f64", "SCALAR", 8),
            ("resultSaveMean", "ptr", "DEV_PTR", 8), ("resultSaveInvVariance", "ptr", "DEV_PTR", 8)]),
        ("cudnnBatchNormalizationForwardInference", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("mode", "i32", "SCALAR", 4),
            ("alpha", "ptr", "HOST_IN", 8), ("beta", "ptr", "HOST_IN", 8),
            ("xDesc", "ptr", "HANDLE", 8), ("x", "ptr", "DEV_PTR", 8),
            ("yDesc", "ptr", "HANDLE", 8), ("y", "ptr", "DEV_PTR", 8),
            ("bnScaleBiasMeanVarDesc", "ptr", "HANDLE", 8),
            ("bnScale", "ptr", "DEV_PTR", 8), ("bnBias", "ptr", "DEV_PTR", 8),
            ("estimatedMean", "ptr", "DEV_PTR", 8), ("estimatedVariance", "ptr", "DEV_PTR", 8),
            ("epsilon", "f64", "SCALAR", 8)]),
        ("cudnnBatchNormalizationBackward", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("mode", "i32", "SCALAR", 4),
            ("alphaDataDiff", "ptr", "HOST_IN", 8), ("betaDataDiff", "ptr", "HOST_IN", 8),
            ("alphaParamDiff", "ptr", "HOST_IN", 8), ("betaParamDiff", "ptr", "HOST_IN", 8),
            ("xDesc", "ptr", "HANDLE", 8), ("x", "ptr", "DEV_PTR", 8),
            ("dyDesc", "ptr", "HANDLE", 8), ("dy", "ptr", "DEV_PTR", 8),
            ("dxDesc", "ptr", "HANDLE", 8), ("dx", "ptr", "DEV_PTR", 8),
            ("bnScaleBiasDiffDesc", "ptr", "HANDLE", 8),
            ("bnScale", "ptr", "DEV_PTR", 8),
            ("resultBnScaleDiff", "ptr", "DEV_PTR", 8), ("resultBnBiasDiff", "ptr", "DEV_PTR", 8),
            ("epsilon", "f64", "SCALAR", 8),
            ("savedMean", "ptr", "DEV_PTR", 8), ("savedInvVariance", "ptr", "DEV_PTR", 8)]),

        # Add tensor (bias add)
        ("cudnnAddTensor", "i32", [
            ("handle", "ptr", "HANDLE", 8),
            ("alpha", "ptr", "HOST_IN", 8), ("aDesc", "ptr", "HANDLE", 8), ("A", "ptr", "DEV_PTR", 8),
            ("beta", "ptr", "HOST_IN", 8), ("cDesc", "ptr", "HANDLE", 8), ("C", "ptr", "DEV_PTR", 8)]),

        # Scale tensor
        ("cudnnScaleTensor", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("yDesc", "ptr", "HANDLE", 8),
            ("y", "ptr", "DEV_PTR", 8), ("alpha", "ptr", "HOST_IN", 8)]),

        # Transform tensor
        ("cudnnTransformTensor", "i32", [
            ("handle", "ptr", "HANDLE", 8),
            ("alpha", "ptr", "HOST_IN", 8), ("xDesc", "ptr", "HANDLE", 8), ("x", "ptr", "DEV_PTR", 8),
            ("beta", "ptr", "HOST_IN", 8), ("yDesc", "ptr", "HANDLE", 8), ("y", "ptr", "DEV_PTR", 8)]),

        # Version
        ("cudnnGetVersion", "u64", []),

        # Dropout (used in training)
        ("cudnnCreateDropoutDescriptor", "i32", [("desc", "ptr", "HANDLE_OUT", 8)]),
        ("cudnnDestroyDropoutDescriptor", "i32", [("desc", "ptr", "HANDLE", 8)]),
        ("cudnnDropoutGetStatesSize", "i32", [("handle", "ptr", "HANDLE", 8), ("sizeInBytes", "ptr", "HOST_OUT", 8)]),

        # RNN
        ("cudnnCreateRNNDescriptor", "i32", [("desc", "ptr", "HANDLE_OUT", 8)]),
        ("cudnnDestroyRNNDescriptor", "i32", [("desc", "ptr", "HANDLE", 8)]),

        # OpTensor
        ("cudnnCreateOpTensorDescriptor", "i32", [("desc", "ptr", "HANDLE_OUT", 8)]),
        ("cudnnDestroyOpTensorDescriptor", "i32", [("desc", "ptr", "HANDLE", 8)]),
        ("cudnnSetOpTensorDescriptor", "i32", [
            ("desc", "ptr", "HANDLE", 8), ("opTensorOp", "i32", "SCALAR", 4),
            ("opTensorCompType", "i32", "SCALAR", 4), ("opTensorNanOpt", "i32", "SCALAR", 4)]),
        ("cudnnOpTensor", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("opTensorDesc", "ptr", "HANDLE", 8),
            ("alpha1", "ptr", "HOST_IN", 8), ("aDesc", "ptr", "HANDLE", 8), ("A", "ptr", "DEV_PTR", 8),
            ("alpha2", "ptr", "HOST_IN", 8), ("bDesc", "ptr", "HANDLE", 8), ("B", "ptr", "DEV_PTR", 8),
            ("beta", "ptr", "HOST_IN", 8), ("cDesc", "ptr", "HANDLE", 8), ("C", "ptr", "DEV_PTR", 8)]),

        # Reduce tensor
        ("cudnnCreateReduceTensorDescriptor", "i32", [("desc", "ptr", "HANDLE_OUT", 8)]),
        ("cudnnDestroyReduceTensorDescriptor", "i32", [("desc", "ptr", "HANDLE", 8)]),
        ("cudnnSetReduceTensorDescriptor", "i32", [
            ("desc", "ptr", "HANDLE", 8), ("op", "i32", "SCALAR", 4),
            ("compType", "i32", "SCALAR", 4), ("nanOpt", "i32", "SCALAR", 4),
            ("indices", "i32", "SCALAR", 4), ("indicesType", "i32", "SCALAR", 4)]),
        ("cudnnGetReductionWorkspaceSize", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("reduceDesc", "ptr", "HANDLE", 8),
            ("aDesc", "ptr", "HANDLE", 8), ("cDesc", "ptr", "HANDLE", 8),
            ("sizeInBytes", "ptr", "HOST_OUT", 8)]),
        ("cudnnReduceTensor", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("reduceDesc", "ptr", "HANDLE", 8),
            ("indices", "ptr", "DEV_PTR", 8), ("indicesSizeInBytes", "u64", "SCALAR", 8),
            ("workspace", "ptr", "DEV_PTR", 8), ("workspaceSizeInBytes", "u64", "SCALAR", 8),
            ("alpha", "ptr", "HOST_IN", 8), ("aDesc", "ptr", "HANDLE", 8), ("A", "ptr", "DEV_PTR", 8),
            ("beta", "ptr", "HOST_IN", 8), ("cDesc", "ptr", "HANDLE", 8), ("C", "ptr", "DEV_PTR", 8)]),

        # Nd descriptors (generic N-dimensional — used by modern PyTorch)
        ("cudnnSetTensorNdDescriptor", "i32", [
            ("desc", "ptr", "HANDLE", 8), ("dataType", "i32", "SCALAR", 4),
            ("nbDims", "i32", "SCALAR", 4),
            ("dimA", "ptr", "HOST_IN", 32), ("strideA", "ptr", "HOST_IN", 32)]),
        ("cudnnSetFilterNdDescriptor", "i32", [
            ("desc", "ptr", "HANDLE", 8), ("dataType", "i32", "SCALAR", 4),
            ("format", "i32", "SCALAR", 4), ("nbDims", "i32", "SCALAR", 4),
            ("filterDimA", "ptr", "HOST_IN", 32)]),
        ("cudnnSetConvolutionNdDescriptor", "i32", [
            ("desc", "ptr", "HANDLE", 8), ("arrayLength", "i32", "SCALAR", 4),
            ("padA", "ptr", "HOST_IN", 24), ("filterStrideA", "ptr", "HOST_IN", 24),
            ("dilationA", "ptr", "HOST_IN", 24),
            ("mode", "i32", "SCALAR", 4), ("computeType", "i32", "SCALAR", 4)]),

        # Algorithm finding
        ("cudnnGetConvolutionForwardAlgorithm_v7", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("srcDesc", "ptr", "HANDLE", 8),
            ("filterDesc", "ptr", "HANDLE", 8), ("convDesc", "ptr", "HANDLE", 8),
            ("destDesc", "ptr", "HANDLE", 8),
            ("requestedAlgoCount", "i32", "SCALAR", 4),
            ("returnedAlgoCount", "ptr", "HOST_OUT", 4),
            ("perfResults", "ptr", "HOST_OUT", 512)]),  # array of perf structs
        ("cudnnGetConvolutionBackwardDataAlgorithm_v7", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("filterDesc", "ptr", "HANDLE", 8),
            ("diffDesc", "ptr", "HANDLE", 8), ("convDesc", "ptr", "HANDLE", 8),
            ("gradDesc", "ptr", "HANDLE", 8),
            ("requestedAlgoCount", "i32", "SCALAR", 4),
            ("returnedAlgoCount", "ptr", "HOST_OUT", 4),
            ("perfResults", "ptr", "HOST_OUT", 512)]),
        ("cudnnGetConvolutionBackwardFilterAlgorithm_v7", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("srcDesc", "ptr", "HANDLE", 8),
            ("diffDesc", "ptr", "HANDLE", 8), ("convDesc", "ptr", "HANDLE", 8),
            ("gradDesc", "ptr", "HANDLE", 8),
            ("requestedAlgoCount", "i32", "SCALAR", 4),
            ("returnedAlgoCount", "ptr", "HOST_OUT", 4),
            ("perfResults", "ptr", "HOST_OUT", 512)]),

        # Workspace sizes for backward
        ("cudnnGetConvolutionBackwardDataWorkspaceSize", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("wDesc", "ptr", "HANDLE", 8),
            ("dyDesc", "ptr", "HANDLE", 8), ("convDesc", "ptr", "HANDLE", 8),
            ("dxDesc", "ptr", "HANDLE", 8),
            ("algo", "i32", "SCALAR", 4), ("sizeInBytes", "ptr", "HOST_OUT", 8)]),
        ("cudnnGetConvolutionBackwardFilterWorkspaceSize", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("xDesc", "ptr", "HANDLE", 8),
            ("dyDesc", "ptr", "HANDLE", 8), ("convDesc", "ptr", "HANDLE", 8),
            ("gradDesc", "ptr", "HANDLE", 8),
            ("algo", "i32", "SCALAR", 4), ("sizeInBytes", "ptr", "HOST_OUT", 8)]),

        # Set tensor value
        ("cudnnSetTensor", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("yDesc", "ptr", "HANDLE", 8),
            ("y", "ptr", "DEV_PTR", 8), ("valuePtr", "ptr", "HOST_IN", 8)]),

        # Pooling backward
        ("cudnnPoolingBackward", "i32", [
            ("handle", "ptr", "HANDLE", 8), ("poolingDesc", "ptr", "HANDLE", 8),
            ("alpha", "ptr", "HOST_IN", 8),
            ("yDesc", "ptr", "HANDLE", 8), ("y", "ptr", "DEV_PTR", 8),
            ("dyDesc", "ptr", "HANDLE", 8), ("dy", "ptr", "DEV_PTR", 8),
            ("xDesc", "ptr", "HANDLE", 8), ("x", "ptr", "DEV_PTR", 8),
            ("beta", "ptr", "HOST_IN", 8), ("dxDesc", "ptr", "HANDLE", 8), ("dx", "ptr", "DEV_PTR", 8)]),

        # LRN (Local Response Normalization)
        ("cudnnCreateLRNDescriptor", "i32", [("desc", "ptr", "HANDLE_OUT", 8)]),
        ("cudnnDestroyLRNDescriptor", "i32", [("desc", "ptr", "HANDLE", 8)]),
        ("cudnnSetLRNDescriptor", "i32", [
            ("desc", "ptr", "HANDLE", 8), ("lrnN", "u32", "SCALAR", 4),
            ("lrnAlpha", "f64", "SCALAR", 8), ("lrnBeta", "f64", "SCALAR", 8), ("lrnK", "f64", "SCALAR", 8)]),
    ]
}

# ── CUDA Runtime additional (full RPC) ──────────────────────
LIBS["cuda_rt_extra"] = {
    "id": 3,
    "header": "cuda_runtime_api.h",
    "link": "cudart",
    "handle_type": None,
    "functions": [
        # Memory
        ("cudaMallocManaged", "i32", [("devPtr", "ptr", "HANDLE_OUT", 8), ("size", "u64", "SCALAR", 8), ("flags", "u32", "SCALAR", 4)]),
        ("cudaMallocHost", "i32", [("ptr", "ptr", "HANDLE_OUT", 8), ("size", "u64", "SCALAR", 8)]),
        ("cudaFreeHost", "i32", [("ptr", "ptr", "HANDLE", 8)]),
        ("cudaMallocAsync", "i32", [("devPtr", "ptr", "HANDLE_OUT", 8), ("size", "u64", "SCALAR", 8), ("stream", "ptr", "DEV_PTR", 8)]),
        ("cudaFreeAsync", "i32", [("devPtr", "ptr", "HANDLE", 8), ("stream", "ptr", "DEV_PTR", 8)]),
        ("cudaMemGetInfo", "i32", [("free", "ptr", "HOST_OUT", 8), ("total", "ptr", "HOST_OUT", 8)]),
        ("cudaMemcpy2D", "i32", [
            ("dst", "ptr", "DEV_PTR", 8), ("dpitch", "u64", "SCALAR", 8),
            ("src", "ptr", "DEV_PTR", 8), ("spitch", "u64", "SCALAR", 8),
            ("width", "u64", "SCALAR", 8), ("height", "u64", "SCALAR", 8), ("kind", "i32", "SCALAR", 4)]),
        ("cudaMemsetAsync", "i32", [
            ("devPtr", "ptr", "DEV_PTR", 8), ("value", "i32", "SCALAR", 4),
            ("count", "u64", "SCALAR", 8), ("stream", "ptr", "DEV_PTR", 8)]),

        # Device management
        ("cudaGetDevice", "i32", [("device", "ptr", "HOST_OUT", 4)]),
        ("cudaDeviceGetStreamPriorityRange", "i32", [
            ("leastPriority", "ptr", "HOST_OUT", 4), ("greatestPriority", "ptr", "HOST_OUT", 4)]),
        ("cudaDeviceCanAccessPeer", "i32", [
            ("canAccessPeer", "ptr", "HOST_OUT", 4), ("device", "i32", "SCALAR", 4), ("peerDevice", "i32", "SCALAR", 4)]),
        ("cudaSetDeviceFlags", "i32", [("flags", "u32", "SCALAR", 4)]),
        ("cudaGetDeviceCount_v2", "i32", [("count", "ptr", "HOST_OUT", 4)]),
        ("cudaRuntimeGetVersion", "i32", [("runtimeVersion", "ptr", "HOST_OUT", 4)]),
        ("cudaDriverGetVersion", "i32", [("driverVersion", "ptr", "HOST_OUT", 4)]),

        # Stream
        ("cudaStreamQuery", "i32", [("stream", "ptr", "DEV_PTR", 8)]),
        ("cudaStreamWaitEvent", "i32", [("stream", "ptr", "DEV_PTR", 8), ("event", "ptr", "DEV_PTR", 8), ("flags", "u32", "SCALAR", 4)]),
        ("cudaStreamAddCallback", "i32", [
            ("stream", "ptr", "DEV_PTR", 8), ("callback", "ptr", "DEV_PTR", 8),
            ("userData", "ptr", "DEV_PTR", 8), ("flags", "u32", "SCALAR", 4)]),

        # Event
        ("cudaEventQuery", "i32", [("event", "ptr", "DEV_PTR", 8)]),

        # Occupancy
        ("cudaOccupancyMaxActiveBlocksPerMultiprocessor", "i32", [
            ("numBlocks", "ptr", "HOST_OUT", 4), ("func", "ptr", "DEV_PTR", 8),
            ("blockSize", "i32", "SCALAR", 4), ("dynamicSMemSize", "u64", "SCALAR", 8)]),

        # Misc
        ("cudaGetSymbolAddress", "i32", [("devPtr", "ptr", "HANDLE_OUT", 8), ("symbol", "ptr", "DEV_PTR", 8)]),
        ("cudaGetSymbolSize", "i32", [("size", "ptr", "HOST_OUT", 8), ("symbol", "ptr", "DEV_PTR", 8)]),
        ("cudaPointerGetAttributes", "i32", [("attributes", "ptr", "HOST_OUT", 64), ("ptr", "ptr", "DEV_PTR", 8)]),

        # Peer access
        ("cudaDeviceEnablePeerAccess", "i32", [("peerDevice", "i32", "SCALAR", 4), ("flags", "u32", "SCALAR", 4)]),
        ("cudaDeviceDisablePeerAccess", "i32", [("peerDevice", "i32", "SCALAR", 4)]),

        # Texture (minimal stubs)
        ("cudaCreateTextureObject", "i32", [
            ("pTexObject", "ptr", "HOST_OUT", 8), ("pResDesc", "ptr", "HOST_IN", 64),
            ("pTexDesc", "ptr", "HOST_IN", 64), ("pResViewDesc", "ptr", "HOST_IN", 64)]),
        ("cudaDestroyTextureObject", "i32", [("texObject", "u64", "SCALAR", 8)]),

        # Graph (minimal — used by PyTorch CUDA Graphs)
        ("cudaGraphCreate", "i32", [("pGraph", "ptr", "HANDLE_OUT", 8), ("flags", "u32", "SCALAR", 4)]),
        ("cudaGraphDestroy", "i32", [("graph", "ptr", "HANDLE", 8)]),
        ("cudaGraphInstantiate", "i32", [
            ("pGraphExec", "ptr", "HANDLE_OUT", 8), ("graph", "ptr", "HANDLE", 8),
            ("flags", "u64", "SCALAR", 8)]),
        ("cudaGraphLaunch", "i32", [("graphExec", "ptr", "HANDLE", 8), ("stream", "ptr", "DEV_PTR", 8)]),
        ("cudaGraphExecDestroy", "i32", [("graphExec", "ptr", "HANDLE", 8)]),

        # Memory pool (CUDA 11.2+)
        ("cudaDeviceGetDefaultMemPool", "i32", [("memPool", "ptr", "HANDLE_OUT", 8), ("device", "i32", "SCALAR", 4)]),
        ("cudaMemPoolTrimTo", "i32", [("memPool", "ptr", "HANDLE", 8), ("minBytesToKeep", "u64", "SCALAR", 8)]),
    ]
}

# ── Code generation ─────────────────────────────────────────

def count_all():
    total = 0
    for lib_name, lib in LIBS.items():
        total += len(lib["functions"])
    return total


def generate_client_stubs(outpath):
    """Generate client-side C++ stubs that serialize args and call RPC."""
    lines = []
    lines.append("/* AUTO-GENERATED by codegen/generate_stubs.py — DO NOT EDIT */")
    lines.append("")
    lines.append('#include <cstring>')
    lines.append('#include <vector>')
    lines.append('#include <cstdint>')
    lines.append('#include "gpushare/protocol.h"')
    lines.append('#include "gpushare/cuda_defs.h"')
    lines.append("")
    lines.append("/* Defined in gpushare_client.cpp */")
    lines.append("extern bool rpc_call(uint16_t opcode, const void *req, uint32_t req_len,")
    lines.append("                     std::vector<uint8_t> &resp, uint16_t *flags = nullptr);")
    lines.append("extern int32_t rpc_simple(uint16_t opcode, const void *req, uint32_t req_len);")
    lines.append("")
    lines.append("#ifdef _WIN32")
    lines.append('  #define GEN_EXPORT __declspec(dllexport)')
    lines.append("#else")
    lines.append('  #define GEN_EXPORT __attribute__((visibility("default")))')
    lines.append("#endif")
    lines.append("")
    lines.append('#ifdef __cplusplus')
    lines.append('extern "C" {')
    lines.append('#endif')
    lines.append("")

    func_id = 0
    for lib_name, lib in LIBS.items():
        lib_id = lib["id"]
        lines.append(f"/* ── {lib_name} ({len(lib['functions'])} functions) ── */")
        lines.append("")

        for fname, ret_type, args in lib["functions"]:
            func_id += 1

            # Build C function signature
            c_args = []
            for aname, atype, akind, asize in args:
                if akind in ("HANDLE", "DEV_PTR"):
                    c_args.append(f"void *{aname}")
                elif akind in ("HANDLE_OUT", "HOST_OUT"):
                    c_args.append(f"void *{aname}")
                elif akind == "HOST_IN":
                    c_args.append(f"const void *{aname}")
                elif akind == "HOST_INOUT":
                    c_args.append(f"void *{aname}")
                elif atype == "i32":
                    c_args.append(f"int {aname}")
                elif atype == "u32":
                    c_args.append(f"unsigned int {aname}")
                elif atype == "i64":
                    c_args.append(f"int64_t {aname}")
                elif atype == "u64":
                    c_args.append(f"uint64_t {aname}")
                elif atype == "f32":
                    c_args.append(f"float {aname}")
                elif atype == "f64":
                    c_args.append(f"double {aname}")
                else:
                    c_args.append(f"void *{aname}")

            ret_c = "int" if ret_type == "i32" else ("uint64_t" if ret_type == "u64" else "int")
            sig = f"GEN_EXPORT {ret_c} {fname}({', '.join(c_args) if c_args else 'void'})"

            lines.append(sig + " {")

            # Serialize: [u8 lib_id][u16 func_id][args...]
            lines.append(f"    /* lib={lib_id}, func={func_id} ({fname}) */")
            lines.append(f"    std::vector<uint8_t> buf;")

            # Header: lib_id(1) + func_id(2)
            lines.append(f"    buf.push_back({lib_id});")
            lines.append(f"    uint16_t fid = {func_id}; buf.insert(buf.end(), (uint8_t*)&fid, (uint8_t*)&fid + 2);")

            # Serialize each arg
            for aname, atype, akind, asize in args:
                if akind == "SCALAR":
                    if asize <= 4 and atype in ("i32", "u32", "f32"):
                        lines.append(f"    {{ auto v = {aname}; buf.insert(buf.end(), (uint8_t*)&v, (uint8_t*)&v + {asize}); }}")
                    else:
                        lines.append(f"    {{ auto v = {aname}; buf.insert(buf.end(), (uint8_t*)&v, (uint8_t*)&v + {asize}); }}")
                elif akind in ("HANDLE", "DEV_PTR"):
                    lines.append(f"    {{ uint64_t h = (uint64_t)(uintptr_t){aname}; buf.insert(buf.end(), (uint8_t*)&h, (uint8_t*)&h + 8); }}")
                elif akind == "HANDLE_OUT":
                    lines.append(f"    /* {aname}: handle output — no input data */")
                elif akind == "HOST_IN":
                    lines.append(f"    if ({aname}) buf.insert(buf.end(), (const uint8_t*){aname}, (const uint8_t*){aname} + {asize});")
                    lines.append(f"    else {{ uint8_t z[{asize}] = {{0}}; buf.insert(buf.end(), z, z + {asize}); }}")
                elif akind in ("HOST_OUT", "HOST_INOUT"):
                    if akind == "HOST_INOUT":
                        lines.append(f"    if ({aname}) buf.insert(buf.end(), (const uint8_t*){aname}, (const uint8_t*){aname} + {asize});")
                    lines.append(f"    /* {aname}: output ({asize} bytes) */")

            # RPC call
            lines.append(f"    std::vector<uint8_t> resp;")
            lines.append(f"    if (!rpc_call(GS_OP_LIB_CALL, buf.data(), (uint32_t)buf.size(), resp))")
            lines.append(f"        return {'999' if ret_type == 'i32' else '0'};")

            # Deserialize response: [i32 return_code][out_data...]
            lines.append(f"    if (resp.size() < 4) return {'999' if ret_type == 'i32' else '0'};")
            lines.append(f"    {ret_c} ret; memcpy(&ret, resp.data(), 4);")

            # Deserialize output args
            offset = 4
            for aname, atype, akind, asize in args:
                if akind == "HANDLE_OUT":
                    lines.append(f"    if ({aname} && resp.size() >= {offset + 8})")
                    lines.append(f"        memcpy({aname}, resp.data() + {offset}, 8);")
                    offset += 8
                elif akind in ("HOST_OUT", "HOST_INOUT"):
                    lines.append(f"    if ({aname} && resp.size() >= {offset + asize})")
                    lines.append(f"        memcpy({aname}, resp.data() + {offset}, {asize});")
                    offset += asize

            lines.append(f"    return ret;")
            lines.append(f"}}")
            lines.append("")

    lines.append('#ifdef __cplusplus')
    lines.append('}')
    lines.append('#endif')

    with open(outpath, "w") as f:
        f.write("\n".join(lines))
    print(f"Generated {outpath}: {func_id} functions, {len(lines)} lines")
    return func_id


def generate_server_dispatch(outpath):
    """Generate server-side dispatch that calls real library functions."""
    lines = []
    lines.append("/* AUTO-GENERATED by codegen/generate_stubs.py — DO NOT EDIT */")
    lines.append("")
    lines.append('#include <cstring>')
    lines.append('#include <cstdint>')
    lines.append('#include <vector>')
    lines.append('#include <unordered_map>')
    lines.append('#include <dlfcn.h>')
    lines.append('#include <cstdio>')
    lines.append("")
    lines.append("/* Handle mapping: client handle (uint64_t) → server real handle (void*) */")
    lines.append("static std::unordered_map<uint64_t, void*> g_handle_map;")
    lines.append("static uint64_t g_next_handle = 0x1000;")
    lines.append("")
    lines.append("static uint64_t register_handle(void *real) {")
    lines.append("    uint64_t h = g_next_handle++;")
    lines.append("    g_handle_map[h] = real;")
    lines.append("    return h;")
    lines.append("}")
    lines.append("")
    lines.append("static void* lookup_handle(uint64_t h) {")
    lines.append("    auto it = g_handle_map.find(h);")
    lines.append("    return (it != g_handle_map.end()) ? it->second : nullptr;")
    lines.append("}")
    lines.append("")
    lines.append("static void remove_handle(uint64_t h) { g_handle_map.erase(h); }")
    lines.append("")
    lines.append("/* Library loading */")

    for lib_name, lib in LIBS.items():
        lines.append(f'static void *g_{lib_name}_lib = nullptr;')
    lines.append("")

    lines.append("static void ensure_libs_loaded() {")
    for lib_name, lib in LIBS.items():
        link = lib["link"]
        lines.append(f'    if (!g_{lib_name}_lib) g_{lib_name}_lib = dlopen("lib{link}.so", RTLD_NOW | RTLD_GLOBAL);')
    lines.append("}")
    lines.append("")

    lines.append("/* Dispatch function — called by server for GS_OP_LIB_CALL */")
    lines.append("bool dispatch_lib_call(const uint8_t *data, uint32_t len,")
    lines.append("                       std::vector<uint8_t> &response) {")
    lines.append("    if (len < 3) return false;")
    lines.append("    ensure_libs_loaded();")
    lines.append("    uint8_t lib_id = data[0];")
    lines.append("    uint16_t func_id; memcpy(&func_id, data + 1, 2);")
    lines.append("    const uint8_t *args = data + 3;")
    lines.append("    uint32_t args_len = len - 3;")
    lines.append("    (void)args_len;")
    lines.append("")
    lines.append("    int32_t ret = 0;")
    lines.append("    response.clear();")
    lines.append("")

    func_id = 0
    for lib_name, lib in LIBS.items():
        lib_id = lib["id"]

        for fname, ret_type, args_def in lib["functions"]:
            func_id += 1

            # Build the real function call
            # Build typedef for function pointer
            def arg_ctype(a):
                _, atype, akind, _ = a
                if akind in ("HANDLE", "DEV_PTR", "HANDLE_OUT", "HOST_IN", "HOST_OUT", "HOST_INOUT"):
                    return "void*"
                if atype in ("i32", "u32"): return "int"
                if atype == "i64": return "int64_t"
                if atype == "u64": return "uint64_t"
                if atype == "f32": return "float"
                if atype == "f64": return "double"
                return "void*"

            ret_ctype = "int" if ret_type == "i32" else "uint64_t"
            arg_types = ", ".join([arg_ctype(a) for a in args_def]) if args_def else "void"

            lines.append(f"    if (lib_id == {lib_id} && func_id == {func_id}) {{ /* {fname} */")
            lines.append(f"      do {{")
            lines.append(f"        typedef {ret_ctype} (*fn_t)({arg_types});")
            lines.append(f'        static fn_t fn = (fn_t)dlsym(g_{lib_name}_lib, "{fname}");')
            lines.append(f"        if (!fn) {{ ret = -1; break; }}")

            # Deserialize args
            offset_var = "off"
            lines.append(f"        uint32_t {offset_var} = 0;")

            call_args = []
            out_vars = []

            for aname, atype, akind, asize in args_def:
                if akind == "SCALAR":
                    if atype in ("i32", "u32"):
                        lines.append(f"        int {aname}; memcpy(&{aname}, args + {offset_var}, 4); {offset_var} += 4;")
                    elif atype in ("i64",):
                        lines.append(f"        int64_t {aname}; memcpy(&{aname}, args + {offset_var}, 8); {offset_var} += 8;")
                    elif atype in ("u64",):
                        lines.append(f"        uint64_t {aname}; memcpy(&{aname}, args + {offset_var}, 8); {offset_var} += 8;")
                    elif atype == "f32":
                        lines.append(f"        float {aname}; memcpy(&{aname}, args + {offset_var}, 4); {offset_var} += 4;")
                    elif atype == "f64":
                        lines.append(f"        double {aname}; memcpy(&{aname}, args + {offset_var}, 8); {offset_var} += 8;")
                    call_args.append(f"(void*)(uintptr_t){aname}" if atype == "ptr" else aname)
                elif akind in ("HANDLE", "DEV_PTR"):
                    lines.append(f"        uint64_t {aname}_h; memcpy(&{aname}_h, args + {offset_var}, 8); {offset_var} += 8;")
                    if akind == "HANDLE":
                        lines.append(f"        void *{aname} = lookup_handle({aname}_h);")
                    else:
                        lines.append(f"        void *{aname} = (void*)(uintptr_t){aname}_h;")
                    call_args.append(aname)
                elif akind == "HANDLE_OUT":
                    lines.append(f"        void *{aname}_real = nullptr;")
                    call_args.append(f"&{aname}_real")
                    out_vars.append((aname, "HANDLE_OUT", asize))
                elif akind == "HOST_IN":
                    lines.append(f"        uint8_t {aname}_buf[{asize}]; memcpy({aname}_buf, args + {offset_var}, {asize}); {offset_var} += {asize};")
                    call_args.append(f"{aname}_buf")
                elif akind == "HOST_OUT":
                    lines.append(f"        uint8_t {aname}_buf[{asize}] = {{0}};")
                    call_args.append(f"{aname}_buf")
                    out_vars.append((aname, "HOST_OUT", asize))
                elif akind == "HOST_INOUT":
                    lines.append(f"        uint8_t {aname}_buf[{asize}]; memcpy({aname}_buf, args + {offset_var}, {asize}); {offset_var} += {asize};")
                    call_args.append(f"{aname}_buf")
                    out_vars.append((aname, "HOST_INOUT", asize))

            # Make the call
            if not args_def:
                lines.append(f"        ret = (int32_t)fn();")
            else:
                call_str = ", ".join(call_args)
                lines.append(f"        ret = (int32_t)fn({call_str});")

            # Build response inside the do-while block: [i32 ret][out_data...]
            lines.append(f"        response.resize(4);")
            lines.append(f"        memcpy(response.data(), &ret, 4);")

            for aname, akind, asize in out_vars:
                if akind == "HANDLE_OUT":
                    lines.append(f"        {{ uint64_t h = register_handle({aname}_real);")
                    lines.append(f"          response.insert(response.end(), (uint8_t*)&h, (uint8_t*)&h + 8); }}")
                elif akind in ("HOST_OUT", "HOST_INOUT"):
                    lines.append(f"        response.insert(response.end(), {aname}_buf, {aname}_buf + {asize});")

            lines.append(f"      }} while(0);")
            lines.append(f"        if (response.empty()) {{ response.resize(4); memcpy(response.data(), &ret, 4); }}")
            lines.append(f"        return true;")
            lines.append(f"    }}")
            lines.append("")

    lines.append('    fprintf(stderr, "[WARN] Unknown lib call: lib=%d func=%d\\n", lib_id, func_id);')
    lines.append("    ret = -1;")
    lines.append("    response.resize(4); memcpy(response.data(), &ret, 4);")
    lines.append("    return true;")
    lines.append("}")
    lines.append("")

    with open(outpath, "w") as f:
        f.write("\n".join(lines))
    print(f"Generated {outpath}: {func_id} dispatch entries, {len(lines)} lines")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    client_out = os.path.join(project_dir, "client", "generated_stubs.cpp")
    server_out = os.path.join(project_dir, "server", "generated_dispatch.cpp")

    print(f"Function database: {count_all()} functions across {len(LIBS)} libraries")
    print()

    n = generate_client_stubs(client_out)
    generate_server_dispatch(server_out)

    print()
    print(f"Total: {n} auto-generated stubs")
    print(f"  cuBLAS: {len(LIBS['cublas']['functions'])} functions")
    print(f"  cuDNN:  {len(LIBS['cudnn']['functions'])} functions")
