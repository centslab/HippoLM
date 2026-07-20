// SPDX-License-Identifier: Apache-2.0
// Common CUDA helpers for the FP8 GEMM kernel.
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess)                                                   \
        {                                                                         \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                                                    \
        }                                                                         \
    } while (0)
