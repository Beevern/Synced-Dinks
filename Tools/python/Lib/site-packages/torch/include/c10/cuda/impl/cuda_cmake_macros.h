#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
#pragma once

// Automatically generated header file for the C10 CUDA library.  Do not
// include this file directly.  Instead, include c10/cuda/CUDAMacros.h

#define C10_CUDA_BUILD_SHARED_LIBS

#else
#error "This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."
#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
