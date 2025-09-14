#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h> 
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace torchlab::ops::matmul {
    void validateOpInput(const at::Tensor& matA, const at::Tensor& matB);
    at::Tensor unsqueezeToDim(const at::Tensor& t, int targetDim);
    std::vector<int64_t> getOutputShape(const at::Tensor& matA, const at::Tensor& matB);
    at::Tensor forward(const at::Tensor& matA, const at::Tensor& matB);
}