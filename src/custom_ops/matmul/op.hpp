#pragma once
#include <torch/extension.h>

namespace torchlab::ops::matmul {
    void copyStridesToDevice(const at::Tensor& t, int **strides_d);
    at::Tensor adjustTensor(const at::Tensor& t, int targetDim);
    at::Tensor forward(const at::Tensor& matA, const at::Tensor& matB);
}