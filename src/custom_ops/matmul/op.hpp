#pragma once
#include <torch/extension.h>

namespace torchlab::ops::matmul {
    at::Tensor forward(const at::Tensor& matA, const at::Tensor& matB);
}