#pragma once
#include <torch/extension.h>

// Forward declaration of CUDA stubs
namespace torchlab::ops::sigmoid{
    at::Tensor forward(const at::Tensor& input);
    at::Tensor backward(const at::Tensor& grad_output, const at::Tensor& saved_output);
}

