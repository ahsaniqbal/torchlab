#include "op.hpp"

/*TORCH_LIBRARY(torchlab, m) {
    m.def("cmatmul(Tensor inputA, Tensor inputB) -> Tensor");
}*/

TORCH_LIBRARY_FRAGMENT(torchlab, m) {
    m.def("cmatmul(Tensor inputA, Tensor inputB) -> Tensor");
}

TORCH_LIBRARY_IMPL(torchlab, CPU, m) {
    m.impl("cmatmul", [](const at::Tensor& inputA, const at::Tensor& inputB){
        TORCH_CHECK(false, "cmatmul not implemented for CPU.");
        return at::Tensor();
    });
}

TORCH_LIBRARY_IMPL(torchlab, CUDA, m) {
    m.impl("cmatmul", &torchlab::ops::matmul::forward);
}