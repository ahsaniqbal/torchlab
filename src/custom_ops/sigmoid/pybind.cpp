#include "op.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &custom::ops::sigmoid::forward, "CustomSigmoid forward (CUDA)");
    m.def("backward", &custom::ops::sigmoid::backward, "CustomSigmoid backward (CUDA)");
}