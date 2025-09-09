#include "op.hpp"

// Dispatcher schema: declares operator signatures
TORCH_LIBRARY(torchlab, m) {
    m.def("customsigmoid(Tensor input) -> Tensor");
}

// CPU dispatch fallback (optional): raise error if called on CPU
TORCH_LIBRARY_IMPL(torchlab, CPU, m) {
    m.impl("customsigmoid", [](const at::Tensor& input) {
        TORCH_CHECK(false, "customsigmoid not implemented for CPU");
        return at::Tensor();
    });
}

// CUDA dispatch for forward
TORCH_LIBRARY_IMPL(torchlab, CUDA, m) {
    m.impl("customsigmoid", &torchlab::ops::sigmoid::forward);
}

// Register autograd formula
namespace torchlab::ops::sigmoid {
    struct SigmoidAutograd : public torch::autograd::Function<torchlab::ops::sigmoid::SigmoidAutograd> {
        static at::Tensor forward(torch::autograd::AutogradContext* ctx, const at::Tensor& input) {
            auto y = torchlab::ops::sigmoid::forward(input);
            ctx->save_for_backward({y});
            return y;
        }
        static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_outputs) {
            auto saved = ctx->get_saved_variables();
            auto y = saved[0];
            auto grad_out = grad_outputs[0];
            auto grad_in = torchlab::ops::sigmoid::backward(grad_out, y);
            return {grad_in};
        }
    };
}


TORCH_LIBRARY_IMPL(torchlab, Autograd, m) {
    m.impl("customsigmoid", [](const at::Tensor& input) {
        return torchlab::ops::sigmoid::SigmoidAutograd::apply(input);
    });
}