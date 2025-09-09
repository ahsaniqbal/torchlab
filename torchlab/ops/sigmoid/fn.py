import torch
import custom_sigmoid_pybind_ext as _ext  # built by setup.py

class CustomSigmoidFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        y = _ext.forward(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_outputs) -> torch.Tensor:
        (y,) = ctx.saved_tensors
        return _ext.backward(grad_outputs, y)
    
def customsigmoid(x):
    return CustomSigmoidFn.apply(x)