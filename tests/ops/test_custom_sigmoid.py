import os
import torch
import pytest
import torchlab

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_matches_torch(dtype):
    x = torch.randn(1024, device="cuda", dtype=dtype)
    y_my = torchlab.customsigmoid(x)
    y_torch = torch.sigmoid(x)

    atol = 1e-6 if dtype == torch.float64 else 1e-6
    rtol = 1e-6 if dtype == torch.float64 else 1e-5

    assert torch.allclose(y_my, y_torch, atol=atol, rtol=rtol), (
        f"Forward mismatch for dtype={dtype}: "
        f"max diff={ (y_my - y_torch).abs().max().item() }"
    )

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_backward_matches_torch(dtype):
    x1 = torch.randn(1024, device="cuda", dtype=dtype, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    target = torch.randn_like(x1)

    y1 = torchlab.customsigmoid(x1)
    loss1 = torch.mean((y1 - target) ** 2)
    loss1.backward()

    y2 = torch.sigmoid(x2)
    loss2 = torch.mean((y2 - target) ** 2)
    loss2.backward()

    atol = 1e-6 if dtype == torch.float64 else 1e-6
    rtol = 1e-6 if dtype == torch.float64 else 1e-5

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol), (
        f"Backward mismatch for dtype={dtype}: "
        f"max diff={ (x1.grad - x2.grad).abs().max().item() }"
    )

def test_gradcheck():
    x = torch.randn(16, device="cuda", dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(torchlab.customsigmoid, (x,), eps=1e-6, atol=1e-6), f"gradcheck failed"