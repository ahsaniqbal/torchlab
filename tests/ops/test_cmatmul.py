import os
import time
import torch
import pytest
import torchlab

def run_and_compare(matA: torch.Tensor, matB: torch.Tensor, time_it=False) -> None:
    if time_it:
        # Warm up
        _ = matA @ matB
        _ = torchlab.cmatmul(matA, matB)

    torch.cuda.synchronize()
    start = time.time()
    result_torch = matA @ matB
    torch.cuda.synchronize()
    time_torch = time.time() - start

    torch.cuda.synchronize()
    start = time.time()
    result_custom = torchlab.cmatmul(matA, matB)
    torch.cuda.synchronize()
    custom_time = time.time() - start

    atol = 1e-6 if matA.dtype == torch.float64 else 1e-6
    rtol = 1e-6 if matA.dtype == torch.float64 else 1e-5

    assert torch.allclose(result_custom, result_torch, atol=atol, rtol=rtol), (
        f"Forward mismatch for dtype={matA.dtype}: "
        f"max diff={ (result_custom - result_torch).abs().max().item() }"
    )

    if time_it:
        print()
        print(f"TorchTime for {matA.dtype}:{time_torch*1e3:.3f} ms")
        print(f"CustomTime for {matA.dtype}:{custom_time*1e3:.3f} ms")

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3,), (4, 2, 3)])
@pytest.mark.parametrize("M, N, K", [(2, 2, 2), (3, 4, 5), (5, 1, 7)])
def test_forward_matches_torch(dtype, batch_shape, M, N, K) -> None:
    matA = torch.randn((*batch_shape, M, N), dtype=dtype).cuda()
    matB = torch.randn((*batch_shape, N, K), dtype=dtype).cuda()
    run_and_compare(matA, matB)

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_broadcasting(dtype):
    # Broadcasting over batch dims
    matA = torch.randn(5, 1, 3, dtype=dtype).cuda()
    matB = torch.randn(3, 4, dtype=dtype).cuda()
    run_and_compare(matA, matB)

    matA = torch.randn(2, 2, 3, 4, dtype=dtype).cuda()
    matB = torch.randn(2, 1, 4, 5, dtype=dtype).cuda()
    run_and_compare(matA, matB)

    matA = torch.randn(2, 2, 3, 4, dtype=dtype).cuda()
    matB = torch.randn(2, 2, 2, 1, 4, 5, dtype=dtype).cuda()
    run_and_compare(matA, matB)

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_on_large_matrices(dtype):
    matA = torch.randn((1024, 1024), dtype=dtype).cuda()
    matB = torch.randn((1024, 1024), dtype=dtype).cuda()

    run_and_compare(matA, matB, time_it=True)