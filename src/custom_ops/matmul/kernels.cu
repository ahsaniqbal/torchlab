#include "ops.hpp"

template<typename scalar_t>
__global__ void cmatmul_forward_kernel(const scalar_t* __restrict__ inputA,
    const scalar_t* __restrict__ inputB, scalar_t* __restrict__ output,
    int64_t rowsA, int64_t colsA, int64_t colsB) {

}


at::Tensor torchlab::ops::matmul::forward(const at::Tensor& inputA, const at::Tensor& inputB) {
    TORCH_CHECK(inputA.is_cuda(), "cmatmul::forward inputA must be a CUDA Tensor.");
    TORCH_CHECK(inputB.is_cuda(), "cmatmul::forward inputB must be a CUDA Tensor.");
    TORCH_CHECK(inputA.device() == inputB.device(),
            "cmatmul::forward both inputs must be on the same device");

    TORCH_CHECK(inputA.layout() == at::kStrided, "cmatmul::forward inputA must be a strided Tensor.");
    TORCH_CHECK(inputB.layout() == at::kStrided, "cmatmul::forward inputB must be a strided Tensor.");

    TORCH_CHECK(inputA.scalar_type() == at::kFloat || inputA.scalar_type() == at::kDouble,
                    "cmatmul::forward inputA must have float or double as a scalar type.");
    TORCH_CHECK(inputB.scalar_type() == at::kFloat || inputB.scalar_type() == at::kDouble,
                    "cmatmul::forward inputB must have float or double as a scalar type.");

    TORCH_CHECK(inputA.scalar_type() == inputB.scalar_type(),
                    "cmatmul::forward both inputs must have same scalar type.");

    TORCH_CHECK(inputA.numel() > 0, "cmatmul::forward inputA must have atleast 1 elements.");
    TORCH_CHECK(inputB.numel() > 0, "cmatmul::forward inputB must have atleast 1 elements.");

    auto shapeA = inputA.sizes();
    auto shapeB = inputB.sizes();

    TORCH_CHECK(shapeA.size() >= 2 && shapeB.size() >= 2 && shapeA[shapeA.size() - 1] == shapeB[shapeB.size() - 2],
                    "cmatmul::forward both inputs should be compatible for matrix multiplication.");
    
    c10::cuda::CUDAGuard guard(inputA.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto batchA = inputA.dim() - 2;
    auto batchB = inputB.dim() - 2;
    auto maxBatch = std::max(batchA, batchB);

    auto outputShape = std::vector<int64_t>(maxBatch + 2, 1);
    outputShape[outputShape.size() - 1] = shapeB[shapeB.size() - 1];
    outputShape[outputShape.size() - 2] = shapeA[shapeA.size() - 2];

    for (int i = 0; i < maxBatch; i++) {
        auto dimA = i >= batchA ? 1 : shapeA[batchA - i - 1];
        auto dimB = i >= batchB ? 1 : shapeB[batchB - i - 1];
        TORCH_CHECK(dimA == dimB || dimA == 1 || dimB == 1,
                    "cmatmul::forward the input tensors can't be broadcastable");
        outputShape[outputShape.size() - (3 + i)] = std::max(dimA, dimB);
    }

    auto output = at::empty(outputShape, inputA.options());

    //kernel call

    return output;
}