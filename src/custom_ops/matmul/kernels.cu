#include <vector>
#include <numeric>
#include "ops.hpp"

__device__ int getBatchOffset(const int* __restrict__ stridesOut,
                              const int* __restrict__ stridesIn, 
                              const int* __restrict__ shapeIn,
                              int linearIndexOut, int rank) {
    int batchOffset = 0;
    int dimIndex = 0;
    for (int i = 0; i < (rank - 2); i++) {
        dimIndex = linearIndexOut / stridesOut[i];
        linearIndexOut %= stridesOut[i];
        batchOffset += (shapeIn[i] == 1 ? 0 : dimIndex) * stridesIn[i];
    }
    return batchOffset;
}




template<typename scalar_t>
__global__ void cmatmul_forward_kernel(const scalar_t* __restrict__ inputA,
    const scalar_t* __restrict__ inputB, scalar_t* __restrict__ output,
    const int* __restrict__ stridesA, const int* __restrict__ stridesB,
    const int* __restrict__ stridesOut, const int* __restrict__ shapeA,
    const int* __restrict__ shapeB, int64_t rank) {

        int batch = blockIdx.z;
        int rowIndex = blockIdx.y * blockDim.y + threadIdx.y; 
        int colIndex = blockIdx.x * blockDim.x + threadIdx.x;

        int rowsA = shapeA[rank - 2];
        int colsA = shapeA[rank - 1];

        int rowsB = shapeB[rank - 2];
        int colsB = shapeB[rank - 1];

        if (rowIndex < rowsA && colIndex < colsB) {
            auto indexOut = batch * rowsA * colsB + rowIndex * colsB + colIndex;

            auto batchOffsetA = getBatchOffset(stridesOut, stridesA, shapeA, indexOut, rank);
            auto batchOffsetB = getBatchOffset(stridesOut, stridesB, shapeB, indexOut, rank);

            auto outValue = scalar_t(0);
            for (int i = 0; i < colsA; i++) {
                auto indexA = batchOffsetA * rowsA * colsA + rowIndex * colsA + i;
                auto indexB = batchOffsetB * rowsB * colsB + i * colsB + colIndex;

                outValue += inputA[indexA] * inputB[indexB];
            }
            output[indexOut] = outValue;
        }
}

at::Tensor torchlab::ops::matmul::unsqueezeToDim(const at::Tensor& t, int targetDim) {
    TORCH_CHECK(t.dim() <= targetDim,
    "expected input tensor to have at most ", targetDim, " dims, got ", t.dim());

    if (t.dim() == targetDim) {
        return t;
    }

    auto out = t;
    auto thisDim = t.dim();
    for (int i=0; i< targetDim - thisDim; i++) {
        out = out.unsqueeze(0);
    }
    return out;
}

void torchlab::ops::matmul::validateOpInput(const at::Tensor& inputA, const at::Tensor& inputB) {
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
}

std::vector<int64_t> torchlab::ops::matmul::getOutputShape(const at::Tensor& matA, const at::Tensor& matB) {
    TORCH_CHECK(matA.dim() == matB.dim(),
                "cmatmul::forward tensors must have same number of dims after alignment, got ",
                matA.dim(), " vs ", matB.dim());

    auto matAShape = matA.sizes();
    auto matBShape = matB.sizes();

    auto outputShape = std::vector<int64_t>(matA.dim(), 1);
    outputShape[outputShape.size() - 1] = matBShape[matBShape.size() - 1];
    outputShape[outputShape.size() - 2] = matAShape[matAShape.size() - 2];

    for (int i = 0; i < outputShape.size() - 2; i++) {
        auto dimA = matAShape[i];
        auto dimB = matBShape[i];
        TORCH_CHECK(dimA == dimB || dimA == 1 || dimB == 1,
                    "cmatmul::forward the input tensors can't be broadcastable");
        outputShape[i] = std::max(dimA, dimB);
    }
    return outputShape;
}

at::Tensor torchlab::ops::matmul::forward(const at::Tensor& inputA, const at::Tensor& inputB) {
    validateOpInput(inputA, inputB);
    
    c10::cuda::CUDAGuard guard(inputA.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto targetDim = std::max(inputA.dim(), inputB.dim());
    auto matA = unsqueezeToDim(inputA, targetDim);
    auto matB = unsqueezeToDim(inputB, targetDim);

    auto outputShape = getOutputShape(matA, matB);

    auto output = at::empty(outputShape, matA.options());

    auto stridesA = at::tensor(matA.strides(), matA.options().dtype(at::kLong));
    auto stridesB = at::tensor(matB.strides(), matB.options().dtype(at::kLong));
    auto stridesOut = at::tensor(output.strides(), output.options().dtype(at::kLong));

    int numThreadsRows = 32;
    int numThreadsCols = 32;
    //kernel call
    AT_DISPATCH_FLOATING_TYPES(matA.scalar_type(), "cmatmul::forward", [&]{
        const scalar_t* dataMatA = matA.data_ptr<scalar_t>();
        const scalar_t* dataMatB = matB.data_ptr<scalar_t>();

        scalar_t* dataOut = output.data_ptr<scalar_t>();

        int64_t batchCount = std::accumulate(outputShape.begin(), outputShape.end() - 2, int64_t(1), std::multiplies<int64_t>());

        dim3 gridConfig = dim3((outputShape[outputShape.size() - 2] + numThreadsRows - 1) / numThreadsRows,
                                (outputShape[outputShape.size() - 1] + numThreadsCols - 1) / numThreadsCols,
                                batchCount);
        dim3 blockConfig = dim3(numThreadsRows, numThreadsCols, 1);
        
        //kernel_launch_here
        //cmatmul_forward_kernel<scalar_t><<<gridConfig, blockConfig, 0, stream>>>(dataMatA, dataMatB, dataOut,
        //    matA.sizes()[matA.dim() - 2], matA.sizes()[matA.dim() - 1], matB.sizes()[matB.dim() - 1], batchCount, stridesOut, stridesA, stridesB, output.dim());

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return output;
}