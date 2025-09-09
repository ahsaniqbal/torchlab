#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h> 
#include <cuda_runtime.h>
#include <math_constants.h>   // for constants like CUDART_INF
//#include <math_functions.h>   // optional; device math decls (often pulled transitively)
#include "op.hpp"

// Kernel declarations (stubs, no implementation yet)
template <typename scalar_t>
__global__ void custom_sigmoid_forward_kernel(const scalar_t* __restrict__ input,
                                         scalar_t* __restrict__ output,
                                         int32_t numel) {
    int32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    //This branching is only to have numerical stable solution 
    //Naive formula sigmoid(x) = 1/(1+exp(-x)), could be for very large negative x
    //It can be proven exp(x)/(1+exp(x)) = 1/(1+exp(-x)), so branching is for stability.
    
    if (index < numel) {
        scalar_t x = input[index];
        if (x >= 0) {
            scalar_t z = exp(-x);
            output[index] = scalar_t(1) / (scalar_t(1) + z);
        }
        else {
            scalar_t z = exp(x);
            output[index] = z / (scalar_t(1) + z);
        }
    }

    //However this branching results in control divergence at warp level. 

}

template <typename scalar_t>
__global__ void custom_sigmoid_backward_kernel(const scalar_t* __restrict__ grad_output,
                                          const scalar_t* __restrict__ saved_output,
                                          scalar_t* __restrict__ grad_input,
                                          int32_t numel) {
    int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel) {
        const scalar_t y = saved_output[index];
        const scalar_t dy = grad_output[index];

        grad_input[index] = dy * y * (scalar_t(1) - y);
    }

}




// Host function called from the C++ glue / pybind module
at::Tensor torchlab::ops::sigmoid::forward(const at::Tensor& input) {
    // Basic guards to fail fast with a helpful error message.
    TORCH_CHECK(input.is_cuda(), "custom_sigmoid_forward: input must be a CUDA tensor");
    TORCH_CHECK(input.layout() == at::kStrided,
                "custom_sigmoid_forward: only strided (dense) tensors are supported");
    TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kDouble,
                "custom_sigmoid_forward: supported dtypes are float32/float64");

    // This will set the *current CUDA device* of the calling host thread
    // to match the device of `input`. That ensures:
    //   1) Any allocations (like `empty_like`) happen on the same GPU as `input`.
    //   2) Kernel launches go to the correct device by default.
    // It’s RAII: when `guard` goes out of scope, the previous device is restored.
    c10::cuda::CUDAGuard guard(input.device());

    // Each CUDA device supports multiple *streams* (ordered queues of work).
    // PyTorch maintains a per-thread, per-device "current stream" that it uses for ops.
    // Grabbing it here means our kernel is enqueued on the same stream, so it executes
    // in-order with surrounding PyTorch ops (no manual syncs needed).
    auto stream = at::cuda::getCurrentCUDAStream();

    // Without making the tensor contiguous, views like transposes/slices can have
    // non-default *strides*. Example: a transpose shares the same underlying storage
    // but changes indexing; it is still a *strided* dense tensor, just not stored
    // in a compact row-major layout. If our kernel assumes linear indexing
    // (i.e., element i sits at base[i]), we must ensure a compact layout.
    // `.contiguous()` returns a compact (row-major) copy if needed; otherwise it’s a no-op.
    auto x = input.contiguous();

    // Create the output tensor with the same shape/dtype/device as `x` (hence as `input`).
    auto y = at::empty_like(x);

    const int64_t n = x.numel();
    if (n == 0) {
        // Nothing to do; return an empty-like tensor of the correct shape/options.
        return y;
    }

    // Kernel launch configuration:
    //  - `threads` threads per block (a common choice is 128/256/512).
    //  - `blocks` so that blocks*threads >= n (full coverage).
    const int threads = 256;
    const int blocks  = static_cast<int>((n + threads - 1) / threads);

    // Runtime dtype → compile-time `scalar_t`. This macro expands to a switch
    // over the input dtype and runs the lambda once with `scalar_t = float`
    // (for kFloat) or `scalar_t = double` (for kDouble). That way we compile
    // a specialized kernel for each supported dtype.
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "custom_sigmoid_forward", [&]{
        // Raw, typed device pointers into the `x` and `y` storages.
        const scalar_t* in_ptr = x.data_ptr<scalar_t>();
        scalar_t*       out_ptr = y.data_ptr<scalar_t>();

        // CUDA launch syntax:
        //   <<< gridDim, blockDim, sharedMemBytes (dynamic), stream >>>
        //
        // Here:
        //   - `blocks` = number of thread blocks
        //   - `threads` = threads per block
        //   - `0` = no *dynamic* shared memory requested (we don’t use extern __shared__)
        //   - `stream` = PyTorch’s current CUDA stream (keeps correct op ordering)
        //
        // Note: depending on your headers, `CUDAStream` can implicitly convert to `cudaStream_t`.
        // If your compiler complains, use `stream.stream()` instead.
        custom_sigmoid_forward_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(in_ptr, out_ptr, n);

        // Check for launch errors. This does *not* add a device-wide sync on success;
        // it only synchronizes the stream if there was an error to surface it.
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });

    return y;
}

at::Tensor torchlab::ops::sigmoid::backward(const at::Tensor& grad_output, const at::Tensor& saved_output) {
    TORCH_CHECK(grad_output.is_cuda(), "custom_sigmoid_backward: grad_output must be a CUDA tensor");
    TORCH_CHECK(saved_output.is_cuda(), "custom_sigmoid_backward: saved_output must be a CUDA tensor");
    TORCH_CHECK(grad_output.device() == saved_output.device(),
            "custom_sigmoid_backward: grad_output and saved_output must be on the same device");
    
    TORCH_CHECK(grad_output.layout() == at::kStrided, 
        "custom_sigmoid_backward: grad_output must be a strided (dense) tensor");
    TORCH_CHECK(saved_output.layout() == at::kStrided, 
        "custom_sigmoid_backward: saved_output must be a strided (dense) tensor");
    
    TORCH_CHECK(grad_output.scalar_type() == at::kFloat || grad_output.scalar_type() == at::kDouble, 
        "custom_sigmoid_backward: the dtype of grad_output must be float/double");
    TORCH_CHECK(saved_output.scalar_type() == at::kFloat || saved_output.scalar_type() == at::kDouble, 
        "custom_sigmoid_backward: the dtype of saved_output must be float/double");
    TORCH_CHECK(grad_output.scalar_type() == saved_output.scalar_type(), 
        "custom_sigmoid_backward: the dtype of grad_output and saved_output must be same");

    TORCH_CHECK(grad_output.sizes() == saved_output.sizes(), 
        "custom_sigmoid_backward: the tensors grad_output and saved_output must have same shape");

    c10::cuda::CUDAGuard guard(grad_output.device());

    auto stream = at::cuda::getCurrentCUDAStream();

    auto grad_output_cont = grad_output.contiguous();
    auto saved_output_cont = saved_output.contiguous();
    auto grad_input = at::empty_like(grad_output_cont);

    auto num_elements = grad_output_cont.numel();

    if (num_elements > 0) {
        const int threads = 256;
        const int blocks = static_cast<int>((num_elements + threads - 1) / threads);

        AT_DISPATCH_FLOATING_TYPES(grad_output_cont.scalar_type(), "custom_sigmoid_backward", [&]{
            const scalar_t* data_grad_output = grad_output_cont.data_ptr<scalar_t>();
            const scalar_t* data_saved_output = saved_output_cont.data_ptr<scalar_t>();
            scalar_t* data_grad_input = grad_input.data_ptr<scalar_t>();

            custom_sigmoid_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(data_grad_output,
                data_saved_output, data_grad_input, num_elements);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    }

    return grad_input; 
}