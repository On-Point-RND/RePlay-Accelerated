#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void multinomial_kernel(
    const float* __restrict__ probs,
    const float* __restrict__ rand_vals,
    int64_t* __restrict__ output,
    const int64_t batch_size,
    const int64_t num_items,
    const int64_t num_samples
) {
    int row = blockIdx.x;
    int sample = threadIdx.x;

    float r = rand_vals[row * num_samples + sample];
    float cum_prob = 0.0f;

    for (int i = 0; i < num_items; ++i) {
        cum_prob += probs[i];
        if (r < cum_prob) {
            output[row * num_samples + sample] = i;
            return;
        }
    }

    output[row * num_samples + sample] = num_items - 1;
}

void multinomial_cuda_forward(
    at::Tensor probs,
    at::Tensor rand_vals,
    at::Tensor output,
    int64_t num_items,
    int64_t num_samples_per_row
) {
    const auto batch_size = rand_vals.size(0);
    const dim3 blocks(batch_size);
    const dim3 threads(num_samples_per_row);

    multinomial_kernel<<<blocks, threads>>>(
        probs.data_ptr<float>(),
        rand_vals.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        batch_size,
        num_items,
        num_samples_per_row
    );
}
