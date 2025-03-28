#include <torch/extension.h>
#include <vector>

void multinomial_cuda_forward(
    at::Tensor probs,
    at::Tensor rand_vals,
    at::Tensor output,
    int64_t num_items,
    int64_t num_samples_per_row
);

void forward(
    at::Tensor probs,
    at::Tensor rand_vals,
    at::Tensor output,
    int64_t num_items,
    int64_t num_samples_per_row
) {
    multinomial_cuda_forward(probs, rand_vals, output, num_items, num_samples_per_row);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA multinomial forward");
}
