from torch.utils.cpp_extension import load
import os

src_dir = os.path.dirname(__file__)

cuda_multinomial = load(
    name="cuda_multinomial",
    sources=[
        os.path.join(src_dir, "cuda_multinomial.cpp"),
        os.path.join(src_dir, "cuda_multinomial_kernel.cu"),
    ],
    verbose=False,
)
