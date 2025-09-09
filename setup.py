from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

setup(
    name="torchlab",
    version="0.1.0",
    description="A CUDA playground for custom PyTorch ops, layers, and modules",
    author="Ahsan Iqbal",
    packages=find_packages(include=["torchlab", "torchlab.*"]),
    package_data={
        "torchlab.ops.sigmoid": ["*.so"],  # install dispatcher .so inside package
    },
    ext_modules=[
        # Option A: PyBind-based extension (behaves like a Python module)
        #CUDAExtension(
        #    name="custom_sigmoid_pybind_ext",
        #    sources=[
        #        "src/custom_ops/sigmoid/pybind.cpp",
        #        "src/custom_ops/sigmoid/kernels.cu",
        #    ],
        #    extra_compile_args={
        #        "cxx": [f"-D_GLIBCXX_USE_CXX11_ABI={torch._C._GLIBCXX_USE_CXX11_ABI}"]
        #    },
        #),
        # Option B: Dispatcher-based extension (pure C++/CUDA shared lib)
        CUDAExtension(
            name="torchlab.torchlab_C",
            sources=[
                "src/custom_ops/sigmoid/dispatch.cpp",
                "src/custom_ops/sigmoid/kernels.cu",
            ],
            extra_compile_args={
                "cxx": [f"-D_GLIBCXX_USE_CXX11_ABI={torch._C._GLIBCXX_USE_CXX11_ABI}"]
                #"cxx": ["-DPy_LIMITED_API=0x03090000"]
            },
            #py_limited_api=True,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
    zip_safe=False,
)
