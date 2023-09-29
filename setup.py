import os
import sys
import glob
import os.path as osp
from itertools import product
from setuptools import setup, find_packages

os.environ["USE_ROCM"] = "0"
os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["MAX_JOBS"] = f"{os.cpu_count()}"
LD_LIBRARY_PATH = os.environ["LD_LIBRARY_PATH"]
if LD_LIBRARY_PATH.startswith(":"):
    LD_LIBRARY_PATH = LD_LIBRARY_PATH[1:]
os.environ["LD_LIBRARY_PATH"] = "/usr/lib:/usr/local/lib:/usr/local/cuda/lib64:" + LD_LIBRARY_PATH

import torch
from torch.__config__ import parallel_info
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME


WITH_CUDA = torch.cuda.is_available() and CUDA_HOME is not None
suffices = ["cpu", "cuda"]


def get_extensions():
    extensions = []

    extensions_dir = osp.join("csrc")
    main_files = glob.glob(osp.join(extensions_dir, "*.cpp"))

    for main, suffix in product(main_files, suffices):
        define_macros = []
        extra_compile_args = {"cxx": ["-O2"]}
        extra_link_args = ["-s"]

        info = parallel_info()
        if (
            "backend: OpenMP" in info
            and "OpenMP not found" not in info
            and sys.platform != "darwin"
        ):
            extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
            if sys.platform == "win32":
                extra_compile_args["cxx"] += ["/openmp"]
            else:
                extra_compile_args["cxx"] += ["-fopenmp"]
        else:
            print("Compiling without OpenMP...")

        if suffix == "cuda":
            define_macros += [("WITH_CUDA", None)]
            nvcc_flags = os.getenv("NVCC_FLAGS", "")
            nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
            nvcc_flags += ["--expt-relaxed-constexpr", "-O2"]
            extra_compile_args["nvcc"] = nvcc_flags

        name = main.split(os.sep)[-1][:-4]
        sources = [main]

        path = osp.join(extensions_dir, "cpu", f"{name}_cpu.cpp")
        if osp.exists(path):
            sources += [path]

        path = osp.join(extensions_dir, "cuda", f"{name}_cuda.cu")
        if suffix == "cuda" and osp.exists(path):
            sources += [path]

        Extension = CppExtension if suffix == "cpu" else CUDAExtension
        extension = Extension(
            f"torch_scatter._{name}_{suffix}",
            sources,
            include_dirs=[
                extensions_dir,
                "/usr/include",
                "/usr/local/include",
                "/usr/local/cuda/include",
            ],
            library_dirs=[
                "/usr/lib",
                "/usr/local/lib",
                "/usr/local/cuda/lib64",
            ],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
        extensions += [extension]

    return extensions


install_requires = []
setup_requires = []
tests_require = ["pytest", "pytest-runner", "pytest-cov"]

setup(
    name="torch_scatter",
    version="2.0.9",
    author="Matthias Fey",
    author_email="matthias.fey@tu-dortmund.de",
    url="https://github.com/rusty1s/pytorch_scatter",
    description="PyTorch Extension Library of Optimized Scatter Operations",
    keywords=["pytorch", "scatter", "segment", "gather"],
    license="MIT",
    python_requires=">=3.6",
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require={"test": tests_require},
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)},
    packages=find_packages(),
)
