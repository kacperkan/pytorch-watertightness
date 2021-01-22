from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Python interface
setup(
    name="pytorch_watertightness",
    version="0.2.0",
    install_requires=["torch"],
    packages=["watertightness"],
    package_dir={"watertightness": "./"},
    ext_modules=[
        CUDAExtension(
            name="watertightness_backend",
            include_dirs=["./"],
            sources=[
                "pybind/bind.cpp",
            ],
            libraries=["watertightness"],
            library_dirs=["objs"],
            # extra_compile_args=['-g']
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    author="Christopher B. Choy",
    author_email="chrischoy@ai.stanford.edu",
    description="Tutorial for Pytorch C++ Extension with a Makefile",
    keywords="Pytorch C++ Extension",
    url="https://github.com/chrischoy/MakePytorchPlusPlus",
    zip_safe=False,
)
