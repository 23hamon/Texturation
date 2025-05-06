from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

setup(
    name="utils_cpp", 
    ext_modules=cythonize([
        Extension(
            "utils_cpp", 
            sources=["wrapper.pyx", "utils_cpp.cpp"],
            language="c++", 
            include_dirs=["/usr/include/eigen3"],  
            extra_compile_args=["-std=c++11"]
        )
    ]),
)
