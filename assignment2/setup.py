from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension('cs231n.im2col_cython', ['cs231n/im2col_cython.pyx'],
        include_dirs = [numpy.get_include()]
    ),
]

setup(
    name='cs231n',
    ext_modules=cythonize(extensions),
    packages=find_packages(),
) 