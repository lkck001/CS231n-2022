from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
from os.path import join, dirname

# Get the directory where setup.py is located
setup_dir = dirname(__file__)

extensions = [
    Extension('im2col_cython', 
              sources=[join(setup_dir, 'im2col_cython.pyx')],
              include_dirs=[numpy.get_include()]
    ),
]

setup(
    ext_modules=cythonize(extensions),
) 