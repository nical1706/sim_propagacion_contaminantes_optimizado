from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        name="simulator_core_no_omp", 
        sources=[
            "cython_module.pyx", 
            "contaminationSeparated.c", 
            "simulatorModule.c",
        ],
        #Obligatorio para que encuentre numpy/arrayobject.h
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3', '-march=native', '-flto', '-fopenmp'],
        extra_link_args=['-O3', '-flto', '-fopenmp']
    )
]

setup(
    name="SimuladorContaminacion",
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"})
)