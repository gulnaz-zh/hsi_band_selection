from setuptools import Extension, setup

import numpy

try:
    from Cython.Build import cythonize
except ImportError as exc:
    raise SystemExit(
        "Cython is required to build the RegionGrowth extension. "
        "Install it with: python -m pip install cython"
    ) from exc


extensions = [
    Extension(
        name="RegionGrowth",
        sources=["RegionGrowth.pyx"],
        include_dirs=[numpy.get_include()],
    )
]


setup(
    name="hsi-region-growing",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
