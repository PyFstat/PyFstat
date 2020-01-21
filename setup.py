#!/usr/bin/env python

from setuptools import setup, find_packages
from os import path
import sys

# check python version
min_python_version = (3, 5, 0)  # (major,minor,micro)
python_version = sys.version_info
print("Running Python version %s.%s.%s" % python_version[:3])
if python_version < min_python_version:
    sys.exit(
        "Python < %s.%s.%s is not supported, aborting setup" % min_python_version[:3]
    )
else:
    print("Confirmed Python version %s.%s.%s or above" % min_python_version[:3])


here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="PyFstat",
    version="1.3",
    author="Gregory Ashton, David Keitel, Reinhard Prix",
    author_email="gregory.ashton@ligo.org",
    license="MIT",
    description="python wrappers for lalpulsar F-statistic code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "pyfstat": [
            "pyCUDAkernels/cudaTransientFstatExpWindow.cu",
            "pyCUDAkernels/cudaTransientFstatRectWindow.cu",
        ]
    },
    python_requires=">=%s.%s.%s" % min_python_version[:3],
    install_requires=[
        "matplotlib",
        "scipy",
        "ptemcee",
        "corner",
        "dill",
        "tqdm",
        "bashplotlib",
        "peakutils",
        "pathos",
        "lalsuite",
    ],
)
