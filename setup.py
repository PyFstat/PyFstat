#!/usr/bin/env python

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="PyFstat",
    version="0.2",
    author="Gregory Ashton",
    author_email="gregory.ashton@ligo.org",
    packages=find_packages(where="pyfstat"),
    include_package_data=True,
    package_data={
        "pyfstat": [
            "pyCUDAkernels/cudaTransientFstatExpWindow.cu",
            "pyCUDAkernels/cudaTransientFstatRectWindow.cu",
        ]
    },
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
    ],
)
