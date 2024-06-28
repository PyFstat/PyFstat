import os
import sys

from setuptools import find_packages, setup

import versioneer

# check python version
min_python_version = (3, 9, 0)  # (major,minor,micro)
next_unsupported_python_version = (3, 13)  # (major,minor) - don't restrict micro
python_version = sys.version_info
if (
    python_version < min_python_version
    or python_version > next_unsupported_python_version
):
    sys.exit(
        f"Python {'.'.join(str(v) for v in python_version[:3])} is not supported, aborting setup."
    )
else:
    print(
        f"Confirmed Python version between [{'.'.join(str(v) for v in min_python_version[:3])},"
        f" {'.'.join(str(v) for v in next_unsupported_python_version[:2])}]"
    )


here = os.path.abspath(os.path.dirname(__file__))
# Get the long description from the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# dependencies
requires = [
    "attrs",
    "corner",
    "dill",
    "matplotlib>=3.3",
    "numpy<2.0",
    "pathos",
    "ptemcee",
    "scipy",
    "setuptools; python_version >= '3.12'",
    "tqdm",
    "versioneer",
]
lalsuite = "lalsuite[lalpulsar]>=7.13"
extras_require = {
    "chainconsumer": ["chainconsumer"],
    "dev": [
        "pre-commit",
    ],
    "docs": [
        "sphinx==5.3.0",
        "sphinx_autodoc_defaultargs==0.1.2",
        "sphinx_autodoc_typehints==1.19.5",
        "sphinx_gallery==0.11.1",
        "sphinx_rtd_theme==0.5.1",
        "m2r2==0.3.3",
        "mistune==0.8.4",
        "Pillow==9.3.0",
    ],
    "pycuda": ["pycuda"],
    "style": [
        "black",
        "flake8",
        "flake8-docstrings",
        "flake8-executable",
        "flake8-isort",
        "isort",
    ],
    "test": ["pytest<8.2.2", "pytest-cov", "flaky", "nbmake"],
    "wheel": ["wheel", "check-wheel-contents"],
}
for key in ["docs", "style", "test", "wheel"]:
    extras_require["dev"] += extras_require[key]

setup(
    name="PyFstat",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Gregory Ashton, David Keitel, Reinhard Prix, Rodrigo Tenorio",
    author_email="gregory.ashton@ligo.org",
    maintainer="David Keitel",
    maintainer_email="david.keitel@ligo.org",
    license="MIT",
    description="a python package for gravitational wave analysis with the F-statistic",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PyFstat/PyFstat",
    project_urls={
        "Changelog": "https://github.com/PyFstat/PyFstat/blob/master/CHANGELOG.md",
        "Documentation": "https://pyfstat.readthedocs.io/",
        "Issue tracker": "https://github.com/PyFstat/PyFstat/issues",
    },
    packages=find_packages(),
    package_data={
        "pyfstat": [
            "pyCUDAkernels/cudaTransientFstatExpWindow.cu",
            "pyCUDAkernels/cudaTransientFstatRectWindow.cu",
        ]
    },
    platforms="POSIX",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=%s.%s.%s" % min_python_version[:3],
    install_requires=requires
    + ([] if os.environ.get("NO_LALSUITE_FROM_PYPI", False) else [lalsuite]),
    extras_require=extras_require,
)
