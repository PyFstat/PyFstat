import os
import sys

from setuptools import find_packages, setup

import versioneer

# check python version
min_python_version = (3, 7, 0)  # (major,minor,micro)
python_version = sys.version_info
print("Running Python version %s.%s.%s" % python_version[:3])
if python_version < min_python_version:
    sys.exit(
        "Python < %s.%s.%s is not supported, aborting setup" % min_python_version[:3]
    )
else:
    print("Confirmed Python version %s.%s.%s or above" % min_python_version[:3])


here = os.path.abspath(os.path.dirname(__file__))
# Get the long description from the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# dependencies
requires = [
    "attrs",
    "corner",
    "dill",
    "matplotlib>=2.1",
    "numpy",
    "pathos",
    "ptemcee",
    "scipy",
    "tqdm",
    "versioneer",
]
lalsuite = "lalsuite>=7.5"
extras_require = {
    "chainconsumer": ["chainconsumer"],
    "dev": [
        "pre-commit",
    ],
    "docs": [
        "sphinx==5.1.1",
        "sphinx_autodoc_typehints==1.19.2",
        "sphinx_rtd_theme==0.5.1",
        "sphinx_gallery==0.11.1",
        "m2r2==0.3.3",
        "mistune==0.8.4",
        "Pillow==9.2.0",
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
    "test": ["pytest", "pytest-cov", "flaky", "nbmake"],
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
