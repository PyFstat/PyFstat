#!/usr/bin/env python

from setuptools import setup, find_packages
from os import path
import sys
import subprocess


def write_version_file(version):
    """ Writes a file with version information to be used at run time

    Parameters
    ----------
    version: str
        A string containing the current version information

    Returns
    -------
    version_file: str
        A path to the version file

    """
    try:
        git_log = subprocess.check_output(
            ['git', 'log', '-1', '--pretty=%h %ai']).decode('utf-8')
        git_diff = (subprocess.check_output(['git', 'diff', '.']) +
                    subprocess.check_output(
                        ['git', 'diff', '--cached', '.'])).decode('utf-8')
        if git_diff == '':
            git_status = '(CLEAN) ' + git_log
        else:
            git_status = '(UNCLEAN) ' + git_log
    except Exception as e:
        print("Unable to obtain git version information, exception: {}"
              .format(e))
        git_status = ''

    version_file = '.version'
    if path.isfile(version_file) is False:
        with open('pyfstat/' + version_file, 'w+') as f:
            f.write('{}: {}'.format(version, git_status))
        print('Done', version_file, version, git_status)

    return version_file


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

VERSION = '1.3'
version_file = write_version_file(VERSION)

setup(
    name="PyFstat",
    version=VERSION,
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
            version_file
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
