#!/usr/bin/env bash
# This script is intended to set up a proper development
# environment according to the PyFstat standad. 
# As explained in the REAMDE, that includes:
# - Making sure you are running under a python virtual environment.
# - Installing pytest to enable local testing.
# - Configuring pre-commit hooks to enforce the use of black and flake8.

# First of all, get this script's path
this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check whether you are running inside a virtual environment (you'd better do so!)
python ${this_dir}/check_if_virtual_environment.py

# Install development tools
pip install -r ${this_dir}/../requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
