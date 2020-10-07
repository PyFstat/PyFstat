#!/usr/bin/env python3
import os
import sys

is_virtual_environment = sys.prefix != sys.base_prefix
is_conda_environment = os.path.exists(os.path.join(sys.prefix, "conda-meta", "history"))

if not (is_virtual_environment or is_conda_environment):
    sys.exit(
        "Sorry, you are not using neither a virtual environment"
        " nor a conda environment; aborting operation."
    )

for name, state in [
    ("Python virtual environment {}", is_virtual_environment),
    ("Conda environment {}", is_conda_environment),
]:
    print(name.format("detected." if state else "not detected."))
print(f"Executable: {sys.executable}")
