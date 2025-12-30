"""
Pytest configuration and fixture definitions.

This file makes fixtures from commons_for_tests.py available to all test files
in this directory and subdirectories.
"""

# Import fixtures from commons_for_tests so pytest can discover them
from commons_for_tests import (
    data_fixture,
    default_binary_parameters,
    default_signal_parameters,
    default_signal_parameters_no_sky,
    default_transient_parameters,
    default_Writer_parameters,
    outdir,
)

# Make fixtures available (they're already decorated with @pytest.fixture)
__all__ = [
    "outdir",
    "data_fixture",
    "default_Writer_parameters",
    "default_signal_parameters",
    "default_signal_parameters_no_sky",
    "default_binary_parameters",
    "default_transient_parameters",
]
