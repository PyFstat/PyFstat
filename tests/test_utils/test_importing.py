import os

import pytest

from pyfstat.utils import safe_X_less_plt


@pytest.mark.skipif(
    "DISPLAY" not in os.environ, reason="Running test on X-less machine"
)
def test_safe_X_less_plt_with_X(caplog):

    # No default backend is defined for matplotlib
    import matplotlib
    import matplotlib.pyplot as plt

    default_backend = matplotlib.get_backend()

    safe_plt = safe_X_less_plt()

    assert safe_plt is plt
    assert matplotlib.get_backend() is default_backend


def test_safe_X_less_plt_without_X(caplog):

    # Remove DISPLAY if it's there
    if "DISPLAY" in os.environ:
        del os.environ["DISPLAY"]
    safe_X_less_plt()

    _, log_level, log_message = caplog.record_tuples[-1]
    # Backend is fixed by the function
    import matplotlib

    assert matplotlib.get_backend().lower() == "agg"
    assert log_level == 20
    assert "No $DISPLAY environment variable" in log_message
