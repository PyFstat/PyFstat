import logging
import sys

from pyfstat.logging import _get_default_logger


def test_get_default_logger():

    # No handlers in root: Should get pyfstat logger
    root_logger = logging.getLogger()
    while root_logger.hasHandlers():
        root_logger.removeHandler(root_logger.handlers[0])

    assert _get_default_logger() is logging.getLogger("pyfstat")

    # Handler in root logger: Should get root logger
    logging.basicConfig(stream=sys.stdout)
    assert _get_default_logger() is logging.getLogger()
