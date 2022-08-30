import logging

# import os
import sys

from pyfstat.logging import _get_default_logger, set_up_logger


def test_get_default_logger():

    # No handlers in root: Should get pyfstat logger
    root_logger = logging.getLogger()
    while root_logger.hasHandlers():
        root_logger.removeHandler(root_logger.handlers[0])

    assert _get_default_logger() is logging.getLogger("pyfstat")

    # Handler in root logger: Should get root logger
    logging.basicConfig(stream=sys.stdout)
    assert _get_default_logger() is logging.getLogger()


def test_set_up_logger():
    # Clear all handlers
    logger = set_up_logger(streams=None, append=False)
    assert not len(logger.handlers)

    # Test default behaviour
    logger = set_up_logger()
    assert len(logger.handlers) == 1
    handler = logger.handlers[0]
    assert type(handler) == logging.StreamHandler
    assert handler.stream is sys.stdout

    # Test behaviour with files
    # logger = set_up_logger(label="test_log", outdir=pytest.tmp_path, append=False)
    # log_file = os.path.join(pytest.tmp_path, label + ".log")
    # assert os.path.isfile(log_file)
    # assert len(logger.handlers) == 1
    # handler = logger.handlers[0]
    # assert type(handler) == logging.FileHandler
    # assert handler.fileBasename == log_file

    # Test append's behaviour
    logger = set_up_logger()
    # logger = set_up_logger(label="test_log", outdir=pytest.tmp_path)
    # assert len(logger.handlers) == 2

    # Check file handler

    # Check stream handler
