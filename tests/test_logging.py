import logging
import os
import sys

from pyfstat.logging import _get_default_logger, set_up_logger


def _check_FileHandler(handler, log_file):
    assert type(handler) == logging.FileHandler
    assert handler.baseFilename == log_file
    assert os.path.isfile(log_file)


def _check_StreamHandler(handler, stream):
    assert type(handler) == logging.StreamHandler
    assert handler.stream is stream


def test__get_default_logger():
    # No handlers in root: Should get pyfstat logger
    root_logger = logging.getLogger()
    while root_logger.hasHandlers():
        root_logger.removeHandler(root_logger.handlers[0])
    assert _get_default_logger() is logging.getLogger("pyfstat")

    # Handler in root logger: Should get root logger
    logging.basicConfig(stream=sys.stdout)
    assert _get_default_logger() is logging.getLogger()


def test_set_up_logger(tmp_path):
    file_args = {"outdir": tmp_path, "label": "test_log"}
    kwargs = {
        "StreamHandler": {"stream": sys.stdout},
        "FileHandler": {
            "log_file": os.path.join(file_args["outdir"], file_args["label"] + ".log")
        },
    }

    # Test default behaviour
    for logger in [
        set_up_logger(),
        set_up_logger(append=False, streams=None, **file_args),
    ]:
        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        name = type(handler).__name__
        globals()[f"_check_{name}"](handler=handler, **kwargs[name])

    # Test append's behaviour
    logger = set_up_logger()
    logger = set_up_logger(**file_args)
    assert len(logger.handlers) == 2
    for handler in logger.handlers:
        name = type(handler).__name__
        globals()[f"_check_{name}"](handler=handler, **kwargs[name])
