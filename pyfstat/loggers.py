import logging
import os
import sys

from pyfstat.helper_functions import get_version_string


def set_up_logger(outdir=None, label="pyfstat", log_level="INFO"):
    """Setup the logger.

    Based on the implementation in Nessai:
    https://github.com/mj-will/nessai/blob/main/nessai/utils/logging.py

    Parameters
    ----------
    outdir : str, optional
        Path to outdir directory.
    label : str, optional
        Label for this instance of the logger.
        Defaults to `pyfstat`, which is the "root" logger of this package.
    log_level : {'ERROR', 'WARNING', 'INFO', 'DEBUG'}, optional
        Level of logging passed to logger.

    Returns
    -------
    obj:`logging.Logger`
        Instance of the Logger class.

    """
    if type(log_level) is str:
        try:
            level = getattr(logging, log_level.upper())
        except AttributeError:
            raise ValueError("log_level {} not understood".format(log_level))
    else:
        level = int(log_level)

    logger = logging.getLogger("pyfstat")
    logger.setLevel(level)

    common_formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(name)s %(levelname)-8s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )

    if not any([type(h) == logging.StreamHandler for h in logger.handlers]):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(common_formatter)
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    if (
        label
        and outdir
        and (not any([type(h) == logging.FileHandler for h in logger.handlers]))
    ):
        os.makedirs(outdir, exist_ok=True)
        log_file = os.path.join(outdir, f"{label}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(common_formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(level)

    logger.info(f"Running PyFstat version {get_version_string()}")

    return logger
