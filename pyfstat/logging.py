"""
PyFstat's logging implementation.

PyFstat main logger is called `pyfstat` and be accessed via::

    import logging
    logger = logging.getLoger('pyfstat')

ELI5 of logging: For all our purposes, there are *logger* objects
and *handler* objects. Loggers are the ones in charge of logging,
hence you call them to emit a logging message with a specific logging
level (e.g. ``logger.info``); handlers are in charge of redirecting that
message to a specific place (e.g. a file or your terminal, which is
usually referred to as a *stream*).

The default behaviour is to attach a ``logging.StreamHandler`` to
the *pyfstat* printing out to ``sys.stdout`` upon importing the package.
If, for any reason, ``logging`` cannot access ``sys.stdout`` at import time,
the exception is reported via ``print`` and no handlers are attached
(i.e. the logger won't print to ``sys.stoud``).

The user can modify the logger's behaviour at run-time using ``set_up_logger``.
This function attaches extra ``logging.StreamHandler`` and ``logging.FileHandler``
handlers to the logger, allowing to redirect loggin messages to either a different
stream or a specific output file specified using the ``outdir, label`` variables
(with the same format as in the rest of the package).

Finally, logging can be disable at run-time by manually configuring the *pyfstat*
logger. For example, the following block of code will suppress logging messages
below ``WARNING``:::

    import logging
    logging.getLoger('pyfstat').setLevel(logging.WARNING)
"""

import logging
import os
import sys
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    import io


def set_up_logger(
    outdir: Optional[str] = None,
    label: Optional[str] = "pyfstat",
    log_level: Literal["ERROR", "WARNING", "INFO", "DEBUG"] = "INFO",
    streams: Optional["io.TextIOWrapper"] = None,
    append: bool = True,
) -> logging.Logger:
    """Add file and stream handlers to the `pyfstat` logger.

    Parameters
    ----------
    outdir:
        Path to outdir directory. If `None`, no file handler will be added.
    label:
        Label for this instance of the logger.
        This is consistent with the rest of the package: ``label'' referes
        to the string prepended at every file produced by a script.
        Required, in conjunction with `outdir`, to add a file handler.
    log_level:
        Level of logging. This level is imposed the logger itself and
        *every single handler* attached to it.
    streams:
        Stream to which logging messages will be passed using a
        StreamHandler object. If `None`, a handler to `sys.stdout`
        will be attached unless it already exists.
        Other common streams include e.g. `sys.stderr`.
    append:
        If True, removes all handlers from the `pyfstat` logger.

    Returns
    -------
    obj:`logging.Logger`
        Instance of the Logger class.

    """
    logger = logging.getLogger("pyfstat")
    logger.setLevel(log_level)

    if not append:
        while logger.hasHandlers():
            logger.removeHandler(logger.hadlers[0])
        stream_names = []
        file_names = []
    else:
        for handler in logger.handlers:
            handler.setLevel(log_level)
        stream_names = [
            handler.stream.name
            for handler in logger.handlers
            if type(handler) == logging.StreamHandler
        ]
        file_names = [
            handler.fileBasename
            for handler in logger.handlers
            if type(handler) == logging.FileHandler
        ]

    streams = streams or ([sys.stdout] if sys.stdout.name not in stream_names else [])

    common_formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(name)s %(levelname)-8s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",  # intended to match LALSuite's format
    )

    for stream in streams:
        if stream.name in stream_names:
            continue
        stream_handler = logging.StreamHandler(stream)
        stream_handler.setFormatter(common_formatter)
        stream_handler.setLevel(log_level)
        logger.addHandler(stream_handler)

    if label and outdir:
        os.makedirs(outdir, exist_ok=True)
        log_file = os.path.join(outdir, f"{label}.log")

        if log_file not in file_names:

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(common_formatter)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)

    return logger
