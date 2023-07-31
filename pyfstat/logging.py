"""
PyFstat's logging implementation.

PyFstat main logger is called `pyfstat` and can be accessed via::

    import logging
    logger = logging.getLoger('pyfstat')

Basics of logging: For all our purposes, there are *logger* objects
and *handler* objects. Loggers are the ones in charge of logging,
hence you call them to emit a logging message with a specific logging
level (e.g. ``logger.info``); handlers are in charge of redirecting that
message to a specific place (e.g. a file or your terminal, which is
usually referred to as a *stream*).

The default behaviour upon importing ``pyfstat`` is to attach a ``logging.StreamHandler`` to
the *pyfstat* logger, printing out to ``sys.stdout``.
This is only done if the root logger has no handlers attached yet;
if it does have at least one handler already,
then we inherit those and do not add any extra handlers by default.
If, for any reason, ``logging`` cannot access ``sys.stdout`` at import time,
the exception is reported via ``print`` and no handlers are attached
(i.e. the logger won't print to ``sys.stdout``).

The user can modify the logger's behaviour at run-time using ``set_up_logger``.
This function attaches extra ``logging.StreamHandler`` and ``logging.FileHandler``
handlers to the logger, allowing to redirect logging messages to a different
stream or a specific output file specified using the ``outdir, label`` variables
(with the same format as in the rest of the package).

Finally, logging can be disabled, or the level changed, at run-time by
manually configuring the *pyfstat* logger. For example, the following block of code
will suppress logging messages below ``WARNING``::

    import logging
    logging.getLogger('pyfstat').setLevel(logging.WARNING)
"""

import logging
import os
import sys
from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    import io


def _get_default_logger() -> logging.Logger:
    root_logger = logging.getLogger()
    return root_logger if root_logger.handlers else set_up_logger()


def set_up_logger(
    outdir: Optional[str] = None,
    label: Optional[str] = "pyfstat",
    log_level: str = "INFO",  # FIXME: Requires Python 3.8 Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = "INFO",
    streams: Optional[Iterable["io.TextIOWrapper"]] = (sys.stdout,),
    append: bool = True,
) -> logging.Logger:
    """Add file and stream handlers to the `pyfstat` logger.

    Handler names generated from ``streams`` and ``outdir, label``
    must be unique and no duplicated handler will be attached by
    this function.

    Parameters
    ----------
    outdir:
        Path to outdir directory. If ``None``, no file handler will be added.
    label:
        Label for the file output handler, i.e.
        the log file will be called `label.log`.
        Required, in conjunction with ``outdir``, to add a file handler.
        Ignored otherwise.
    log_level:
        Level of logging. This level is imposed on the logger itself and
        *every single handler* attached to it.
    streams:
        Stream to which logging messages will be passed using a
        StreamHandler object. By default, log to ``sys.stdout``.
        Other common streams include e.g. ``sys.stderr``.
    append:
        If ``False``, removes all handlers from the `pyfstat` logger
        before adding new ones. This removal is not propagated to
        handlers on the `root` logger.

    Returns
    -------
    logger: logging.Logger
        Configured instance of the ``logging.Logger`` class.

    """
    logger = logging.getLogger("pyfstat")
    logger.setLevel(log_level)

    if not append:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    else:
        for handler in logger.handlers:
            handler.setLevel(log_level)

    stream_names = [
        handler.stream.name
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler)
    ]
    file_names = [
        handler.baseFilename
        for handler in logger.handlers
        if isinstance(handler, logging.FileHandler)
    ]

    common_formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(name)s %(levelname)-8s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",  # intended to match LALSuite's format
    )

    for stream in streams or []:
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
