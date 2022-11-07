import logging
import os
import subprocess
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def run_commandline(
    cl: str, raise_error: bool = True, return_output: bool = False
) -> Optional[subprocess.CompletedProcess]:
    """Run a string command as a subprocess.

    Parameters
    ----------
    cl:
        Command to run
    raise_error:
        If True, raise an error if the subprocess fails.
        If False, just log the error, continue, and return ``None``.
    return_output:
        If True, return the ``subprocess.CompletedProcess`` object.
        If False, return ``None``.

    Returns
    ----------
    out: subprocess.CompletedProcess or None
        The ```subprocess.CompletedProcess`` of the subprocess
        if ``return_output=True``. ``None`` if ``return_output=False``
        or on  failed execution if ``raise_error=False``.
    """

    logger.info("Now executing: " + cl)
    if "|" in cl:
        logger.warning(
            "Pipe ('|') found in commandline, errors may not be  properly caught!"
        )
    try:
        completed_process = subprocess.run(
            cl,
            check=True,
            shell=True,
            capture_output=True,
            text=True,
        )
        msg = completed_process.stdout
        if msg:
            [logger.info(line) for line in msg.splitlines()]
        msg = completed_process.stderr
        if msg:
            [logger.error(line) for line in msg.splitlines()]
        if return_output:
            return completed_process
    except subprocess.CalledProcessError as e:
        msg = getattr(e, "output", None)
        if msg:
            [logger.info(line) for line in msg.splitlines()]
        logger.error(f"Execution failed: {e}")
        msg = getattr(e, "stderr", None)
        if msg:
            [logger.error(line) for line in msg.splitlines()]
        if raise_error:
            raise

    return None


def match_commandlines(cl1, cl2, be_strict_about_full_executable_path=False):
    """Check if two commandline strings match element-by-element, regardless of order.

    Parameters
    ----------
    cl1, cl2: str
        Commandline strings of `executable --key1=val1 --key2=val2` etc format.
    be_strict_about_full_executable_path: bool
        If False (default), only checks the basename of the executable.
        If True, requires its full path to match.

    Returns
    -------
    match: bool
        Whether the executable and all `key=val` pairs of the two strings matched.
    """
    cl1s = cl1.split(" ")
    cl2s = cl2.split(" ")
    # first item will be the executable name
    # by default be generous here and do not worry about full paths
    if not be_strict_about_full_executable_path:
        cl1s[0] = os.path.basename(cl1s[0])
        cl2s[0] = os.path.basename(cl2s[0])
    unmatched = np.setxor1d(cl1s, cl2s)
    return len(unmatched) == 0
