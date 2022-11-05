import logging

import lalpulsar
import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaincc

logger = logging.getLogger(__name__)


def get_BSGL_setup(
    numDetectors: int,
    Fstar0sc: float,
    numSegments: int = 1,
    oLGX: ArrayLike = None,
    useLogCorrection: bool = True,
) -> lalpulsar.BSGLSetup:
    """
    Get a ``lalpulsar.BSGLSetup`` setup struct.

    This encapsulates all parameters defining a line-robust statistic,
    ready to use in ``lalpulsar.ComputeBSGL()``.

    Parameters
    -------
    numDetectors:
        Number of detectors.
    Fstar0sc:
        The (semi-coherent) prior transition-scale parameter.
    numSegments:
        Number of segments in a semi-coherent search (=1 for coherent search).
    oLGX:
        Prior per-detector line odds.
        If ``None``, interpreted as `oLGX=1/numDetectors` for all X.
        Else, must be either of `len(oLGX)==numDetectors`
        or `len(oLGX)==lalpulsar.PULSAR_MAX_DETECTORS`
        with all `X>numDetectors` equal to 0.
    useLogCorrection:
        Include log-term correction (slower) or not (faster, less accurate)?
    Returns
    -------
    BSGLSetup:
        The initialised setup struct.
    """
    if numDetectors < 2:
        raise ValueError("numDetectors must be at least 2 for BSGL.")
    if numDetectors > lalpulsar.PULSAR_MAX_DETECTORS:
        raise ValueError(
            f"numDetectors={numDetectors} >"
            f" lalpulsar.PULSAR_MAX_DETECTORS={lalpulsar.PULSAR_MAX_DETECTORS}."
        )
    if oLGX is None:
        # default agnostic oLGX and always with log correction term.
        oLGX = np.zeros(lalpulsar.PULSAR_MAX_DETECTORS)
        oLGX[:numDetectors] = 1.0 / numDetectors
    else:
        if not len(oLGX) in [numDetectors, lalpulsar.PULSAR_MAX_DETECTORS]:
            raise ValueError(
                f"len(oLGX) must be either numDetectors={numDetectors} or lalpulsar.PULSAR_MAX_DETECTORS={lalpulsar.PULSAR_MAX_DETECTORS}."
            )
        if len(oLGX) > numDetectors and np.any(oLGX[numDetectors:] != 0):
            raise ValueError(
                f"oLGX={oLGX} has length"
                f" lalpulsar.PULSAR_MAX_DETECTORS={lalpulsar.PULSAR_MAX_DETECTORS},"
                " but it must not have non-0 entries beyond"
                f" numDetectors={numDetectors}."
            )
        if len(oLGX) < lalpulsar.PULSAR_MAX_DETECTORS:
            oLGX = np.concatenate(
                (oLGX, np.zeros(lalpulsar.PULSAR_MAX_DETECTORS - len(oLGX)))
            )
    logger.info(f"Setting up BSGL statistic with Fstar0sc={Fstar0sc} and oLGX={oLGX}.")
    BSGLSetup = lalpulsar.CreateBSGLSetup(
        numDetectors=numDetectors,
        Fstar0sc=Fstar0sc,
        oLGX=oLGX,
        useLogCorrection=useLogCorrection,
        numSegments=numSegments,
    )
    return BSGLSetup


def compute_Fstar0sc_from_p_val_threshold(
    p_val_threshold: float = 1e-6,
    numSegments: int = 1,
    maxFstar0: float = 1000,
    numSteps: int = 10000,
) -> float:
    """
    Computes a BSGL tuning parameter Fstar0sc from a p-value threshold.

    See https://arxiv.org/abs/1311.5738 for definitions.

    This uses a simple numerical inversion of the gammaincc function,
    returning the Fstar0sc value closest to reproducing the requested p-value.

    Parameters
    -------
    p_val_threshold:
        Threshold on p-value.
    numSegments:
        Number of segments in a semi-coherent search (=1 for coherent search).
    maxFstar0:
        Maximum value to be tried in the numerical inversion.
    numSteps:
        Number of steps to try in the numerical inversion.
    Returns
    -------
    Fstar0sc:
        The (semi-coherent) prior transition-scale parameter.
    """

    Fstar0s = np.linspace(0, maxFstar0, numSteps)
    p_vals = gammaincc(2 * numSegments, Fstar0s)
    Fstar0sc = Fstar0s[np.argmin(np.abs(p_vals - p_val_threshold))]
    if Fstar0sc == Fstar0s[-1]:
        raise ValueError("Max Fstar0 exceeded")
    logger.info(f"Using Fstar0sc={Fstar0sc} for BSGL statistic.")
    return Fstar0sc
