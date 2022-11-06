import logging
from typing import Sequence, Tuple, Union

import lalpulsar
import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaincc

logger = logging.getLogger(__name__)

"""Supported detection statistics and synonyms for them
The key is the canonical name to be used internally and in outputs;
the listed synonyms are allowed inputs
(e.g. to be parsed by :func:`~pyfstat.utils.detstats.parse_detstats`).
Lookup can be performed with :func:`~pyfstat.utils.detstats.get_canonical_detstat_name`.
"""
detstat_synonyms = {
    "twoF": ["2F"],
    "twoFX": ["2FX"],
    "maxTwoF": ["max2F"],
    "lnBtSG": ["BtSG", "logBtSG", "logBstat"],
    "log10BSGL": ["BSGL", "logBSGL"],
}


def parse_detstats(detstats: Sequence[Union[str, dict]]) -> Tuple[list, dict]:
    """Parse a requested list of detection statistics and, if required, their parameters.

    NOTE: additional statistics required by one of the requested ones may get added,
    e.g. BSGL automatically adds twoF and twoFX.

    Parameters
    ----------
    detstats:
        The list of statistics,
        optionally including parameters.
        Each entry in the input can be either a single string
        (currently supported:
        twoF, twoFX, maxTwoF, BtSG)
        or a single-item dictionary where the key is the statistic name
        (currently supported: BSGL)
        and the value is a dictionary of keyword arguments
        further defining the statistic.
        Example:
        ``detstats=["twoF", {"BSGL": {"Fstar0sc": 15}}]``
        If only parameter-free statistics are included,
        a comma-separated string is also allowed.

    Returns
    -------
    detstats_parsed:
        A list of detection statistics names only.
    params:
        A dictionary of statistics with parameters,
        with each key matching a `detstats_parsed` entry
        and each value a dictionary of parameters.
    """
    if not isinstance(detstats, Sequence):
        raise ValueError("`detstats` must be a list or other Sequence type.")
    if isinstance(detstats, str):
        detstats = detstats.split(",")
    detstaterr = "Entries of `detstats` list must be keys or single-item dicts."
    detstats_parsed = []
    params = {}
    # FIXME: if we ever get more than 1 stat with parameters,
    # we should implement lookups for their names,
    # required stats and required parameters.
    for stat in detstats:
        if isinstance(stat, str):
            if get_canonical_detstat_name(stat) == get_canonical_detstat_name("BSGL"):
                raise ValueError(
                    "For BSGL statistic, please pass extra parameters,"
                    " at least `'BSGL': {'Fstar0sc': 15}`."
                    " See pyfstat.utils.get_BSGL_setup."
                )
            detstats_parsed.append(get_canonical_detstat_name(stat))
        elif isinstance(stat, dict):
            if len(stat) > 1:
                raise ValueError(detstaterr)
            statname = list(stat.keys())[0]
            if not get_canonical_detstat_name(statname) == get_canonical_detstat_name(
                "BSGL"
            ):
                raise ValueError(
                    f"{statname} does not require parameters, please do not pass a dict for it."
                )
            detstats_parsed.append(get_canonical_detstat_name(statname))
            if detstats_parsed[-1] == get_canonical_detstat_name("BSGL"):
                params[detstats_parsed[-1]] = list(stat.values())[0]
        else:
            raise ValueError(detstaterr)
    if get_canonical_detstat_name("BSGL") in detstats_parsed:
        for required_stat in ["twoF", "twoFX"]:
            if required_stat not in detstats_parsed:
                detstats_parsed.append(required_stat)
        if "Fstar0sc" not in params[get_canonical_detstat_name("BSGL")].keys():
            raise ValueError("BSGL requires the `Fstar0sc` parameter.")
    return detstats_parsed, params


def get_canonical_detstat_name(statname: str) -> str:
    """
    Normalize detection statistic names to our canonical choices.

    Used to convert abbreviated detection statistic input keys into more explicit
    keys as used internally and for output.
    For example, we specify which base the logarithm of a Bayes factor takes.

    Parameters
    ----------
    statname:
        The detection statistic name to make canonical.

    Returns
    -------
    canonical_name: str
        The canonical name,
        if input `statname` can be matched.
    """
    if statname in detstat_synonyms.keys():
        return statname
    else:
        for key, val in detstat_synonyms.items():
            if statname in val:
                return key
        raise ValueError(f"Unsupported detection statistic: {statname}")


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
    ----------
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
    BSGLSetup: lalpulsar.BSGLSetup
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
    ----------
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
    Fstar0sc: float
        The (semi-coherent) prior transition-scale parameter.
    """
    if p_val_threshold < 0 or p_val_threshold > 1:
        raise ValueError("p_val_threshold must be in [0,1]")
    if maxFstar0 < 0:
        raise ValueError("maxFstar0 must be >=0")
    Fstar0s = np.linspace(0, maxFstar0, numSteps)
    p_vals = gammaincc(2 * numSegments, Fstar0s)
    Fstar0sc = Fstar0s[np.argmin(np.abs(p_vals - p_val_threshold))]
    if Fstar0sc == Fstar0s[-1]:
        raise RuntimeError("Max Fstar0 exceeded")
    logger.info(f"Using Fstar0sc={Fstar0sc} for BSGL statistic.")
    return Fstar0sc
