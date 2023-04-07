import logging
import os

import lal
import lalpulsar
import numpy as np

from .cli import run_commandline

logger = logging.getLogger(__name__)


def get_covering_band(
    tref,
    tstart,
    tend,
    F0,
    F1,
    F2,
    F0band=0.0,
    F1band=0.0,
    F2band=0.0,
    maxOrbitAsini=0.0,
    minOrbitPeriod=0.0,
    maxOrbitEcc=0.0,
):
    """Get the covering band for CW signals for given time and parameter ranges.

    This uses the lalpulsar function `XLALCWSignalCoveringBand()`,
    accounting for
    the spin evolution of the signals within the given [F0,F1,F2] ranges,
    the maximum possible Dopper modulation due to detector motion
    (i.e. for the worst-case sky locations),
    and for worst-case binary orbital motion.

    Parameters
    ----------
    tref: int
        Reference time (in GPS seconds) for the signal parameters.
    tstart: int
        Start time (in GPS seconds) for the signal evolution to consider.
    tend: int
        End time (in GPS seconds) for the signal evolution to consider.
    F0, F1, F1: float
        Minimum frequency and spin-down of signals to be covered.
    F0band, F1band, F1band: float
        Ranges of frequency and spin-down of signals to be covered.
    maxOrbitAsini: float
        Largest orbital projected semi-major axis to be covered.
    minOrbitPeriod: float
        Shortest orbital period to be covered.
    maxOrbitEcc: float
        Highest orbital eccentricity to be covered.

    Returns
    -------
    minCoverFreq, maxCoverFreq: float
        Estimates of the minimum and maximum frequencies of the signals
        from the given parameter ranges over the `[tstart,tend]` duration.
    """
    tref = lal.LIGOTimeGPS(tref)
    tstart = lal.LIGOTimeGPS(tstart)
    tend = lal.LIGOTimeGPS(tend)
    psr = lalpulsar.PulsarSpinRange()
    psr.fkdot[0] = F0
    psr.fkdot[1] = F1
    psr.fkdot[2] = F2
    psr.fkdotBand[0] = F0band
    psr.fkdotBand[1] = F1band
    psr.fkdotBand[2] = F2band
    psr.refTime = tref
    minCoverFreq, maxCoverFreq = lalpulsar.CWSignalCoveringBand(
        tstart, tend, psr, maxOrbitAsini, minOrbitPeriod, maxOrbitEcc
    )
    if (
        np.isnan(minCoverFreq)
        or np.isnan(maxCoverFreq)
        or minCoverFreq <= 0.0
        or maxCoverFreq <= 0.0
        or maxCoverFreq < minCoverFreq
    ):
        raise RuntimeError(
            "Got invalid pair minCoverFreq={}, maxCoverFreq={} from"
            " lalpulsar.CWSignalCoveringBand.".format(minCoverFreq, maxCoverFreq)
        )
    return minCoverFreq, maxCoverFreq


def generate_loudest_file(
    max_params,
    tref,
    outdir,
    label,
    sftfilepattern,
    minStartTime=None,
    maxStartTime=None,
    transientWindowType=None,
    earth_ephem=None,
    sun_ephem=None,
):
    """Use ComputeFstatistic_v2 executable to produce a .loudest file.

    Parameters
    -------
    max_params: dict
        Dictionary of a single parameter-space point.
        This needs to already have been translated to lal conventions
        and must NOT include detection statistic entries!
    tref: int
        Reference time of the max_params.
    outdir: str
        Directory to place the .loudest file in.
    label: str
        Search name bit to be used in the output filename.
    sftfilepattern: str
        Pattern to match SFTs using wildcards (`*?`) and ranges [0-9];
        mutiple patterns can be given separated by colons.
    minStartTime, maxStartTime: int or None
        GPS seconds of the start time and end time;
        default: use al available data.
    transientWindowType: str or None
        optional: transient window type,
        needs to go with t0 and tau parameters inside max_params.
    earth_ephem: str or None
        optional: user-set Earth ephemeris file
    sun_ephem: str or None
        optional: user-set Sun ephemeris file

    Returns
    -------
    loudest_file: str
        The filename of the CFSv2 output file.
    """
    logger.info("Running CFSv2 to get .loudest file")
    if np.any([key in max_params for key in ["delta_F0", "delta_F1", "tglitch"]]):
        raise RuntimeError("CFSv2 --outputLoudest cannot deal with glitch parameters.")
    if transientWindowType:
        logger.warning(
            "CFSv2 --outputLoudest always reports the maximum of the"
            " standard CW 2F-statistic, not the transient max2F."
        )

    loudest_file = os.path.join(outdir, label + ".loudest")
    cmd = "lalpulsar_ComputeFstatistic_v2"
    CFSv2_params = {
        "DataFiles": f'"{sftfilepattern}"',
        "outputLoudest": loudest_file,
        "refTime": tref,
    }
    CFSv2_params.update(max_params)
    opt_params = {
        "minStartTime": minStartTime,
        "maxStartTime": maxStartTime,
        "transient-WindowType": transientWindowType,
        "ephemEarth": earth_ephem,
        "ephemSun": sun_ephem,
    }
    CFSv2_params.update({key: val for key, val in opt_params.items() if val})
    cmd += " " + " ".join([f"--{key}={val}" for key, val in CFSv2_params.items()])

    run_commandline(cmd, return_output=False)
    return loudest_file
