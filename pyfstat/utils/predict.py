import logging
import os

import numpy as np

from .cli import run_commandline
from .converting import parse_list_of_numbers
from .ephemeris import get_ephemeris_files
from .io import read_par

logger = logging.getLogger(__name__)


def predict_fstat(
    h0=None,
    cosi=None,
    psi=None,
    Alpha=None,
    Delta=None,
    F0=None,
    sftfilepattern=None,
    timestampsFiles=None,
    minStartTime=None,
    duration=None,
    IFOs=None,
    assumeSqrtSX=None,
    tempory_filename="fs.tmp",
    earth_ephem=None,
    sun_ephem=None,
    transientWindowType="none",
    transientStartTime=None,
    transientTau=None,
):
    """Wrapper to PredictFstat executable for predicting expected F-stat values.

    Parameters
    ----------
    h0, cosi, psi, Alpha, Delta : float
        Signal parameters, see `lalpulsar_PredictFstat --help` for more info.
    F0: float or None
        Signal frequency.
        Only needed for noise floor estimation when given `sftfilepattern`
        but `assumeSqrtSX=None`.
        The actual F-stat prediction is frequency-independent.
    sftfilepattern : str or None
        Pattern matching the SFT files to use for inferring
        detectors, timestamps and/or estimating the noise floor.
    timestampsFiles : str or None
        Comma-separated list of per-detector files containing timestamps to use.
        Only used if `sftfilepattern=None`.
    minStartTime, duration : int or None
        If `sftfilepattern` given: used as optional constraints.
        If `timestampsFiles` given: ignored.
        If neither given: used as the interval for prediction.
    IFOs : str or None
        Comma-separated list of detectors.
        Required if `sftfilepattern=None`,
        ignored otherwise.
    assumeSqrtSX : float or str
        Assume stationary per-detector noise-floor instead of estimating from SFTs.
        Single float or str value: use same for all IFOs.
        Comma-separated string: must match `len(IFOs)`
        and/or the data in `sftfilepattern`.
        Detectors will be paired to list elements following alphabetical order.
        Required if `sftfilepattern=None`,
        optional otherwise..
    tempory_filename : str
        Temporary file used for `PredictFstat` output,
        will be deleted at the end.
    earth_ephem, sun_ephem : str or None
        Ephemerides files, defaults will be used if `None`.
    transientWindowType: str
        Optional parameter for transient signals,
        see `lalpulsar_PredictFstat --help`.
        Default of `none` means a classical Continuous Wave signal.
    transientStartTime, transientTau: int or None
        Optional parameters for transient signals,
        see `lalpulsar_PredictFstat --help`.

    Returns
    -------
    twoF_expected, twoF_sigma : float
        The expectation and standard deviation of 2F.
    """

    cl_pfs = []
    cl_pfs.append("lalpulsar_PredictFstat")

    pars = {"h0": h0, "cosi": cosi, "psi": psi, "Alpha": Alpha, "Delta": Delta}
    cl_pfs.extend([f"--{key}={val}" for key, val in pars.items() if val is not None])

    if sftfilepattern is None:
        if IFOs is None or assumeSqrtSX is None:
            raise ValueError("Without sftfilepattern, need IFOs and assumeSqrtSX!")
        cl_pfs.append("--IFOs={}".format(IFOs))
        if timestampsFiles is None:
            if minStartTime is None or duration is None:
                raise ValueError(
                    "Without sftfilepattern, need timestampsFiles or [minStartTime,duration]!"
                )
            else:
                cl_pfs.append("--minStartTime={}".format(minStartTime))
                cl_pfs.append("--duration={}".format(duration))
        else:
            cl_pfs.append("--timestampsFiles={}".format(timestampsFiles))
    else:
        cl_pfs.append("--DataFiles='{}'".format(sftfilepattern))
        if minStartTime is not None:
            cl_pfs.append("--minStartTime={}".format(minStartTime))
            if duration is not None:
                cl_pfs.append("--maxStartTime={}".format(minStartTime + duration))
        if assumeSqrtSX is None:
            if F0 is None:
                raise ValueError(
                    "With sftfilepattern but without assumeSqrtSX,"
                    " we need F0 to estimate noise floor."
                )
            cl_pfs.append("--Freq={}".format(F0))
    if assumeSqrtSX is not None:
        if np.any([s <= 0 for s in parse_list_of_numbers(assumeSqrtSX)]):
            raise ValueError("assumeSqrtSX must be >0!")
        cl_pfs.append("--assumeSqrtSX={}".format(assumeSqrtSX))

    cl_pfs.append("--outputFstat={}".format(tempory_filename))

    earth_ephem_default, sun_ephem_default = get_ephemeris_files()
    if earth_ephem is None:
        earth_ephem = earth_ephem_default
    if sun_ephem is None:
        sun_ephem = sun_ephem_default
    cl_pfs.append("--ephemEarth='{}'".format(earth_ephem))
    cl_pfs.append("--ephemSun='{}'".format(sun_ephem))

    if transientWindowType != "none":
        cl_pfs.append("--transientWindowType='{}'".format(transientWindowType))
        cl_pfs.append("--transientStartTime='{}'".format(transientStartTime))
        cl_pfs.append("--transientTau='{}'".format(transientTau))

    cl_pfs = " ".join(cl_pfs)
    run_commandline(cl_pfs)
    d = read_par(filename=tempory_filename)
    twoF_expected = float(d["twoF_expected"])
    twoF_sigma = float(d["twoF_sigma"])
    os.remove(tempory_filename)
    return twoF_expected, twoF_sigma


def get_predict_fstat_parameters_from_dict(signal_parameters, transientWindowType=None):
    """Extract a subset of parameters as needed for predicting F-stats.
    Given a dictionary with arbitrary signal parameters,
    this extracts only those ones required by `helper_functions.predict_fstat()`:
    Freq, Alpha, Delta, h0, cosi, psi.
    Also preserves transient parameters, if included in the input dict.

    Parameters
    ----------
    signal_parameters: dict
        Dictionary containing at least those signal parameters required by
        helper_functions.predict_fstat.
        This dictionary's keys must follow
        the PyFstat convention (e.g. F0 instead of Freq).
    transientWindowType: str
        Transient window type to store in the output dict.
        Currently required because the typical input dicts
        produced by various PyFstat functions
        tend not to store this property.
        If there is a key with this name already, its value will be overwritten.
    Returns
    -------
    predict_fstat_params: dict
        The dictionary of selected parameters.
    """
    required_keys = ["F0", "Alpha", "Delta", "h0", "cosi", "psi"]
    transient_keys = {
        "transientWindowType": "transientWindowType",
        "transient_tstart": "transientStartTime",
        "transient_duration": "transientTau",
    }
    predict_fstat_params = {key: signal_parameters[key] for key in required_keys}
    for key, val in transient_keys.items():
        if key in signal_parameters:
            predict_fstat_params[val] = signal_parameters[key]
        elif val in signal_parameters:
            predict_fstat_params[val] = signal_parameters[val]
    if transientWindowType is not None:
        predict_fstat_params["transientWindowType"] = transientWindowType
    return predict_fstat_params
