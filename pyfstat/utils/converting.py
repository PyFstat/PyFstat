from datetime import datetime, timezone

import lal
import numpy as np


def get_dictionary_from_lines(lines, comments, raise_error):
    """Return a dictionary of key=val pairs for each line in a list.

    Parameters
    ----------
    lines: list of strings
        The list of lines to parse.
    comments: str or list of strings
        Characters denoting that a row is a comment.
    raise_error: bool
        If True, raise an error for lines which are not comments,
        but cannot be read.
        Note that CFSv2 "loudest" files contain complex numbers which fill raise
        an error unless this is set to False.

    Returns
    -------
    d: dict
        The `key=val` pairs as a dictionary.

    """
    d = {}
    for line in lines:
        if line[0] not in comments and len(line.split("=")) == 2:
            try:
                key, val = line.rstrip("\n").split("=")
                key = key.strip()
                val = val.strip()
                if (val[0] in ["'", '"']) and (val[-1] in ["'", '"']):
                    d[key] = val.lstrip('"').lstrip("'").rstrip('"').rstrip("'")
                else:
                    try:
                        d[key] = np.float64(eval(val.rstrip("; ")))
                    except NameError:
                        d[key] = val.rstrip("; ")
                    # FIXME: learn how to deal with complex numbers
            except SyntaxError:
                if raise_error:
                    raise IOError("Line {} not understood".format(line))
                pass
    return d


def parse_list_of_numbers(val):
    """Convert a number, list of numbers or comma-separated str into a list of numbers.

    This is useful e.g. for `sqrtSX` inputs where the user can be expected
    to try either type of input.

    Parameters
    -------
    val: float, list or str
        The input to be parsed.

    Returns
    -------
    out: list
        The parsed list.
    """
    try:
        out = list(
            map(float, val.split(",") if hasattr(val, "split") else np.atleast_1d(val))
        )
    except Exception as e:
        raise ValueError(
            f"Could not parse '{val}' as a number or list of numbers."
            f" Got exception: {e}."
        )
    return out


def gps_to_datestr_utc(gps):
    """Convert an integer count of GPS seconds to a UTC date-time string.

    This uses the locale's default string formatting as per `datetime.strftime()`.
    It is intended just for informing the user and may not be as reliable
    in all situations as `lal[apps]_tconvert`.
    If you want to do any postprocessing of the date-time string,
    for safety you should probably call that commandline tool.

    Parameters
    -------
    gps: int
        Integer seconds since GPS seconds.

    Returns
    -------
    dtstr: str
        A string representation of date-time in UTC and locale format.

    """
    utc = lal.GPSToUTC(gps)
    dt = datetime(
        year=utc[0],
        month=utc[1],
        day=utc[2],
        hour=utc[3],
        minute=utc[4],
        second=utc[5],
        microsecond=0,
        tzinfo=timezone.utc,
    )
    return dt.strftime("%c %Z")


def convert_h0_cosi_to_aPlus_aCross(h0, cosi):
    """
    Converts amplitude parameters from a pair of `(h0,cosi)` to a pair of `(aPlus,aCross)`.

    See e.g. Eq. (30) of https://dcc.ligo.org/T0900149-v6/public .

    If both inputs are single numbers,
    both outputs will be as well.
    If at least one input is a list or np.array,
    both outputs will be np.arrays.

    Parameters
    -------
    h0: float, list or np.array
        Nominal GW amplitude.
    cosi: float, list or np.array
        Cosine of the source inclination w.r.t. line of sight.

    Returns
    -------
    aPlus: float or np.array
        Plus polarization amplitude.
    aCross: float or np.array
        Cross polarization amplitude.
    """

    h0 = np.atleast_1d(h0)
    cosi = np.atleast_1d(cosi)

    if np.any(h0 < 0):
        raise ValueError("not valid for h0<0")
    if np.any(np.abs(cosi) > 1):
        raise ValueError("not valid for abs(cosi)>1")

    aPlus = 0.5 * h0 * (1.0 + cosi**2)
    aCross = h0 * cosi

    if aPlus.size == 1:
        aPlus = aPlus[0]
    if cosi.size == 1:
        aCross = aCross[0]
    return aPlus, aCross


def convert_aPlus_aCross_to_h0_cosi(aPlus, aCross):
    """
    Converts amplitude parameters from a pair of `(aPlus,aCross)` to a pair of `(h0,cosi)`.

    Inverse to ``convert_h0_cosi_to_aPlus_aCross()``.

    Conversion in this direction is only well-defined if `aPlus >= abs(aCross) >= 0`,
    as expected for GWs from neutron stars at twice the spin frequency,
    but not necessarily in all other CW emission scenarios.
    See e.g. Eq. (32) of https://dcc.ligo.org/T0900149-v6/public .

    If both inputs are single numbers,
    both outputs will be as well.
    If at least one input is a list or np.array,
    both outputs will be np.arrays.

    Parameters
    -------
    aPlus: float, list or np.array
        Plus polarization amplitude
        (must be `>= abs(aCross)` and `>= 0`).
    aCross: float, list or np.array
        Cross polarization amplitude.

    Returns
    -------
    h0: float or np.array
        Nominal GW amplitude.
    cosi: float or np.array
        Cosine of the source inclination w.r.t. line of sight.
    """

    aPlus = np.atleast_1d(aPlus)
    aCross = np.atleast_1d(aCross)

    if np.any(aPlus < 0):
        raise ValueError("not valid for aPlus<0")
    if np.any(np.abs(aCross) > aPlus):
        raise ValueError("not valid for abs(aCross)>aPlus")

    h0 = aPlus + np.sqrt(aPlus**2 - aCross**2)

    # vectorized version of if-else on h0>0
    cosi = np.divide(aCross, h0, out=np.zeros_like(aCross), where=(h0 > 0))

    if h0.size == 1:
        h0 = h0[0]
    if cosi.size == 1:
        cosi = cosi[0]
    return h0, cosi
