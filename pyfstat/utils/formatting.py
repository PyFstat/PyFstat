import lalpulsar
import numpy as np


def round_to_n(x, n):
    """Simple rounding function for getting a fixed number of digits.

    Parameters
    ----------
    x: float
        The number to round.
    n: int
        The number of digits to round to
        (before plus after the decimal separator).

    Returns
    ----------
    rounded: float
        The rounded number.
    """
    if not x:
        return 0
    power = -int(np.floor(np.log10(abs(x)))) + (n - 1)
    factor = 10**power
    return round(x * factor) / factor


def texify_float(x, d=2):
    """Format float numbers nicely for LaTeX output, including rounding.

    Numbers with absolute values between 0.01 and 100 will be returned
    in plain float format,
    while smaller or larger numbers will be returned in powers-of-ten notation.

    Parameters
    ----------
    x: float
        The number to round and format.
    n: int
        The number of digits to round to
        (before plus after the decimal separator).

    Returns
    ----------
    formatted: str
        The formatted string.
    """
    if x == 0:
        return 0
    if isinstance(x, str):
        return x
    x = round_to_n(x, d)
    if 0.01 < abs(x) < 100:
        return str(x)
    else:
        power = int(np.floor(np.log10(abs(x))))
        stem = np.round(x / 10**power, d)
        if d == 1:
            stem = int(stem)
        return r"${}{{\times}}10^{{{}}}$".format(stem, power)


def get_doppler_params_output_format(keys, fmt_str="%.16g"):
    """Set a canonical output precision for frequency evolution parameters.

    The default format (`%.16g`) is the same as
    the `write_FstatCandidate_to_fp()` function of
    the `ComputeFstatistic_v2` executable.

    This assigns that format to each parameter name in `keys`
    which matches a hardcoded list of known standard 'Doppler' parameters,
    and ignores any others.

    Parameters
    -------
    keys: dict
        The parameter keys for which to select formats.
    fmt_str: str
        fprintf-style format specifier for a single value.

    Returns
    -------
    fmt_dict: dict
        A dictionary assigning the default format to each parameter key
        from the hardcoded list of standard 'Doppler' parameters.
    """
    doppler_keys = [f"F{k}" for k in range(lalpulsar.PULSAR_MAX_SPINS)]
    doppler_keys += [
        "Alpha",
        "Delta",
        "asini",
        "period",
        "ecc",
        "tp",
        "argp",
    ]
    fmt_dict = {k: fmt_str for k in keys if k in doppler_keys}
    return fmt_dict
