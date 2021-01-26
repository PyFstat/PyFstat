"""
A collection of helpful functions to facilitate ease-of-use of PyFstat.

Most of these are used internally by other parts of the package
and are of interest mostly only for developers,
but others can also be helpful for end users.
"""

import os
import sys
import subprocess
import argparse
import logging
import inspect
import peakutils
import shutil
from functools import wraps
import numpy as np
import lal
import lalpulsar
from ._version import get_versions

# workaround for matplotlib on X-less remote logins
if "DISPLAY" in os.environ:
    import matplotlib.pyplot as plt
else:
    logging.info(
        'No $DISPLAY environment variable found, so importing \
                  matplotlib.pyplot with non-interactive "Agg" backend.'
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


def set_up_optional_tqdm():
    """Provides local replacement for tqdm if it cannot be imported."""
    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(x, *args, **kwargs):
            return x

    return tqdm


def set_up_matplotlib_defaults():
    """Sets some defaults for matplotlib plotting."""
    plt.rcParams["text.usetex"] = shutil.which("latex") is not None
    plt.rcParams["axes.formatter.useoffset"] = False


def set_up_command_line_arguments():
    """Parse global commandline arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase output verbosity [logging.DEBUG]",
    )
    parser.add_argument(
        "-q",
        "--quite",
        action="store_true",
        help="Decrease output verbosity [logging.WARNING]",
    )
    parser.add_argument(
        "--no-interactive", help="Don't use interactive", action="store_true"
    )
    parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        help="Force clean data, never use cached data",
    )
    fu_parser = parser.add_argument_group(
        "follow-up options", "Options related to MCMCFollowUpSearch"
    )
    fu_parser.add_argument(
        "-s",
        "--setup-only",
        action="store_true",
        help="Only generate the setup file, don't run",
    )
    fu_parser.add_argument(
        "--no-template-counting",
        action="store_true",
        help="No counting of templates, useful if the setup is predefined",
    )
    parser.add_argument(
        "-N",
        type=int,
        default=3,
        metavar="N",
        help="Number of threads to use when running in parallel",
    )
    parser.add_argument("unittest_args", nargs="*")
    args, unknown = parser.parse_known_args()
    sys.argv[1:] = args.unittest_args

    if args.quite or args.no_interactive:

        def tqdm(x, *args, **kwargs):
            return x

    else:
        tqdm = set_up_optional_tqdm()

    logger = logging.getLogger()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s: %(message)s", datefmt="%H:%M")
    )

    if args.quite:
        logger.setLevel(logging.WARNING)
        stream_handler.setLevel(logging.WARNING)
    elif args.verbose:
        logger.setLevel(logging.DEBUG)
        stream_handler.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        stream_handler.setLevel(logging.INFO)

    logger.addHandler(stream_handler)
    return args, tqdm


def get_ephemeris_files():
    """Set the ephemeris files to use for the Earth and Sun.

    This looks first for a configuration file `~/.pyfstat.conf`
    and next in the $LALPULSAR_DATADIR environment variable.

    If neither source provides the necessary files,
    a warning is emitted and the user can still run PyFstat searches,
    but must then include ephemeris options manually on each class instantiation.

    The 'DE405' ephemerides version provided with lalpulsar is expected.

    Returns
    ----------
    earth_ephem, sun_ephem: str
        Paths of the two files containing positions of Earth and Sun.
    """
    config_file = os.path.join(os.path.expanduser("~"), ".pyfstat.conf")
    env_var = "LALPULSAR_DATADIR"
    please = "Please provide the ephemerides paths when initialising searches."
    if os.path.isfile(config_file):
        d = {}
        with open(config_file, "r") as f:
            for line in f:
                k, v = line.split("=")
                k = k.replace(" ", "")
                for item in [" ", "'", '"', "\n"]:
                    v = v.replace(item, "")
                d[k] = v
        try:
            earth_ephem = d["earth_ephem"]
            sun_ephem = d["sun_ephem"]
        except KeyError:
            logging.warning(
                "No [earth/sun]_ephem found in " + config_file + ". " + please
            )
            earth_ephem = None
            sun_ephem = None
    elif env_var in list(os.environ.keys()):
        ephem_version = "DE405"
        earth_ephem = os.path.join(
            os.environ[env_var], "earth00-40-{:s}.dat.gz".format(ephem_version)
        )
        sun_ephem = os.path.join(
            os.environ[env_var], "sun00-40-{:s}.dat.gz".format(ephem_version)
        )
        if not (os.path.isfile(earth_ephem) and os.path.isfile(sun_ephem)):
            earth_ephem = os.path.join(
                os.environ[env_var], "earth00-19-{:s}.dat.gz".format(ephem_version)
            )
            sun_ephem = os.path.join(
                os.environ[env_var], "sun00-19-{:s}.dat.gz".format(ephem_version)
            )
            if not (os.path.isfile(earth_ephem) and os.path.isfile(sun_ephem)):
                logging.warning(
                    "Default [earth/sun]00-[19/40]-" + ephem_version + " ephemerides "
                    "not found in the " + os.environ[env_var] + " directory. " + please
                )
                earth_ephem = None
                sun_ephem = None
    else:
        logging.warning(
            "No " + config_file + " file or $" + env_var + " environment "
            "variable found. " + please
        )
        earth_ephem = None
        sun_ephem = None
    return earth_ephem, sun_ephem


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
    factor = 10 ** power
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
    if type(x) == str:
        return x
    x = round_to_n(x, d)
    if 0.01 < abs(x) < 100:
        return str(x)
    else:
        power = int(np.floor(np.log10(abs(x))))
        stem = np.round(x / 10 ** power, d)
        if d == 1:
            stem = int(stem)
        return r"${}{{\times}}10^{{{}}}$".format(stem, power)


def initializer(func):
    """Decorator to automatically assign the parameters of a class instantiation to self."""
    argspec = inspect.getfullargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(argspec.args[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(argspec.args), reversed(argspec.defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper


def get_peak_values(frequencies, twoF, threshold_2F, F0=None, F0range=None):
    """Find a set of local peaks of twoF values over a 1D frequency list.

    Parameters
    ----------
    frequencies: np.ndarray
        1D array of frequency values.
    twoF: np.ndarray
        Corresponding 1D array of detection statistic values.
    threshold_2F: float
        Only report peaks above this threshold.
    F0, F0range: float or None
        Only report peaks within [F0-F0range,F0+F0range].

    Returns
    ----------
    F0maxs: np.ndarray
        1D array of peak frequencies.
    twoFmaxs: np.ndarray
        1D array of peak twoF values.
    freq_errs: np.ndarray
        1D array of peak frequency estimation error,
        taken as the spacing between neighbouring input frequencies.
    """
    if F0:
        if F0range is None:
            raise ValueError("If given F0, also need F0range.")
        cut_idxs = np.abs(frequencies - F0) < F0range
        frequencies = frequencies[cut_idxs]
        twoF = twoF[cut_idxs]
    idxs = peakutils.indexes(twoF, thres=1.0 * threshold_2F / np.max(twoF))
    F0maxs = frequencies[idxs]
    twoFmaxs = twoF[idxs]
    freq_err = frequencies[1] - frequencies[0]
    return F0maxs, twoFmaxs, freq_err * np.ones(len(idxs))


def get_comb_values(F0, frequencies, twoF, period, N=4):
    """Check which points of a [frequencies,twoF] set correspond to a certain comb of frequencies.

    Parameters
    ----------
    F0: float
        Base frequency for the comb.
    frequencies: np.ndarray
        1D array of frequency values.
    twoF: np.ndarray
        Corresponding 1D array of detection statistic values.
    period: str
        Modulation type of the comb, either `sidereal` or `terrestrial`.
    N: int
        Number of comb harmonics.

    Returns
    ----------
    comb_frequencies: np.ndarray
        1D array of relative frequency offsets for the comb harmonics.
    comb_twoFs: np.ndarray
        1D array of twoF values at the closest-matching frequencies.
    freq_errs: np.ndarray
        1D array of frequency match error,
        taken as the spacing between neighbouring input frequencies.
    """
    if period == "sidereal":
        period = 23 * 60 * 60 + 56 * 60 + 4.0616
    elif period == "terrestrial":
        period = 86400
    else:
        raise ValueError("period must be either 'sidereal' or 'terrestrial'.")
    freq_err = frequencies[1] - frequencies[0]
    comb_frequencies = [n * 1 / period for n in range(-N, N + 1)]
    comb_idxs = [np.argmin(np.abs(frequencies - F0 - F)) for F in comb_frequencies]
    return comb_frequencies, twoF[comb_idxs], freq_err * np.ones(len(comb_idxs))


def run_commandline(cl, log_level=20, raise_error=True, return_output=True):
    """Run a string cmd as a subprocess, check for errors and return output.

    Parameters
    ----------
    cl: str
        Command to run
    log_level: int
        Sets the logging level for some of this function's messages.
        See https://docs.python.org/library/logging.html#logging-levels
        Default is '20' (INFO).
        FIXME: Not used for all messages.
    raise_error: bool
        If True, raise an error if the subprocess fails.
        If False, continue and just return `0`.
    return_output: bool
        If True, return the captured output of the subprocess (stdout and stderr).
        If False, return nothing on successful execution.

    Returns
    ----------
    out: str or int, optional
        The captured output of the subprocess (stdout and stderr)
        if `return_output=True`.
        0 on failed execution if `raise_error=False`.
    """

    logging.log(log_level, "Now executing: " + cl)
    if "|" in cl:
        logging.warning(
            "Pipe ('|') found in commandline, errors may not be" " properly caught!"
        )
    try:
        if return_output:
            out = subprocess.check_output(
                cl,  # what to run
                stderr=subprocess.STDOUT,  # catch errors
                shell=True,  # proper environment etc
                universal_newlines=True,  # properly display linebreaks in error/output printing
            )
        else:
            subprocess.check_call(cl, shell=True)
    except subprocess.CalledProcessError as e:
        logging.log(40, "Execution failed: {}".format(e))
        if e.output:
            logging.log(40, e.output)
        if raise_error:
            raise
        elif return_output:
            out = 0
    os.system("\n")
    if return_output:
        return out


def convert_array_to_gsl_matrix(array):
    """Convert a numpy array to a LAL-wrapped GSL matrix.

    Parameters
    ----------
    array: np.ndarray
        The array to convert.
        `array.shape` must have 2 dimensions.

    Returns
    ----------
    gsl_matrix: lal.gsl_matrix
        The LAL-wrapped GSL matrix object.
    """
    gsl_matrix = lal.gsl_matrix(*array.shape)
    gsl_matrix.data = array
    return gsl_matrix


def get_sft_array(sftfilepattern, F0=None, dF0=None):
    """Return the raw data (absolute values) from a set of SFTs.

    FIXME: currently only returns data for first detector.

    Parameters
    ----------
    sftfilepattern: str
            Pattern to match SFTs using wildcards (`*?`) and ranges [0-9];
            multiple patterns can be given separated by colons.
    F0, dF0: float or None
        Restrict frequency range to `[F0-dF0,F0+dF0]`.

    Returns
    ----------
    times: np.ndarray
        The SFT starttimes as a 1D array.
    freqs: np.ndarray
        The frequency bins in each SFT.
        These will be the same for each SFT,
        so only a single 1D array is returned.
    data: np.ndarray
        A 2D array of the absolute values of the SFT data
        in each frequency bin at each timestamp.
    """

    if F0 is None and dF0 is None:
        fMin = -1
        fMax = -1
    elif F0 is None or dF0 is None:
        raise ValueError("Need either none or both of F0, dF0.")
    else:
        fMin = F0 - dF0
        fMax = F0 + dF0

    SFTCatalog = lalpulsar.SFTdataFind(sftfilepattern, lalpulsar.SFTConstraints())
    MultiSFTs = lalpulsar.LoadMultiSFTs(SFTCatalog, fMin, fMax)
    ndet = MultiSFTs.length
    if ndet > 1:
        logging.warning(
            "Loaded SFTs from {:d} detectors, only using the first.".format(ndet)
        )

    SFTs = MultiSFTs.data[0]
    times = np.array([sft.epoch.gpsSeconds for sft in SFTs.data])
    data = [np.abs(sft.data.data) for sft in SFTs.data]
    data = np.array(data).T
    nbins, nsfts = data.shape

    sft0 = SFTs.data[0]
    freqs = np.linspace(sft0.f0, sft0.f0 + (nbins - 1) * sft0.deltaF, nbins)

    return times, freqs, data


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


def get_version_string():
    """
    Get the canonical version string of the currently running PyFstat instance.

    Returns
    -------
    version: str
        The version string.
    """
    return get_versions()["version"]


def get_doppler_params_output_format(keys):
    """Set a canonical output precision for frequency evolution parameters.

    This uses the same format (`%.16g`) as
    the `write_FstatCandidate_to_fp()` function of
    `lalapps_ComputeFstatistic_v2`.

    This assigns that format to each parameter name in `keys`
    which matches a hardcoded list of known standard 'Doppler' parameters,
    and ignores any others.

    Parameters
    -------
    keys: dict
        The parameter keys for which to select formats.

    Returns
    -------
    fmt: dict
        A dictionary assigning the default format to each parameter key
        from the hardcoded list of standard 'Doppler' parameters.
    """
    CFSv2_fmt = "%.16g"
    doppler_keys = [
        "F0",
        "F1",
        "F2",
        "Alpha",
        "Delta",
        "asini",
        "period",
        "ecc",
        "tp",
        "argp",
    ]
    fmt = {k: CFSv2_fmt for k in keys if k in doppler_keys}
    return fmt


def read_txt_file_with_header(f, names=True, comments="#"):
    """Wrapper to np.genfromtxt with smarter handling of variable-length commented headers.

    The header is identified as an uninterrupted block of lines
    from the beginning of the file,
    each starting with the given `comments` character.

    After identifying a header of length `Nhead`,
    this function then tells `np.genfromtxt()` to skip `Nhead-1` lines
    (to allow for reading field names from the last commented line
    before the actual data starts).

    Parameters
    -------
    f: str
        Name of the file to read.
    names: bool
        Passed on to `np.genfromtxt()`:
        If True, the field names are read from the last header line.
    comments: str
        The character used to indicate the start of a comment.
        Also passed on to `np.genfromtxt()`.

    Returns
    -------
    data: np.ndarray
        The data array read from the file after skipping the header.
    """
    with open(f, "r") as f_opened:
        Nhead = 0
        for line in f_opened:
            if not line.startswith(comments):
                break
            Nhead += 1
    data = np.atleast_1d(
        np.genfromtxt(f, skip_header=Nhead - 1, names=names, comments=comments)
    )
    return data


def get_lalapps_commandline_from_SFTDescriptor(descriptor):
    """Extract a lalapps commandline from the 'comment' entry of a SFT descriptor.

    Most SFT creation tools save their commandline into that entry,
    so we can extract it and reuse it to reproduce that data.

    Parameters
    ----------
    descriptor: SFTDescriptor
        Element of a `lalpulsar.SFTCatalog` structure.

    Returns
    -------
    cmd: str
        A lalapps commandline string,
        or an empty string if 'lalapps' not found in comment.
    """
    comment = getattr(descriptor, "comment", None)
    if comment is None:
        return ""
    comment_lines = comment.split("\n")
    # get the first line with the right substring
    # (iterate until it's found)
    return next((line for line in comment_lines if "lalapps" in line), "")


def read_parameters_dict_lines_from_file_header(
    outfile, comments="#", strip_spaces=True
):
    """Load a list of pretty-printed parameters dictionary lines from a commented file header.

    Returns a list of lines from a commented file header
    that match the pretty-printed parameters dictionary format
    as generated by `BaseSearchClass.get_output_file_header()`.
    The opening/closing bracket lines (`{`,`}`) are not included.
    Newline characters at the end of each line are stripped.

    Parameters
    ----------
    outfile: str
        Name of a PyFstat-produced output file.
    comments: str
        Comment character used to start header lines.
    strip_spaces: bool
        Whether to strip leading/trailing spaces.

    Returns
    -------
    dict_lines: list
        A list of unparsed pprinted dictionary entries.
    """
    dict_lines = []
    with open(outfile, "r") as f_opened:
        in_dict = False
        for line in f_opened:
            if not line.startswith(comments):
                raise IOError(
                    "Encountered end of {:s}-commented header before finding closing '}}' of parameters dictionary in file '{:s}'.".format(
                        comments, outfile
                    )
                )
            elif line.startswith(comments + " {"):
                in_dict = True
            elif line.startswith(comments + " }"):
                break
            elif in_dict:
                line = line.lstrip(comments).rstrip("\n")
                if strip_spaces:
                    line = line.strip(" ")
                dict_lines.append(line)
    if len(dict_lines) == 0:
        raise IOError(
            "Could not parse non-empty parameters dictionary from file '{:s}'.".format(
                outfile
            )
        )
    return dict_lines


def get_parameters_dict_from_file_header(outfile, comments="#", eval_values=False):
    """Load a parameters dict from a commented file header.

    Returns a parameters dictionary,
    as generated by `BaseSearchClass.get_output_file_header()`,
    from an output file header.
    Always returns a proper python dictionary,
    but the values will be unparsed strings if not requested otherwise.

    Parameters
    ----------
    outfile: str
        Name of a PyFstat-produced output file.
    comments: str
        Comment character used to start header lines.
    eval_values: bool
        If False, return dictionary values as unparsed strings.
        If True, evaluate each of them.
        DANGER! Only do this if you trust the source of the file!

    Returns
    -------
    params_dict: dictionary
        A dictionary of parameters
        (with values either as unparsed strings, or evaluated).
    """
    if eval_values:
        logging.warning(
            "Will evaluate dictionary values read from file '{:s}'.".format(outfile)
        )
    params_dict = {}
    dict_lines = read_parameters_dict_lines_from_file_header(
        outfile, comments="#", strip_spaces=True
    )
    for line in dict_lines:
        line_split = line.rstrip(",").split(":")
        # check for a few possible corrupt formats,
        # though we can't be exhaustive here...
        if (
            (len(line_split) != 2)
            or np.any([len(s) == 0 for s in line_split])
            or (line_split[-1] == ",")
            or (line_split[0][0] != "'")
            or (line_split[0][-1] != "'")
        ):
            raise IOError(
                "Line '{:s}' is not of the expected format (# 'key': val,').".format(
                    line
                )
            )
        key = line_split[0].strip("'")
        val = line_split[1].strip(" ")
        if eval_values:
            val = eval(val)  # DANGER
        params_dict[key] = val
    return params_dict


def read_par(
    filename=None,
    label=None,
    outdir=None,
    suffix="par",
    comments=["%", "#"],
    raise_error=False,
):
    """Read in a .par or .loudest file, returns a dictionary of the key=val pairs.

    Notes
    -----
    This can also be used to read in `.loudest` files
    produced by `lalapps_ComputeFstatistic_v2`,
    or any file which has rows of `key=val` data
    (in which the val can be understood using `eval(val)`).

    Parameters
    ----------
    filename : str
        Filename (path) containing rows of `key=val` data to read in.
    label, outdir, suffix : str, optional
        If filename is `None`, form the file to read as `outdir/label.suffix`.
    comments : str or list of strings, optional
        Characters denoting that a row is a comment.
    raise_error : bool, optional
        If True, raise an error for lines which are not comments, but cannot
        be read.

    Returns
    -------
    d: dict
        The `key=val` pairs as a dictionary.

    """
    if filename is None:
        filename = os.path.join(outdir, "{}.{}".format(label, suffix))
    if os.path.isfile(filename) is False:
        raise ValueError("No file {} found".format(filename))
    d = {}
    with open(filename, "r") as f:
        d = get_dictionary_from_lines(f, comments, raise_error)
    return d


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


def get_predict_fstat_parameters_from_dict(signal_parameters):
    """Extract a subset of parameters as needed for predicting F-stats.

    Given a dictionary with arbitrary signal parameters,
    this extracts only those ones required by `helper_functions.predict_fstat()`:
    Freq, Alpha, Delta, h0, cosi, psi.

    Parameters
    ----------
    signal_parameters: dict
        Dictionary containing at least those signal parameters required by
        helper_functions.predict_fstat.
        This dictionary's keys must follow
        the PyFstat convention (e.g. F0 instead of Freq).

    Returns
    -------
    predict_fstat_params: dict
        The dictionary of selected parameters.
    """
    predict_fstat_params = {
        key: signal_parameters[key]
        for key in ["F0", "Alpha", "Delta", "h0", "cosi", "psi"]
    }
    return predict_fstat_params


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
    """Wrapper to lalapps_PredictFstat for predicting expected F-stat values.

    Parameters
    ----------
    h0, cosi, psi, Alpha, Delta : float
        Signal parameters, see `lalapps_PredictFstat --help` for more info.
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
        Temporary file used for `lalapps_PredictFstat` output,
        will be deleted at the end.
    earth_ephem, sun_ephem : str or None
        Ephemerides files, defaults will be used if `None`.
    transientWindowType: str
        Optional parameter for transient signals,
        see `lalapps_PredictFstat --help`.
        Default of `none` means a classical Continuous Wave signal.
    transientStartTime, transientTau: int or None
        Optional parameters for transient signals,
        see `lalapps_PredictFstat --help`.

    Returns
    -------
    twoF_expected, twoF_sigma : float
        The expectation and standard deviation of 2F.
    """

    cl_pfs = []
    cl_pfs.append("lalapps_PredictFstat")

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
