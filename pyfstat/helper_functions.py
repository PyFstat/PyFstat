"""
Provides helpful functions to facilitate ease-of-use of pyfstat
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
from scipy.stats.distributions import ncx2
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
    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(x, *args, **kwargs):
            return x

    return tqdm


def set_up_matplotlib_defaults():
    plt.switch_backend("Agg")
    plt.rcParams["text.usetex"] = shutil.which("latex") is not None
    plt.rcParams["axes.formatter.useoffset"] = False


def set_up_command_line_arguments():
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
    """ Returns the earth_ephem and sun_ephem """
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
        except:
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
    if not x:
        return 0
    power = -int(np.floor(np.log10(abs(x)))) + (n - 1)
    factor = 10 ** power
    return round(x * factor) / factor


def texify_float(x, d=2):
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
    """ Decorator function to automatically assign the parameters to self """
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
    if F0:
        cut_idxs = np.abs(frequencies - F0) < F0range
        frequencies = frequencies[cut_idxs]
        twoF = twoF[cut_idxs]
    idxs = peakutils.indexes(twoF, thres=1.0 * threshold_2F / np.max(twoF))
    F0maxs = frequencies[idxs]
    twoFmaxs = twoF[idxs]
    freq_err = frequencies[1] - frequencies[0]
    return F0maxs, twoFmaxs, freq_err * np.ones(len(idxs))


def get_comb_values(F0, frequencies, twoF, period, N=4):
    if period == "sidereal":
        period = 23 * 60 * 60 + 56 * 60 + 4.0616
    elif period == "terrestrial":
        period = 86400
    freq_err = frequencies[1] - frequencies[0]
    comb_frequencies = [n * 1 / period for n in range(-N, N + 1)]
    comb_idxs = [np.argmin(np.abs(frequencies - F0 - F)) for F in comb_frequencies]
    return comb_frequencies, twoF[comb_idxs], freq_err * np.ones(len(comb_idxs))


def compute_P_twoFstarcheck(twoFstarcheck, twoFcheck, M0, plot=False):
    """ Returns the unnormalised pdf of twoFstarcheck given twoFcheck """
    upper = 4 + twoFstarcheck + 0.5 * (2 * (4 * M0 + 2 * twoFcheck))
    rho2starcheck = np.linspace(1e-1, upper, 500)
    integrand = ncx2.pdf(twoFstarcheck, 4 * M0, rho2starcheck) * ncx2.pdf(
        twoFcheck, 4, rho2starcheck
    )
    if plot:
        fig, ax = plt.subplots()
        ax.plot(rho2starcheck, integrand)
        fig.savefig("test")
    return np.trapz(integrand, rho2starcheck)


def compute_pstar(twoFcheck_obs, twoFstarcheck_obs, m0, plot=False):
    M0 = 2 * m0 + 1
    upper = 4 + twoFcheck_obs + (2 * (4 * M0 + 2 * twoFcheck_obs))
    twoFstarcheck_vals = np.linspace(1e-1, upper, 500)
    P_twoFstarcheck = np.array(
        [
            compute_P_twoFstarcheck(twoFstarcheck, twoFcheck_obs, M0)
            for twoFstarcheck in twoFstarcheck_vals
        ]
    )
    C = np.trapz(P_twoFstarcheck, twoFstarcheck_vals)
    idx = np.argmin(np.abs(twoFstarcheck_vals - twoFstarcheck_obs))
    if plot:
        fig, ax = plt.subplots()
        ax.plot(twoFstarcheck_vals, P_twoFstarcheck)
        ax.fill_between(twoFstarcheck_vals[: idx + 1], 0, P_twoFstarcheck[: idx + 1])
        ax.axvline(twoFstarcheck_vals[idx])
        fig.savefig("test")
    pstar_l = np.trapz(P_twoFstarcheck[: idx + 1] / C, twoFstarcheck_vals[: idx + 1])
    return 2 * np.min([pstar_l, 1 - pstar_l])


def run_commandline(cl, log_level=20, raise_error=True, return_output=True):
    """Run a string cmd as a subprocess, check for errors and return output.

    Parameters
    ----------
    cl: str
        Command to run
    log_level: int
        See https://docs.python.org/library/logging.html#logging-levels
        default is '20' (INFO)

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
            process = subprocess.check_call(cl, shell=True)
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
    gsl_matrix = lal.gsl_matrix(*array.shape)
    gsl_matrix.data = array
    return gsl_matrix


def get_sft_array(sftfilepattern, data_duration=None, F0=None, dF0=None):
    """ Return the raw data (absolute value) from a set of SFTs

    FIXME: currently only returns data for first detector
    """

    if F0 is None and dF0 is None:
        fMin = -1
        fMax = -1
    elif F0 is None or dF0 is None:
        raise ValueError("Need either none or both of F0, dF0.")
    else:
        fMin = F0 - dF0
        fMax = F0 + dF0

    if data_duration is not None:
        logging.warning(
            "Option 'data_duration' for get_sft_array()"
            " is no longer in use and will be removed."
        )

    SFTCatalog = lalpulsar.SFTdataFind(sftfilepattern, lalpulsar.SFTConstraints())
    MultiSFTs = lalpulsar.LoadMultiSFTs(SFTCatalog, fMin, fMax)
    ndet = MultiSFTs.length
    if ndet > 1:
        logging.warning(
            "Loaded SFTs from {:d} detectors, only using the first.".format(ndet)
        )

    SFTs = MultiSFTs.data[0]
    times = [sft.epoch.gpsSeconds for sft in SFTs.data]
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
    """ Get the covering band using XLALCWSignalCoveringBand

    Parameters
    ----------
    tref, tstart, tend: int
        The reference, start, and end times of interest
    F0, F1, F1: float
        Minimum frequency and spin-down of signals to be covered
    F0band, F1band, F1band: float
        Ranges of frequency and spin-down of signals to be covered
    maxOrbitAsini: float
        Largest orbital projected semi-major axis to be covered
    minOrbitPeriod: float
        Shortest orbital period to be covered
    maxOrbitEcc: float
        Highest orbital eccentricity to be covered

    Returns
    -------
    F0min, F0max: float
        Estimates of the minimum and maximum frequencies of the signal during
        the search

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


def twoFDMoffThreshold(
    twoFon, knee=400, twoFDMoffthreshold_below_threshold=62, prefactor=0.9, offset=0.5
):
    """ Calculation of the 2F_DMoff threshold, see Eq 2 of arXiv:1707.5286 """
    if twoFon <= knee:
        return twoFDMoffthreshold_below_threshold
    else:
        return 10 ** (prefactor * np.log10(twoFon - offset))


def match_commandlines(cl1, cl2, be_strict_about_full_executable_path=False):
    """ Check if two commandlines match element-by-element, regardless of order """
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
    return get_versions()["version"]


def get_doppler_params_output_format(keys):
    # use same format for writing out search parameters
    # as write_FstatCandidate_to_fp() function of lalapps_CFSv2
    fmt = []
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

    for k in keys:
        if k in doppler_keys:
            fmt += [CFSv2_fmt]
    return fmt


def read_txt_file_with_header(f, comments="#"):
    # wrapper to np.genfromtxt with smarter header handling
    with open(f, "r") as f_opened:
        Nhead = 0
        for line in f_opened:
            if not line.startswith(comments):
                break
            Nhead += 1
    data = np.genfromtxt(f, skip_header=Nhead - 1, names=True, comments=comments)

    return data


def get_lalapps_commandline_from_SFTDescriptor(descriptor):
    """ get a lalapps commandline from SFT descriptor "comment" entry

    Parameters
    ----------
    descriptor: SFTDescriptor
        element of a lalpulsar SFTCatalog

    Returns
    -------
    cmd: str
        a lalapps commandline, or empty string if "lalapps" not found in comment
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
    """ load a list of pretty-printed parameters dictionary lines from a commented file header

    Returns a list of lines from a commented file header
    that match the pretty-printed parameters dictionary format
    as generated by BaseSearchClass.get_output_file_header().
    The opening/closing bracket lines ("{","}") are not included.
    Newline characters at the end of each line are stripped.

    Parameters
    ----------
    outfile: str
        name of a PyFstat-produced output file
    comments: str
        comment character used to start header lines
    strip_spaces: bool
        whether to strip leading/trailing spaces

    Returns
    -------
    dict_lines: list
        a list of unparsed pprinted dictionary entries
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
    """ load a parameters dict from a commented file header

    Returns a parameters dictionary,
    as generated by BaseSearchClass.get_output_file_header(),
    from an output file header.
    Always returns a proper python dictionary,
    but the values will be unparsed strings if not requested otherwise.

    Parameters
    ----------
    outfile: str
        name of a PyFstat-produced output file
    comments: str
        comment character used to start header lines
    eval_values: bool
        If False, return dictionary values as strings.
        If True, evaluate them. DANGER! Only do this if you trust the source.

    Returns
    -------
    params_dict: dictionary
        a dictionary of parameters
        (with values either as unparsed strings, or evaluated)
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
