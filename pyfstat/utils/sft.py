import logging
import os
from typing import Dict, Optional, Tuple

import lalpulsar
import matplotlib.pyplot as plt
import numpy as np

from pyfstat.logging import set_up_logger

logger = logging.getLogger(__name__)


def get_sft_as_arrays(
    sftfilepattern: str,
    fMin: Optional[float] = None,
    fMax: Optional[float] = None,
    constraints: Optional[lalpulsar.SFTConstraints] = None,
) -> Tuple[np.ndarray, Dict, Dict]:
    """
    Read binary SFT files into NumPy arrays.

    Parameters
    ----------
    sftfilepattern:
        Pattern to match SFTs using wildcards (`*?`) and ranges [0-9];
        multiple patterns can be given separated by colons.
    fMin, fMax:
        Restrict frequency range to `[fMin, fMax]`.
        If None, retrieve the full frequency range.
    constraints:
        Constrains to be fed into XLALSFTdataFind to specify detector,
        GPS time range or timestamps to be retrieved.

    Returns
    ----------
    freqs: np.ndarray
        The frequency bins in each SFT. These will be the same for each SFT,
        so only a single 1D array is returned.
    times: Dict
        The SFT start times as a dictionary of 1D arrays, one for each detector.
        Keys correspond to the official detector names as returned by
        lalpulsar.ListIFOsInCatalog.
    data: Dict
        A dictionary of 2D arrays of the complex Fourier amplitudes of the SFT data
        for each detector in each frequency bin at each timestamp.
        Keys correspond to the official detector names as returned by
        lalpulsar.ListIFOsInCatalog.
    """
    constraints = constraints or lalpulsar.SFTConstraints()
    if fMin is None and fMax is None:
        fMin = fMax = -1
    elif fMin is None or fMax is None:
        raise ValueError("Need either none or both of fMin, fMax.")

    sft_catalog = lalpulsar.SFTdataFind(sftfilepattern, constraints)
    ifo_labels = lalpulsar.ListIFOsInCatalog(sft_catalog)

    logger.info(
        f"Loading {sft_catalog.length} SFTs from {', '.join(ifo_labels.data)}..."
    )
    multi_sfts = lalpulsar.LoadMultiSFTs(sft_catalog, fMin, fMax)
    logger.debug("done!")

    times = {}
    amplitudes = {}

    old_frequencies = None
    for ind, ifo in enumerate(ifo_labels.data):
        sfts = multi_sfts.data[ind]

        times[ifo] = np.array([sft.epoch.gpsSeconds for sft in sfts.data])
        amplitudes[ifo] = np.array([sft.data.data for sft in sfts.data]).T

        nbins, nsfts = amplitudes[ifo].shape

        logger.debug(f"{nsfts} retrieved from {ifo}.")

        f0 = sfts.data[0].f0
        df = sfts.data[0].deltaF
        frequencies = np.linspace(f0, f0 + (nbins - 1) * df, nbins)

        if (old_frequencies is not None) and not np.allclose(
            frequencies, old_frequencies
        ):
            raise ValueError(
                f"Frequencies don't match between {ifo_labels.data[ind - 1]} and {ifo}"
            )
        old_frequencies = frequencies

    return frequencies, times, amplitudes


def get_commandline_from_SFTDescriptor(descriptor):
    """Extract a commandline from the 'comment' entry of a SFT descriptor.

    Most LALSuite SFT creation tools save their commandline into that entry,
    so we can extract it and reuse it to reproduce that data.

    Since lalapps 9.0.0 / lalpulsar 5.0.0
    the relevant executables have been moved to lalpulsar,
    but we allow for lalapps backwards compatibility here,

    Parameters
    ----------
    descriptor: SFTDescriptor
        Element of a `lalpulsar.SFTCatalog` structure.

    Returns
    -------
    cmd: str
        A lalapps/lalpulsar commandline string,
        or an empty string if no match in comment.
    """
    comment = getattr(descriptor, "comment", None)
    if comment is None:
        return ""
    comment_lines = comment.split("\n")
    # get the first line with the right substring
    # (iterate until it's found)
    return next(
        (line for line in comment_lines if "lalpulsar" in line or "lalapps" in line), ""
    )


def get_official_sft_filename(
    IFO,
    numSFTs,
    Tsft,
    tstart,
    duration,
    label=None,
    window_type=None,
    window_param=None,
):
    """Wrapper to predict the canonical lalpulsar names for SFT files.

    Parameters
    ----------
    IFO: str
        Two-char detector name, e.g. `H1`.
    numSFTs: int
        numSFTs	number of SFTs in SFT-file
    Tsft: int
        time-baseline in (integer) seconds
    tstart: int
        GPS seconds of first SFT start time
    duration: int
        total time-spanned by all SFTs in seconds
    label: str or None
        optional 'Misc' entry in the SFT 'D' field
    window_type: str or None
        window function applied to SFTs
    window_param: float or None
        additional parameter for some window functions

    Returns
    -------
    filename: str
        The canonical SFT file name for the input parameters.
    """
    spec = lalpulsar.SFTFilenameSpec()
    lalpulsar.FillSFTFilenameSpecStrings(
        spec=spec,
        path=None,
        extn=None,
        detector=IFO,
        window_type=window_type,
        privMisc=label,
        pubObsKind=None,
        pubChannel=None,
    )
    spec.window_param = window_param or 0
    spec.numSFTs = numSFTs
    spec.SFTtimebase = Tsft
    spec.gpsStart = tstart
    # possible gotcha: duration may be different if nanoseconds of sft-epochs are non-zero
    # (see SFTfileIO.c in lalpulsar)
    spec.SFTspan = duration
    return lalpulsar.BuildSFTFilenameFromSpec(spec)


def plot_spectrogram(
    sftfilepattern: str,
    savefig: Optional[bool] = False,
    outdir: Optional[str] = ".",
    label: Optional[str] = None,
    quantity: Optional[str] = "power",
    detector: Optional[str] = "H1",
    sqrtSX: Optional[float] = None,
    fMin: Optional[float] = None,
    fMax: Optional[float] = None,
    constraints: Optional[lalpulsar.SFTConstraints] = None,
    **kwargs,
):
    """
    Compute spectrograms of a set of SFTs.
    This is useful to produce visualizations of the Doppler modulation of a CW signal.

    Parameters
    ----------
    sftfilepattern:
        Pattern to match SFTs using wildcards (`*?`) and ranges [0-9];
        multiple patterns can be given separated by colons.
    savefig:
        If true, save the figure in `outdir`.
        If false, return an axis object without saving to disk.
    outdir:
        Output folder.
    label:
        Output filename.
    quantity:
        Magnitude to be plotted.
        It can be "power" for power, "normpower" for normalized power, "Re" for the real part of
        the SFTs, and "Im" for the imaginary part of the SFTs.
        Set to "Power" by default.
    detector:
        Name of the detector of the data that will be plot; it can be "H1", for LIGO Hanford, or "L1", for LIGO Livingstone.
        Set to "H1" by default.

    sqrtSX:
        ???
    fMin, fMax:
        Restrict frequency range to `[fMin, fMax]`.
        If None, retrieve the full frequency range.
    constraints:
        Constrains to be fed into XLALSFTdataFind to specify detector,
        GPS time range or timestamps to be retrieved.
    kwarg: dict
        Other kwargs, only used to be passed to matplotlip.

    Returns
    -------
    ax: matplotlib.axes._subplots_AxesSubplot
        The axes object containing the plot.
    """

    logger = set_up_logger(label=label, outdir=outdir)

    logger.info("Loading SFT data")
    frequency, timestamps, fourier_data = get_sft_as_arrays(
        sftfilepattern, fMin, fMax, constraints
    )

    if detector != "H1" and detector != "L1":
        raise ValueError("Introduce a valid detector: 'H1' or 'L1'")

    if savefig:
        if label is None:
            raise ValueError("Label needed to save the figure")
        else:
            plotfile = os.path.join(outdir, label + ".png")
            logger.info(f"Plotting to file: {plotfile}")

    plt.rcParams["axes.grid"] = False  # turn off the gridlines
    fig, ax = plt.subplots(figsize=(0.8 * 16, 0.8 * 9))
    ax.set(xlabel="Time [days]", ylabel="Frequency [Hz]")

    time_in_days = (timestamps[detector] - timestamps[detector][0]) / 86400

    if quantity == "power":
        logger.info("Computing power")
        sft_power = fourier_data[detector].real ** 2 + fourier_data[detector].imag ** 2
        q = sft_power
        logger.info("Computing power: done")
        label = "Power"

    elif quantity == "normpower":
        if sqrtSX is None:
            raise ValueError("Value of sqrtSX needed to compute the normalized power.")
        else:
            logger.info("Computing normalized power")
            sft_power = (
                fourier_data[detector].real ** 2 + fourier_data[detector].imag ** 2
            )
            Tsft = 1800  # CHANGE!!
            normalized_power = 2 * sft_power / (Tsft * sqrtSX**2)
            q = normalized_power
            logger.info("Computing normalized power: done")
            label = "Normalized Power"

    elif quantity == "Re":
        q = fourier_data[detector].real
        ax.set_title("SFT Real part")
        label = "Fourier amplitude"

    elif quantity == "Im":
        q = fourier_data[detector].imag
        ax.set_title("SFT Imaginary part")
        label = "Fourier amplitude"

    else:
        raise ValueError(
            "String `quantity` not accepted. Please, introduce a valid string."
        )

    c = ax.pcolormesh(
        time_in_days,
        frequency,
        q,
        cmap="inferno_r",
        shading="nearest",
    )
    fig.colorbar(c, label=label)
    plt.tight_layout()
    if savefig:
        fig.savefig(plotfile)

    return ax
