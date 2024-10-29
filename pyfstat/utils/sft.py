import logging
import os
from typing import Dict, Optional, Tuple

import lal
import lalpulsar
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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


def get_sft_constraints_from_tstart_duration(
    tstart: Optional[int] = None,
    duration: Optional[int] = None,
    timestamps: Optional[dict] = None,
) -> lalpulsar.SFTConstraints:
    """
    Use start and duration to set up a lalpulsar.SFTConstraints
    object.

    Parameters
    ----------
    tstart:
        GPS seconds of first SFT start time
    duration:
        Total time-spanned by all SFTs in seconds.
    timestamps:
        The SFT start times as a dictionary of 1D arrays, one for each detector.
        Keys correspond to the official detector names as returned by lalpulsar.ListIFOsInCatalog

    Returns
    -------
    SFTConstraint: lalpulsar.SFTConstraints
        Constraints to be fed into XLALSFTdataFind to specify detector, GPS time range or timestamps to be retrieved.
    """
    SFTConstraint = lalpulsar.SFTConstraints()

    if tstart is None:
        SFTConstraint.minStartTime = None
        SFTConstraint.maxStartTime = None
    elif duration is None:
        SFTConstraint.maxStartTime = None
    else:
        SFTConstraint.minStartTime = lal.LIGOTimeGPS(tstart)
        SFTConstraint.maxStartTime = SFTConstraint.minStartTime + duration

    SFTConstraint.timestamps = None  # FIXME: not currently supported
    if timestamps is not None:  # pragma: no cover
        raise NotImplementedError("Timestamps not yet supported in this function.")

    logger.info(
        "SFT Constraints: [minStartTime:{}, maxStartTime:{}]".format(
            SFTConstraint.minStartTime,
            SFTConstraint.maxStartTime,
        )
    )

    return SFTConstraint


def plot_spectrogram(
    sftfilepattern: str,
    detector: str,
    savefig: Optional[bool] = False,
    outdir: Optional[str] = ".",
    label: Optional[str] = None,
    quantity: Optional[str] = "power",
    sqrtSX: Optional[float] = None,
    fMin: Optional[float] = None,
    fMax: Optional[float] = None,
    constraints: Optional[lalpulsar.SFTConstraints] = None,
    figsize: Optional[Tuple] = (16, 9),
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Compute spectrograms of a set of SFTs.
    In case the signal contains gaps, these are replaced by "nans", so in the plot they appear in white.
    This is useful to produce visualizations of the Doppler modulation of a CW signal.

    Parameters
    ----------
    sftfilepattern:
        Pattern to match SFTs using wildcards (`*?`) and ranges [0-9];
        multiple patterns can be given separated by colons.
    detector:
        Name of the detector of the data that will be plot.
    savefig:
        If True, save the figure in `outdir`.
        If False, return an axis object without saving to disk.
    outdir:
        Output folder.
    label:
        Output filename.
    quantity:
        Magnitude to be plotted.
        It can be "power" for SFT power,
        "normpower" for normalized power (`2*power/(Tsft*sqrtSX**2)`),
        "real" for the real part of the SFTs,
        and "imag" for the imaginary part of the SFTs.
        Set to "power" by default.
    sqrtSX:
        Amplitude spectral density of the data.
        Only needed if `quantity = "normpower".
    fMin, fMax:
        Restrict frequency range to `[fMin, fMax]`.
        If None, retrieve the full frequency range.
    constraints:
        Constraints to be fed into XLALSFTdataFind to specify detector,
        GPS time range or timestamps to be retrieved.
    figsize:
        Size of the figure, as a tuple of (horizontal,vertical) measurements (standard matplotlib convention).
    kwarg: dict
        Other kwargs, only used to be passed to `matplotlib.pcolormesh`.

    Returns
    -------
    ax: matplotlib.axes.Axes
        The axes object containing the plot.
    """

    logger.info("Loading SFT data")
    frequency, timestamps, fourier_data = get_sft_as_arrays(
        sftfilepattern, fMin, fMax, constraints
    )

    if savefig:
        if label is None:  # pragma: no cover
            raise ValueError("Label needed to save the figure")
        else:
            plotfile = os.path.join(outdir, label + ".png")

    if detector not in timestamps:  # pragma: no cover
        raise ValueError(
            f"Detector {detector} not found in timestamps, available detectors are {timestamps.keys()}"
        )

    # Compute Tsft
    constraints = constraints or lalpulsar.SFTConstraints()
    multi_sft_catalog = lalpulsar.GetMultiSFTCatalogView(
        lalpulsar.SFTdataFind(sftfilepattern, constraints)
    )
    Tsft = int(round(1.0 / multi_sft_catalog.data[0].data[0].header.deltaF))
    logger.info(f"Extracted Tsft={Tsft}")

    # Fill up gaps with Nans
    gap_length = np.diff(timestamps[detector]) - Tsft
    gap_data = [fourier_data[detector][:, 0]]
    gap_timestamps = [timestamps[detector][0]]

    gaps = False
    for ind, gap in enumerate(gap_length):
        if gap > 0:
            gaps = True
            num_nans = gap // Tsft
            remainder = gap % Tsft

            for i in range(num_nans):
                gap_data.append(
                    np.full_like(fourier_data[detector][:, ind], np.nan + 1j * np.nan)
                )
                gap_timestamps.append(timestamps[detector][ind] + (i + 1) * Tsft)

            if remainder > 0:
                gap_data.append(
                    np.full_like(fourier_data[detector][:, ind], np.nan + 1j * np.nan)
                )
                gap_timestamps.append(
                    timestamps[detector][ind] + (num_nans * Tsft) + remainder
                )

        gap_data.append(fourier_data[detector][:, ind + 1])
        gap_timestamps.append(timestamps[detector][ind + 1])

    if gaps is True:
        timestamps = {detector: np.hstack(gap_timestamps)}
        fourier_data = {detector: np.vstack(gap_data).T}

    # Initialize plot
    plt.rcParams["axes.grid"] = False  # turn off the gridlines
    fig, ax = plt.subplots(figsize=figsize)
    ax.set(xlabel="Time [days]", ylabel="Frequency [Hz]")

    time_in_days = (timestamps[detector] - timestamps[detector][0]) / 86400

    if "power" in quantity:
        logger.info("Computing SFT power")
        q = fourier_data[detector].real ** 2 + fourier_data[detector].imag ** 2
        if quantity == "normpower":
            if sqrtSX is None:  # pragma: no cover
                raise ValueError(
                    "Value of sqrtSX needed to compute the normalized power."
                )
            q *= 2 / (Tsft * sqrtSX**2)
            label = "Normalized power"
        else:
            label = "Power"

    elif quantity == "real":
        q = fourier_data[detector].real
        ax.set_title("SFT real part")
        label = "Fourier amplitude"

    elif quantity == "imag":
        q = fourier_data[detector].imag
        ax.set_title("SFT imaginary part")
        label = "Fourier amplitude"

    else:  # pragma: no cover
        raise ValueError(
            f"Quantity '{quantity}' not accepted. Please, introduce a supported quantity."
        )

    if savefig:
        logger.info(f"Plotting to file: {plotfile}")
    else:  # pragma: no cover
        logger.info("Plotting, will return axes object")
    c = ax.pcolormesh(
        time_in_days,
        frequency,
        q,
        cmap=kwargs["cmap"] if "cmap" in kwargs else "inferno_r",
        shading="nearest",
    )
    fig.colorbar(c, label=label)
    plt.tight_layout()
    if savefig:
        fig.savefig(plotfile)

    return ax
