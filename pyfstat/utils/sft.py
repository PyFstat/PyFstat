import logging
from typing import Optional, Tuple

import lalpulsar
import numpy as np

logger = logging.getLogger(__name__)


def get_sft_as_arrays(
    sftfilepattern: str,
    fMin: Optional[float] = None,
    fMax: Optional[float] = None,
    constraints: Optional[lalpulsar.SFTConstraints] = None,
) -> Tuple[np.ndarray, dict, dict]:

    """
    Read binary SFT files into NumPy arrays.

    Parameters
    ----------
    sftfilepattern:
        Pattern to match SFTs using wildcards (`*?`) and ranges [0-9];
        multiple patterns can be given separated by colons.
    fMin, fMax:
        Restrict frequency range to `[fMin, fMax]`.
        If None, retreive the full frequency range.
    constraints:
        Constrains to be fed into XLALSFTdataFind to specify detector,
        GPS time range or timestamps to be retrieved.

    Returns
    ----------
    freqs:
        The frequency bins in each SFT. These will be the same for each SFT,
        so only a single 1D array is returned.
    times:
        The SFT start times as a dictionary of 1D arrays, one for each detector.
        Keys correspond to the official detector names as returned by
        lalpulsar.ListIFOsInCatalog.
    data:
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
                f"Frequencies don't match between {ifo_labels.data[ind-1]} and {ifo}"
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
