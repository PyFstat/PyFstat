"""PyFstat tools to generate and manipulate data in the form of SFTs."""

import glob
import logging
import os
import pkgutil

import lal
import lalpulsar
import numpy as np
from tqdm import tqdm, trange

import pyfstat.utils as utils
from pyfstat.core import BaseSearchClass, SearchForSignalWithJumps

logger = logging.getLogger(__name__)


class Writer(BaseSearchClass):
    """The main class for generating data in the form of SFTs.

    Short Fourier Transforms (SFTs) are a standard data format used in LALSuite,
    containing the Fourier transform of strain data over a duration Tsft.

    SFT data can be generated from scratch, including Gaussian noise and/or
    simulated CW signals or transient signals.
    Existing SFTs (real data or previously simulated) can also be reused through
    the `noiseSFTs` option, allowing to 'inject' additional signals into them.

    This class currently relies on the `Makefakedata_v5` executable
    which will be run in a subprocess.
    See `lalpulsar_Makefakedata_v5 --help`
    for more detailed help with some of the parameters.
    """

    mfd = "lalpulsar_Makefakedata_v5"
    """The executable; can be overridden by child classes."""

    signal_parameter_labels = [
        "tref",
        "F0",
        "F1",
        "F2",
        "Alpha",
        "Delta",
        "h0",
        "cosi",
        "psi",
        "phi",
        "transientWindowType",
        "transientStartTime",
        "transientTau",
    ]
    """Default convention of labels for the various signal parameters."""

    gps_time_and_string_formats_as_LAL = {
        "refTime": ":10.9f",
        "transientWindowType": ":s",
        "transientStartTime": ":10.0f",
        "transientTau": ":10.0f",
    }
    """Dictionary to ensure proper format handling for some special parameters.

    GPS times should NOT be parsed using scientific notation.
    LAL routines would silently parse them wrongly.
    """

    required_signal_parameters = [
        # leaving out "F1","F2","psi","phi","tref" as they have defaults
        "F0",
        "Alpha",
        "Delta",
        "cosi",
    ]
    """List of parameters required for a successful execution of Makefakedata_v5.
    The rest of available parameters are not required as they have default values
    silently given by Makefakedata_v5
    """

    @utils.initializer
    def __init__(
        self,
        label="PyFstat",
        tstart=None,
        duration=None,
        tref=None,
        F0=None,
        F1=0,
        F2=0,
        Alpha=None,
        Delta=None,
        h0=None,
        cosi=None,
        psi=0.0,
        phi=0,
        Tsft=1800,
        outdir=".",
        sqrtSX=None,
        noiseSFTs=None,
        SFTWindowType=None,
        SFTWindowParam=None,
        SFTWindowBeta=None,
        Band=None,
        detectors=None,
        earth_ephem=None,
        sun_ephem=None,
        transientWindowType="none",
        transientStartTime=None,
        transientTau=None,
        randSeed=None,
        timestamps=None,
    ):
        """
        Parameters
        ----------
        label: string
            A human-readable label to be used in naming the output files.
            NOTE: to agree with the v3 SFT naming specification
            ( https://dcc.ligo.org/T040164-v2/public )
            label can only contain ASCII alphanumeric characters
            in their "description" field,
            i.e. no underscores, hyphens etc.
            Also, internally
            a "channel"/"frame" name is constructed as `IFO:label`,
            which may not exceed 64 characters,
            so a label may only be 60 characters long.
        tstart: int
            Starting GPS epoch of the data set.
            If `noiseSFT` are given, this is used as a LALPulsar
            `SFTConstraint <https://lscsoft.docs.ligo.org/lalsuite/lalpulsar/struct_s_f_t_constraints.html>`_.
            NOTE: mutually exclusive with `timestamps`.
        duration: int
            Duration (in GPS seconds) of the total data set.
            If `noiseSFT` are given, this is used as a LALPulsar
            `SFTConstraint <https://lscsoft.docs.ligo.org/lalsuite/lalpulsar/struct_s_f_t_constraints.html>`_.
            NOTE: mutually exclusive with `timestamps`.
        tref: float or None
            Reference time for simulated signals.
            Default is `None`, which sets the reference time to `tstart`.
        F0: float or None
            Frequency of a signal to inject.
            Also used (if `Band` is not `None`) as center of frequency band.
            Also needed when noise-only (`h0=None` or `h0==0`)
            but no `noiseSFTs` given,
            in which case it is also used as center of frequency band.
        F1, F2, Alpha, Delta, h0, cosi, psi, phi: float or None
            Additional frequency evolution and amplitude parameters for a signal.
            If `h0=None` or `h0=0`, these are all ignored.
            If `h0>0`, then at least `[Alpha,Delta,cosi]` need to be set explicitly.
        Tsft: int
            The SFT duration in seconds.
            Will be ignored if `noiseSFTs` are given.
        outdir: str
            The directory where files are written to.
            Default: current working directory.
        sqrtSX: float or list or str or None
            Single-sided PSD values for generating fake Gaussian noise.
            Single float or str value: use same for all detectors.
            List or comma-separated string: must match len(detectors).
            Detectors will be paired to list elements following alphabetical order.
        noiseSFTs: str or None
            Existing SFT files on top of which signals will be injected.
            If not `None`, additional constraints can be applied
            using the arguments `tstart` and `duration`.
            NOTE: mutually exclusive with `timestamps`.
        SFTWindowType: str or None
            LAL name of the windowing function to apply to the data.
        SFTWindowParam: float
            Optional parameter for some windowing functions.
        SFTWindowBeta: float
            Defunct alias to `SFTWindowParam`.
            Will be removed in a future release.
        Band: float or None
            If float, and `F0` is also not `None`, then output SFTs cover
            `[F0-Band/2,F0+Band/2]`.
            If `None` and `noiseSFTs` given, use their bandwidth.
            If `None` and no `noiseSFTs` given,
            a minimal covering band for a perfectly-matched
            single-template ComputeFstat analysis is estimated.
        detectors: str or None
            Comma-separated list of detectors to generate data for.
            May be required depending on `timestamps`; see its documentation.
        earth_ephem, sun_ephem: str or None
            Paths of the two files containing positions of Earth and Sun.
            If None, will check standard sources as per
            utils.get_ephemeris_files().
        transientWindowType: str
            If `none`, a fully persistent CW signal is simulated.
            If `rect` or `exp`, a transient signal with the corresponding
            amplitude evolution is simulated.
        transientStartTime: int or None
            Start time for a transient signal.
        transientTau: int or None
            Duration (`rect` case) or decay time (`exp` case) of a transient signal.
        randSeed: int or None
            Optionally fix the random seed of Gaussian noise generation
            for reproducibility.
        timestamps: str or dict
            Dictionary of timestamps (each key must refer to a detector),
            a single list of timestamps
            (will be replicated for all detectors; `detectors` must be set),
            or comma-separated list of per-detector timestamps files
            (simple text files,
            comments must use `%`,
            the first column is interpreted as SFT start times
            and additional columns are ignored;
            `detectors` must be set,
            and the length and order must match).
            Timestamps must be integers;
            otherwise, will be implicitly cast by this method and MFDv5.
            NOTE: mutually exclusive with [`tstart`, `duration`]
            and with `noiseSFTs`.
        """

        self.set_ephemeris_files(earth_ephem, sun_ephem)
        self._basic_setup()
        self._parse_args_consistent_with_mfd()
        self.calculate_fmin_Band()

    def _get_sft_constraints_from_tstart_duration(self):
        """
        Use start and duration to set up a lalpulsar.SFTConstraints
        object. This method is only used if noiseSFTs is not None.
        """
        SFTConstraint = lalpulsar.SFTConstraints()

        if self.tstart is None:
            SFTConstraint.minStartTime = None
            SFTConstraint.maxStartTime = None
        elif self.duration is None:
            SFTConstraint.maxStartTime = None
        else:
            SFTConstraint.minStartTime = lal.LIGOTimeGPS(self.tstart)
            SFTConstraint.maxStartTime = SFTConstraint.minStartTime + self.duration

        SFTConstraint.timestamps = None  # FIXME: not currently supported

        logger.info(
            "SFT Constraints: [minStartTime:{}, maxStartTime:{}]".format(
                SFTConstraint.minStartTime,
                SFTConstraint.maxStartTime,
            )
        )

        return SFTConstraint

    def _get_setup_from_tstart_duration(self):
        """
        Default behavior: If no noiseSFTs are given, use the input parameters (tstart,
        duration, detectors and Tsft) to make fake data.
        """
        self.tstart = int(self.tstart)
        self.duration = int(self.duration)

        IFOs = self.detectors.split(",")
        # when duration is not a multiple of Tsft, to match MFDv5
        # we need to round the number *up*
        # and also include the overlapping bit at the end in the duration
        numSFTs = int(np.ceil(float(self.duration) / self.Tsft))  # per IFO
        effective_duration = numSFTs * self.Tsft

        self.sftfilenames = [
            utils.get_official_sft_filename(
                dets,
                numSFTs,
                self.Tsft,
                self.tstart,
                effective_duration,
                self.label,
            )
            for ind, dets in enumerate(IFOs)
        ]

    def _get_setup_from_noiseSFTs(self):
        """
        If noiseSFTs are given, use them to obtain relevant data parameters (tstart,
        duration, detectors and Tsft). The corresponding input values will be used to
        set up a lalpulsar.SFTConstraints object to be imposed to the SFTs. Keep in
        mind that Tsft will also be checked to be consistent accross all SFTs (this is
        not implemented through SFTConstraints but through a simple list check).
        """
        SFTConstraint = self._get_sft_constraints_from_tstart_duration()
        noise_multi_sft_catalog = lalpulsar.GetMultiSFTCatalogView(
            lalpulsar.SFTdataFind(self.noiseSFTs, SFTConstraint)
        )
        if noise_multi_sft_catalog.length == 0:
            raise IOError("Got empty SFT catalog.")

        # Information to be extracted from the SFTs themselves
        IFOs = []
        tstart = []
        tend = []
        Tsft = []
        self.sftfilenames = []  # This refers to the MFD output!

        for ifo_catalog in noise_multi_sft_catalog.data:
            ifo_name = lalpulsar.ListIFOsInCatalog(ifo_catalog).data[0]

            time_stamps = lalpulsar.TimestampsFromSFTCatalog(ifo_catalog)
            this_Tsft = int(round(1.0 / ifo_catalog.data[0].header.deltaF))
            this_start_time = time_stamps.data[0].gpsSeconds
            this_end_time = time_stamps.data[-1].gpsSeconds + this_Tsft

            self.sftfilenames.append(
                utils.get_official_sft_filename(
                    ifo_name,
                    time_stamps.length,  # ifo_catalog.length fails for NB case
                    this_Tsft,
                    this_start_time,
                    this_end_time - this_start_time,
                    self.label,
                )
            )

            IFOs.append(ifo_name)
            tstart.append(this_start_time)
            tend.append(this_end_time)
            Tsft.append(this_Tsft)

        # Get the "overall" values of the search
        Tsft = np.unique(Tsft)
        if len(Tsft) != 1:
            raise ValueError(f"SFTs contain different basetimes: {Tsft}")
        if Tsft[0] != self.Tsft:
            logger.warning(
                f"Overwriting self.Tsft={self.Tsft}"
                f" with value {Tsft[0]} read from noiseSFTs."
            )
        self.Tsft = Tsft[0]
        self.tstart = min(tstart)
        self.duration = max(tend) - self.tstart
        self.detectors = ",".join(IFOs)

    def _get_setup_from_timestamps(self):
        """
        If timestamps are given, use them to obtain relevant data parameters
        (tstart, duration; but not detectors and Tsft as in the noiseSFTs case).

        We ignore any extra columns (e.g. nanoseconds, end times)
        after the first (SFT start times)
        and implicitly assume that all SFTs are the same length.
        """
        self._parse_timestamps()
        IFOs = self.detectors.split(",")
        # at this point, it's definitely a comma-separated string
        tsfiles = self.timestamps.split(",")
        if len(IFOs) != len(tsfiles):
            raise ValueError(
                f"Length of detectors=='{self.detectors}'"
                f" does not match that of timestamps=='{self.timestamps}'"
                f" ({len(IFOs)}!={len(tsfiles)})"
            )
        tstart = []
        tend = []
        self.sftfilenames = []  # This refers to the MFD output!
        for X, IFO in enumerate(IFOs):
            tsX = np.genfromtxt(tsfiles[X], comments="%")
            if tsX.ndim > 1:
                logger.warning(
                    f"Timestamps file {tsfiles[X]} has more than 1 column,"
                    " we will ignore the rest."
                )
                tsX = tsX[:, 0]
            if not tsX[0].is_integer() or not tsX[-1].is_integer():
                logger.warning(
                    "Detected non-integer timestamps in timestamp file."
                    " We will floor start and end times to the nearest integer"
                    " for the SFT name,"
                    " and let lalpulsar_Makefakedata_v5 handle the rest."
                )

            this_start_time = int(tsX[0])
            this_end_time = int(tsX[-1]) + self.Tsft
            tstart.append(this_start_time)
            tend.append(this_end_time)
            self.sftfilenames.append(
                utils.get_official_sft_filename(
                    IFO,
                    len(tsX),
                    self.Tsft,
                    this_start_time,
                    this_end_time - this_start_time,
                    self.label,
                )
            )
        self.tstart = min(tstart)
        self.duration = max(tend) - self.tstart

    def _parse_timestamps(self):
        """
        Timestamps can be given either as a timestamp file or as an actual list
        of timestamps. The former case ignores this function, whereas the second
        one requires us to actually construct the timestamps file.
        """
        if self.detectors is None and not isinstance(self.timestamps, dict):
            raise ValueError(
                "Detector names must be given either as a key in"
                " a `timestamps` dict or explicitly via `detectors`."
            )

        if isinstance(self.timestamps, str):
            numTS = len(self.timestamps.split(","))
            numDets = len(self.detectors.split(","))
            if not numTS == numDets:
                raise ValueError(
                    "Inconsistent length of comma-separated"
                    f" `timestamps` and `detectors`: {numTS}!={numDets}"
                )
            return

        if isinstance(self.timestamps, dict):
            # Each key should correspond to `detectors` if given;
            # otherwise, construct detectors from the given keys.
            ifos = list(self.timestamps.keys())
            input_timestamps = self.timestamps.values()

            if self.detectors is not None:
                ifos_in_detectors = self.detectors.split(",")
                if np.setdiff1d(ifos, ifos_in_detectors).size:
                    raise ValueError(
                        f"Detector names in timestamps dictionary ({ifos}) "
                        f"are inconsistent with detector names given via keyword ({ifos_in_detectors})."
                    )
            else:
                self.detectors = ",".join(ifos)
        else:
            # Otherwise, assume it's a single list of timestamps,
            # and replicate it for each detector.
            ifos = self.detectors.split(",")
            input_timestamps = [self.timestamps for i in ifos]

        # If this point was reached, it means we should create timestamps files.
        timestamp_files = []
        for ind, ts in enumerate(input_timestamps):
            output_file = os.path.join(
                self.outdir, f"{self.label}_timestamps_{ifos[ind]}.csv"
            )
            np.savetxt(output_file, ts.reshape(-1, 1), fmt="%d")
            timestamp_files.append(output_file)
        self.timestamps = ",".join(timestamp_files)

    def _basic_setup(self):
        """Basic parameters handling, path setup etc."""

        if not self.label.isalnum():
            raise ValueError(
                f"Label '{self.label}' is not alphanumeric,"
                " which is incompatible with the SFTv3 naming specification"
                " ( https://dcc.ligo.org/T040164-v2/public )."
                " Please avoid underscores, hyphens etc."
            )
        if len(self.label) > 60:
            raise ValueError(
                f"Label {self.label} is too long to comply with SFT naming rules"
                f" ({len(self.label)}>60)."
            )

        os.makedirs(self.outdir, exist_ok=True)
        self.config_file_name = os.path.join(self.outdir, self.label + ".cff")
        self.theta = np.array([self.phi, self.F0, self.F1, self.F2])

        if self.h0 and np.any(
            [getattr(self, k, None) is None for k in self.required_signal_parameters]
        ):
            raise ValueError(
                "If h0>0, also need all of ({:s})".format(
                    ",".join(self.required_signal_parameters)
                )
            )

        incompatible_with_TS = ["tstart", "duration", "noiseSFTs"]
        TS_required_options = ["Tsft"]
        no_noiseSFTs_options = ["tstart", "duration", "Tsft", "detectors"]

        if getattr(self, "timestamps", None) is not None:
            if np.any(
                [getattr(self, k, None) is not None for k in incompatible_with_TS]
            ):
                raise ValueError(
                    "timestamps option is incompatible with"
                    f" ({','.join(incompatible_with_TS)})."
                )
            if np.any([getattr(self, k, None) is None for k in TS_required_options]):
                raise ValueError(
                    "With timestamps option, need also all of"
                    f" ({','.join(TS_required_options)})."
                )
            self._get_setup_from_timestamps()
        elif self.noiseSFTs is not None:
            logger.info(
                "noiseSFTs is not None: Inferring tstart, duration, Tsft. "
                "Input tstart and duration will be treated as SFT constraints "
                "using lalpulsar.SFTConstraints; Tsft will be checked for "
                "internal consistency accross input SFTs."
            )
            self._get_setup_from_noiseSFTs()
        elif np.any([getattr(self, k, None) is None for k in no_noiseSFTs_options]):
            raise ValueError(
                "Need either noiseSFTs, timestamps or all of ({:s}).".format(
                    ",".join(no_noiseSFTs_options)
                )
            )
        else:
            self._get_setup_from_tstart_duration()

        self.sftfilenames = [os.path.join(self.outdir, fn) for fn in self.sftfilenames]
        self.sftfilepath = ";".join(self.sftfilenames)

        if self.tref is None:
            self.tref = self.tstart

        if getattr(self, "SFTWindowBeta", None):
            raise ValueError(
                "Option 'SFTWindowBeta' is defunct, please use 'SFTWindowParam'."
            )
        if getattr(self, "SFTWindowType", None):
            try:
                lal.CheckNamedWindow(
                    self.SFTWindowType, self.SFTWindowParam is not None
                )
            except RuntimeError:
                raise ValueError(
                    "XLAL error on checking SFT window options."
                    f" Likely either SFTWindowType={self.SFTWindowType} is not a recognised window name,"
                    " or it requires also setting an SFTWindowParam."
                )

    @property
    def tend(self):
        """`
        Defined as `self.start + self.duration`.

        If stored as an attribute, there would be the risk of it going out of
        sync with the other two values.
        """
        return self.tstart + self.duration

    def _parse_args_consistent_with_mfd(self):
        """Internal method to ensure parameters are handled consistently with MFD."""
        self.signal_parameters = self.translate_keys_to_lal(
            {
                key: self.__dict__[key]
                for key in self.signal_parameter_labels
                if self.__dict__.get(key, None) is not None
            }
        )

        self.signal_formats = {
            key: self.gps_time_and_string_formats_as_LAL.get(key, ":1.18e")
            for key in self.signal_parameters
        }

    def calculate_fmin_Band(self):
        """Set fmin and Band for the output SFTs to cover.

        Either uses the user-provided `Band` and puts `F0` in the middle;
        does nothing to later reuse the full bandwidth of `noiseSFTs`
        (only if using MFDv5);
        or if `F0!=None`, `noiseSFTs=None` and `Band=None`
        it estimates a minimal band for just the injected signal:
        F-stat covering band plus extra bins for demod default parameters.
        This way a perfectly matched single-template `ComputeFstat` analysis
        should run through perfectly on the returned SFTs.
        For any wider-band or mismatched search, one needs to set `Band` manually.
        If using MFDv4, at least `F0` is required even if `noiseSFTs!=None`.
        """
        if self.F0 is None and self.Band is not None:
            raise ValueError("Band option can only be set if F0 is also given.")
        elif self.F0 is not None and self.Band is not None:
            self.fmin = self.F0 - 0.5 * self.Band
        elif self.noiseSFTs and not self.mfd.endswith("v4"):
            logger.info("Generating SFTs with full bandwidth from noiseSFTs.")
        elif self.F0 is None:
            err_msg = "Need F0 and Band,"
            if not self.mfd.endswith("v4"):
                err_msg += " or noiseSFTs,"
            err_msg += " or at least F0 to auto-estimate bandwidth around it."
            if self.mfd.endswith("v4"):
                err_msg += (
                    f" Since we are using {self.mfd}, we need this even with noiseSFTs."
                )
            raise ValueError(err_msg)
        else:
            extraBins = (
                # matching extraBinsFull in XLALCreateFstatInput():
                # https://lscsoft.docs.ligo.org/lalsuite/lalpulsar/_compute_fstat_8c_source.html#l00490
                lalpulsar.FstatOptionalArgsDefaults.Dterms
                + int(lalpulsar.FstatOptionalArgsDefaults.runningMedianWindow / 2)
                + 1
            )
            logger.info(
                "Estimating required SFT frequency range from properties"
                " of signal to inject plus {:d} extra bins either side"
                " (corresponding to default F-statistic settings).".format(extraBins)
            )
            minCoverFreq, maxCoverFreq = utils.get_covering_band(
                tref=self.tref,
                tstart=self.tstart,
                tend=self.tend,
                F0=self.F0,
                F1=self.F1,
                F2=self.F2,
                F0band=0.0,
                F1band=0.0,
                F2band=0.0,
                maxOrbitAsini=getattr(self, "asini", 0.0),
                minOrbitPeriod=getattr(self, "period", 0.0),
                maxOrbitEcc=getattr(self, "ecc", 0.0),
            )
            self.fmin = minCoverFreq - extraBins / self.Tsft
            self.Band = maxCoverFreq - minCoverFreq + 2 * extraBins / self.Tsft
        if hasattr(self, "fmin"):
            logger.info(
                "Generating SFTs with fmin={}, Band={}".format(self.fmin, self.Band)
            )

    def _get_single_config_line(self, i):
        """Formatting for signal injection parameters."""
        config_line = "[TS{}]\n".format(i)
        config_line += "\n".join(
            [
                "{} = {{{}}}".format(key, self.signal_formats[key]).format(
                    self.signal_parameters[key]
                )
                for key in self.signal_parameters.keys()
            ]
        )
        config_line += "\n"

        return config_line

    def make_cff(self, verbose=False):
        """Generates a .cff file including signal injection parameters.

        This will be saved to `self.config_file_name`.

        Parameters
        ----------
        verbose: boolean
            If true, increase logging verbosity.
        """

        content = self._get_single_config_line(0)

        if verbose:
            logger.info("Injection parameters:")
            logger.info(content.rstrip("\n"))

        if self._check_if_cff_file_needs_rewriting(content):
            logger.info("Writing config file: {:s}".format(self.config_file_name))
            config_file = open(self.config_file_name, "w+")
            config_file.write(content)
            config_file.close()

    def check_cached_data_okay_to_use(self, cl_mfd):
        """Check if SFT files already exist that can be re-used.

        This does not check the actual data contents of the SFTs,
        but only the following criteria:

         * filename

         * if injecting a signal, that the .cff file is older than the SFTs
           (but its contents should have been checked separately)

         * that the commandline stored in the (first) SFT header matches

        Parameters
        ----------
        cl_mfd: str
            The commandline we'd execute if not finding matching files.
        """

        need_new = "Will create new SFT file(s)."

        logger.info("Checking if we can re-use existing SFT data file(s)...")
        for sftfile in self.sftfilenames:
            if os.path.isfile(sftfile) is False:
                logger.info(
                    "...no SFT file matching '{}' found. {}".format(sftfile, need_new)
                )
                return False
        logger.info("...OK: file(s) found matching '{}'.".format(sftfile))

        if os.path.isfile(self.config_file_name):
            if np.any(
                [
                    os.path.getmtime(sftfile) < os.path.getmtime(self.config_file_name)
                    for sftfile in self.sftfilenames
                ]
            ):
                logger.info(
                    (
                        "...the config file '{}' has been modified since"
                        " creation of the SFT file(s) '{}'. {}"
                    ).format(self.config_file_name, self.sftfilepath, need_new)
                )
                return False
            else:
                logger.info(
                    "...OK: The config file '{}' is older than the SFT file(s)"
                    " '{}'.".format(self.config_file_name, self.sftfilepath)
                )
                # NOTE: at this point we assume it's safe to re-use, since
                # _check_if_cff_file_needs_rewriting()
                # should have already been called before
        elif "injectionSources" in cl_mfd:
            raise RuntimeError(
                "Commandline requires file '{}' but it is missing.".format(
                    self.config_file_name
                )
            )

        logger.info("...checking new commandline against existing SFT header(s)...")
        # here we check one SFT header from each SFT file,
        # assuming that any concatenated file has been sanely constructed with
        # matching CLs
        for sftfile in self.sftfilenames:
            catalog = lalpulsar.SFTdataFind(sftfile, None)
            cl_old = utils.get_commandline_from_SFTDescriptor(catalog.data[0])
            if len(cl_old) == 0:
                logger.info(
                    "......could not obtain comparison commandline from first SFT"
                    " header in old file '{}'. {}".format(sftfile, need_new)
                )
                return False
            if not utils.match_commandlines(cl_old, cl_mfd):
                logger.info(
                    "......commandlines unmatched for first SFT in old"
                    " file '{}':".format(sftfile)
                )
                logger.info(cl_old)
                logger.info(cl_mfd)
                logger.info(need_new)
                return False
        logger.info("......OK: Commandline matched with old SFT header(s).")
        logger.info(
            "...all data consistency checks passed: Looks like existing"
            " SFT data matches current options, will re-use it!"
        )
        return True

    def _check_if_cff_file_needs_rewriting(self, content):
        """Check if the .cff file has changed.

        Returns True if the file should be overwritten - where possible avoid
        overwriting to allow cached data to be used
        """
        logger.info("Checking if we can re-use injection config file...")
        if os.path.isfile(self.config_file_name) is False:
            logger.info("...no config file {} found.".format(self.config_file_name))
            return True
        else:
            logger.info(
                "...OK: config file {} already exists.".format(self.config_file_name)
            )

        with open(self.config_file_name, "r") as f:
            file_content = f.read()
            if file_content == content:
                logger.info(
                    "...OK: file contents match, no update of {} required.".format(
                        self.config_file_name
                    )
                )
                return False
            else:
                logger.info(
                    "...file contents unmatched, updating {}.".format(
                        self.config_file_name
                    )
                )
                return True

    def make_data(self, verbose=False):
        """A convenience wrapper to generate a cff file and then SFTs."""
        if self.h0:
            self.make_cff(verbose)
        else:
            logger.info("Got h0=0, not writing an injection .cff file.")
        self.run_makefakedata()

    def run_makefakedata(self):
        """Generate the SFT data calling Makefakedata_v5 executable.

        This first builds the full commandline,
        then calls `check_cached_data_okay_to_use()`
        to see if equivalent data files already exist,
        and else runs the actual generation code.
        """
        cl_mfd = self._build_MFD_command_line()

        check_ok = self.check_cached_data_okay_to_use(cl_mfd)
        if check_ok is False:
            utils.run_commandline(cl_mfd)
            if not np.all([os.path.isfile(f) for f in self.sftfilenames]):
                raise IOError(
                    f"It seems we successfully ran {self.mfd},"
                    f" but did not get the expected SFT file path(s): {self.sftfilepath}."
                    f" What we have in the output directory '{self.outdir}' is:"
                    f" {os.listdir(self.outdir)}"
                )
            logger.info(f"Successfully wrote SFTs to: {self.sftfilepath}")
            logger.info("Now validating each SFT file...")
            for sft in self.sftfilenames:
                lalpulsar.ValidateSFTFile(sft)

    def _build_MFD_command_line(self):
        cl_mfd = [self.mfd]
        cl_mfd.append("--outSingleSFT=TRUE")
        cl_mfd.append('--outSFTdir="{}"'.format(self.outdir))
        cl_mfd.append('--outLabel="{}"'.format(self.label))

        if self.noiseSFTs is not None:
            if self.sqrtSX and np.any(
                [s > 0 for s in utils.parse_list_of_numbers(self.sqrtSX)]
            ):
                logger.warning(
                    "In addition to using noiseSFTs, you are adding "
                    "Gaussian noise with sqrtSX={} "
                    "Please, make sure this is what you intend to do.".format(
                        self.sqrtSX
                    )
                )
            cl_mfd.append('--noiseSFTs="{}"'.format(self.noiseSFTs))
        else:
            cl_mfd.append(
                "--IFOs={}".format(
                    ",".join(['"{}"'.format(d) for d in self.detectors.split(",")])
                )
            )
        if self.sqrtSX:
            cl_mfd.append('--sqrtSX="{}"'.format(self.sqrtSX))

        if self.SFTWindowType is not None:
            cl_mfd.append('--SFTWindowType="{}"'.format(self.SFTWindowType))
            cl_mfd.append("--SFTWindowParam={}".format(self.SFTWindowParam))
        if getattr(self, "timestamps", None) is not None:
            cl_mfd.append("--timestampsFiles={}".format(self.timestamps))
        else:
            cl_mfd.append("--startTime={}".format(self.tstart))
            cl_mfd.append("--duration={}".format(self.duration))
        if hasattr(self, "fmin") and self.fmin:
            cl_mfd.append("--fmin={:.16g}".format(self.fmin))
        if hasattr(self, "Band") and self.Band:
            cl_mfd.append("--Band={:.16g}".format(self.Band))
        cl_mfd.append("--Tsft={}".format(self.Tsft))
        if self.h0:
            cl_mfd.append('--injectionSources="{}"'.format(self.config_file_name))
        earth_ephem = getattr(self, "earth_ephem", None)
        sun_ephem = getattr(self, "sun_ephem", None)
        if earth_ephem is not None:
            cl_mfd.append('--ephemEarth="{}"'.format(earth_ephem))
        if sun_ephem is not None:
            cl_mfd.append('--ephemSun="{}"'.format(sun_ephem))
        if self.randSeed:
            cl_mfd.append("--randSeed={}".format(self.randSeed))

        return " ".join(cl_mfd)

    def predict_fstat(self, assumeSqrtSX=None):
        """Predict the expected F-statistic value for the injection parameters.

        Through utils.predict_fstat(), this wraps
        the PredictFstat executable.

        Parameters
        ----------
        assumeSqrtSX: float, str or None
            If None, PSD is estimated from self.sftfilepath.
            Else, assume this stationary per-detector noise-floor instead.
            Single float or str value: use same for all IFOs.
            Comma-separated string: must match len(self.detectors)
            and the data in self.sftfilepath.
            Detectors will be paired to list elements following alphabetical order.
        """
        twoF_expected, twoF_sigma = utils.predict_fstat(
            h0=self.h0,
            cosi=self.cosi,
            psi=self.psi,
            Alpha=self.Alpha,
            Delta=self.Delta,
            F0=self.F0,
            sftfilepattern=self.sftfilepath,
            minStartTime=self.tstart,
            duration=self.duration,
            IFOs=self.detectors,
            assumeSqrtSX=(assumeSqrtSX or self.sqrtSX),
            tempory_filename=os.path.join(self.outdir, self.label + ".tmp"),
            earth_ephem=self.earth_ephem,
            sun_ephem=self.sun_ephem,
            transientWindowType=self.transientWindowType,
            transientStartTime=self.transientStartTime,
            transientTau=self.transientTau,
        )
        return twoF_expected


class BinaryModulatedWriter(Writer):
    """Special Writer variant for simulating a CW signal for a source in a binary system."""

    @utils.initializer
    def __init__(
        self,
        label="PyFstat",
        tstart=None,
        duration=None,
        tref=None,
        F0=None,
        F1=0,
        F2=0,
        Alpha=None,
        Delta=None,
        tp=0.0,
        argp=0.0,
        asini=0.0,
        ecc=0.0,
        period=0.0,
        h0=None,
        cosi=None,
        psi=0.0,
        phi=0,
        Tsft=1800,
        outdir=".",
        sqrtSX=None,
        noiseSFTs=None,
        SFTWindowType=None,
        SFTWindowParam=None,
        SFTWindowBeta=None,
        Band=None,
        detectors=None,
        earth_ephem=None,
        sun_ephem=None,
        transientWindowType="none",
        transientStartTime=None,
        transientTau=None,
        randSeed=None,
        timestamps=None,
    ):
        """
        Most parameters are the same as for the basic `Writer` class,
        only the additional ones are documented here:

        Parameters
        ----------
        tp, argp, asini, ecc, period:
            binary orbit parameters
        """
        self.signal_parameter_labels = super().signal_parameter_labels + [
            "tp",
            "argp",
            "asini",
            "ecc",
            "period",
        ]
        self.gps_time_and_string_formats_as_LAL["orbitTp"] = ":10.9f"

        super().__init__(
            label=label,
            tstart=tstart,
            duration=duration,
            tref=tref,
            F0=F0,
            F1=F1,
            F2=F2,
            Alpha=Alpha,
            Delta=Delta,
            h0=h0,
            cosi=cosi,
            psi=psi,
            phi=phi,
            Tsft=Tsft,
            outdir=outdir,
            sqrtSX=sqrtSX,
            SFTWindowType=SFTWindowType,
            SFTWindowParam=SFTWindowParam,
            SFTWindowBeta=SFTWindowBeta,
            noiseSFTs=noiseSFTs,
            Band=Band,
            detectors=detectors,
            earth_ephem=earth_ephem,
            sun_ephem=sun_ephem,
            transientWindowType=transientWindowType,
            transientStartTime=transientStartTime,
            transientTau=transientTau,
        )


class LineWriter(Writer):
    """Inject a simulated line-like detector artifact into SFT data.

    A (transient) line is defined as a constant amplitude and constant excess power artifact in the data.

    In practice, it corresponds to a CW without Doppler or antenna-pattern-induced amplitude modulation.

    NOTE: This functionality is implemented via `Makefakedata_v4`'s `lineFeature` option.
    This version of MFD only supports one interferometer at a time.

    NOTE: All signal parameters except for `h0`, `Freq`, `phi0` and transient parameters will be ignored.
    """

    mfd = "lalpulsar_Makefakedata_v4"
    """The executable (older version that supports the `--lineFeature` option)."""

    required_signal_parameters = [
        "F0",
        "phi",
        "h0",
    ]
    """Required parameters for Makefakedata_v4 to success. Any other parameter is
    silently given a default value by Makefakedata_v4.
    """
    signal_parameters_labels = required_signal_parameters + [
        "transientWindowType",
        "transientStartTime",
        "transientTau",
    ]
    """Other signal parameters will be removed before passing to Makefakedata_v4."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.detectors is None:
            raise ValueError("Makefakedata_v4 requires detector name to be given")
        elif len(self.detectors.split(",")) > 1:
            raise NotImplementedError(
                "Makefakedata_v4 does not support more than one detector at a time. "
                "Multi-detector behaviour can be reproduced by calling the procedure "
                "on single-detector SFT sets once at a time."
            )

    def _parse_args_consistent_with_mfd(self):
        """
        Adapt input arguments.
        Take care of minor inconsistencies between MFD_v4 and MFD_v5
        """
        super()._parse_args_consistent_with_mfd()

        # FIXME: There should be a smoother way to translate keys
        lal_required_signal_parameters = self.translate_keys_to_lal(
            dict(
                zip(
                    self.required_signal_parameters,
                    [0] * len(self.required_signal_parameters),
                )
            )
        )

        if any(
            key not in lal_required_signal_parameters for key in self.signal_parameters
        ):
            logger.warning(
                "Injection of line artifacts only uses the following parameters:\n"
                f"{self.required_signal_parameters}.\n"
                "Any other parameter will be purged from this class now"
            )
            params_to_purge = list(
                set(self.signal_parameters) - set(lal_required_signal_parameters)
            )
            logger.info(
                "Purging input parameters that are not meaningful for LineWriter: {}".format(
                    params_to_purge
                )
            )
            for key in params_to_purge:
                self.signal_parameters.pop(key)

        if "transientTau" in self.signal_parameters:
            self.signal_parameters["transientTauDays"] = (
                self.signal_parameters.pop("transientTau") / 86400.0
            )
            self.signal_formats["transientTauDays"] = self.signal_formats.pop(
                "transientTau"
            )

    def _build_MFD_command_line(self):
        """Generate the SFT data calling Makefakedata_v4."""

        cl_mfd = [self.mfd]

        cl_mfd.append("--lineFeature=TRUE")
        cl_mfd.append("--outSingleSFT=TRUE")
        cl_mfd.append('--outSFTbname="{}"'.format(self.sftfilenames[0]))
        cl_mfd.append('--IFO="{}"'.format(self.detectors))

        if self.noiseSFTs is not None and self.SFTWindowType is None:
            raise ValueError(
                "SFTWindowType is required when using noiseSFTs. "
                "Please, make sure you understand the window function used "
                "to produce noiseSFTs."
            )
        elif self.noiseSFTs is not None:
            if self.sqrtSX and np.any(
                [s > 0 for s in utils.parse_list_of_numbers(self.sqrtSX)]
            ):
                logger.warning(
                    "In addition to using noiseSFTs, you are adding "
                    "Gaussian noise with sqrtSX={} "
                    "Please, make sure this is what you intend to do.".format(
                        self.sqrtSX
                    )
                )
            cl_mfd.append('--noiseSFTs="{}"'.format(self.noiseSFTs))
        if self.sqrtSX:
            cl_mfd.append("--noiseSqrtSh={}".format(self.sqrtSX))

        if self.SFTWindowType is not None:
            cl_mfd.append('--window="{}"'.format(self.SFTWindowType))
            cl_mfd.append("--windowParam={}".format(self.SFTWindowParam))
        cl_mfd.append("--startTime={}".format(self.tstart))
        cl_mfd.append("--duration={}".format(self.duration))
        if getattr(self, "fmin", None):
            cl_mfd.append("--fmin={:.16g}".format(self.fmin))
        if getattr(self, "Band", None):
            cl_mfd.append("--Band={:.16g}".format(self.Band))
        cl_mfd.append("--Tsft={}".format(self.Tsft))
        if self.h0:
            cl_mfd.append(
                " ".join(
                    f"--{key}={value:.16g}"
                    for key, value in self.signal_parameters.items()
                )
            )
            cl_mfd.append("--cosi=0")  # Required by MFDv4

        earth_ephem = getattr(self, "earth_ephem", None)
        sun_ephem = getattr(self, "sun_ephem", None)
        if earth_ephem is not None:
            cl_mfd.append('--ephemEarth="{}"'.format(earth_ephem))
        if sun_ephem is not None:
            cl_mfd.append('--ephemSun="{}"'.format(sun_ephem))
        if self.randSeed:
            cl_mfd.append("--randSeed={}".format(self.randSeed))

        return " ".join(cl_mfd)


class GlitchWriter(SearchForSignalWithJumps, Writer):
    """Special Writer variant for simulating a CW signal containing a timing glitch."""

    @utils.initializer
    def __init__(
        self,
        label="PyFstat",
        tstart=None,
        duration=None,
        dtglitch=None,
        delta_phi=0,
        delta_F0=0,
        delta_F1=0,
        delta_F2=0,
        tref=None,
        F0=None,
        F1=0,
        F2=0,
        Alpha=None,
        Delta=None,
        h0=None,
        cosi=None,
        psi=0.0,
        phi=0,
        Tsft=1800,
        outdir=".",
        sqrtSX=None,
        noiseSFTs=None,
        SFTWindowType=None,
        SFTWindowParam=None,
        SFTWindowBeta=None,
        Band=None,
        detectors=None,
        earth_ephem=None,
        sun_ephem=None,
        transientWindowType="rect",
        randSeed=None,
        timestamps=None,
    ):
        """
        Most parameters are the same as for the basic `Writer` class,
        only the additional ones are documented here:

        Parameters
        ----------
        dtglitch: float or None
            Time (in GPS seconds) of the glitch after `tstart`.
            To create data without a glitch, set `dtglitch=None`.
        delta_phi, delta_F0, delta_F1: float
            Instantaneous glitch magnitudes in rad, Hz, and Hz/s respectively.
        """

        self.set_ephemeris_files(earth_ephem, sun_ephem)
        self._basic_setup()
        self.calculate_fmin_Band()

        shapes = np.array(
            [
                np.shape(x)
                for x in [self.delta_phi, self.delta_F0, self.delta_F1, self.delta_F2]
            ]
        )
        if not np.all(shapes == shapes[0]):
            raise ValueError("all delta_* must be the same shape: {}".format(shapes))

        for d in self.delta_phi, self.delta_F0, self.delta_F1, self.delta_F2:
            if np.size(d) == 1:
                d = np.atleast_1d(d)

        if self.dtglitch is None:
            self.tbounds = [self.tstart, self.tend]
        else:
            self.dtglitch = np.atleast_1d(self.dtglitch)
            self.tglitch = self.tstart + self.dtglitch
            self.tbounds = np.concatenate(([self.tstart], self.tglitch, [self.tend]))
        logger.info("Using segment boundaries {}".format(self.tbounds))

        tbs = np.array(self.tbounds)
        self.durations = tbs[1:] - tbs[:-1]

        self.delta_thetas = np.atleast_2d(
            np.array([delta_phi, delta_F0, delta_F1, delta_F2]).T
        )

    def _get_base_template(self, i, Alpha, Delta, h0, cosi, psi, phi, F0, F1, F2, tref):
        """FIXME: ported over from Writer,
        should be replaced by a more elegant re-use of _parse_args_consistent_with_mfd
        """
        return """[TS{}]
Alpha = {:1.18e}
Delta = {:1.18e}
h0 = {:1.18e}
cosi = {:1.18e}
psi = {:1.18e}
phi0 = {:1.18e}
Freq = {:1.18e}
f1dot = {:1.18e}
f2dot = {:1.18e}
refTime = {:10.6f}"""

    def _get_single_config_line_cw(
        self, i, Alpha, Delta, h0, cosi, psi, phi, F0, F1, F2, tref
    ):
        template = (
            self._get_base_template(
                i, Alpha, Delta, h0, cosi, psi, phi, F0, F1, F2, tref
            )
            + """\n"""
        )
        return template.format(i, Alpha, Delta, h0, cosi, psi, phi, F0, F1, F2, tref)

    def _get_single_config_line_tcw(
        self,
        i,
        Alpha,
        Delta,
        h0,
        cosi,
        psi,
        phi,
        F0,
        F1,
        F2,
        tref,
        window,
        transientStartTime,
        transientTau,
    ):
        template = (
            self._get_base_template(
                i, Alpha, Delta, h0, cosi, psi, phi, F0, F1, F2, tref
            )
            + """
transientWindowType = {:s}
transientStartTime = {:10.0f}
transientTau = {:10.0f}\n"""
        )
        return template.format(
            i,
            Alpha,
            Delta,
            h0,
            cosi,
            psi,
            phi,
            F0,
            F1,
            F2,
            tref,
            window,
            transientStartTime,
            transientTau,
        )

    def _get_single_config_line(
        self,
        i,
        Alpha,
        Delta,
        h0,
        cosi,
        psi,
        phi,
        F0,
        F1,
        F2,
        tref,
        window,
        transientStartTime,
        transientTau,
    ):
        if window == "none":
            return self._get_single_config_line_cw(
                i, Alpha, Delta, h0, cosi, psi, phi, F0, F1, F2, tref
            )
        else:
            return self._get_single_config_line_tcw(
                i,
                Alpha,
                Delta,
                h0,
                cosi,
                psi,
                phi,
                F0,
                F1,
                F2,
                tref,
                window,
                transientStartTime,
                transientTau,
            )

    def make_cff(self, verbose=False):
        """Generates a .cff file including signal injection parameters, including a glitch.

        This will be saved to `self.config_file_name`.

        Parameters
        ----------
        verbose: boolean
            If true, increase logging verbosity.
        """

        thetas = self._calculate_thetas(self.theta, self.delta_thetas, self.tbounds)

        content = ""
        for i, (t, d, ts) in enumerate(zip(thetas, self.durations, self.tbounds[:-1])):
            line = self._get_single_config_line(
                i,
                self.Alpha,
                self.Delta,
                self.h0,
                self.cosi,
                self.psi,
                t[0],
                t[1],
                t[2],
                t[3],
                self.tref,
                self.transientWindowType,
                ts,
                d,
            )

            content += line

        if verbose:
            logger.info("Injection parameters:")
            logger.info(content.rstrip("\n"))

        if self._check_if_cff_file_needs_rewriting(content):
            logger.info("Writing config file: {:s}".format(self.config_file_name))
            config_file = open(self.config_file_name, "w+")
            config_file.write(content)
            config_file.close()


class FrequencyModulatedArtifactWriter(Writer):
    """Specialized Writer variant to generate SFTs containing simulated instrumental artifacts.

    Contrary to the main `Writer` class, this calls the older
    `Makefakedata_v4` executable which supports the special `--lineFeature` option.
    See `lalpulsar_Makefakedata_v4 --help`
    for more detailed help with some of the parameters.
    """

    @utils.initializer
    def __init__(
        self,
        label,
        outdir=".",
        tstart=700000000,
        duration=86400,
        F0=30,
        F1=0,
        tref=None,
        h0=10,
        Tsft=1800,
        sqrtSX=0.0,
        Band=4,
        Pmod=lal.DAYSID_SI,
        Pmod_phi=0,
        Pmod_amp=1,
        Alpha=None,
        Delta=None,
        detectors=None,
        earth_ephem=None,
        sun_ephem=None,
        randSeed=None,
    ):
        """
        Parameters
        ----------
        label: string
            A human-readable label to be used in naming the output files.
        outdir: str
            The directory where files are written to.
            Default: current working directory.
        tstart: int
            Starting GPS epoch of the data set.
        duration: int
            Duration (in GPS seconds) of the total data set.
        F0: float
            Frequency of the artifact.
        F1: float
            Frequency drift of the artifact.
        tref: float or None
            Reference time for simulated signals.
            Default is `None`, which sets the reference time to `tstart`.
        h0: float
            Amplitude of the artifact.
        Tsft: int
            The SFT duration in seconds.
            Will be ignored if `noiseSFTs` are given.
        sqrtSX: float
            Background detector noise level.
        Band: float
            Output SFTs cover
            `[F0-Band/2,F0+Band/2]`.
        Pmod: float
            Modulation period of the artifact.
        Pmod_phi, Pmod_amp: float
            Additional parameters for modulation of the artifact.
        Alpha, Delta: float or None
            If not none: add an orbital modulation to the artifact
            corresponding to a signal from that sky position, in radians.
        detectors: str or None
            Comma-separated list of detectors to generate data for.
        earth_ephem, sun_ephem: str or None
            Paths of the two files containing positions of Earth and Sun.
            If None, will check standard sources as per
            utils.get_ephemeris_files().
        randSeed: int or None
            Optionally fix the random seed of Gaussian noise generation
            for reproducibility.
        """

        self.phi = 0
        self.F2 = 0

        self.cosi = 0
        self.noiseSFTs = None

        if type(self.detectors) is not str or len(self.detectors.split(",")) > 1:
            raise ValueError("'detectors' must be  a single-IFO string")

        # The _basic_setup() method inherited from Writer
        # requires additional CW signal parameters if h0>0,
        # which an artifact doesn't need to have,
        # hence we temporarily unset it here as a workaround
        # and restore it after the method call.
        h0_value = self.h0
        self.h0 = None
        self._basic_setup()
        self.h0 = h0_value

        self.set_ephemeris_files(earth_ephem, sun_ephem)
        self.tstart = int(tstart)
        self.duration = int(duration)

        if os.path.isdir(self.outdir) is False:
            os.makedirs(self.outdir)
        if tref is None:
            raise ValueError("Input `tref` not specified")

        self.nsfts = int(np.ceil(self.duration / self.Tsft))
        self.calculate_fmin_Band()

        self.Fmax = F0

        if Alpha is not None and Delta is not None:
            self.n = np.array(
                [
                    np.cos(Alpha) * np.cos(Delta),
                    np.sin(Alpha) * np.cos(Delta),
                    np.sin(Delta),
                ]
            )

    def get_frequency(self, t):
        """Evolve the artifact frequency in time.

        This includes a drift term and optionally,
        if `Alpha` and `Delta` are not `None`,
        a simulated orbital modulation.

        Parameters
        ----------
        t: float
            Time stamp to evaluate the frequency at.

        Returns
        -------
        f: float
            Frequency at time `t`.
        """
        DeltaFDrift = self.F1 * (t - self.tref)

        phir = 2 * np.pi * t / self.Pmod + self.Pmod_phi

        if self.Alpha is not None and self.Delta is not None:
            spin_posvel = lalpulsar.PosVel3D_t()
            orbit_posvel = lalpulsar.PosVel3D_t()
            det = lal.CachedDetectors[4]
            ephems = lalpulsar.InitBarycenter(self.earth_ephem, self.sun_ephem)
            lalpulsar.DetectorPosVel(
                spin_posvel,
                orbit_posvel,
                lal.LIGOTimeGPS(t),
                det,
                ephems,
                lalpulsar.DETMOTION_ORBIT,
            )
            # Pos and vel returned in units of c
            DeltaFOrbital = np.dot(self.n, orbit_posvel.vel) * self.Fmax

            if self.detectors == "H1":
                Lambda = lal.LHO_4K_DETECTOR_LATITUDE_RAD
            elif self.detectors == "L1":
                Lambda = lal.LLO_4K_DETECTOR_LATITUDE_RAD
            else:
                raise ValueError(
                    "Simulated orbital modulation"
                    " (triggered by Alpha, Delta parameters)"
                    " is currently only supported for detectors H1 or L1."
                )

            DeltaFSpin = (
                self.Pmod_amp
                * lal.REARTH_SI
                / lal.C_SI
                * 2
                * np.pi
                / self.Pmod
                * (np.cos(self.Delta) * np.cos(Lambda) * np.sin(self.Alpha - phir))
                * self.Fmax
            )
        else:
            DeltaFOrbital = 0
            DeltaFSpin = 2 * np.pi * self.Pmod_amp / self.Pmod * np.cos(phir)

        f = self.F0 + DeltaFDrift + DeltaFOrbital + DeltaFSpin
        return f

    def get_h0(self, t):
        """Evaluate the artifact amplitude at a given time.

        NOTE: Here it's actually implemented as a constant!

        Parameters
        ----------
        t: float
            Time stamp to evaluate at.

        Returns
        -------
        h0: float
            Amplitude at time `t`.
        """
        return self.h0

    def concatenate_sft_files(self):
        """Merges the individual SFT files via splitSFTs executable."""

        SFTFilename = (
            f"{self.detectors[0]}-{self.nsfts}_{self.detectors}_{self.Tsft}SFT_mfdv4"
        )
        # We don't try to reproduce the NB filename convention exactly,
        # as there could be always rounding offsets with the number of bins,
        # instead we use wildcards there.
        outfreq = int(np.floor(self.fmin))
        outwidth = int(np.floor(self.Band))
        SFTFilename += f"_NBF{outfreq:04d}Hz*W{outwidth:04d}Hz*"
        SFTFilename += f"-{self.tstart}-{self.duration}.sft"
        SFTFile_fullpath = os.path.join(self.outdir, SFTFilename)
        if os.path.isfile(SFTFile_fullpath):
            logger.info(
                f"Removing previous file(s) {SFTFile_fullpath} (no caching implemented)."
            )
            os.remove(SFTFile_fullpath)

        inpattern = os.path.join(self.tmp_outdir, "*sft")
        cl_splitSFTS = "lalpulsar_splitSFTs"
        cl_splitSFTS += " -fs {} -fb {} -fe {} -n {} -- {}".format(
            self.fmin, self.Band, self.fmin + self.Band, self.outdir, inpattern
        )
        utils.run_commandline(cl_splitSFTS)
        utils.run_commandline(f"rm -r {self.tmp_outdir}")
        outglob = glob.glob(SFTFile_fullpath)
        if len(outglob) != 1:
            raise IOError(
                "Expected to produce exactly 1 merged file"
                f" matching pattern '{SFTFile_fullpath}',"
                f" but got {len(outglob)} matches: {outglob}"
                " Something went wrong!"
            )
        self.sftfilepath = outglob[0]
        logger.info(f"Successfully wrote SFTs to: {self.sftfilepath}")

    def pre_compute_evolution(self):
        """Precomputes evolution parameters for the artifact.

        This computes midtimes, frequencies, phases and amplitudes
        over the list of SFT timestamps.
        """
        logger.info("Precomputing evolution parameters")
        self.lineFreqs = []
        self.linePhis = []
        self.lineh0s = []
        self.mid_times = []

        linePhi = 0
        lineFreq_old = 0

        for i in trange(self.nsfts):
            mid_time = self.tstart + (i + 0.5) * self.Tsft
            lineFreq = self.get_frequency(mid_time)

            self.mid_times.append(mid_time)
            self.lineFreqs.append(lineFreq)
            self.linePhis.append(
                linePhi + np.pi * self.Tsft * (lineFreq_old + lineFreq)
            )
            self.lineh0s.append(self.get_h0(mid_time))

            lineFreq_old = lineFreq

    def make_ith_sft(self, i):
        """Call MFDv4 to create a single SFT with evolved artifact parameters."""
        self.run_makefakedata_v4(
            self.mid_times[i],
            self.lineFreqs[i],
            self.linePhis[i],
            self.lineh0s[i],
            self.tmp_outdir,
        )

    def make_data(self, num_threads=1):
        """Create a full multi-SFT data set.

        This loops over SFTs and generate them serially or in parallel,
        then contatenates the results together at the end.

        Parameters
        ----------
        num_processes: int
            Number threads to use when running in parallel.
            Verbatim implementation of the former `args.N`.
        """

        self.tmp_outdir = os.path.join(self.outdir, self.label + "_tmp")
        if os.path.isdir(self.tmp_outdir) is True:
            raise ValueError(
                "Temporary directory {} already exists, please rename".format(
                    self.tmp_outdir
                )
            )
        else:
            os.makedirs(self.tmp_outdir)

        self.pre_compute_evolution()

        logger.info("Generating SFTs")

        if num_threads > 1 and pkgutil.find_loader("pathos") is not None:
            import pathos.pools

            logger.info("Using {} threads".format(num_threads))
            try:
                with pathos.pools.ProcessPool(num_threads) as p:
                    list(
                        tqdm(
                            p.imap(self.make_ith_sft, list(range(self.nsfts))),
                            total=self.nsfts,
                        )
                    )
            except KeyboardInterrupt:
                p.terminate()
        else:
            logger.info(
                "No multiprocessing requested or `pathos` not install, cont."
                " without multiprocessing"
            )
            for i in trange(self.nsfts):
                self.make_ith_sft(i)

        self.concatenate_sft_files()

    def run_makefakedata_v4(self, mid_time, lineFreq, linePhi, h0, tmp_outdir):
        """Generate SFT data using the MFDv4 code with the --lineFeature option."""
        cl_mfd = []
        cl_mfd.append("lalpulsar_Makefakedata_v4")
        cl_mfd.append("--outSingleSFT=FALSE")
        cl_mfd.append('--outSFTbname="{}"'.format(tmp_outdir))
        cl_mfd.append("--IFO={}".format(self.detectors))
        cl_mfd.append('--noiseSqrtSh="{}"'.format(self.sqrtSX))
        cl_mfd.append("--startTime={:0.0f}".format(mid_time - self.Tsft / 2.0))
        cl_mfd.append("--refTime={:0.0f}".format(mid_time))
        cl_mfd.append("--duration={}".format(self.Tsft))
        cl_mfd.append("--fmin={:.16g}".format(self.fmin))
        cl_mfd.append("--Band={:.16g}".format(self.Band))
        cl_mfd.append("--Tsft={}".format(self.Tsft))
        cl_mfd.append("--Freq={}".format(lineFreq))
        cl_mfd.append("--phi0={}".format(linePhi))
        cl_mfd.append("--h0={}".format(h0))
        cl_mfd.append("--cosi={}".format(self.cosi))
        cl_mfd.append("--lineFeature=TRUE")
        earth_ephem = getattr(self, "earth_ephem", None)
        sun_ephem = getattr(self, "sun_ephem", None)
        if earth_ephem is not None:
            cl_mfd.append('--ephemEarth="{}"'.format(earth_ephem))
        if sun_ephem is not None:
            cl_mfd.append('--ephemSun="{}"'.format(sun_ephem))
        if self.randSeed:
            cl_mfd.append("--randSeed={}".format(self.randSeed))
        cl_mfd = " ".join(cl_mfd)
        utils.run_commandline(cl_mfd)


class FrequencyAmplitudeModulatedArtifactWriter(FrequencyModulatedArtifactWriter):
    """A variant of FrequencyModulatedArtifactWriter with evolving amplitude."""

    def get_h0(self, t):
        """Evaluate the artifact amplitude at a given time.

        NOTE: Here it's actually changing over time!

        Parameters
        ----------
        t: float
            Time stamp to evaluate at.

        Returns
        -------
        h0: float
            Amplitude at time `t`.
        """
        return self.h0 * np.sin(2 * np.pi * t / self.Pmod + self.Pmod_phi)
