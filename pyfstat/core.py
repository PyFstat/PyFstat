""" The core tools used in pyfstat """

import getpass
import glob
import logging
import os
import socket
from datetime import datetime
from pprint import pformat
from weakref import finalize

import lal
import lalpulsar
import numpy as np
import scipy.optimize
import scipy.special

import pyfstat.tcw_fstat_map_funcs as tcw
import pyfstat.utils as utils

from ._version import get_versions

plt = utils.safe_X_less_plt()


logger = logging.getLogger(__name__)

detector_colors = {"h1": "C0", "l1": "C1"}


class BaseSearchClass:
    """The base class providing parent methods to other PyFstat classes.

    This does not actually have any 'search' functionality,
    which needs to be added by child classes
    along with full initialization and any other custom methods.
    """

    def __new__(cls, *args, **kwargs):
        logger.info(f"Creating {cls.__name__} object...")
        instance = super().__new__(cls)
        return instance

    def _get_list_of_matching_sfts(self):
        """Returns a list of sfts matching the attribute sftfilepattern"""
        sftfilepatternlist = np.atleast_1d(self.sftfilepattern.split(";"))
        matches = [glob.glob(p) for p in sftfilepatternlist]
        matches = [item for sublist in matches for item in sublist]
        if len(matches) > 0:
            return matches
        else:
            raise IOError("No sfts found matching {}".format(self.sftfilepattern))

    def set_ephemeris_files(self, earth_ephem=None, sun_ephem=None):
        """Set the ephemeris files to use for the Earth and Sun.

        NOTE: If not given explicit arguments,
        default values from utils.get_ephemeris_files()
        are used.

        Parameters
        ----------
        earth_ephem, sun_ephem: str
            Paths of the two files containing positions of Earth and Sun,
            respectively at evenly spaced times, as passed to CreateFstatInput
        """
        earth_ephem_default, sun_ephem_default = utils.get_ephemeris_files()
        self.earth_ephem = earth_ephem or earth_ephem_default
        self.sun_ephem = sun_ephem or sun_ephem_default

    def _set_init_params_dict(self, argsdict):
        """Store the initial input arguments, e.g. for logging output."""
        argsdict.pop("self")
        self.init_params_dict = argsdict

    def pprint_init_params_dict(self):
        """Pretty-print a parameters dictionary for output file headers.

        Returns
        -------
        pretty_init_parameters: list
            A list of lines to be printed,
            including opening/closing "{" and "}",
            consistent indentation,
            as well as end-of-line commas,
            but no comment markers at start of lines.
        """
        pretty_init_parameters = pformat(
            self.init_params_dict, indent=2, width=74
        ).split("\n")
        pretty_init_parameters = (
            ["{"]
            + [pretty_init_parameters[0].replace("{", " ")]
            + pretty_init_parameters[1:-1]
            + [pretty_init_parameters[-1].rstrip("}")]
            + ["}"]
        )
        return pretty_init_parameters

    def get_output_file_header(self):
        """Constructs a meta-information header for text output files.

        This will include
        PyFstat and LALSuite versioning,
        information about when/where/how the code was run,
        and input parameters of the instantiated class.

        Returns
        -------
        header: list
            A list of formatted header lines.

        """
        header = [
            "date: {}".format(str(datetime.now())),
            "user: {}".format(getpass.getuser()),
            "hostname: {}".format(socket.gethostname()),
            "PyFstat: {}".format(get_versions()["version"]),
        ]
        lalVCSinfo = lal.VCSInfoString(lalpulsar.PulsarVCSInfoList, 0, "")
        header += filter(None, lalVCSinfo.split("\n"))
        header += [
            "search: {}".format(type(self).__name__),
            "parameters: ",
        ]
        header += self.pprint_init_params_dict()
        return header

    def read_par(
        self, filename=None, label=None, outdir=None, suffix="par", raise_error=True
    ):
        """Read a `key=val` file and return a dictionary.

        Parameters
        ----------
        filename: str or None
            Filename (path) containing rows of `key=val` data to read in.
        label, outdir, suffix : str or None
            If filename is None, form the file to read as `outdir/label.suffix`.
        raise_error : bool
            If True, raise an error for lines which are not comments,
            but cannot be read.

        Returns
        -------
        params_dict: dict
            A dictionary of the parsed `key=val` pairs.

        """
        params_dict = utils.read_par(
            filename=filename,
            label=label or getattr(self, "label", None),
            outdir=outdir or getattr(self, "outdir", None),
            suffix=suffix,
            raise_error=raise_error,
        )
        return params_dict

    @staticmethod
    def translate_keys_to_lal(dictionary):
        """Convert input keys into lalpulsar convention.

        In PyFstat's convention, input keys (search parameter names)
        are F0, F1, F2, ...,
        while lalpulsar functions prefer to use Freq, f1dot, f2dot, ....

        Since lalpulsar keys are only used internally to call lalpulsar routines,
        this function is provided so the keys can be translated on the fly.

        Parameters
        ----------
        dictionary: dict
            Dictionary to translate. A copy will be made (and returned)
            before translation takes place.

        Returns
        -------
        translated_dict: dict
            Copy of "dictionary" with new keys according to lalpulsar convention.
        """

        translation = {
            "F0": "Freq",
            "F1": "f1dot",
            "F2": "f2dot",
            "phi": "phi0",
            "tref": "refTime",
            "asini": "orbitasini",
            "period": "orbitPeriod",
            "tp": "orbitTp",
            "argp": "orbitArgp",
            "ecc": "orbitEcc",
            "transient_tstart": "transient-t0Epoch",
            "transient_duration": "transient-tau",
        }

        keys_to_translate = [key for key in dictionary.keys() if key in translation]

        translated_dict = dictionary.copy()
        for key in keys_to_translate:
            translated_dict[translation[key]] = translated_dict.pop(key)
        return translated_dict


class ComputeFstat(BaseSearchClass):
    """Base search class providing an interface to `lalpulsar.ComputeFstat`.

    In most cases, users should be using one of the higher-level search classes
    from the grid_based_searches or mcmc_based_searches modules instead.

    See the lalpulsar documentation at https://lscsoft.docs.ligo.org/lalsuite/lalpulsar/group___compute_fstat__h.html
    and R. Prix, The F-statistic and its implementation in ComputeFstatistic_v2 ( https://dcc.ligo.org/T0900149/public )
    for details of the lalpulsar module and the meaning of various technical concepts
    as embodied by some of the class's parameters.

    Normally this will read in existing data through the `sftfilepattern` argument,
    but if that option is `None` and the necessary alternative arguments are used,
    it can also generate simulated data (including noise and/or signals) on the fly.

    NOTE that the detection statistics that can be computed from an instance of this class
    depend on the `BSGL`, `BtSG` and `transientWindowType` arguments given at initialisation.
    See `get_fullycoherent_detstat()` and `get_transient_detstats()` for details.
    To change what you want to compute,
    you may need to initialise a new instance with different options.

    NOTE for GPU users (`tCWFstatMapVersion="pycuda"`):
    This class tries to conveniently deal with GPU context management behind the scenes.
    A known problematic case is if you try to instantiate it twice from the same
    session/script. If you then get some messages like
    `RuntimeError: make_default_context()`
    and `invalid device context`,
    that is because the GPU is still blocked from the first instance when
    you try to initiate the second.
    To avoid this problem, use context management::

        with pyfstat.ComputeFstat(
            [...],
            tCWFstatMapVersion="pycuda",
        ) as search:
            search.get_fullycoherent_detstat([...])

    or manually call the `search.finalizer_()` method where needed.
    """

    @utils.initializer
    def __init__(
        self,
        tref,
        sftfilepattern=None,
        minStartTime=None,
        maxStartTime=None,
        Tsft=1800,
        binary=False,
        singleFstats=False,
        BSGL=False,
        BtSG=False,
        transientWindowType=None,
        t0Band=None,
        tauBand=None,
        tauMin=None,
        dt0=None,
        dtau=None,
        detectors=None,
        minCoverFreq=None,
        maxCoverFreq=None,
        search_ranges=None,
        injectSources=None,
        injectSqrtSX=None,
        randSeed=None,
        assumeSqrtSX=None,
        SSBprec=None,
        RngMedWindow=None,
        tCWFstatMapVersion="lal",
        cudaDeviceName=None,
        computeAtoms=False,
        earth_ephem=None,
        sun_ephem=None,
        allowedMismatchFromSFTLength=None,
    ):
        """
        Parameters
        ----------
        tref : int
            GPS seconds of the reference time.
        sftfilepattern : str
            Pattern to match SFTs using wildcards (`*?`) and ranges [0-9];
            mutiple patterns can be given separated by colons.
        minStartTime, maxStartTime : int
            Only use SFTs with timestamps starting from within this range,
            following the XLALCWGPSinRange convention:
            half-open intervals [minStartTime,maxStartTime].
        Tsft: int
            SFT duration in seconds.
            Only required if `sftfilepattern=None` and hence simulted data is
            generated on the fly.
        binary : bool
            If true, search over binary parameters.
        singleFstats : bool
            If true, also compute the single-detector twoF values.
        BSGL : bool
            If true, compute the log10BSGL statistic rather than the twoF value.
            For details, see Keitel et al (PRD 89, 064023, 2014):
            https://arxiv.org/abs/1311.5738
            Note this automatically sets `singleFstats=True` as well.
            Tuning parameters are currently hardcoded:

            * `Fstar0=15` for coherent searches.

            * A p-value of 1e-6 and correspondingly recalculated Fstar0
              for semicoherent searches.

            * Uniform per-detector prior line-vs-Gaussian odds.
        BtSG: bool
            If true and `transientWindowType` is not `None`,
            compute the transient
            :math:`\\ln\\mathcal{B}_{\\mathrm{tS}/\\mathrm{G}}`
            statistic from Prix, Giampanis & Messenger (PRD 84, 023007, 2011)
            (tCWFstatMap marginalised over uniform t0, tau priors).
            rather than the maxTwoF value.
        transientWindowType: str
            If `rect` or `exp`,
            allow for the Fstat to be computed over a transient range.
            (`none` instead of `None` explicitly calls the transient-window
            function, but with the full range, for debugging.)
            (If not None, will also force atoms regardless of computeAtoms option.)
        t0Band, tauBand: int
            Search ranges for transient start-time t0 and duration tau.
            If >0, search t0 in (minStartTime,minStartTime+t0Band)
            and tau in (tauMin,2*Tsft+tauBand).
            If =0, only compute the continuous-wave Fstat with t0=minStartTime,
            tau=maxStartTime-minStartTime.
        tauMin: int
            Minimum transient duration to cover,
            defaults to 2*Tsft.
        dt0: int
            Grid resolution in transient start-time,
            defaults to Tsft.
        dtau: int
            Grid resolution in transient duration,
            defaults to Tsft.
        detectors : str
            Two-character references to the detectors for which to use data.
            Specify `None` for no constraint.
            For multiple detectors, separate by commas.
        minCoverFreq, maxCoverFreq : float
            The min and max cover frequency passed to lalpulsar.CreateFstatInput.
            For negative values, these will be used as offsets from the min/max
            frequency contained in the sftfilepattern.
            If either is `None`, the search_ranges argument is used to estimate them.
            If the automatic estimation fails and you do not have a good idea
            what to set these two options to, setting both to -0.5 will
            reproduce the default behaviour of PyFstat <=1.4 and may be a
            reasonably safe fallback in many cases.
        search_ranges: dict
            Dictionary of ranges in all search parameters,
            only used to estimate frequency band passed to lalpulsar.CreateFstatInput,
            if minCoverFreq, maxCoverFreq are not specified (==`None`).
            For actually running searches,
            grids/points will have to be passed separately to the .run() method.
            The entry for each parameter must be a list of length 1, 2 or 3:
            [single_value], [min,max] or [min,max,step].
        injectSources : dict or str
            Either a dictionary of the signal parameters to inject,
            or a string pointing to a .cff file defining a signal.
        injectSqrtSX : float or list or str
            Single-sided PSD values for generating fake Gaussian noise on the fly.
            Single float or str value: use same for all IFOs.
            List or comma-separated string: must match len(detectors)
            and/or the data in sftfilepattern.
            Detectors will be paired to list elements following alphabetical order.
        randSeed : int or None
            random seed for on-the-fly noise generation using `injectSqrtSX`.
            Setting this to 0 or None is equivalent; both will randomise the seed,
            following the behaviour of XLALAddGaussianNoise(),
            while any number not equal to 0 will produce a reproducible noise realisation.
        assumeSqrtSX : float or list or str
            Don't estimate noise-floors but assume this (stationary) single-sided PSD.
            Single float or str value: use same for all IFOs.
            List or comma-separated string: must match len(detectors)
            and/or the data in sftfilepattern.
            Detectors will be paired to list elements following alphabetical order.
            If working with signal-only data, please set assumeSqrtSX=1 .
        SSBprec : int
            Flag to set the Solar System Barycentring (SSB) calculation in lalpulsar:
            0=Newtonian, 1=relativistic,
            2=relativistic optimised, 3=DMoff, 4=NO_SPIN
        RngMedWindow : int
           Running-Median window size for F-statistic noise normalization
           (number of SFT bins).
        tCWFstatMapVersion: str
            Choose between implementations of the transient F-statistic funcionality:
            standard `lal` implementation,
            `pycuda` for GPU version,
            and some others only for devel/debug.
        cudaDeviceName: str
            GPU name to be matched against drv.Device output,
            only for `tCWFstatMapVersion=pycuda`.
        computeAtoms: bool
            Request calculation of 'F-statistic atoms' regardless of transientWindowType.
        earth_ephem: str
            Earth ephemeris file path.
            If None, will check standard sources as per
            utils.get_ephemeris_files().
        sun_ephem: str
            Sun ephemeris file path.
            If None, will check standard sources as per
            utils.get_ephemeris_files().
        allowedMismatchFromSFTLength: float
            Maximum allowed mismatch from SFTs being too long
            [Default: what's hardcoded in XLALFstatMaximumSFTLength]
        """

        self._setup_finalizer()
        self._set_init_params_dict(locals())
        self.set_ephemeris_files(earth_ephem, sun_ephem)
        self.init_computefstatistic()
        self.output_file_header = self.get_output_file_header()
        self.get_det_stat = self.get_fullycoherent_detstat
        self.allowedMismatchFromSFTLength = allowedMismatchFromSFTLength

    def _setup_finalizer(self):
        """
        Setup for proper cleanup at end of context in pycuda case.

        Users should normally *not* have to call self._finalizer() manually:
        the `finalize` call is enough to set up python garbage collection,
        and we only store it as an attribute for debugging/testing purposes.

        However, if one wants to initialise two or more of these objects from
        a single script, one has to manually clean up after each one by either
        using context management or calling the `.finalizer_()` method.
        """
        if "cuda" in self.tCWFstatMapVersion:
            logger.debug(
                f"Setting up GPU context finalizer for {self.tCWFstatMapVersion} transient maps."
            )
            self._finalizer = finalize(self, self._finalize_gpu_context)

    def _finalize_gpu_context(self):
        """Clean up at the end of context manager style usage."""
        logger.debug("Leaving the ComputeFStat context...")
        if hasattr(self, "gpu_context") and self.gpu_context:
            logger.debug("Detaching GPU context...")
            # this is needed because we use pyCuda without autoinit
            self.gpu_context.detach()

    def __enter__(self):
        """Enables context manager style calling."""
        logger.debug("Entering the ComputeFstat context...")
        return self

    def __exit__(self, *args, **kwargs):
        """Clean up at the end of context manager style usage."""
        logger.debug("Leaving the ComputeFStat context...")
        if "cuda" in self.tCWFstatMapVersion:
            self._finalizer()

    def _get_SFTCatalog(self):
        """Load the SFTCatalog

        If sftfilepattern is specified, load the data. If not, attempt to
        create data on the fly.
        """
        if hasattr(self, "SFTCatalog"):
            logger.info("Already have SFTCatalog.")
            return
        if self.sftfilepattern is None:
            logger.info("No sftfilepattern given, making fake SFTCatalog.")
            for k in ["minStartTime", "maxStartTime", "detectors"]:
                if getattr(self, k) is None:
                    raise ValueError(
                        "If sftfilepattern==None, you must provide" " '{}'.".format(k)
                    )
            C1 = getattr(self, "injectSources", None) is None
            C2 = getattr(self, "injectSqrtSX", None) is None
            C3 = getattr(self, "Tsft", None) is None
            if (C1 and C2) or C3:
                raise ValueError(
                    "If sftfilepattern==None, you must specify Tsft and"
                    " either one of injectSources or injectSqrtSX."
                )
            SFTCatalog = lalpulsar.SFTCatalog()
            Toverlap = 0
            self.detector_names = self.detectors.split(",")
            self.numDetectors = len(self.detector_names)
            detNames = lal.CreateStringVector(*[d for d in self.detector_names])
            # MakeMultiTimestamps follows the same [minStartTime,maxStartTime)
            # convention as the SFT library, so we can pass Tspan like this
            Tspan = self.maxStartTime - self.minStartTime
            multiTimestamps = lalpulsar.MakeMultiTimestamps(
                self.minStartTime, Tspan, self.Tsft, Toverlap, detNames.length
            )
            SFTCatalog = lalpulsar.MultiAddToFakeSFTCatalog(
                SFTCatalog, detNames, multiTimestamps
            )
            self.SFTCatalog = SFTCatalog
            return

        logger.info("Initialising SFTCatalog from sftfilepattern.")
        constraints = lalpulsar.SFTConstraints()
        constr_str = []
        if self.detectors:
            if "," in self.detectors:
                logger.warning(
                    "Multiple-detector constraints not available,"
                    " using all available data."
                )
            else:
                constraints.detector = self.detectors
                constr_str.append("detector=" + constraints.detector)
        if self.minStartTime:
            constraints.minStartTime = lal.LIGOTimeGPS(self.minStartTime)
            constr_str.append("minStartTime={}".format(self.minStartTime))
        if self.maxStartTime:
            constraints.maxStartTime = lal.LIGOTimeGPS(self.maxStartTime)
            constr_str.append("maxStartTime={}".format(self.maxStartTime))
        logger.info(
            "Loading data matching SFT file name pattern '{}'"
            " with constraints {}.".format(self.sftfilepattern, ", ".join(constr_str))
        )
        self.SFTCatalog = lalpulsar.SFTdataFind(self.sftfilepattern, constraints)
        if self.SFTCatalog.length == 0:
            raise IOError("No SFTs found.")
        Tsft_from_catalog = int(1.0 / self.SFTCatalog.data[0].header.deltaF)
        if Tsft_from_catalog != self.Tsft:
            logger.info(
                "Overwriting pre-set Tsft={:d} with {:d} obtained from SFTs.".format(
                    self.Tsft, Tsft_from_catalog
                )
            )
        self.Tsft = Tsft_from_catalog

        # NOTE: in multi-IFO case, this will be a joint list of timestamps
        # over all IFOs, probably sorted and not cleaned for uniqueness.
        SFT_timestamps = [d.header.epoch for d in self.SFTCatalog.data]
        self.SFT_timestamps = [float(s) for s in SFT_timestamps]
        if len(SFT_timestamps) == 0:
            raise ValueError("Failed to load any data")

        dtstr1 = utils.gps_to_datestr_utc(int(SFT_timestamps[0]))
        dtstr2 = utils.gps_to_datestr_utc(int(SFT_timestamps[-1]))
        logger.info(
            f"Data contains SFT timestamps from {SFT_timestamps[0]} ({dtstr1})"
            f" to (including) {SFT_timestamps[-1]} ({dtstr2})."
        )

        if self.minStartTime is None:
            self.minStartTime = int(SFT_timestamps[0])
        if self.maxStartTime is None:
            # XLALCWGPSinRange() convention: half-open intervals,
            # maxStartTime must always be > last actual SFT timestamp
            self.maxStartTime = int(SFT_timestamps[-1]) + self.Tsft

        self.detector_names = list(set([d.header.name for d in self.SFTCatalog.data]))
        self.numDetectors = len(self.detector_names)
        if self.numDetectors == 0:
            raise ValueError("No data loaded.")
        logger.info(
            "Loaded {} SFTs from {} detectors: {}".format(
                len(SFT_timestamps), self.numDetectors, self.detector_names
            )
        )

    def init_computefstatistic(self):
        """Initialization step for the F-stastic computation internals.

        This sets up the special input and output structures the lalpulsar module needs,
        the ephemerides,
        optional on-the-fly signal injections,
        and extra options for multi-detector consistency checks and transient searches.

        All inputs are taken from the pre-initialized object,
        so this function does not have additional arguments of its own.
        """

        self._get_SFTCatalog()

        # some sanity checks on user options
        if self.BSGL:  # pragma: no cover
            if len(self.detector_names) < 2:
                raise ValueError("Can't use BSGL with single detector data")
            if getattr(self, "BtSG", False):
                raise ValueError("Please choose only one of [BSGL,BtSG].")

        logger.info("Initialising ephems")
        ephems = lalpulsar.InitBarycenter(self.earth_ephem, self.sun_ephem)

        logger.info("Initialising Fstat arguments")
        dFreq = 0
        self.whatToCompute = lalpulsar.FSTATQ_2F
        if self.transientWindowType or self.computeAtoms:
            self.whatToCompute += lalpulsar.FSTATQ_ATOMS_PER_DET

        FstatOAs = lalpulsar.FstatOptionalArgs()
        if self.SSBprec:
            logger.info("Using SSBprec={}".format(self.SSBprec))
            FstatOAs.SSBprec = self.SSBprec
        else:
            FstatOAs.SSBprec = lalpulsar.FstatOptionalArgsDefaults.SSBprec
        FstatOAs.Dterms = lalpulsar.FstatOptionalArgsDefaults.Dterms
        if self.RngMedWindow:
            FstatOAs.runningMedianWindow = self.RngMedWindow
        else:
            FstatOAs.runningMedianWindow = (
                lalpulsar.FstatOptionalArgsDefaults.runningMedianWindow
            )
        FstatOAs.FstatMethod = lalpulsar.FstatOptionalArgsDefaults.FstatMethod
        if self.assumeSqrtSX is None:
            FstatOAs.assumeSqrtSX = lalpulsar.FstatOptionalArgsDefaults.assumeSqrtSX
        else:
            mnf = lalpulsar.MultiNoiseFloor()
            assumeSqrtSX = utils.parse_list_of_numbers(self.assumeSqrtSX)
            mnf.sqrtSn[: len(assumeSqrtSX)] = assumeSqrtSX
            mnf.length = len(assumeSqrtSX)
            FstatOAs.assumeSqrtSX = mnf
        FstatOAs.prevInput = lalpulsar.FstatOptionalArgsDefaults.prevInput
        FstatOAs.collectTiming = lalpulsar.FstatOptionalArgsDefaults.collectTiming
        if self.allowedMismatchFromSFTLength:
            FstatOAs.allowedMismatchFromSFTLength = self.allowedMismatchFromSFTLength

        if hasattr(self, "injectSources") and isinstance(self.injectSources, dict):
            logger.info("Injecting source with params: {}".format(self.injectSources))
            PPV = lalpulsar.CreatePulsarParamsVector(1)
            PP = PPV.data[0]
            h0 = self.injectSources["h0"]
            cosi = self.injectSources["cosi"]
            use_aPlus = "aPlus" in dir(PP.Amp)
            if use_aPlus:  # lalsuite interface changed in aff93c45
                PP.Amp.aPlus = 0.5 * h0 * (1.0 + cosi**2)
                PP.Amp.aCross = h0 * cosi
            else:
                PP.Amp.h0 = h0
                PP.Amp.cosi = cosi

            PP.Amp.phi0 = self.injectSources["phi"]
            PP.Amp.psi = self.injectSources["psi"]
            PP.Doppler.Alpha = self.injectSources["Alpha"]
            PP.Doppler.Delta = self.injectSources["Delta"]
            if "fkdot" in self.injectSources:
                PP.Doppler.fkdot = np.array(self.injectSources["fkdot"])
            else:
                PP.Doppler.fkdot = np.zeros(lalpulsar.PULSAR_MAX_SPINS)
                for i, key in enumerate(["F0", "F1", "F2"]):
                    PP.Doppler.fkdot[i] = self.injectSources[key]
            PP.Doppler.refTime = self.tref
            if "t0" not in self.injectSources:
                PP.Transient.type = lalpulsar.TRANSIENT_NONE
            FstatOAs.injectSources = PPV
        elif hasattr(self, "injectSources") and isinstance(self.injectSources, str):
            logger.info(
                "Injecting source from param file: {}".format(self.injectSources)
            )
            PPV = lalpulsar.PulsarParamsFromFile(self.injectSources, self.tref)
            FstatOAs.injectSources = PPV
        else:
            FstatOAs.injectSources = lalpulsar.FstatOptionalArgsDefaults.injectSources
        if hasattr(self, "injectSqrtSX") and self.injectSqrtSX is not None:
            self.injectSqrtSX = utils.parse_list_of_numbers(self.injectSqrtSX)
            if len(self.injectSqrtSX) != len(self.detector_names):
                raise ValueError(
                    "injectSqrtSX must be of same length as detector_names ({}!={})".format(
                        len(self.injectSqrtSX), len(self.detector_names)
                    )
                )
            FstatOAs.injectSqrtSX = lalpulsar.MultiNoiseFloor()
            FstatOAs.injectSqrtSX.length = len(self.injectSqrtSX)
            FstatOAs.injectSqrtSX.sqrtSn[: FstatOAs.injectSqrtSX.length] = (
                self.injectSqrtSX
            )
        else:
            FstatOAs.injectSqrtSX = lalpulsar.FstatOptionalArgsDefaults.injectSqrtSX
        # Here we are treating 0 and None as equivalent
        # (use default, which is 0 and means "randomise the seed").
        # See XLALAddGaussianNoise().
        FstatOAs.randSeed = (
            getattr(self, "randSeed", None)
            or lalpulsar.FstatOptionalArgsDefaults.randSeed
        )
        self._set_min_max_cover_freqs()

        logger.info("Initialising FstatInput")
        self.FstatInput = lalpulsar.CreateFstatInput(
            self.SFTCatalog,
            self.minCoverFreq,
            self.maxCoverFreq,
            dFreq,
            ephems,
            FstatOAs,
        )

        logger.info("Initialising PulsarDoplerParams")
        PulsarDopplerParams = lalpulsar.PulsarDopplerParams()
        PulsarDopplerParams.refTime = self.tref
        PulsarDopplerParams.Alpha = 1
        PulsarDopplerParams.Delta = 1
        PulsarDopplerParams.fkdot = np.zeros(lalpulsar.PULSAR_MAX_SPINS)
        self.PulsarDopplerParams = PulsarDopplerParams

        logger.info("Initialising FstatResults")
        self.FstatResults = lalpulsar.FstatResults()

        # always initialise the twoFX array,
        # but only actually compute it if requested
        self.twoF = 0
        self.twoFX = np.zeros(lalpulsar.PULSAR_MAX_DETECTORS)
        self.singleFstats = self.singleFstats or self.BSGL  # BSGL implies twoFX
        if self.singleFstats:
            self.whatToCompute += lalpulsar.FSTATQ_2F_PER_DET

        if self.BSGL:
            logger.info("Initialising BSGL")
            self.log10BSGL = np.nan
            # Tuning parameters - to be reviewed
            # We use a fixed Fstar0 for coherent searches,
            # and recompute it from a fixed p-value for the semicoherent case.
            nsegs_eff = max([getattr(self, "nsegs", 1), getattr(self, "nglitch", 1)])
            if nsegs_eff > 1:
                p_val_threshold = 1e-6
                Fstar0s = np.linspace(0, 1000, 10000)
                p_vals = scipy.special.gammaincc(2 * nsegs_eff, Fstar0s)
                self.Fstar0 = Fstar0s[np.argmin(np.abs(p_vals - p_val_threshold))]
                if self.Fstar0 == Fstar0s[-1]:
                    raise ValueError("Max Fstar0 exceeded")
            else:
                self.Fstar0 = 15.0
            logger.info("Using Fstar0 of {:1.2f}".format(self.Fstar0))
            # assume uniform per-detector prior line-vs-Gaussian odds
            self.oLGX = np.zeros(lalpulsar.PULSAR_MAX_DETECTORS)
            self.oLGX[: self.numDetectors] = 1.0 / self.numDetectors
            self.BSGLSetup = lalpulsar.CreateBSGLSetup(
                numDetectors=self.numDetectors,
                Fstar0sc=self.Fstar0,
                oLGX=self.oLGX,
                useLogCorrection=True,
                numSegments=getattr(self, "nsegs", 1),
            )

        if self.transientWindowType:
            logger.info(
                f"Initialising transient parameters for window type '{self.transientWindowType}'"
            )
            self.maxTwoF = 0
            if getattr(self, "BtSG", False):
                self.lnBtSG = np.nan
            self.windowRange = lalpulsar.transientWindowRange_t()
            transientWindowTypes = {
                "none": lalpulsar.TRANSIENT_NONE,
                "rect": lalpulsar.TRANSIENT_RECTANGULAR,
                "exp": lalpulsar.TRANSIENT_EXPONENTIAL,
            }
            if self.transientWindowType in transientWindowTypes:
                self.windowRange.type = transientWindowTypes[self.transientWindowType]
            else:
                raise ValueError(
                    "Unknown window-type ({}) passed as input, [{}] allows.".format(
                        self.transientWindowType, ", ".join(transientWindowTypes)
                    )
                )

            # default spacing
            self.windowRange.dt0 = self.Tsft
            self.windowRange.dtau = self.Tsft

            # special treatment of window_type = none
            # ==> replace by rectangular window spanning all the data
            if self.windowRange.type == lalpulsar.TRANSIENT_NONE:
                self.windowRange.t0 = int(self.minStartTime)
                self.windowRange.t0Band = 0
                self.windowRange.tau = int(self.maxStartTime - self.minStartTime)
                self.windowRange.tauBand = 0
            else:  # user-set bands and spacings
                if getattr(self, "t0Band", None) is None:
                    self.windowRange.t0Band = 0
                else:
                    if not isinstance(self.t0Band, int):
                        logger.warning(
                            "Casting non-integer t0Band={} to int...".format(
                                self.t0Band
                            )
                        )
                        self.t0Band = int(self.t0Band)
                    self.windowRange.t0Band = self.t0Band
                    if self.dt0:
                        self.windowRange.dt0 = self.dt0
                if getattr(self, "tauBand", None) is None:
                    self.windowRange.tauBand = 0
                else:
                    if not isinstance(self.tauBand, int):
                        logger.warning(
                            "Casting non-integer tauBand={} to int...".format(
                                self.tauBand
                            )
                        )
                        self.tauBand = int(self.tauBand)
                    self.windowRange.tauBand = self.tauBand
                    if self.dtau:
                        self.windowRange.dtau = self.dtau
                    if self.tauMin is None:
                        self.windowRange.tau = int(2 * self.Tsft)
                    else:
                        if not isinstance(self.tauMin, int):
                            logger.warning(
                                "Casting non-integer tauMin={} to int...".format(
                                    self.tauMin
                                )
                            )
                            self.tauMin = int(self.tauMin)
                        self.windowRange.tau = self.tauMin

            logger.info("Initialising transient FstatMap features...")
            (
                self.tCWFstatMapFeatures,
                self.gpu_context,
            ) = tcw.init_transient_fstat_map_features(
                self.tCWFstatMapVersion, self.cudaDeviceName
            )

            if self.BSGL:
                self.twoFXatMaxTwoF = np.zeros(lalpulsar.PULSAR_MAX_DETECTORS)

    def _set_min_max_cover_freqs(self):
        # decide on which minCoverFreq and maxCoverFreq to use:
        # either from direct user input, estimate_min_max_CoverFreq(), or SFTs
        if self.sftfilepattern is not None:
            minFreq_SFTs, maxFreq_SFTs = self._get_min_max_freq_from_SFTCatalog()
        if (self.minCoverFreq is None) != (self.maxCoverFreq is None):
            raise ValueError(
                "Please use either both or none of [minCoverFreq,maxCoverFreq]."
            )
        elif (
            self.minCoverFreq is None
            and self.maxCoverFreq is None
            and self.search_ranges is None
        ):
            raise ValueError(
                "Please use either search_ranges or both of [minCoverFreq,maxCoverFreq]."
            )
        elif self.minCoverFreq is None or self.maxCoverFreq is None:
            logger.info(
                "[minCoverFreq,maxCoverFreq] not provided, trying to estimate"
                " from search ranges."
            )
            self.estimate_min_max_CoverFreq()
        elif (self.minCoverFreq < 0.0) or (self.maxCoverFreq < 0.0):
            if self.sftfilepattern is None:
                raise ValueError(
                    "If sftfilepattern==None, cannot use negative values for"
                    " minCoverFreq or maxCoverFreq (interpreted as offsets from"
                    " min/max SFT frequency)."
                    " Please use actual frequency values for both,"
                    " or set both to None (automated estimation)."
                )
            if self.minCoverFreq < 0.0:
                logger.info(
                    "minCoverFreq={:f} provided, using as offset from min(SFTs).".format(
                        self.minCoverFreq
                    )
                )
                # to set *above* min, since minCoverFreq is negative: subtract it
                self.minCoverFreq = minFreq_SFTs - self.minCoverFreq
            if self.maxCoverFreq < 0.0:
                logger.info(
                    "maxCoverFreq={:f} provided, using as offset from max(SFTs).".format(
                        self.maxCoverFreq
                    )
                )
                # to set *below* max, since minCoverFreq is negative: add it
                self.maxCoverFreq = maxFreq_SFTs + self.maxCoverFreq
        if (self.sftfilepattern is not None) and (
            (self.minCoverFreq < minFreq_SFTs) or (self.maxCoverFreq > maxFreq_SFTs)
        ):
            raise ValueError(
                "[minCoverFreq,maxCoverFreq]=[{:f},{:f}] Hz incompatible with"
                " SFT files content [{:f},{:f}] Hz".format(
                    self.minCoverFreq, self.maxCoverFreq, minFreq_SFTs, maxFreq_SFTs
                )
            )
        logger.info(
            "Using minCoverFreq={} and maxCoverFreq={}.".format(
                self.minCoverFreq, self.maxCoverFreq
            )
        )

    def _get_min_max_freq_from_SFTCatalog(self):
        fAs = [d.header.f0 for d in self.SFTCatalog.data]
        minFreq_SFTs = np.min(fAs)
        fBs = [
            d.header.f0 + (d.numBins - 1) * d.header.deltaF
            for d in self.SFTCatalog.data
        ]
        maxFreq_SFTs = np.max(fBs)
        return minFreq_SFTs, maxFreq_SFTs

    def estimate_min_max_CoverFreq(self):
        """Extract spanned spin-range at reference -time from the template bank.

        To use this method, self.search_ranges must be a dictionary of lists per search parameter
        which can be either [single_value], [min,max] or [min,max,step].
        """
        if type(self.search_ranges) is not dict:
            raise ValueError("Need a dictionary for search_ranges!")
        range_keys = list(self.search_ranges.keys())
        required_keys = ["Alpha", "Delta", "F0"]
        if len(np.setdiff1d(required_keys, range_keys)) > 0:
            raise ValueError(
                "Required keys not found in search_ranges: {}".format(
                    np.setdiff1d(required_keys, range_keys)
                )
            )
        for key in range_keys:
            if (
                type(self.search_ranges[key]) is not list
                or len(self.search_ranges[key]) == 0
                or len(self.search_ranges[key]) > 3
            ):
                raise ValueError(
                    "search_ranges entry for {:s}"
                    " is not a list of a known format"
                    " (either [single_value], [min,max]"
                    " or [min,max,step]): {}".format(key, self.search_ranges[key])
                )
        # start by constructing a DopplerRegion structure
        # which will be needed to conservatively account for sky-position dependent
        # Doppler shifts of the frequency range to be covered
        searchRegion = lalpulsar.DopplerRegion()
        # sky region
        Alpha = self.search_ranges["Alpha"][0]
        AlphaBand = (
            self.search_ranges["Alpha"][1] - Alpha
            if len(self.search_ranges["Alpha"]) >= 2
            else 0.0
        )
        Delta = self.search_ranges["Delta"][0]
        DeltaBand = (
            self.search_ranges["Delta"][1] - Delta
            if len(self.search_ranges["Delta"]) >= 2
            else 0.0
        )
        searchRegion.skyRegionString = lalpulsar.SkySquare2String(
            Alpha,
            Delta,
            AlphaBand,
            DeltaBand,
        )
        searchRegion.refTime = self.tref
        # frequency and spindowns
        searchRegion.fkdot = np.zeros(lalpulsar.PULSAR_MAX_SPINS)
        searchRegion.fkdotBand = np.zeros(lalpulsar.PULSAR_MAX_SPINS)
        for k in range(3):
            Fk = "F{:d}".format(k)
            if Fk in range_keys:
                searchRegion.fkdot[k] = self.search_ranges[Fk][0]
                searchRegion.fkdotBand[k] = (
                    self.search_ranges[Fk][1] - self.search_ranges[Fk][0]
                    if len(self.search_ranges[Fk]) >= 2
                    else 0.0
                )
        # now construct DopplerFullScan from searchRegion
        scanInit = lalpulsar.DopplerFullScanInit()
        scanInit.searchRegion = searchRegion
        scanInit.stepSizes = lalpulsar.PulsarDopplerParams()
        scanInit.stepSizes.refTime = self.tref
        scanInit.stepSizes.Alpha = (
            self.search_ranges["Alpha"][-1]
            if len(self.search_ranges["Alpha"]) == 3
            else 0.001  # fallback, irrelevant for band estimate but must be > 0
        )
        scanInit.stepSizes.Delta = (
            self.search_ranges["Delta"][-1]
            if len(self.search_ranges["Delta"]) == 3
            else 0.001  # fallback, irrelevant for band estimate but must be > 0
        )
        scanInit.stepSizes.fkdot = np.zeros(lalpulsar.PULSAR_MAX_SPINS)
        for k in range(3):
            if Fk in range_keys:
                Fk = "F{:d}".format(k)
                scanInit.stepSizes.fkdot[k] = (
                    self.search_ranges[Fk][-1]
                    if len(self.search_ranges[Fk]) == 3
                    else 0.0
                )
        scanInit.startTime = self.minStartTime
        scanInit.Tspan = float(self.maxStartTime - self.minStartTime)
        scanState = lalpulsar.InitDopplerFullScan(scanInit)
        # now obtain the PulsarSpinRange extended over all relevant Doppler shifts
        spinRangeRef = lalpulsar.PulsarSpinRange()
        lalpulsar.GetDopplerSpinRange(spinRangeRef, scanState)
        # optional: binary parameters
        if "asini" in range_keys:
            if len(self.search_ranges["asini"]) >= 2:
                maxOrbitAsini = self.search_ranges["asini"][1]
            else:
                maxOrbitAsini = self.search_ranges["asini"][0]
        else:
            maxOrbitAsini = 0.0
        if "period" in range_keys:
            minOrbitPeriod = self.search_ranges["period"][0]
        else:
            minOrbitPeriod = 0.0
        if "ecc" in range_keys:
            if len(self.search_ranges["ecc"]) >= 2:
                maxOrbitEcc = self.search_ranges["ecc"][1]
            else:
                maxOrbitEcc = self.search_ranges["ecc"][0]
        else:
            maxOrbitEcc = 0.0
        # finally call the wrapped lalpulsar estimation function with the
        # extended PulsarSpinRange and optional binary parameters
        self.minCoverFreq, self.maxCoverFreq = utils.get_covering_band(
            tref=self.tref,
            tstart=self.minStartTime,
            tend=self.maxStartTime,
            F0=spinRangeRef.fkdot[0],
            F1=spinRangeRef.fkdot[1],
            F2=spinRangeRef.fkdot[2],
            F0band=spinRangeRef.fkdotBand[0],
            F1band=spinRangeRef.fkdotBand[1],
            F2band=spinRangeRef.fkdotBand[2],
            maxOrbitAsini=maxOrbitAsini,
            minOrbitPeriod=minOrbitPeriod,
            maxOrbitEcc=maxOrbitEcc,
        )

    def get_fullycoherent_detstat(
        self,
        F0,
        F1,
        F2,
        Alpha,
        Delta,
        asini=None,
        period=None,
        ecc=None,
        tp=None,
        argp=None,
        tstart=None,
        tend=None,
    ):
        """Computes the detection statistic(s) fully-coherently at a single point.

        Currently supported statistics:

        * twoF (CW)

        * log10BSGL (CW or transient)

        * maxTwoF (transient)

        * lnBtSG (transient)

        All computed statistics are stored as attributes,
        but only one statistic is returned.

        As the basic statistic of this class, `twoF` is always computed
        and stored as `self.twoF` as well,
        and it is the default return value.

        If `self.singleFstats`, additionally the single-detector
        2F-stat values are stored in `self.twoFX`.

        If `self.BSGL`, the `log10BSGL` statistic for CWs is additionally stored,
        and it is returned instead of `twoF`.

        If transient parameters are enabled (`self.transientWindowType` is set),
        `maxTwoF` will always be computed and stored,
        and returned by default.
        Depending on the `self.BSGL` and `self.BtSG` options,
        either `log10BSGL` (a transient version of it, superseding the CW version)
        or `lnBtSG` will also be computed, stored,
        and returned instead of `maxTwoF`.
        The full transient-F-stat map is also computed here,
        but stored in `self.FstatMap`, not returned.

        Parameters
        ----------
        F0, F1, F2, Alpha, Delta: float
            Parameters at which to compute the statistic.
        asini, period, ecc, tp, argp: float, optional
            Optional: Binary parameters at which to compute the statistic.
        tstart, tend: int or None
            GPS times to restrict the range of data used.
            If None: falls back to self.minStartTime and self.maxStartTime.
            This is only passed on to `self.get_transient_detstats()`,
            i.e. only used if `self.transientWindowType` is set.

        Returns
        -------
        stat: float
            A single value of the main detection statistic
            at the input parameter values.
        """
        self.get_fullycoherent_twoF(
            F0, F1, F2, Alpha, Delta, asini, period, ecc, tp, argp
        )
        if not self.transientWindowType:
            if self.singleFstats:
                self.get_fullycoherent_single_IFO_twoFs()
            if not self.BSGL:
                return self.twoF
            self.get_fullycoherent_log10BSGL()
            return self.log10BSGL
        return self.get_transient_detstats(
            tstart=tstart,
            tend=tend,
        )

    def get_fullycoherent_twoF(
        self,
        F0,
        F1,
        F2,
        Alpha,
        Delta,
        asini=None,
        period=None,
        ecc=None,
        tp=None,
        argp=None,
    ):
        """Computes the fully-coherent 2F statistic at a single point.

        NOTE: This always uses the full data set as defined when initialising
        the search object.
        If you want to restrict the range of data used for a single 2F computation,
        you need to set a `self.transientWindowType` and then call
        `self.get_fullycoherent_detstat()` with `tstart` and `tend` options
        instead of this funcion.

        Parameters
        ----------
        F0, F1, F2, Alpha, Delta: float
            Parameters at which to compute the statistic.
        asini, period, ecc, tp, argp: float, optional
            Optional: Binary parameters at which to compute the statistic.

        Returns
        -------
        twoF: float
            A single value of the fully-coherent 2F statistic
            at the input parameter values.
            Also stored as `self.twoF`.
        """
        self.PulsarDopplerParams.fkdot = np.zeros(lalpulsar.PULSAR_MAX_SPINS)
        self.PulsarDopplerParams.fkdot[:3] = [F0, F1, F2]
        self.PulsarDopplerParams.Alpha = float(Alpha)
        self.PulsarDopplerParams.Delta = float(Delta)
        if self.binary:
            self.PulsarDopplerParams.asini = float(asini)
            self.PulsarDopplerParams.period = float(period)
            self.PulsarDopplerParams.ecc = float(ecc)
            self.PulsarDopplerParams.tp = float(tp)
            self.PulsarDopplerParams.argp = float(argp)

        lalpulsar.ComputeFstat(
            Fstats=self.FstatResults,
            input=self.FstatInput,
            doppler=self.PulsarDopplerParams,
            numFreqBins=1,
            whatToCompute=self.whatToCompute,
        )
        # We operate on a single frequency bin, so we grab the 0 component
        # of what is internally a twoF array.
        self.twoF = float(self.FstatResults.twoF[0])
        return self.twoF

    def get_fullycoherent_single_IFO_twoFs(self):
        """Computes single-detector F-stats at a single point.

        This requires `self.get_fullycoherent_twoF()` to be run first.

        Returns
        -------
        twoFX: list
            A list of the single-detector detection statistics twoF.
            Also stored as `self.twoFX`.
        """
        if not self.singleFstats:
            raise RuntimeError(
                "This function is available only if singleFstats or BSGL options were set."
            )
        self.twoFX[: self.FstatResults.numDetectors] = np.concatenate(
            [
                self.FstatResults.twoFPerDet(X)
                for X in range(self.FstatResults.numDetectors)
            ]
        )
        return self.twoFX

    def get_fullycoherent_log10BSGL(self):
        """Computes the line-robust statistic log10BSGL at a single point.

        This requires `self.get_fullycoherent_twoF()`
        and `self.get_fullycoherent_single_IFO_twoFs()`
        to be run first.

        Returns
        -------
        log10BSGL: float
            A single value of the detection statistic log10BSGL
            at the input parameter values.
            Also stored as `self.log10BSGL`.
        """
        self.log10BSGL = lalpulsar.ComputeBSGL(self.twoF, self.twoFX, self.BSGLSetup)
        return self.log10BSGL

    def get_transient_detstats(
        self,
        tstart=None,
        tend=None,
    ):
        """Computes one or more transient detection statistics at a single point.

        This requires `self.get_fullycoherent_twoF()` to be run first.

        All computed statistics will be stored as attributes of `self`,
        but only one (`twoF`, `log10BSGL` or `lnBtSG`) will be the return value.

        The full transient-F-stat map will also be computed here,
        but stored in `self.FstatMap`, not returned.

        Parameters
        ----------
        tstart, tend: int or None
            GPS times to restrict the range of data used.
            If None: falls back to self.minStartTime and self.maxStartTime.

        Returns
        -------
        detstat: float
            A single value of the main chosen detection statistic
            (maxTwoF, log10BSGL or lnBtSG)
            at the input parameter values.
        """

        tstart = tstart or self.minStartTime
        tend = tend or self.maxStartTime
        if tstart is None or tend is None:
            raise ValueError(
                "Need tstart or self.minStartTime, and tend or self.maxStartTime!"
            )
        self.windowRange.t0 = int(tstart)  # TYPE UINT4
        if self.windowRange.tauBand == 0:
            self.windowRange.tau = int(tend - tstart)  # TYPE UINT4

        self.FstatMap, self.timingFstatMap = tcw.call_compute_transient_fstat_map(
            version=self.tCWFstatMapVersion,
            features=self.tCWFstatMapFeatures,
            multiFstatAtoms=self.FstatResults.multiFatoms[0],  # single frequency bin
            windowRange=self.windowRange,
            BtSG=getattr(self, "BtSG", False),
        )

        # get the maximum twoF over the transient window range
        self.maxTwoF = 2 * self.FstatMap.maxF
        if np.isnan(self.maxTwoF):
            self.maxTwoF = 0

        if getattr(self, "BtSG", False):
            self.lnBtSG = self.FstatMap.lnBtSG
            return self.lnBtSG
        elif self.BSGL:
            self.get_transient_log10BSGL()
            return self.log10BSGL
        else:
            return self.maxTwoF

    def get_transient_maxTwoFstat(
        self,
        tstart=None,
        tend=None,
    ):
        """Computes the transient maxTwoF statistic at a single point.

        This requires `self.get_fullycoherent_twoF()` to be run first,
        and is itself now only a backwards compatibility / convenience
        wrapper around the more general `get_transient_detstats().`

        The full transient-F-stat map will also be computed here,
        but stored in `self.FstatMap`, not returned.

        Parameters
        ----------
        tstart, tend: int or None
            GPS times to restrict the range of data used.
            If None: falls back to self.minStartTime and self.maxStartTime.

        Returns
        -------
        maxTwoF: float
            A single value of the detection statistic
            at the input parameter values.
            Also stored as `self.maxTwoF`.
        """
        self.get_transient_detstats(tstart=tstart, tend=tend)
        return self.maxTwoF

    def get_transient_log10BSGL(self):
        """Computes a transient detection statistic log10BSGL at a single point.

        This should normally be called through `get_transient_detstats()`,
        but if called stand-alone,
        it requires `self.get_transient_maxTwoFstat()` to be run first.

        The single-detector 2F-stat values
        used for that computation (at the index of `maxTwoF`)
        are saved in `self.twoFXatMaxTwoF`,
        not returned.

        Returns
        -------
        log10BSGL: float
            A single value of the detection statistic log10BSGL
            at the input parameter values.
            Also stored as `self.log10BSGL`.
        """
        # First, we need to also compute per-detector F_mn maps.
        # For now, we use the t0,tau index that maximises the multi-detector F
        # to return BSGL for a signal with those parameters.
        # FIXME: should we instead compute BSGL over the whole F_mn
        # and return the maximum of that?
        idx_maxTwoF = self.FstatMap.get_maxF_idx()
        for X in range(self.FstatResults.numDetectors):
            # The [0] index on the multiFatoms here is over frequency bins;
            # we always operate on a single bin.
            singleIFOmultiFatoms = utils.extract_singleIFOmultiFatoms_from_multiAtoms(
                self.FstatResults.multiFatoms[0], X
            )
            FXstatMap, timingFXstatMap = tcw.call_compute_transient_fstat_map(
                self.tCWFstatMapVersion,
                self.tCWFstatMapFeatures,
                singleIFOmultiFatoms,
                self.windowRange,
                BtSG=False,
            )
            self.twoFXatMaxTwoF[X] = 2 * FXstatMap.F_mn[idx_maxTwoF]
        self.log10BSGL = lalpulsar.ComputeBSGL(
            self.maxTwoF, self.twoFXatMaxTwoF, self.BSGLSetup
        )
        return self.log10BSGL

    def _set_up_cumulative_times(self, tstart, tend, num_segments):
        """Construct time arrays to be used in cumulative twoF computations.

        This allows calculate_twoF_cumulative and predict_twoF_cumulative to use
        the same convention (although the number of segments on use is generally
        different due to the computing time required by predict_twoF_cumulative).

        First segment is hardcoded to spann 2 * self.Tsft. Last segment embraces
        the whole data stream.

        Parameters
        ----------
        tstart, tend: int or None
            GPS times to restrict the range of data used;
            if None: falls back to self.minStartTime and self.maxStartTime;
            if outside those: auto-truncated
        num_segments: int
            Number of segments to split [tstart,tend] into
        """
        tstart = max(tstart, self.minStartTime) if tstart else self.minStartTime
        tend = min(tend, self.maxStartTime) if tend else self.maxStartTime
        min_duration = 2 * self.Tsft
        max_duration = tend - tstart
        cumulative_durations = np.linspace(min_duration, max_duration, num_segments)

        return tstart, tend, cumulative_durations

    def calculate_twoF_cumulative(
        self,
        F0,
        F1,
        F2,
        Alpha,
        Delta,
        asini=None,
        period=None,
        ecc=None,
        tp=None,
        argp=None,
        tstart=None,
        tend=None,
        transient_tstart=None,
        transient_duration=None,
        num_segments=1000,
    ):
        """Calculate the cumulative twoF over subsets of the observation span.

        This means that we consider sub-"segments" of the [tstart,tend] interval,
        each starting at the overall tstart and with increasing durations,
        and compute the 2F for each of these, which for a true CW signal should
        increase roughly with duration towards the full value.

        Parameters
        ----------
        F0, F1, F2, Alpha, Delta: float
            Parameters at which to compute the cumulative twoF.
        asini, period, ecc, tp, argp: float, optional
            Optional: Binary parameters at which to compute the cumulative 2F.
        tstart, tend: int or None
            GPS times to restrict the range of data used.
            If None: falls back to self.minStartTime and self.maxStartTime;.
            If outside those: auto-truncated.
        num_segments: int
            Number of segments to split [tstart,tend] into.
        transient_tstart, transient_duration: float or None
            These are not actually used by this function,
            but just included so a parameters dict can be safely passed.
        Returns
        -------
        cumulative_durations : ndarray of shape (num_segments,)
            Offsets of each segment's tend from the overall tstart.
        twoFs : ndarray of shape (num_segments,)
            Values of twoF computed over
            [[tstart,tstart+duration] for duration in cumulative_durations].
        """

        reset_old_window = None
        if not self.transientWindowType:
            reset_old_window = self.transientWindowType
            self.transientWindowType = "rect"
            self.init_computefstatistic()

        tstart, tend, cumulative_durations = self._set_up_cumulative_times(
            tstart, tend, num_segments
        )

        twoFs = [
            self.get_fullycoherent_detstat(
                tstart=tstart,
                tend=tstart + duration,
                F0=F0,
                F1=F1,
                F2=F2,
                Alpha=Alpha,
                Delta=Delta,
                asini=asini,
                period=period,
                ecc=ecc,
                tp=tp,
                argp=argp,
            )
            for duration in cumulative_durations
        ]

        if reset_old_window is not None:
            self.transientWindowType = reset_old_window
            self.init_computefstatistic()

        return tstart, cumulative_durations, np.array(twoFs)

    def predict_twoF_cumulative(
        self,
        F0,
        Alpha,
        Delta,
        h0,
        cosi,
        psi,
        tstart=None,
        tend=None,
        num_segments=10,
        **predict_fstat_kwargs,
    ):
        """Calculate expected 2F, with uncertainty, over subsets of the observation span.

        This yields the expected behaviour that calculate_twoF_cumulative() can
        be compared against: 2F for CW signals increases with duration
        as we take longer and longer subsets of the total observation span.

        Parameters
        ----------
        F0, Alpha, Delta, h0, cosi, psi: float
            Parameters at which to compute the cumulative predicted twoF.
        tstart, tend: int or None
            GPS times to restrict the range of data used.
            If None: falls back to self.minStartTime and self.maxStartTime.
            If outside those: auto-truncated.
        num_segments: int
            Number of segments to split [tstart,tend] into.
        predict_fstat_kwargs:
            Other kwargs to be passed to utils.predict_fstat().

        Returns
        -------
        tstart: int
            GPS start time of the observation span.
        cumulative_durations: ndarray of shape (num_segments,)
            Offsets of each segment's tend from the overall tstart.
        pfs: ndarray of size (num_segments,)
            Predicted 2F for each segment.
        pfs_sigma: ndarray of size (num_segments,)
            Standard deviations of predicted 2F.

        """
        tstart, tend, cumulative_durations = self._set_up_cumulative_times(
            tstart, tend, num_segments
        )
        out = [
            utils.predict_fstat(
                minStartTime=tstart,
                duration=duration,
                sftfilepattern=self.sftfilepattern,
                h0=h0,
                cosi=cosi,
                psi=psi,
                Alpha=Alpha,
                Delta=Delta,
                F0=F0,
                **predict_fstat_kwargs,
            )
            for duration in cumulative_durations
        ]
        pfs, pfs_sigma = np.array(out).T
        return tstart, cumulative_durations, pfs, pfs_sigma

    def plot_twoF_cumulative(
        self,
        CFS_input,
        PFS_input=None,
        tstart=None,
        tend=None,
        num_segments_CFS=1000,
        num_segments_PFS=10,
        custom_ax_kwargs=None,
        savefig=False,
        label=None,
        outdir=None,
        **PFS_kwargs,
    ):
        """Plot how 2F accumulates over time.

        This compares the accumulation on the actual data set ('CFS', from self.calculate_twoF_cumulative())
        against (optionally) the average expectation ('PFS', from self.predict_twoF_cumulative()).

        Parameters
        ----------
        CFS_input: dict
            Input arguments for self.calculate_twoF_cumulative()
            (besides [tstart, tend, num_segments]).
        PFS_input: dict
            Input arguments for self.predict_twoF_cumulative()
            (besides [tstart, tend, num_segments]).
            If None: do not calculate predicted 2F.
        tstart, tend: int or None
            GPS times to restrict the range of data used.
            If None: falls back to self.minStartTime and self.maxStartTime.
            If outside those: auto-truncated.
        num_segments_(CFS|PFS) : int
            Number of time segments to (compute|predict) twoF.
        custom_ax_kwargs : dict
            Optional axis formatting options.
        savefig : bool
            If true, save the figure in `outdir`.
            If false, return an axis object without saving to disk.
        label: str
            Output filename will be constructed by appending `_twoFcumulative.png`
            to this label. (Ignored unless `savefig=true`.)
        outdir: str
            Output folder (ignored unless `savefig=true`).
        PFS_kwargs: dict
            Other kwargs to be passed to self.predict_twoF_cumulative().

        Returns
        -------
        ax : matplotlib.axes._subplots_AxesSubplot, optional
            The axes object containing the plot.
        """

        # Compute cumulative twoF
        actual_tstart_CFS, taus_CFS, twoFs = self.calculate_twoF_cumulative(
            tstart=tstart,
            tend=tend,
            num_segments=num_segments_CFS,
            **CFS_input,
        )
        taus_CFS_days = taus_CFS / 86400.0

        # Set up plot-related objects
        axis_kwargs = {
            "xlabel": f"Days from $t_\\mathrm{{start}}={actual_tstart_CFS:.0f}$",
            "ylabel": (
                "$\\log_{10}(\\mathrm{BSGL})_{\\mathrm{cumulative}}$"
                if self.BSGL
                else "$\\widetilde{2\\mathcal{F}}_{\\mathrm{cumulative}}$"
            ),
            "xlim": (0, taus_CFS_days[-1]),
        }
        plot_label = (
            f"Cumulative 2F {num_segments_CFS:d} segments"
            f" ({(taus_CFS_days[1] - taus_CFS_days[0]):.2g} days per segment)"
        )

        if custom_ax_kwargs is not None:
            for kwarg in "xlabel", "ylabel":
                if kwarg in custom_ax_kwargs:
                    logger.warning(
                        f"Be careful, overwriting {kwarg} {axis_kwargs[kwarg]}"
                        " with {custom_ax_kwargs[kwarg]}: Check out the units!"
                    )
            axis_kwargs.update(custom_ax_kwargs or {})
            plot_label = custom_ax_kwargs.pop("label", plot_label)

        fig, ax = plt.subplots()
        ax.grid()
        ax.set(**axis_kwargs)

        ax.plot(taus_CFS_days, twoFs, label=plot_label, color="k")

        # Predict cumulative twoF and plot if required
        if PFS_input is not None:
            actual_tstart_PFS, taus_PFS, pfs, pfs_sigma = self.predict_twoF_cumulative(
                tstart=tstart,
                tend=tend,
                num_segments=num_segments_PFS,
                **PFS_input,
                **PFS_kwargs,
            )
            taus_PFS_days = taus_PFS / 86400.0
            assert actual_tstart_CFS == actual_tstart_PFS, (
                "CFS and PFS starting time differs: This shouldn't be the case. "
                "Did you change conventions?"
            )

            ax.fill_between(
                taus_PFS_days,
                pfs - pfs_sigma,
                pfs + pfs_sigma,
                color="cyan",
                label=(
                    "Predicted $\\langle 2\\mathcal{F} \\rangle \\pm 1\\sigma$ band"
                ),
                zorder=-10,
                alpha=0.2,
            )

        if "transient_tstart" in CFS_input and "transient_duration" in CFS_input:
            ax.axvspan(
                (CFS_input["transient_tstart"] - actual_tstart_CFS) / 86400.0,
                (
                    CFS_input["transient_tstart"]
                    + CFS_input["transient_duration"]
                    - actual_tstart_CFS
                )
                / 86400.0,
                color="lightgrey",
                alpha=0.5,
                label="transient duration",
            )

        ax.legend(loc="best")
        if savefig:
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, label + "_twoFcumulative.png"))
            plt.close()
        return ax

    def write_atoms_to_file(self, fnamebase="", comments="%%"):
        """Save F-statistic atoms (time-dependent quantities) for a given parameter-space point.

        Parameters
        ----------
        fnamebase: str
            Basis for output filename, full name will be
            `{fnamebase}_Fstatatoms_{dopplerName}.dat`
            where `dopplerName` is a canonical lalpulsar formatting of the
            'Doppler' parameter space point (frequency-evolution parameters).
        comments: str
            Comments marker character(s) to be prepended to header lines.
            Note that the column headers line
            (last line of the header before the atoms data)
            is printed by lalpulsar, with `%%` as comments marker,
            so (different from most other PyFstat functions)
            the default here is `%%` too.
        """
        multiFatoms = getattr(self.FstatResults, "multiFatoms", None)
        if multiFatoms and multiFatoms[0]:
            dopplerName = lalpulsar.PulsarDopplerParams2String(self.PulsarDopplerParams)
            # fnameAtoms = os.path.join(self.outdir,'Fstatatoms_%s.dat' % dopplerName)
            fnameAtoms = f"{fnamebase}_Fstatatoms_{dopplerName}.dat"
            fo = lal.FileOpen(fnameAtoms, "w")
            for hline in self.output_file_header:
                lal.FilePuts(f"{comments} {hline}\n", fo)
            lalpulsar.write_MultiFstatAtoms_to_fp(fo, multiFatoms[0])
            del fo  # instead of lal.FileClose() which is not SWIG-exported
        else:
            raise RuntimeError(
                "Cannot print atoms vector to file: no FstatResults.multiFatoms, or it is None!"
            )


class SemiCoherentSearch(ComputeFstat):
    """A simple semi-coherent search class.

    This will split the data set into multiple segments,
    run a coherent F-stat search over each,
    and produce a final semi-coherent detection statistic as the sum over segments.

    This does not include any concept of refinement between the two steps,
    as some grid-based semi-coherent search algorithms do;
    both the per-segment coherent F-statistics and the incoherent sum
    are done at the same parameter space point.

    The implementation is based on a simple trick using the transient F-stat map
    functionality: basic F-stat atoms are computed only once over the full data set,
    then the transient code with rectangular 'windows' is used to compute the
    per-segment F-stats, and these are summed to get the semi-coherent result.
    """

    @utils.initializer
    def __init__(
        self,
        label,
        outdir,
        tref,
        nsegs=None,
        sftfilepattern=None,
        binary=False,
        singleFstats=False,
        BSGL=False,
        minStartTime=None,
        maxStartTime=None,
        Tsft=1800,
        minCoverFreq=None,
        maxCoverFreq=None,
        search_ranges=None,
        detectors=None,
        injectSources=None,
        injectSqrtSX=None,
        randSeed=None,
        assumeSqrtSX=None,
        SSBprec=None,
        RngMedWindow=None,
        earth_ephem=None,
        sun_ephem=None,
        allowedMismatchFromSFTLength=None,
    ):
        """
        Only parameters with a special meaning for SemiCoherentSearch itself
        are explicitly documented here.
        For all other parameters inherited from pyfstat.ComputeFStat
        see the documentation of that class.

        Parameters
        ----------
        label, outdir: str
            A label and directory to read/write data from/to.
        tref: int
            GPS seconds of the reference time.
        nsegs: int
            The (fixed) number of segments to split the data set into.
        sftfilepattern: str
            Pattern to match SFTs using wildcards (`*?`) and ranges [0-9];
            multiple patterns can be given separated by colons.
        minStartTime, maxStartTime : int
            Only use SFTs with timestamps starting from this range,
            following the XLALCWGPSinRange convention:
            half-open intervals [minStartTime,maxStartTime].
            Also used to set up segment boundaries, i.e.
            `maxStartTime-minStartTime` will be divided by `nsegs`
            to obtain the per-segment coherence time `Tcoh`.
        """

        self.set_ephemeris_files(earth_ephem, sun_ephem)
        self.transientWindowType = None  # will use semicoherentWindowRange instead
        self.computeAtoms = True  # for semicoh 2F from ComputeTransientFstatMap()
        self.tCWFstatMapVersion = "lal"
        self.cudaDeviceName = None
        self.allowedMismatchFromSFTLength = allowedMismatchFromSFTLength
        self.init_computefstatistic()
        self.init_semicoherent_parameters()
        if self.singleFstats:
            self.twoFX_per_segment = np.zeros(
                (lalpulsar.PULSAR_MAX_DETECTORS, self.nsegs)
            )
        self.get_det_stat = self.get_semicoherent_det_stat

    def _init_semicoherent_window_range(self):
        """
        Use this window to compute the semicoherent Fstat using TransientFstatMaps.
        This way we are able to decouple the semicoherent computation from the
        actual usage of a transient window.
        """
        self.semicoherentWindowRange = lalpulsar.transientWindowRange_t()
        self.semicoherentWindowRange.type = lalpulsar.TRANSIENT_RECTANGULAR

        # Range [t0, t0+t0Band] step dt0
        self.semicoherentWindowRange.t0 = int(self.tboundaries[0])
        self.semicoherentWindowRange.t0Band = int(
            self.tboundaries[-1] - self.tboundaries[0] - self.Tcoh
        )
        self.semicoherentWindowRange.dt0 = int(self.Tcoh)

        # Range [tau, tau + tauBand] step dtau
        # Watch out: dtau must be !=0, but tauBand==0 is allowed
        self.semicoherentWindowRange.tau = int(self.Tcoh)
        self.semicoherentWindowRange.tauBand = int(0)
        self.semicoherentWindowRange.dtau = int(1)  # Irrelevant

    def init_semicoherent_parameters(self):
        """Set up a list of equal-length segments and the corresponding transient windows.

        For a requested number of segments `self.nsegs`,
        `self.tboundaries` will have `self.nsegs+1` entries
        covering `[self.minStartTime,self.maxStartTime]`
        and `self.Tcoh` will be the total duration divided by `self.nsegs`.

        Each segment is required to be at least two SFTs long.f
        """
        logger.info(
            (
                "Initialising semicoherent parameters from"
                " minStartTime={:d} to maxStartTime={:d} in {:d} segments..."
            ).format(self.minStartTime, self.maxStartTime, self.nsegs)
        )
        self.tboundaries = np.linspace(
            self.minStartTime, self.maxStartTime, self.nsegs + 1
        )
        self.Tcoh = self.tboundaries[1] - self.tboundaries[0]
        logger.info(
            ("Obtained {:d} segments of length Tcoh={:f}s (={:f}d).").format(
                self.nsegs, self.Tcoh, self.Tcoh / 86400.0
            )
        )
        logger.debug("Segment boundaries: {}".format(self.tboundaries))
        if self.Tcoh < 2 * self.Tsft:
            raise RuntimeError(
                "Per-segment coherent time {} may not be < Tsft={}"
                " to avoid degenerate F-statistic computations".format(
                    self.Tcoh, self.Tsft
                )
            )
        # FIXME: We can only easily do the next sanity check for a single
        # detector, since self.SFT_timestamps is a joint list for multi-IFO case
        # and the lower-level error checking of XLAL is complicated in that case.
        # But even in the multi-IFO case, if the last segment does not include
        # enough data, there will still be an error message (just uglier) from
        # XLALComputeTransientFstatMap()
        if (
            (len(self.detector_names) == 1)
            and hasattr(self, "SFT_timestamps")
            and (self.tboundaries[-2] > self.SFT_timestamps[-2])
        ):
            raise RuntimeError(
                "Each segment must contain at least 2 SFTs to avoid degenerate"
                " F-statistic computations, but last segment start time {}"
                " is after second-to-last SFT timestamp {}.".format(
                    self.tboundaries[-2], self.SFT_timestamps[-2]
                )
            )
        self._init_semicoherent_window_range()

    def get_semicoherent_det_stat(
        self,
        F0,
        F1,
        F2,
        Alpha,
        Delta,
        asini=None,
        period=None,
        ecc=None,
        tp=None,
        argp=None,
        record_segments=False,
    ):
        """Computes the detection statistic (twoF or log10BSGL) semi-coherently at a single point.

        As the basic statistic of this class, `self.twoF` is always computed.
        If `self.singleFstats`, additionally the single-detector 2F-stat values
        are saved in `self.twoFX` and (optionally) `self.twoFX_per_segment`.

        Parameters
        ----------
        F0, F1, F2, Alpha, Delta: float
            Parameters at which to compute the statistic.
        asini, period, ecc, tp, argp: float, optional
            Optional: Binary parameters at which to compute the statistic.
        record_segments: boolean
            If True, store the per-segment F-stat values as `self.twoF_per_segment`
            and (if `self.singleFstats`) the per-detector per-segment F-stats
            as `self.twoFX_per_segment`.

        Returns
        -------
        stat: float
            A single value of the detection statistic (semi-coherent twoF or log10BSGL)
            at the input parameter values.
            Also stored as `self.twoF` or `self.log10BSGL`.
        """

        self.get_semicoherent_twoF(
            F0, F1, F2, Alpha, Delta, asini, period, ecc, tp, argp, record_segments
        )

        if self.singleFstats:
            self.get_semicoherent_single_IFO_twoFs(record_segments)
        if not self.BSGL:
            return self.twoF
        else:
            return self.get_semicoherent_log10BSGL()

    def get_semicoherent_twoF(
        self,
        F0,
        F1,
        F2,
        Alpha,
        Delta,
        asini=None,
        period=None,
        ecc=None,
        tp=None,
        argp=None,
        record_segments=False,
    ):
        """Computes the semi-coherent twoF statistic at a single point.

        Parameters
        ----------
        F0, F1, F2, Alpha, Delta: float
            Parameters at which to compute the statistic.
        asini, period, ecc, tp, argp: float, optional
            Optional: Binary parameters at which to compute the statistic.
        record_segments: boolean
            If True, store the per-segment F-stat values as `self.twoF_per_segment`.

        Returns
        -------
        twoF: float
            A single value of the semi-coherent twoF statistic
            at the input parameter values.
            Also stored as `self.twoF`.
        """

        self.PulsarDopplerParams.fkdot = np.zeros(lalpulsar.PULSAR_MAX_SPINS)
        self.PulsarDopplerParams.fkdot[:3] = [F0, F1, F2]
        self.PulsarDopplerParams.Alpha = float(Alpha)
        self.PulsarDopplerParams.Delta = float(Delta)
        if self.binary:
            self.PulsarDopplerParams.asini = float(asini)
            self.PulsarDopplerParams.period = float(period)
            self.PulsarDopplerParams.ecc = float(ecc)
            self.PulsarDopplerParams.tp = float(tp)
            self.PulsarDopplerParams.argp = float(argp)

        lalpulsar.ComputeFstat(
            Fstats=self.FstatResults,
            input=self.FstatInput,
            doppler=self.PulsarDopplerParams,
            numFreqBins=1,
            whatToCompute=self.whatToCompute,
        )

        twoF_per_segment = self._get_per_segment_twoF()
        self.twoF = twoF_per_segment.sum()

        if np.isnan(self.twoF):
            logger.debug(
                "NaNs in per-segment 2F treated as zero"
                " and semi-coherent 2F re-computed."
            )
            twoF_per_segment = np.nan_to_num(twoF_per_segment, nan=0.0)
            self.twoF = twoF_per_segment.sum()

        if record_segments:
            self.twoF_per_segment = twoF_per_segment

        return self.twoF

    def get_semicoherent_single_IFO_twoFs(self, record_segments=False):
        """Computes the semi-coherent single-detector F-statss at a single point.

        This requires `self.get_semicoherent_twoF()` to be run first.

        Parameters
        ----------
        record_segments: boolean
            If True, store the per-detector per-segment F-stat values
            as `self.twoFX_per_segment`.


        Returns
        -------
        twoFX: list
            A list of the single-detector detection statistics twoF.
            Also stored as `self.twoFX`.
        """
        if not self.singleFstats:
            raise RuntimeError(
                "This function is available only if singleFstats or BSGL options were set."
            )
        for X in range(self.FstatResults.numDetectors):
            # The [0] index on the multiFatoms here is over frequency bins;
            # we always operate on a single bin.
            singleIFOmultiFatoms = utils.extract_singleIFOmultiFatoms_from_multiAtoms(
                self.FstatResults.multiFatoms[0], X
            )
            FXstatMap = lalpulsar.ComputeTransientFstatMap(
                multiFstatAtoms=singleIFOmultiFatoms,
                windowRange=self.semicoherentWindowRange,
                useFReg=False,
            )
            twoFX_per_segment = 2 * FXstatMap.F_mn.data[:, 0]
            self.twoFX[X] = twoFX_per_segment.sum()
            if np.isnan(self.twoFX[X]):
                logger.debug(
                    "NaNs in per-segment per-detector 2F treated as zero"
                    " and sum re-computed."
                )
                twoFX_per_segment = np.nan_to_num(twoFX_per_segment, nan=0.0)
                self.twoFX[X] = twoFX_per_segment.sum()
            if record_segments:
                self.twoFX_per_segment[: self.FstatResults.numDetectors, :] = (
                    twoFX_per_segment
                )
        return self.twoFX

    def get_semicoherent_log10BSGL(self):
        """Computes the semi-coherent log10BSGL statistic at a single point.

        This requires `self.get_semicoherent_twoF()`
        and `self.get_semicoherent_single_IFO_twoFs()`
        to be run first.

        Returns
        -------
        log10BSGL: float
            A single value of the semi-coherent log10BSGL statistic
            at the input parameter values.
            Also stored as `self.log10BSGL`.
        """
        self.log10BSGL = lalpulsar.ComputeBSGL(self.twoF, self.twoFX, self.BSGLSetup)
        if np.isnan(self.log10BSGL):
            logger.debug("NaNs in semi-coherent log10BSGL treated as zero")
            self.log10BSGL = 0.0
        return self.log10BSGL

    def _get_per_segment_twoF(self):
        Fmap = lalpulsar.ComputeTransientFstatMap(
            multiFstatAtoms=self.FstatResults.multiFatoms[0],
            windowRange=self.semicoherentWindowRange,
            useFReg=False,
        )
        twoF = 2 * Fmap.F_mn.data[:, 0]
        return twoF


class SearchForSignalWithJumps(BaseSearchClass):
    """Internal helper class with some useful methods for glitches or timing noise.

    Users should never need to interact with this class,
    just with the derived search classes.
    """

    def _shift_matrix(self, n, dT):
        """Generate the shift matrix

        Parameters
        ----------
        n : int
            The dimension of the shift-matrix to generate
        dT : float
            The time delta of the shift matrix

        Returns
        -------
        m : ndarray, shape (n,)
            The shift matrix.

        """
        m = np.zeros((n, n))
        factorial = np.math.factorial
        for i in range(n):
            for j in range(n):
                if i == j:
                    m[i, j] = 1.0
                elif i > j:
                    m[i, j] = 0.0
                else:
                    if i == 0:
                        m[i, j] = 2 * np.pi * float(dT) ** (j - i) / factorial(j - i)
                    else:
                        m[i, j] = float(dT) ** (j - i) / factorial(j - i)
        return m

    def _shift_coefficients(self, theta, dT):
        """Shift a set of coefficients by dT

        Parameters
        ----------
        theta : array-like, shape (n,)
            Vector of the expansion coefficients to transform starting from the
            lowest degree e.g [phi, F0, F1,...].
        dT : float
            Difference between the two reference times as tref_new - tref_old.

        Returns
        -------
        theta_new : ndarray, shape (n,)
            Vector of the coefficients as evaluated as the new reference time.
        """
        n = len(theta)
        m = self._shift_matrix(n, dT)
        return np.dot(m, theta)

    def _calculate_thetas(self, theta, delta_thetas, tbounds, theta0_idx=0):
        """Calculates the set of thetas given delta_thetas, the jumps

        This is used when generating data containing glitches or timing noise.
        Specifically, the source parameters of the signal are not constant in
        time, but jump by `delta_theta` at `tbounds`.

        Parameters
        ----------
        theta : array_like
            The source parameters of size (n,).
        delta_thetas : array_like
            The jumps in the source parameters of size (m, n) where m is the
            number of jumps.
        tbounds : array_like
            Time boundaries of the jumps of size (m+2,).
        theta0_idx : int
            Index of the segment for which the theta are defined.

        Returns
        -------
        ndarray
            The set of thetas, shape (m+1, n).

        """
        thetas = [theta]
        for i, dt in enumerate(delta_thetas):
            if i < theta0_idx:
                pre_theta_at_ith_glitch = self._shift_coefficients(
                    thetas[0], tbounds[i + 1] - self.tref
                )
                post_theta_at_ith_glitch = pre_theta_at_ith_glitch - dt
                thetas.insert(
                    0,
                    self._shift_coefficients(
                        post_theta_at_ith_glitch, self.tref - tbounds[i + 1]
                    ),
                )

            elif i >= theta0_idx:
                pre_theta_at_ith_glitch = self._shift_coefficients(
                    thetas[i], tbounds[i + 1] - self.tref
                )
                post_theta_at_ith_glitch = pre_theta_at_ith_glitch + dt
                thetas.append(
                    self._shift_coefficients(
                        post_theta_at_ith_glitch, self.tref - tbounds[i + 1]
                    )
                )
        self.thetas_at_tref = thetas
        return thetas


class SemiCoherentGlitchSearch(SearchForSignalWithJumps, ComputeFstat):
    """A semi-coherent search for CW signals from sources with timing glitches.

    This implements a basic semi-coherent F-stat search in which the data
    is divided into segments either side of the proposed glitch epochs and the
    fully-coherent F-stat in each segment is summed to give the semi-coherent
    F-stat.
    """

    @utils.initializer
    def __init__(
        self,
        label,
        outdir,
        tref,
        minStartTime,
        maxStartTime,
        Tsft=1800,
        nglitch=1,
        sftfilepattern=None,
        theta0_idx=0,
        singleFstats=False,
        BSGL=False,
        minCoverFreq=None,
        maxCoverFreq=None,
        search_ranges=None,
        assumeSqrtSX=None,
        detectors=None,
        SSBprec=None,
        RngMedWindow=None,
        injectSources=None,
        earth_ephem=None,
        sun_ephem=None,
        allowedMismatchFromSFTLength=None,
    ):
        """
        Only parameters with a special meaning for SemiCoherentGlitchSearch itself
        are explicitly documented here.
        For all other parameters inherited from pyfstat.ComputeFStat
        see the documentation of that class.

        Parameters
        ----------
        label, outdir: str
            A label and directory to read/write data from/to.
        tref, minStartTime, maxStartTime: int
            GPS seconds of the reference time, and start and end of the data.
        nglitch: int
            The (fixed) number of glitches.
            This is also allowed to be zero, but occasionally this causes issues,
            in which case please use the basic ComputeFstat class instead.
        sftfilepattern: str
            Pattern to match SFTs using wildcards (`*?`) and ranges [0-9];
            multiple patterns can be given separated by colons.
        theta0_idx: int
            Index (zero-based) of which segment the theta (searched parameters)
            refer to.
            This is useful if providing a tight prior on theta to allow the
            signal to jump to theta (and not just from).
        """

        if self.BSGL:
            raise ValueError(
                f"BSGL option currently not supported by {self.__class__.__name__}."
            )

        self.set_ephemeris_files(earth_ephem, sun_ephem)
        self.transientWindowType = "rect"
        self.t0Band = None
        self.tauBand = None
        self.tCWFstatMapVersion = "lal"
        self.cudaDeviceName = None
        self.binary = False
        self.init_computefstatistic()
        self.get_det_stat = self.get_semicoherent_nglitch_twoF

    def get_semicoherent_nglitch_twoF(self, F0, F1, F2, Alpha, Delta, *args):
        """Returns the semi-coherent glitch summed twoF.

        Parameters
        ----------
        F0, F1, F2, Alpha, Delta: float
            Parameters at which to compute the statistic.
        args: dict
            Additional arguments for the glitch parameters;
            see the source code for full details.

        Returns
        -------
        twoFSum: float
            A single value of the semi-coherent summed detection statistic
            at the input parameter values.
        """

        args = list(args)
        tboundaries = [self.minStartTime] + args[-self.nglitch :] + [self.maxStartTime]
        delta_F0s = args[-3 * self.nglitch : -2 * self.nglitch]
        delta_F1s = args[-2 * self.nglitch : -self.nglitch]
        delta_F2 = np.zeros(len(delta_F0s))
        delta_phi = np.zeros(len(delta_F0s))
        theta = [0, F0, F1, F2]
        delta_thetas = np.atleast_2d(
            np.array([delta_phi, delta_F0s, delta_F1s, delta_F2]).T
        )

        thetas = self._calculate_thetas(
            theta, delta_thetas, tboundaries, theta0_idx=self.theta0_idx
        )

        twoFSum = 0
        for i, theta_i_at_tref in enumerate(thetas):
            ts, te = tboundaries[i], tboundaries[i + 1]
            if te - ts > 1800:
                twoFVal = self.get_fullycoherent_detstat(
                    F0=theta_i_at_tref[1],
                    F1=theta_i_at_tref[2],
                    F2=theta_i_at_tref[3],
                    Alpha=Alpha,
                    Delta=Delta,
                    tstart=ts,
                    tend=te,
                )
                twoFSum += twoFVal

        if np.isfinite(twoFSum):
            return twoFSum
        else:
            return -np.inf

    def compute_glitch_fstat_single(
        self, F0, F1, F2, Alpha, Delta, delta_F0, delta_F1, tglitch
    ):
        """Returns the semi-coherent glitch summed twoF for nglitch=1.

        NOTE: OBSOLETE, used only for testing.
        """

        theta = [F0, F1, F2]
        delta_theta = [delta_F0, delta_F1, 0]
        tref = self.tref

        theta_at_glitch = self._shift_coefficients(theta, tglitch - tref)
        theta_post_glitch_at_glitch = theta_at_glitch + delta_theta
        theta_post_glitch = self._shift_coefficients(
            theta_post_glitch_at_glitch, tref - tglitch
        )

        twoFsegA = self.get_fullycoherent_detstat(
            F0=theta[0],
            F1=theta[1],
            F2=theta[2],
            Alpha=Alpha,
            Delta=Delta,
            tstart=self.minStartTime,
            tend=tglitch,
        )

        if tglitch == self.maxStartTime:
            return twoFsegA

        twoFsegB = self.get_fullycoherent_detstat(
            F0=theta_post_glitch[0],
            F1=theta_post_glitch[1],
            F2=theta_post_glitch[2],
            Alpha=Alpha,
            Delta=Delta,
            tstart=tglitch,
            tend=self.maxStartTime,
        )

        return twoFsegA + twoFsegB


class DeprecatedClass:
    """Outdated classes are marked for future removal by inheriting from this."""

    def __new__(cls, *args, **kwargs):
        logger.warning(
            f"The {cls.__name__} class is no longer maintained"
            " and will be removed in an upcoming release of PyFstat!"
            " If you rely on this class and/or are interested in taking over"
            " maintenance, please report an issue at:"
            " https://github.com/PyFstat/PyFstat/issues",
        )
        return super().__new__(cls, *args, **kwargs)


class DefunctClass:
    """Removed classes are retained for a while but marked by inheriting from this."""

    last_supported_version = None
    pr_welcome = True

    def __new__(cls, *args, **kwargs):
        defunct_message = f"The {cls.__name__} class is no longer included in PyFstat!"
        if cls.last_supported_version:
            defunct_message += (
                f" Last supported version was {cls.last_supported_version}."
            )
        if cls.pr_welcome:
            defunct_message += " Pull requests to reinstate the class are welcome."
        raise NotImplementedError(defunct_message)
