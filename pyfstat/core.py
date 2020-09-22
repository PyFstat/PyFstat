""" The core tools used in pyfstat """


import os
import logging
import copy
from pprint import pformat

import glob
import numpy as np
import scipy.special
import scipy.optimize
from datetime import datetime
import getpass
import socket

import lal
import lalpulsar
import pyfstat.helper_functions as helper_functions
import pyfstat.tcw_fstat_map_funcs as tcw

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

helper_functions.set_up_matplotlib_defaults()
args, tqdm = helper_functions.set_up_command_line_arguments()
detector_colors = {"h1": "C0", "l1": "C1"}


class BaseSearchClass:
    """ The base search class providing parent methods to other searches """

    def __new__(cls, *args, **kwargs):
        logging.info(f"Creating {cls.__name__} object...")
        instance = super().__new__(cls)
        return instance

    def _add_log_file(self, header=None):
        """ Log output to a file, requires class to have outdir and label """
        header = [] if header is None else header
        logfilename = os.path.join(self.outdir, self.label + ".log")
        with open(logfilename, "w") as fp:
            for hline in header:
                fp.write("# {:s}\n".format(hline))
        fh = logging.FileHandler(logfilename)
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s: %(message)s", datefmt="%y-%m-%d %H:%M"
            )
        )
        logging.getLogger().addHandler(fh)

    def _get_list_of_matching_sfts(self):
        """ Returns a list of sfts matching the attribute sftfilepattern """
        sftfilepatternlist = np.atleast_1d(self.sftfilepattern.split(";"))
        matches = [glob.glob(p) for p in sftfilepatternlist]
        matches = [item for sublist in matches for item in sublist]
        if len(matches) > 0:
            return matches
        else:
            raise IOError("No sfts found matching {}".format(self.sftfilepattern))

    def set_ephemeris_files(self, earth_ephem=None, sun_ephem=None):
        """Set the ephemeris files to use for the Earth and Sun

        Parameters
        ----------
        earth_ephem, sun_ephem: str
            Paths of the two files containing positions of Earth and Sun,
            respectively at evenly spaced times, as passed to CreateFstatInput

        Note: If not manually set, default values from get_ephemeris_files()
              are used (looking in ~/.pyfstat or $LALPULSAR_DATADIR)

        """

        earth_ephem_default, sun_ephem_default = helper_functions.get_ephemeris_files()

        if earth_ephem is None:
            self.earth_ephem = earth_ephem_default
        else:
            self.earth_ephem = earth_ephem
        if sun_ephem is None:
            self.sun_ephem = sun_ephem_default
        else:
            self.sun_ephem = sun_ephem

    def _set_init_params_dict(self, argsdict):
        argsdict.pop("self")
        self.init_params_dict = argsdict

    def pprint_init_params_dict(self):
        """
        Pretty-print a parameters dictionary for output file headers.

        Returns a list of lines to be printed,
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
            + pretty_init_parameters[1:-2]
            + [pretty_init_parameters[-1].rstrip("}")]
            + ["}"]
        )
        return pretty_init_parameters

    def get_output_file_header(self):
        header = [
            "date: {}".format(str(datetime.now())),
            "user: {}".format(getpass.getuser()),
            "hostname: {}".format(socket.gethostname()),
            "PyFstat: {}".format(helper_functions.get_version_string()),
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
        params_dict = helper_functions.read_par(
            filename=filename,
            label=label or getattr(self, "label", None),
            outdir=outdir or getattr(self, "outdir", None),
            suffix=suffix,
            raise_error=raise_error,
        )
        return params_dict

    def translate_keys_to_lal(self, dictionary):
        """Convert input keys into lal input keys

        Input keys are F0, F1, F2, ..., while LAL functions
        prefer to use Freq, f1dot, f2dot, ....

        Since lal keys are only used to call for lal routines,
        it makes sense to have this function defined this way
        so it can be called on the fly.

        Parameters
        ----------
        dictionary: dict
            Dictionary to translate. A copy will be made (an returned)
            before translation takes place.

        Returns
        -------
        translated_dict: dict
            Copy of "dictionary" with new keys according to lal.
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
    """ Base class providing interface to `lalpulsar.ComputeFstat` """

    @helper_functions.initializer
    def __init__(
        self,
        tref,
        sftfilepattern=None,
        minStartTime=None,
        maxStartTime=None,
        Tsft=1800,
        binary=False,
        BSGL=False,
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
        assumeSqrtSX=None,
        SSBprec=None,
        RngMedWindow=None,
        tCWFstatMapVersion="lal",
        cudaDeviceName=None,
        computeAtoms=False,
        earth_ephem=None,
        sun_ephem=None,
    ):
        """
        Parameters
        ----------
        tref : int
            GPS seconds of the reference time.
        sftfilepattern : str
            Pattern to match SFTs using wildcards (*?) and ranges [0-9];
            mutiple patterns can be given separated by colons.
        minStartTime, maxStartTime : int
            Only use SFTs with timestamps starting from this range,
            following the XLALCWGPSinRange convention:
            half-open intervals [minStartTime,maxStartTime]
        binary : bool
            If true, search of binary parameters.
        BSGL : bool
            If true, compute the BSGL rather than the twoF value.
        transientWindowType: str
            If 'rect' or 'exp',
            allow for the Fstat to be computed over a transient range.
            ('none' instead of None explicitly calls the transient-window
            function, but with the full range, for debugging)
            (if not None, will also force atoms regardless of computeAtoms option)
        t0Band, tauBand: int
            if >0, search t0 in (minStartTime,minStartTime+t0Band)
                   and tau in (tauMin,2*Tsft+tauBand).
            if =0, only compute CW Fstat with t0=minStartTime,
                   tau=maxStartTime-minStartTime.
        tauMin: int
            defaults to 2*Tsft
        dt0, dtau: int
            grid resolutions in transient start-time and duration,
            both default to Tsft
        detectors : str
            Two-character references to the detectors for which to use data.
            Specify None for no constraint.
            For multiple detectors, separate by comma.
        minCoverFreq, maxCoverFreq : float
            The min and max cover frequency passed to CreateFstatInput.
            For negative values, these will be used as offsets from the min/max
            frequency contained in the sftfilepattern.
            If either is None, search_ranges is used to estimate them.
            If the automatic estimation fails and you do not have a good idea
            what to set these two options to, setting both to -0.5 will
            reproduce the default behaviour of PyFstat <=1.4 and may be a
            reasonably safe fallback in many cases.
        search_ranges: dict
            Dictionary of ranges in all search parameters,
            only used to estimate frequency band passed to CreateFstatInput,
            if minCoverFreq, maxCoverFreq are not specified (=='None').
            For actually running searches,
            grids/points will have to be passed separately to the .run() method.
            The entry for each parameter must be a list of length 1, 2 or 3:
            [single_value], [min,max] or [min,max,step].
        injectSources : dict or str
            Either a dictionary of the values to inject, or a string pointing
            to the .cff file to inject
        injectSqrtSX :
            Per-IFO single-sided PSD values for generating fake Gaussian noise on the fly
        assumeSqrtSX : float
            Don't estimate noise-floors but assume (stationary) per-IFO
            sqrt{SX} (if single value: use for all IFOs). If signal only,
            set sqrtSX=1
        SSBprec : int
            Flag to set the SSB calculation: 0=Newtonian, 1=relativistic,
            2=relativisitic optimised, 3=DMoff, 4=NO_SPIN
        RngMedWindow : int
           Running-Median window size (number of bins)
        tCWFstatMapVersion: str
            Choose between standard 'lal' implementation,
            'pycuda' for gpu, and some others for devel/debug.
        cudaDeviceName: str
            GPU name to be matched against drv.Device output.
        computeAtoms: bool
            request atoms calculations regardless of transientWindowType
        earth_ephem: str
            Earth ephemeris file path
            if None, will check standard sources as per
            helper_functions.get_ephemeris_files()
        sun_ephem: str
            Sun ephemeris file path
            if None, will check standard sources as per
            helper_functions.get_ephemeris_files()
        """

        self._set_init_params_dict(locals())
        self.set_ephemeris_files(earth_ephem, sun_ephem)
        self.init_computefstatistic()
        self.output_file_header = self.get_output_file_header()

    def _get_SFTCatalog(self):
        """Load the SFTCatalog

        If sftfilepattern is specified, load the data. If not, attempt to
        create data on the fly.

        """
        if hasattr(self, "SFTCatalog"):
            return
        if self.sftfilepattern is None:
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

        logging.info("Initialising SFTCatalog")
        constraints = lalpulsar.SFTConstraints()
        constr_str = []
        if self.detectors:
            if "," in self.detectors:
                logging.warning(
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
        logging.info(
            "Loading data matching SFT file name pattern '{}'"
            " with constraints {}.".format(self.sftfilepattern, ", ".join(constr_str))
        )
        self.SFTCatalog = lalpulsar.SFTdataFind(self.sftfilepattern, constraints)
        Tsft_from_catalog = int(1.0 / self.SFTCatalog.data[0].header.deltaF)
        if Tsft_from_catalog != self.Tsft:
            logging.info(
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
        if args.quite is False and args.no_interactive is False:
            try:
                from bashplotlib.histogram import plot_hist

                print("Data timestamps histogram:")
                plot_hist(SFT_timestamps, height=5, bincount=50)
            except ImportError:
                pass

        cl_tconv1 = "lalapps_tconvert {}".format(int(SFT_timestamps[0]))
        output = helper_functions.run_commandline(cl_tconv1, log_level=logging.DEBUG)
        tconvert1 = output.rstrip("\n")
        cl_tconv2 = "lalapps_tconvert {}".format(int(SFT_timestamps[-1]))
        output = helper_functions.run_commandline(cl_tconv2, log_level=logging.DEBUG)
        tconvert2 = output.rstrip("\n")
        logging.info(
            "Data contains SFT timestamps from {} ({}) to (including) {} ({})".format(
                int(SFT_timestamps[0]), tconvert1, int(SFT_timestamps[-1]), tconvert2
            )
        )

        if self.minStartTime is None:
            self.minStartTime = int(SFT_timestamps[0])
        if self.maxStartTime is None:
            # XLALCWGPSinRange() convention: half-open intervals,
            # maxStartTime must always be > last actual SFT timestamp
            self.maxStartTime = int(SFT_timestamps[-1]) + self.Tsft

        self.detector_names = list(set([d.header.name for d in self.SFTCatalog.data]))
        if len(self.detector_names) == 0:
            raise ValueError("No data loaded.")
        logging.info(
            "Loaded {} SFTs from detectors {}".format(
                len(SFT_timestamps), self.detector_names
            )
        )

    def init_computefstatistic(self):
        """ Initialisation step of run_computefstatistic for a single point or search range """

        self._get_SFTCatalog()

        logging.info("Initialising ephems")
        ephems = lalpulsar.InitBarycenter(self.earth_ephem, self.sun_ephem)

        logging.info("Initialising Fstat arguments")
        dFreq = 0
        self.whatToCompute = lalpulsar.FSTATQ_2F
        if self.transientWindowType or self.computeAtoms:
            self.whatToCompute += lalpulsar.FSTATQ_ATOMS_PER_DET

        FstatOAs = lalpulsar.FstatOptionalArgs()
        FstatOAs.randSeed = lalpulsar.FstatOptionalArgsDefaults.randSeed
        if self.SSBprec:
            logging.info("Using SSBprec={}".format(self.SSBprec))
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
            assumeSqrtSX = np.atleast_1d(self.assumeSqrtSX)
            mnf.sqrtSn[: len(assumeSqrtSX)] = assumeSqrtSX
            mnf.length = len(assumeSqrtSX)
            FstatOAs.assumeSqrtSX = mnf
        FstatOAs.prevInput = lalpulsar.FstatOptionalArgsDefaults.prevInput
        FstatOAs.collectTiming = lalpulsar.FstatOptionalArgsDefaults.collectTiming

        if hasattr(self, "injectSources") and type(self.injectSources) == dict:
            logging.info("Injecting source with params: {}".format(self.injectSources))
            PPV = lalpulsar.CreatePulsarParamsVector(1)
            PP = PPV.data[0]
            h0 = self.injectSources["h0"]
            cosi = self.injectSources["cosi"]
            use_aPlus = "aPlus" in dir(PP.Amp)
            print("use_aPlus = {}".format(use_aPlus))
            if use_aPlus:  # lalsuite interface changed in aff93c45
                PP.Amp.aPlus = 0.5 * h0 * (1.0 + cosi ** 2)
                PP.Amp.aCross = h0 * cosi
            else:
                PP.Amp.h0 = h0
                PP.Amp.cosi = cosi

            PP.Amp.phi0 = self.injectSources["phi0"]
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
        elif hasattr(self, "injectSources") and type(self.injectSources) == str:
            logging.info(
                "Injecting source from param file: {}".format(self.injectSources)
            )
            PPV = lalpulsar.PulsarParamsFromFile(self.injectSources, self.tref)
            FstatOAs.injectSources = PPV
        else:
            FstatOAs.injectSources = lalpulsar.FstatOptionalArgsDefaults.injectSources
        if hasattr(self, "injectSqrtSX") and self.injectSqrtSX is not None:
            self.injectSqrtSX = np.atleast_1d(self.injectSqrtSX)
            if len(self.injectSqrtSX) != len(self.detector_names):
                raise ValueError(
                    "injectSqrtSX must be of same length as detector_names ({}!={})".format(
                        len(self.injectSqrtSX), len(self.detector_names)
                    )
                )
            FstatOAs.injectSqrtSX = lalpulsar.MultiNoiseFloor()
            FstatOAs.injectSqrtSX.length = len(self.injectSqrtSX)
            FstatOAs.injectSqrtSX.sqrtSn[
                : FstatOAs.injectSqrtSX.length
            ] = self.injectSqrtSX
        else:
            FstatOAs.injectSqrtSX = lalpulsar.FstatOptionalArgsDefaults.injectSqrtSX
        self._set_min_max_cover_freqs()

        logging.info("Initialising FstatInput")
        self.FstatInput = lalpulsar.CreateFstatInput(
            self.SFTCatalog,
            self.minCoverFreq,
            self.maxCoverFreq,
            dFreq,
            ephems,
            FstatOAs,
        )

        logging.info("Initialising PulsarDoplerParams")
        PulsarDopplerParams = lalpulsar.PulsarDopplerParams()
        PulsarDopplerParams.refTime = self.tref
        PulsarDopplerParams.Alpha = 1
        PulsarDopplerParams.Delta = 1
        PulsarDopplerParams.fkdot = np.array([0, 0, 0, 0, 0, 0, 0])
        self.PulsarDopplerParams = PulsarDopplerParams

        logging.info("Initialising FstatResults")
        self.FstatResults = lalpulsar.FstatResults()

        if self.BSGL:
            if len(self.detector_names) < 2:
                raise ValueError("Can't use BSGL with single detectors data")
            else:
                logging.info("Initialising BSGL")

            # Tuning parameters - to be reviewed
            numDetectors = 2
            if hasattr(self, "nsegs"):
                p_val_threshold = 1e-6
                Fstar0s = np.linspace(0, 1000, 10000)
                p_vals = scipy.special.gammaincc(2 * self.nsegs, Fstar0s)
                Fstar0 = Fstar0s[np.argmin(np.abs(p_vals - p_val_threshold))]
                if Fstar0 == Fstar0s[-1]:
                    raise ValueError("Max Fstar0 exceeded")
            else:
                Fstar0 = 15.0
            logging.info("Using Fstar0 of {:1.2f}".format(Fstar0))
            oLGX = np.zeros(10)
            oLGX[:numDetectors] = 1.0 / numDetectors
            self.BSGLSetup = lalpulsar.CreateBSGLSetup(
                numDetectors, Fstar0, oLGX, True, 1
            )
            self.twoFX = np.zeros(10)
            self.whatToCompute += lalpulsar.FSTATQ_2F_PER_DET

        if self.transientWindowType:
            logging.info("Initialising transient parameters")
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
                        logging.warn(
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
                        logging.warn(
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
                            logging.warn(
                                "Casting non-integer tauMin={} to int...".format(
                                    self.tauMin
                                )
                            )
                            self.tauMin = int(self.tauMin)
                        self.windowRange.tau = self.tauMin

            logging.info("Initialising transient FstatMap features...")
            (
                self.tCWFstatMapFeatures,
                self.gpu_context,
            ) = tcw.init_transient_fstat_map_features(
                self.tCWFstatMapVersion == "pycuda", self.cudaDeviceName
            )

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
            logging.info(
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
                logging.info(
                    "minCoverFreq={:f} provided, using as offset from min(SFTs).".format(
                        self.minCoverFreq
                    )
                )
                # to set *above* min, since minCoverFreq is negative: subtract it
                self.minCoverFreq = minFreq_SFTs - self.minCoverFreq
            if self.maxCoverFreq < 0.0:
                logging.info(
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
        logging.info(
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
        # extract spanned spin-range at reference-time from the template-bank
        # input self.search_ranges must be a dictionary of lists per search parameter
        # which can be either [single_value], [min,max] or [min,max,step].
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
        self.minCoverFreq, self.maxCoverFreq = helper_functions.get_covering_band(
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

    def get_fullycoherent_twoF(
        self,
        tstart,
        tend,
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
        """ Returns twoF or ln(BSGL) fully-coherently at a single point """
        self.PulsarDopplerParams.fkdot = np.array([F0, F1, F2, 0, 0, 0, 0])
        self.PulsarDopplerParams.Alpha = float(Alpha)
        self.PulsarDopplerParams.Delta = float(Delta)
        if self.binary:
            self.PulsarDopplerParams.asini = float(asini)
            self.PulsarDopplerParams.period = float(period)
            self.PulsarDopplerParams.ecc = float(ecc)
            self.PulsarDopplerParams.tp = float(tp)
            self.PulsarDopplerParams.argp = float(argp)

        lalpulsar.ComputeFstat(
            self.FstatResults,
            self.FstatInput,
            self.PulsarDopplerParams,
            1,
            self.whatToCompute,
        )

        if not self.transientWindowType:
            if self.BSGL is False:
                return self.FstatResults.twoF[0]

            twoF = np.float(self.FstatResults.twoF[0])
            self.twoFX[0] = self.FstatResults.twoFPerDet(0)
            self.twoFX[1] = self.FstatResults.twoFPerDet(1)
            log10_BSGL = lalpulsar.ComputeBSGL(twoF, self.twoFX, self.BSGLSetup)
            return log10_BSGL / np.log10(np.exp(1))

        self.windowRange.t0 = int(tstart)  # TYPE UINT4
        if self.windowRange.tauBand == 0:
            # true single-template search also in transient params:
            # actual (t0,tau) window was set with tstart, tend before
            self.windowRange.tau = int(tend - tstart)  # TYPE UINT4

        self.FstatMap, self.timingFstatMap = tcw.call_compute_transient_fstat_map(
            self.tCWFstatMapVersion,
            self.tCWFstatMapFeatures,
            self.FstatResults.multiFatoms[0],
            self.windowRange,
        )
        if self.tCWFstatMapVersion == "lal":
            F_mn = self.FstatMap.F_mn.data
        else:
            F_mn = self.FstatMap.F_mn

        twoF = 2 * np.max(F_mn)
        if self.BSGL is False:
            if np.isnan(twoF):
                return 0
            else:
                return twoF

        FstatResults_single = copy.copy(self.FstatResults)
        FstatResults_single.numDetectors = 1
        FstatResults_single.data = self.FstatResults.multiFatoms[0].data[0]
        FS0 = lalpulsar.ComputeTransientFstatMap(
            FstatResults_single.multiFatoms[0], self.windowRange, False
        )
        FstatResults_single.data = self.FstatResults.multiFatoms[0].data[1]
        FS1 = lalpulsar.ComputeTransientFstatMap(
            FstatResults_single.multiFatoms[0], self.windowRange, False
        )

        # for now, use the Doppler parameter with
        # multi-detector F maximised over t0,tau
        # to return BSGL
        # FIXME: should we instead compute BSGL over the whole F_mn
        # and return the maximum of that?
        idx_maxTwoF = np.argmax(F_mn)

        self.twoFX[0] = 2 * FS0.F_mn.data[idx_maxTwoF]
        self.twoFX[1] = 2 * FS1.F_mn.data[idx_maxTwoF]
        log10_BSGL = lalpulsar.ComputeBSGL(twoF, self.twoFX, self.BSGLSetup)

        return log10_BSGL / np.log10(np.exp(1))

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
        cumulative_fstat_segments=1000,
    ):
        """Calculate the cumulative twoF over subsets of the observation span.

        This means that we consider sub-"segments" of the [tstart,tend] interval,
        each starting at the overall tstart and with increasing durations,
        and compute the 2F for each of these, which for a true CW signal should
        increase roughly with sqrt(duration) towards the full value.

        Parameters
        ----------
        F0, F1, F2, Alpha, Delta: float
            Parameters at which to compute the cumulative twoF
        asini, period, ecc, tp, argp: float, optional
            Optional: Binary parameters at which to compute the cumulative 2F
        tstart, tend: int
            GPS times to restrict the range of data used - automatically
            truncated to the span of data available
            FIXME: tend currently unused
        cumulative_fstat_segments: int
            Number of segments to split [tstart,tend] into

        Returns
        -------
        taus : ndarray of shape (cumulative_fstat_segments,)
            Offsets of each segment's tend from the overall tstart.
        twoFs : ndarray of shape (cumulative_fstat_segments,)
            Values of twoF computed over [[tstart,tstart+tau] for tau in taus].

        Notes
        -----
        The minimum cumulative twoF is hard-coded to be computed over
        the first 6 hours from either the first timestamp in the data (if
        tstart is smaller than it) or tstart.

        """
        SFTminStartTime = self.SFT_timestamps[0]
        SFTmaxStartTime = self.SFT_timestamps[-1]
        tstart = np.max([SFTminStartTime, tstart])
        min_tau = np.max([SFTminStartTime - tstart, 0]) + 3600 * 6
        max_tau = SFTmaxStartTime - tstart
        taus = np.linspace(min_tau, max_tau, cumulative_fstat_segments)
        twoFs = []

        if not self.transientWindowType:
            self.transientWindowType = "rect"
            self.init_computefstatistic()

        for tau in taus:
            detstat = self.get_fullycoherent_twoF(
                tstart=tstart,
                tend=tstart + tau,
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
            twoFs.append(detstat)

        return taus, np.array(twoFs)

    def _calculate_predict_fstat_cumulative(
        self, N, label=None, outdir=None, pfs_input=None
    ):
        """Calculate expected 2F, with uncertainty, over subsets of the observation span.

        This yields the expected behaviour that calculate_twoF_cumulative() can
        be compared again: 2F for CW signals increases with sqrt(duration)
        as we take longer and longer subsets of the total observation span.

        Parameters
        ----------
        N : int
            Number of timesteps to use between minStartTime and maxStartTime.
        pfs_input : dict, optional
            Input kwargs to predict_fstat. Each key should correspond to one
            of the arguments required by PFS (h0, cosi, psi, Alpha, Delta, Freq).
            If None, will try to read parameters from a file instead.
        label, outdir : str, optional
            Alternative to pfs_input. The label and directory to read in the
            .loudest file from. This assumes `loudest` file was created using the
            PyFstat convention os.path.join(outdir, label + ".loudest").
            Ignored if pfs_input is not None.

        Returns
        -------
        times : ndarray of size (N,)
            Endtimes of the sub-segments.
        pfs : ndarray of size (N,)
            Predicted 2F for each segment.
        pfs_sigma : ndarray of size (N,)
            Standard deviations of predicted 2F.

        """

        if pfs_input is None:
            if os.path.isfile(os.path.join(outdir, label + ".loudest")) is False:
                raise ValueError("Need a loudest file to add the predicted Fstat")
            loudest = self.read_par(
                label=label, outdir=outdir, suffix="loudest", raise_error=False
            )
            pfs_input = {
                key: loudest[key]
                for key in ["h0", "cosi", "psi", "Alpha", "Delta", "Freq"]
            }
        times = np.linspace(self.minStartTime, self.maxStartTime, N + 1)[1:]
        times = np.insert(times, 0, self.minStartTime + 86400 / 2.0)
        out = [
            helper_functions.predict_fstat(
                minStartTime=self.minStartTime,
                duration=t - self.minStartTime,
                sftfilepattern=self.sftfilepattern,
                **pfs_input,
            )
            for t in times
        ]
        pfs, pfs_sigma = np.array(out).T
        return times, pfs, pfs_sigma

    def plot_twoF_cumulative(
        self,
        label,
        outdir,
        signal_parameters=None,
        predict_fstat_segments=15,
        custom_axis_kwargs=None,
        plot_label=None,
        savefig=True,
        **calculate_twoF_cumulative_kwargs,
    ):
        """Plot how 2F accumulates over time, and compare with expectation.

        Parameters
        ----------
        label, outdir : str
        add_pfs : bool
            If true, plot the predicted 2F and standard deviation
        signal_parameters: dict
            Dictionary with parameters at which to predict Fstat.
            At least (h0, cosi, psi, Alpha, Delta, F0) keys must be
            present (any other key will simply be ignored).
            Alternatively, these can be read from a ".loudest" file,
            see _calculate_predict_fstat_cumulative() for details.
        predict_fstat_segments : int
            Number of points to use for PredictFStat.
        custom_axis_kwargs : dict
            Optional axis formatting options.
        plot_label : str
            Legend label for the cumulative 2F values computed from data.
        savefig : bool
            If true, save the figure in outdir and return taus, twoFs.
            If false, return axes object.
        calculate_twoF_cumulative_kwargs : dict
            Will be passed on to calculate_twoF_cumulative().

        Returns
        -------
        taus : ndarray of shape (predict_fstat_segments,)
            If savefig, the times up to which the cumulative 2Fs are computed.
        twoFs : ndarray of shape (predict_fstat_segments,)
            If savefig, the cumulative 2F values computed.
        ax : matplotlib.axes._subplots_AxesSubplot, optional
            If savefig is False, the axes object containing the plot.

        """

        taus, twoFs = self.calculate_twoF_cumulative(**calculate_twoF_cumulative_kwargs)
        taus_days = taus / 86400.0

        axis_kwargs = {
            "xlabel": r"Days from $t_{{\rm start}}={:.0f}$".format(
                calculate_twoF_cumulative_kwargs["tstart"]
            ),
            "ylabel": r"$\log_{10}(\mathrm{BSGL})_{\rm cumulative}$"
            if self.BSGL
            else r"$\widetilde{2\mathcal{F}}_{\rm cumulative}$",
            "xlim": (0, taus_days[-1]),
        }
        if custom_axis_kwargs is not None:
            for kwarg in "xlabel", "ylabel":
                if kwarg in custom_axis_kwargs:
                    logging.warning(
                        f"Be careful, overwriting {kwarg} {axis_kwargs[kwarg]}"
                        " with {custom_axis_kwargs[kwarg]}: Check out the units!"
                    )
            axis_kwargs.update(custom_axis_kwargs or {})

        plot_label = plot_label or (
            f"Cumulative 2F {taus.shape[0]:d} segments"
            f" ({(taus_days[1] - taus_days[0]):.2g} days per segment)"
        )

        fig, ax = plt.subplots()
        ax.grid()
        ax.set(**axis_kwargs)

        ax.plot(taus_days, twoFs, label=plot_label, color="k")

        if signal_parameters:
            logging.info(
                f"Compute PFS using signal parameters {signal_parameters} and"
                f" {predict_fstat_segments} segments"
            )
            pfs_input = dict(
                h0=signal_parameters["h0"],
                cosi=signal_parameters["cosi"],
                psi=signal_parameters["psi"],
                Alpha=signal_parameters["Alpha"],
                Delta=signal_parameters["Delta"],
                Freq=signal_parameters["F0"],
            )

            times, pfs, pfs_sigma = self._calculate_predict_fstat_cumulative(
                N=predict_fstat_segments,
                label=label,
                outdir=outdir,
                pfs_input=pfs_input,
            )
            ax.fill_between(
                (times - self.minStartTime) / 86400.0,
                pfs - pfs_sigma,
                pfs + pfs_sigma,
                color="cyan",
                label=(
                    r"Predicted $\langle 2\mathcal{F} " r"\rangle\pm $ 1-$\sigma$ band"
                ),
                zorder=-10,
                alpha=0.2,
            )

        ax.legend(loc="best")
        if savefig:
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, label + "_twoFcumulative.png"))
            plt.close()
            return taus, twoFs
        else:
            return ax

    def get_full_CFSv2_output(self, tstart, tend, F0, F1, F2, Alpha, Delta, tref):
        """ Basic wrapper around CFSv2 to get the full (h0..) output """
        cl_CFSv2 = "lalapps_ComputeFstatistic_v2 --minStartTime={} --maxStartTime={} --Freq={} --f1dot={} --f2dot={} --Alpha={} --Delta={} --refTime={} --DataFiles='{}' --outputLoudest='{}' --ephemEarth={} --ephemSun={}"
        LoudestFile = "loudest.temp"
        helper_functions.run_commandline(
            cl_CFSv2.format(
                tstart,
                tend,
                F0,
                F1,
                F2,
                Alpha,
                Delta,
                tref,
                self.sftfilepattern,
                LoudestFile,
                self.earth_ephem,
                self.sun_ephem,
            )
        )
        loudest = self.read_par(filename=LoudestFile, raise_error=False)
        os.remove(LoudestFile)
        return loudest

    def write_atoms_to_file(self, fnamebase=""):
        multiFatoms = getattr(self.FstatResults, "multiFatoms", None)
        if multiFatoms and multiFatoms[0]:
            dopplerName = lalpulsar.PulsarDopplerParams2String(self.PulsarDopplerParams)
            # fnameAtoms = os.path.join(self.outdir,'Fstatatoms_%s.dat' % dopplerName)
            fnameAtoms = fnamebase + "_Fstatatoms_%s.dat" % dopplerName
            fo = lal.FileOpen(fnameAtoms, "w")
            for hline in self.output_file_header:
                lal.FilePuts("# {:s}\n".format(hline), fo)
            lalpulsar.write_MultiFstatAtoms_to_fp(fo, multiFatoms[0])
            del fo  # instead of lal.FileClose() which is not SWIG-exported
        else:
            raise RuntimeError(
                "Cannot print atoms vector to file: no FstatResults.multiFatoms, or it is None!"
            )

    def __del__(self):
        """
        In pyCuda case without autoinit,
        we need to make sure the context is removed at the end
        """
        if hasattr(self, "gpu_context") and self.gpu_context:
            self.gpu_context.detach()


class SemiCoherentSearch(ComputeFstat):
    """ A semi-coherent search """

    @helper_functions.initializer
    def __init__(
        self,
        label,
        outdir,
        tref,
        nsegs=None,
        sftfilepattern=None,
        binary=False,
        BSGL=False,
        minStartTime=None,
        maxStartTime=None,
        Tsft=1800,
        minCoverFreq=None,
        maxCoverFreq=None,
        search_ranges=None,
        detectors=None,
        injectSources=None,
        assumeSqrtSX=None,
        SSBprec=None,
        RngMedWindow=None,
        earth_ephem=None,
        sun_ephem=None,
    ):
        """
        Parameters
        ----------
        label, outdir: str
            A label and directory to read/write data from/to.
        tref: int
            GPS seconds of the reference time.
        minStartTime, maxStartTime : int
            Only use SFTs with timestamps starting from this range,
            following the XLALCWGPSinRange convention:
            half-open intervals [minStartTime,maxStartTime].
            Also used to set up segment boundaries, i.e.
            maxStartTime-minStartTime will be divided by nsegs
            to obtain the per-segment coherence time Tcoh.
        nsegs: int
            The (fixed) number of segments
        sftfilepattern: str
            Pattern to match SFTs using wildcards (*?) and ranges [0-9];
            multiple patterns can be given separated by colons.

        For all other parameters, see pyfstat.ComputeFStat.
        """

        self.fs_file_name = os.path.join(self.outdir, self.label + "_FS.dat")
        self.set_ephemeris_files(earth_ephem, sun_ephem)
        self.transientWindowType = None  # will use semicoherentWindowRange instead
        self.computeAtoms = True  # for semicoh 2F from ComputeTransientFstatMap()
        self.tCWFstatMapVersion = "lal"
        self.cudaDeviceName = None
        self.init_computefstatistic()
        self.init_semicoherent_parameters()

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
        logging.info(
            (
                "Initialising semicoherent parameters from"
                " minStartTime={:d} to maxStartTime={:d} in {:d} segments..."
            ).format(self.minStartTime, self.maxStartTime, self.nsegs)
        )
        self.tboundaries = np.linspace(
            self.minStartTime, self.maxStartTime, self.nsegs + 1
        )
        self.Tcoh = self.tboundaries[1] - self.tboundaries[0]
        logging.info(
            ("Obtained {:d} segments of length Tcoh={:f}s (={:f}d).").format(
                self.nsegs, self.Tcoh, self.Tcoh / 86400.0
            )
        )
        logging.debug("Segment boundaries: {}".format(self.tboundaries))
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
        """ Returns twoF or ln(BSGL) semi-coherently at a single point """

        self.PulsarDopplerParams.fkdot = np.array([F0, F1, F2, 0, 0, 0, 0])
        self.PulsarDopplerParams.Alpha = float(Alpha)
        self.PulsarDopplerParams.Delta = float(Delta)
        if self.binary:
            self.PulsarDopplerParams.asini = float(asini)
            self.PulsarDopplerParams.period = float(period)
            self.PulsarDopplerParams.ecc = float(ecc)
            self.PulsarDopplerParams.tp = float(tp)
            self.PulsarDopplerParams.argp = float(argp)

        lalpulsar.ComputeFstat(
            self.FstatResults,
            self.FstatInput,
            self.PulsarDopplerParams,
            1,
            self.whatToCompute,
        )

        twoF_per_segment = self._get_per_segment_twoF()
        twoF = twoF_per_segment.sum()

        if np.isnan(twoF):
            logging.debug(
                "NaNs in per-segment 2F treated as zero"
                " and semi-coherent 2F re-computed."
            )
            twoF_per_segment = np.nan_to_num(twoF_per_segment, nan=0.0)
            twoF = twoF_per_segment.sum()

        if record_segments:
            self.twoF_per_segment = twoF_per_segment

        if self.BSGL is False:
            return twoF
        else:
            for X in range(self.FstatResults.numDetectors):
                FstatResults_single = copy.copy(self.FstatResults)
                FstatResults_single.numDetectors = 1
                FstatResults_single.data = self.FstatResults.multiFatoms[0].data[X]
                FSX = lalpulsar.ComputeTransientFstatMap(
                    FstatResults_single.multiFatoms[0],
                    self.semicoherentWindowRange,
                    False,
                )
                twoFX_per_segment = 2 * FSX.F_mn.data[:, 0]
                self.twoFX[X] = twoFX_per_segment.sum()
                if np.isnan(self.twoFX[X]):
                    logging.debug(
                        "NaNs in per-segment per-detector 2F treated as zero"
                        " and sum re-computed."
                    )
                    twoFX_per_segment = np.nan_to_num(twoFX_per_segment, nan=0.0)
                    self.twoFX[X] = twoFX_per_segment.sum()
            log10_BSGL = lalpulsar.ComputeBSGL(twoF, self.twoFX, self.BSGLSetup)
            ln_BSGL = log10_BSGL * np.log(10.0)
            if np.isnan(ln_BSGL):
                logging.debug("NaNs in semi-coherent ln(BSGL) treated as zero")
                ln_BSGL = 0.0
            return ln_BSGL

    def _get_per_segment_twoF(self):
        Fmap = lalpulsar.ComputeTransientFstatMap(
            self.FstatResults.multiFatoms[0], self.semicoherentWindowRange, False
        )
        twoF = 2 * Fmap.F_mn.data[:, 0]
        return twoF


class SearchForSignalWithJumps(BaseSearchClass):
    """ A class which just adds some useful methods for glitches or timing noise """

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
    """A semi-coherent glitch search

    This implements a basic `semi-coherent glitch F-stat in which the data
    is divided into segments either side of the proposed glitches and the
    fully-coherent F-stat in each segment is summed to give the semi-coherent
    F-stat
    """

    @helper_functions.initializer
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
    ):
        """
        Parameters
        ----------
        label, outdir: str
            A label and directory to read/write data from/to.
        tref, minStartTime, maxStartTime: int
            GPS seconds of the reference time, and start and end of the data.
        nglitch: int
            The (fixed) number of glitches; this can zero, but occasionally
            this causes issue (in which case just use ComputeFstat).
        sftfilepattern: str
            Pattern to match SFTs using wildcards (*?) and ranges [0-9];
            mutiple patterns can be given separated by colons.
        theta0_idx, int
            Index (zero-based) of which segment the theta refers to - uyseful
            if providing a tight prior on theta to allow the signal to jump
            too theta (and not just from)

        For all other parameters, see pyfstat.ComputeFStat.
        """

        self.fs_file_name = os.path.join(self.outdir, self.label + "_FS.dat")
        self.set_ephemeris_files(earth_ephem, sun_ephem)
        self.transientWindowType = "rect"
        self.t0Band = None
        self.tauBand = None
        self.tCWFstatMapVersion = "lal"
        self.cudaDeviceName = None
        self.binary = False
        self.init_computefstatistic()

    def get_semicoherent_nglitch_twoF(self, F0, F1, F2, Alpha, Delta, *args):
        """ Returns the semi-coherent glitch summed twoF """

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
                twoFVal = self.get_fullycoherent_twoF(
                    ts,
                    te,
                    theta_i_at_tref[1],
                    theta_i_at_tref[2],
                    theta_i_at_tref[3],
                    Alpha,
                    Delta,
                )
                twoFSum += twoFVal

        if np.isfinite(twoFSum):
            return twoFSum
        else:
            return -np.inf

    def compute_glitch_fstat_single(
        self, F0, F1, F2, Alpha, Delta, delta_F0, delta_F1, tglitch
    ):
        """Returns the semi-coherent glitch summed twoF for nglitch=1

        Note: OBSOLETE, used only for testing
        """

        theta = [F0, F1, F2]
        delta_theta = [delta_F0, delta_F1, 0]
        tref = self.tref

        theta_at_glitch = self._shift_coefficients(theta, tglitch - tref)
        theta_post_glitch_at_glitch = theta_at_glitch + delta_theta
        theta_post_glitch = self._shift_coefficients(
            theta_post_glitch_at_glitch, tref - tglitch
        )

        twoFsegA = self.get_fullycoherent_twoF(
            self.minStartTime, tglitch, theta[0], theta[1], theta[2], Alpha, Delta
        )

        if tglitch == self.maxStartTime:
            return twoFsegA

        twoFsegB = self.get_fullycoherent_twoF(
            tglitch,
            self.maxStartTime,
            theta_post_glitch[0],
            theta_post_glitch[1],
            theta_post_glitch[2],
            Alpha,
            Delta,
        )

        return twoFsegA + twoFsegB
