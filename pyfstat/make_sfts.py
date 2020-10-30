""" pyfstat tools to generate sfts """


import numpy as np
import functools
import logging
import os
import glob
import pkgutil

import lal
import lalpulsar

from pyfstat.core import (
    BaseSearchClass,
    SearchForSignalWithJumps,
    tqdm,
    args,
)
import pyfstat.helper_functions as helper_functions


class KeyboardInterruptError(Exception):
    pass


class InjectionParametersGenerator:
    """
    Draw injection parameter samples from the specified priors and return
    them in the proper format.
    """

    def __init__(self, priors=None, seed=None):
        """
        priors: dict
            Each key refers to one of the signal's parameters (following the PyFstat convention).
            Priors can be given as values in three formats (by order of evaluation):
                1) Callable without required arguments
                    {"ParameterA": np.random.uniform}
                2) Dict containing numpy.random distribution as key and kwargs in a dict as value
                    {"ParameterA": {"uniform": {"low": 0, "high":1}}}
                3) Constant value to be returned as is
                    {"ParameterA": 1.0}
        seed:
            `seed` argument to be feed to numpy.random.default_rng.
        """
        self.set_seed(seed)
        self.set_priors(priors or {})

    def set_priors(self, new_priors):
        """Set priors to draw parameter space points from """
        if type(new_priors) is not dict:
            raise ValueError(
                "new_priors is not a dict type.\nPlease, check "
                "this class' docstring to learn about the expected format: "
                f"{self.__init__.__doc__}"
            )

        self.priors = {}
        self._update_priors(new_priors)

    def set_seed(self, seed):
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)

    def _update_priors(self, new_priors):
        for parameter_name, parameter_prior in new_priors.items():
            if callable(parameter_prior):
                self.priors[parameter_name] = parameter_prior
            elif isinstance(parameter_prior, dict):
                rng_function_name = next(iter(parameter_prior))
                rng_function = getattr(self._rng, rng_function_name)
                rng_kwargs = parameter_prior[rng_function_name]
                self.priors[parameter_name] = functools.partial(
                    rng_function, **rng_kwargs
                )
            else:  # Assume it is something to be returned as is
                self.priors[parameter_name] = functools.partial(
                    lambda x: x, parameter_prior
                )

    def draw(self):
        injection_parameters = {
            parameter_name: parameter_prior()
            for parameter_name, parameter_prior in self.priors.items()
        }
        return injection_parameters

    def __call__(self):
        return self.draw()


class AllSkyInjectionParametersGenerator(InjectionParametersGenerator):
    """Like InjectionParametersGenerator, but with hardcoded priors to perform
    all sky searches. It assumes 1) PyFstat notation and 2) Equatorial coordinates"""

    def set_priors(self, new_priors):
        self._check_if_updating_sky_priors(new_priors)
        super().set_priors({**new_priors, **self.restricted_priors})

    def set_seed(self, seed):
        super().set_seed(seed)
        self.restricted_priors = {
            # This is required because numpy has no arcsin distro
            "Alpha": lambda: self._rng.uniform(low=0.0, high=2 * np.pi),
            "Delta": lambda: np.arcsin(self._rng.uniform(low=-1.0, high=1.0)),
        }

    def _check_if_updating_sky_priors(self, new_priors):
        if any(
            restricted_key in new_priors
            for restricted_key in self.restricted_priors.keys()
        ):
            logging.warning(
                "Ignoring specified sky priors (Alpha, Delta)."
                "This class is explicitly coded to prevent that from happening. Please, restore "
                "to InjectionParametersGenerator if that's really what you want to do."
            )


class Writer(BaseSearchClass):
    """ Instance object for generating SFTs """

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
    ]
    gps_time_and_string_formats_as_LAL = {
        # GPS times should NOT be parsed using scientific notation
        # LAL routines silently parse them wrongly
        "refTime": ":10.9f",
        "transientWindowType": ":s",
    }

    @helper_functions.initializer
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
        SFTWindowBeta=0.0,
        Band=None,
        detectors=None,
        earth_ephem=None,
        sun_ephem=None,
        transientWindowType="none",
        transientStartTime=None,
        transientTau=None,
        randSeed=None,
    ):
        """
        Parameters
        ----------
        label: string
            a human-readable label to be used in naming the output files
        tstart, duration : int
            start and duration (in gps seconds) of the total observation span
        tref: float or None
            reference time (default is None, which sets the reference time to
            tstart)
        F0: float or None
            frequency of signal to inject,
            also used (if Band is not None) as center of frequency band;
            also needed when noise-only (h0==None or ==0)
            but no noiseSFTs given,
            then again used as center of frequency band.
        F1, F2, Alpha, Delta, h0, cosi, psi, phi: float or None
            frequency evolution and amplitude parameters for injection
            if h0==None or h0==0, these are all ignored
            if h0>0, then Alpha, Delta, cosi need to be set explicitly
        Tsft: int
            The SFT duration in seconds.
            Will be ignored if noiseSFTs are given.
        noiseSFTs: str or None
            SFT on top of which signals will be injected.
            If not None, additional constraints can be applied using the arguments
            tstart and duration.
        Band: float or None
            If float, and F0 is also not None, then output SFTs cover
            [F0-Band/2,F0+Band/2].
            If None and noiseSFTs given, use their bandwidth.
            If None and no noiseSFTs given,
            a minimal covering band for a perfectly-matched
            single-template ComputeFstat analysis is estimated.
        see `lalapps_Makefakedata_v5 --help` for help with the other paramaters
        """

        self.set_ephemeris_files(earth_ephem, sun_ephem)
        self.basic_setup()
        self.parse_args_consistent_with_mfd()
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

        logging.info(
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
            lalpulsar.OfficialSFTFilename(
                dets[0],
                dets[1],
                numSFTs,
                self.Tsft,
                self.tstart,
                effective_duration,
                self.label,
            )
            for ind, dets in enumerate(IFOs)
        ]

    def _gps_to_int(self, tGPS):
        """Simple LIGOTimeGPS helper, SWIG-LAL does not import this functionality"""
        return tGPS.gpsSeconds

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
            this_start_time = self._gps_to_int(time_stamps.data[0])
            this_end_time = self._gps_to_int(time_stamps.data[-1]) + this_Tsft

            self.sftfilenames.append(
                lalpulsar.OfficialSFTFilename(
                    ifo_name[0],
                    ifo_name[1],
                    ifo_catalog.length,
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
            logging.warning(
                f"Overwriting self.Tsft={self.Tsft}"
                f" with value {Tsft[0]} read from noiseSFTs."
            )
        self.Tsft = Tsft[0]
        self.tstart = min(tstart)
        self.duration = max(tend) - self.tstart
        self.detectors = ",".join(IFOs)

    def basic_setup(self):
        os.makedirs(self.outdir, exist_ok=True)
        self.config_file_name = os.path.join(self.outdir, self.label + ".cff")
        self.theta = np.array([self.phi, self.F0, self.F1, self.F2])

        required_signal_params = [
            # leaving out "F1","F2","psi","phi","tref" as they have defaults
            "F0",
            "Alpha",
            "Delta",
            "cosi",
        ]
        if self.h0 and np.any(
            [getattr(self, k, None) is None for k in required_signal_params]
        ):
            raise ValueError(
                "If h0>0, also need all of ({:s})".format(
                    ",".join(required_signal_params)
                )
            )

        no_noiseSFTs_options = ["tstart", "duration", "detectors"]
        if self.noiseSFTs is not None:
            logging.warning(
                "noiseSFTs is not None: Inferring tstart, duration, Tsft. "
                "Input tstart and duration will be treated as SFT constraints "
                "using lalpulsar.SFTConstraints; Tsft will be checked for "
                "internal consistency accross input SFTs."
            )
            self._get_setup_from_noiseSFTs()
        elif np.any([getattr(self, k) is None for k in no_noiseSFTs_options]):
            raise ValueError(
                "Need either noiseSFTs or all of ({:s}).".format(
                    ",".join(no_noiseSFTs_options)
                )
            )
        else:
            self._get_setup_from_tstart_duration()

        self.sftfilepath = ";".join(
            [os.path.join(self.outdir, fn) for fn in self.sftfilenames]
        )

        if self.tref is None:
            self.tref = self.tstart

    def tend(self):
        return self.tstart + self.duration

    def parse_args_consistent_with_mfd(self):
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

        if self.signal_parameters["transientWindowType"] != "none":
            self.signal_parameters["transientStartTime"] = self.transientStartTime
            self.signal_formats["transientStartTime"] = ":10.0f"
            self.signal_parameters["transientTau"] = self.transientTau
            self.signal_formats["transientTau"] = ":10.0f"

    def calculate_fmin_Band(self):
        """
        Set fmin and Band for the output SFTs to cover.

        Either uses the user-provided Band and puts F0 in the middle,
        does nothing to later reuse full bandwidth of noiseSFTs,
        or if F0!=None, noiseSFTs==None and Band==None
        it estimates a minimal band for just the injected signal:
        F-stat covering band plus extra bins for demod default parameters.
        This way a perfectly matched single-template ComputeFstat analysis
        should run through perfectly on the SFTs.
        For any wider-band or mismatched search, the set Band manually.

        If you want to use noiseSFTs but auto-estimate a minimal band,
        call helper_functions.get_covering_band() yourself
        and pass the results as fmin, Band.
        """
        if self.F0 is not None and self.Band is not None:
            self.fmin = self.F0 - 0.5 * self.Band
        elif self.noiseSFTs:
            logging.info("Generating SFTs with full bandwidth from noiseSFTs.")
        elif self.F0 is None:
            raise ValueError(
                "Need F0 and Band, or one of (F0 or noiseSFTs)"
                " to auto-estimate bandwidth."
            )
        else:
            extraBins = (
                # matching extraBinsFull in XLALCreateFstatInput():
                # https://lscsoft.docs.ligo.org/lalsuite/lalpulsar/_compute_fstat_8c_source.html#l00490
                lalpulsar.FstatOptionalArgsDefaults.Dterms
                + int(lalpulsar.FstatOptionalArgsDefaults.runningMedianWindow / 2)
                + 1
            )
            logging.info(
                "Estimating required SFT frequency range from properties"
                " of signal to inject plus {:d} extra bins either side"
                " (corresponding to default F-statistic settings).".format(extraBins)
            )
            minCoverFreq, maxCoverFreq = helper_functions.get_covering_band(
                tref=self.tref,
                tstart=self.tstart,
                tend=self.tend(),
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
            logging.info(
                "Generating SFTs with fmin={}, Band={}".format(self.fmin, self.Band)
            )

    def get_single_config_line(self, i):
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
        """
        Generates a .cff file

        """

        content = self.get_single_config_line(0)

        if verbose:
            logging.info("Injection parameters:")
            logging.info(content.rstrip("\n"))

        if self.check_if_cff_file_needs_rewriting(content):
            logging.info("Writing config file: {:s}".format(self.config_file_name))
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
        """

        need_new = "Will create new SFT file(s)."

        logging.info("Checking if we can re-use existing SFT data file(s)...")
        sftfiles = self.sftfilepath.split(";")
        for sftfile in sftfiles:
            if os.path.isfile(sftfile) is False:
                logging.info(
                    "...no SFT file matching '{}' found. {}".format(sftfile, need_new)
                )
                return False
        logging.info("...OK: file(s) found matching '{}'.".format(sftfile))

        if "injectionSources" in cl_mfd:
            if os.path.isfile(self.config_file_name):
                if np.any(
                    [
                        os.path.getmtime(sftfile)
                        < os.path.getmtime(self.config_file_name)
                        for sftfile in sftfiles
                    ]
                ):
                    logging.info(
                        (
                            "...the config file '{}' has been modified since"
                            " creation of the SFT file(s) '{}'. {}"
                        ).format(self.config_file_name, self.sftfilepath, need_new)
                    )
                    return False
                else:
                    logging.info(
                        "...OK: The config file '{}' is older than the SFT file(s)"
                        " '{}'.".format(self.config_file_name, self.sftfilepath)
                    )
                    # NOTE: at this point we assume it's safe to re-use, since
                    # check_if_cff_file_needs_rewriting()
                    # should have already been called before
            else:
                raise RuntimeError(
                    "Commandline requires file '{}' but it is missing.".format(
                        self.config_file_name
                    )
                )

        logging.info("...checking new commandline against existing SFT header(s)...")
        # here we check one SFT header from each SFT file,
        # assuming that any concatenated file has been sanely constructed with
        # matching CLs
        for sftfile in sftfiles:
            catalog = lalpulsar.SFTdataFind(sftfile, None)
            cl_old = helper_functions.get_lalapps_commandline_from_SFTDescriptor(
                catalog.data[0]
            )
            if len(cl_old) == 0:
                logging.info(
                    "......could not obtain comparison commandline from first SFT"
                    " header in old file '{}'. {}".format(sftfile, need_new)
                )
                return False
            if not helper_functions.match_commandlines(cl_old, cl_mfd):
                logging.info(
                    "......commandlines unmatched for first SFT in old"
                    " file '{}'. {}".format(sftfile, need_new)
                )
                return False
        logging.info("......OK: Commandline matched with old SFT header(s).")
        logging.info(
            "...all data consistency checks passed: Looks like existing"
            " SFT data matches current options, will re-use it!"
        )
        return True

    def check_if_cff_file_needs_rewriting(self, content):
        """Check if the .cff file has changed

        Returns True if the file should be overwritten - where possible avoid
        overwriting to allow cached data to be used
        """
        logging.info("Checking if we can re-use injection config file...")
        if os.path.isfile(self.config_file_name) is False:
            logging.info("...no config file {} found.".format(self.config_file_name))
            return True
        else:
            logging.info(
                "...OK: config file {} already exists.".format(self.config_file_name)
            )

        with open(self.config_file_name, "r") as f:
            file_content = f.read()
            if file_content == content:
                logging.info(
                    "...OK: file contents match, no update of {} required.".format(
                        self.config_file_name
                    )
                )
                return False
            else:
                logging.info(
                    "...file contents unmatched, updating {}.".format(
                        self.config_file_name
                    )
                )
                return True

    def make_data(self, verbose=False):
        """ A convienience wrapper to generate a cff file then sfts """
        if self.h0:
            self.make_cff(verbose)
        else:
            logging.info("Got h0=0, not writing an injection .cff file.")
        self.run_makefakedata()

    def run_makefakedata(self):
        """ Generate the sft data from the configuration file """

        # Remove old data:
        try:
            os.unlink(os.path.join(self.outdir, "*" + self.label + "*.sft"))
        except OSError:
            pass

        mfd = "lalapps_Makefakedata_v5"
        cl_mfd = [mfd]
        cl_mfd.append("--outSingleSFT=TRUE")
        cl_mfd.append('--outSFTdir="{}"'.format(self.outdir))
        cl_mfd.append('--outLabel="{}"'.format(self.label))

        if self.noiseSFTs is not None and self.SFTWindowType is None:
            raise ValueError(
                "SFTWindowType is required when using noiseSFTs. "
                "Please, make sure you understand the window function used "
                "to produce noiseSFTs."
            )
        elif self.noiseSFTs is not None:
            if self.sqrtSX and np.any(
                [s > 0 for s in helper_functions.parse_list_of_numbers(self.sqrtSX)]
            ):
                logging.warning(
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
            cl_mfd.append("--SFTWindowBeta={}".format(self.SFTWindowBeta))
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

        cl_mfd = " ".join(cl_mfd)
        check_ok = self.check_cached_data_okay_to_use(cl_mfd)
        if check_ok is False:
            helper_functions.run_commandline(cl_mfd)
            if np.all([os.path.isfile(f) for f in self.sftfilepath.split(";")]):
                logging.info(f"Successfully wrote SFTs to: {self.sftfilepath}")
            else:
                raise RuntimeError(
                    f"It seems we successfully ran {mfd},"
                    f" but did not get the expected SFT file path(s): {self.sftfilepath}."
                )

    def predict_fstat(self, assumeSqrtSX=None):
        """ Wrapper to lalapps_PredictFstat """
        twoF_expected, twoF_sigma = helper_functions.predict_fstat(
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
    """ Instance object for generating SFTs containing a continuous wave signal for a source in a binary system """

    @helper_functions.initializer
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
        SFTWindowBeta=0.0,
        Band=None,
        detectors=None,
        earth_ephem=None,
        sun_ephem=None,
        transientWindowType="none",
        transientStartTime=None,
        transientTau=None,
        randSeed=None,
    ):
        """
        Parameters
        ----------
        label: string
            a human-readable label to be used in naming the output files
        tstart, duration : int
            start and duration (in gps seconds) of the total observation span
        tref: float or None
            reference time (default is None, which sets the reference time to
            tstart)
        F0, F1, F2, Alpha, Delta, tp, argp, asini, ecc, period, h0, cosi, psi, phi: float
            frequency, sky-position, binary orbit and amplitude parameters
        Tsft: float
            the sft duration
        see `lalapps_Makefakedata_v5 --help` for help with the other paramaters
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


class GlitchWriter(SearchForSignalWithJumps, Writer):
    """ Instance object for generating SFTs containing glitch signals """

    @helper_functions.initializer
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
        SFTWindowBeta=0.0,
        Band=None,
        detectors=None,
        earth_ephem=None,
        sun_ephem=None,
        transientWindowType="rect",
        randSeed=None,
    ):
        """
        Parameters
        ----------
        label: string
            a human-readable label to be used in naming the output files
        tstart, duration : float
            start and duration (in gps seconds) of the total observation span
        dtglitch: float
            time (in gps seconds) of the glitch after tstart. To create data
            without a glitch, set dtglitch=None
        delta_phi, delta_F0, delta_F1: float
            instanteneous glitch magnitudes in rad, Hz, and Hz/s respectively
        tref: float or None
            reference time (default is None, which sets the reference time to
            tstart)
        F0, F1, F2, Alpha, Delta, h0, cosi, psi, phi: float
            frequency, sky-position, and amplitude parameters
        Tsft: float
            the sft duration

        see `lalapps_Makefakedata_v5 --help` for help with the other paramaters
        """

        self.set_ephemeris_files(earth_ephem, sun_ephem)
        self.basic_setup()
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
            self.tbounds = [self.tstart, self.tend()]
        else:
            self.dtglitch = np.atleast_1d(self.dtglitch)
            self.tglitch = self.tstart + self.dtglitch
            self.tbounds = np.concatenate(([self.tstart], self.tglitch, [self.tend()]))
        logging.info("Using segment boundaries {}".format(self.tbounds))

        tbs = np.array(self.tbounds)
        self.durations = tbs[1:] - tbs[:-1]

        self.delta_thetas = np.atleast_2d(
            np.array([delta_phi, delta_F0, delta_F1, delta_F2]).T
        )

    def get_base_template(self, i, Alpha, Delta, h0, cosi, psi, phi, F0, F1, F2, tref):
        """FIXME: ported over from Writer,
        should be replaced by a more elegant re-use of parse_args_consistent_with_mfd
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

    def get_single_config_line_cw(
        self, i, Alpha, Delta, h0, cosi, psi, phi, F0, F1, F2, tref
    ):
        template = (
            self.get_base_template(
                i, Alpha, Delta, h0, cosi, psi, phi, F0, F1, F2, tref
            )
            + """\n"""
        )
        return template.format(i, Alpha, Delta, h0, cosi, psi, phi, F0, F1, F2, tref)

    def get_single_config_line_tcw(
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
            self.get_base_template(
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

    def get_single_config_line(
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
            return self.get_single_config_line_cw(
                i, Alpha, Delta, h0, cosi, psi, phi, F0, F1, F2, tref
            )
        else:
            return self.get_single_config_line_tcw(
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
        """
        Generates an .cff file for a 'glitching' signal

        """

        thetas = self._calculate_thetas(self.theta, self.delta_thetas, self.tbounds)

        content = ""
        for i, (t, d, ts) in enumerate(zip(thetas, self.durations, self.tbounds[:-1])):
            line = self.get_single_config_line(
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
            logging.info("Injection parameters:")
            logging.info(content.rstrip("\n"))

        if self.check_if_cff_file_needs_rewriting(content):
            logging.info("Writing config file: {:s}".format(self.config_file_name))
            config_file = open(self.config_file_name, "w+")
            config_file.write(content)
            config_file.close()


class FrequencyModulatedArtifactWriter(Writer):
    """ Instance object for generating SFTs containing artifacts """

    @helper_functions.initializer
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
        tstart, duration : int
            start and duration times (in gps seconds) of the total observation
        Pmod, F0, F1 h0: float
            Modulation period, freq, freq-drift, and h0 of the artifact
        Alpha, Delta: float
            Sky position, in radians, of a signal of which to add the orbital
            modulation to the artifact, if not `None`.
        Tsft: float
            the sft duration
        sqrtSX: float
            Background IFO noise

        see `lalapps_Makefakedata_v4 --help` for help with the other paramaters
        """

        self.phi = 0
        self.F2 = 0

        self.basic_setup()
        self.set_ephemeris_files(earth_ephem, sun_ephem)
        self.tstart = int(tstart)
        self.duration = int(duration)

        if os.path.isdir(self.outdir) is False:
            os.makedirs(self.outdir)
        if tref is None:
            raise ValueError("Input `tref` not specified")

        self.nsfts = int(np.ceil(self.duration / self.Tsft))
        self.calculate_fmin_Band()

        self.cosi = 0
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
                    "This class currently only supports detectors H1 or L1."
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
        return self.h0

    def concatenate_sft_files(self):
        SFTFilename = lalpulsar.OfficialSFTFilename(
            self.detectors[0],
            self.detectors[1],
            self.nsfts,
            self.Tsft,
            int(self.tstart),
            int(self.duration),
            self.label,
        )
        SFTFile_fullpath = os.path.join(self.outdir, SFTFilename)

        # If the file already exists, simply remove it for now (no caching
        # implemented)
        helper_functions.run_commandline(
            "rm {}".format(SFTFile_fullpath), raise_error=False, log_level=10
        )

        inpattern = os.path.join(self.tmp_outdir, "*sft")
        cl_splitSFTS = "lalapps_splitSFTs -fs {} -fb {} -fe {} -o {} -i {}".format(
            self.fmin, self.Band, self.fmin + self.Band, SFTFile_fullpath, inpattern
        )
        helper_functions.run_commandline(cl_splitSFTS)
        helper_functions.run_commandline("rm {} -r".format(self.tmp_outdir))
        files = glob.glob(SFTFile_fullpath + "*")
        if len(files) == 1:
            fn = files[0]
            fn_new = fn.split(".")[0] + ".sft"
            helper_functions.run_commandline("mv {} {}".format(fn, fn_new))
        else:
            raise IOError(
                "Attempted to rename file, but multiple files found: {}".format(files)
            )

    def pre_compute_evolution(self):
        logging.info("Precomputing evolution parameters")
        self.lineFreqs = []
        self.linePhis = []
        self.lineh0s = []
        self.mid_times = []

        linePhi = 0
        lineFreq_old = 0
        for i in tqdm(list(range(self.nsfts))):
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
        try:
            self.run_makefakedata_v4(
                self.mid_times[i],
                self.lineFreqs[i],
                self.linePhis[i],
                self.lineh0s[i],
                self.tmp_outdir,
            )
        except KeyboardInterrupt:
            raise KeyboardInterruptError()

    def make_data(self):
        self.duration = self.Tsft

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

        logging.info("Generating SFTs")

        if args.N > 1 and pkgutil.find_loader("pathos") is not None:
            import pathos.pools

            logging.info("Using {} threads".format(args.N))
            try:
                with pathos.pools.ProcessPool(args.N) as p:
                    list(
                        tqdm(
                            p.imap(self.make_ith_sft, list(range(self.nsfts))),
                            total=self.nsfts,
                        )
                    )
            except KeyboardInterrupt:
                p.terminate()
        else:
            logging.info(
                "No multiprocessing requested or `pathos` not install, cont."
                " without multiprocessing"
            )
            for i in tqdm(list(range(self.nsfts))):
                self.make_ith_sft(i)

        self.concatenate_sft_files()

    def run_makefakedata_v4(self, mid_time, lineFreq, linePhi, h0, tmp_outdir):
        """ Generate the sft data using the --lineFeature option """
        cl_mfd = []
        cl_mfd.append("lalapps_Makefakedata_v4")
        cl_mfd.append("--outSingleSFT=FALSE")
        cl_mfd.append('--outSFTbname="{}"'.format(tmp_outdir))
        cl_mfd.append("--IFO={}".format(self.detectors))
        cl_mfd.append('--noiseSqrtSh="{}"'.format(self.sqrtSX))
        cl_mfd.append("--startTime={:0.0f}".format(mid_time - self.Tsft / 2.0))
        cl_mfd.append("--refTime={:0.0f}".format(mid_time))
        cl_mfd.append("--duration={}".format(self.duration))
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
        helper_functions.run_commandline(cl_mfd, log_level=10)


class FrequencyAmplitudeModulatedArtifactWriter(FrequencyModulatedArtifactWriter):
    """ Instance object for generating SFTs containing artifacts """

    def get_h0(self, t):
        return self.h0 * np.sin(2 * np.pi * t / self.Pmod + self.Pmod_phi)
