""" pyfstat tools to generate sfts """


import numpy as np
import logging
import os
import glob
import pkgutil

import lal
import lalpulsar

from pyfstat.core import (
    BaseSearchClass,
    tqdm,
    args,
    predict_fstat,
    translate_keys_to_lal,
)
import pyfstat.helper_functions as helper_functions


class KeyboardInterruptError(Exception):
    pass


class Writer(BaseSearchClass):
    """ Instance object for generating SFTs """

    @helper_functions.initializer
    def __init__(
        self,
        label="Test",
        tstart=700000000,
        duration=100 * 86400,
        tref=None,
        F0=30,
        F1=1e-10,
        F2=0,
        Alpha=5e-3,
        Delta=6e-2,
        h0=0.1,
        cosi=0.0,
        psi=0.0,
        phi=0,
        Tsft=1800,
        outdir=".",
        sqrtSX=0.0,
        noiseSFTs=None,
        SFTWindowType=None,
        SFTWindowBeta=0.0,
        Band=4,
        detectors="H1",
        minStartTime=None,
        maxStartTime=None,
        add_noise=True,
        transientWindowType="none",
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
        F0, F1, F2, Alpha, Delta, h0, cosi, psi, phi: float
            frequency, sky-position, and amplitude parameters
        Tsft: float
            the sft duration
        minStartTime, maxStartTime: float
            if not None, the total span of data, this can be used to generate
            transient signals

        see `lalapps_Makefakedata_v5 --help` for help with the other paramaters
        """

        self.set_ephemeris_files()
        self.basic_setup()
        self.calculate_fmin_Band()

        self.tbounds = [self.tstart, self.tend]
        logging.info("Using segment boundaries {}".format(self.tbounds))

    def basic_setup(self):
        self.tstart = int(self.tstart)
        self.duration = int(self.duration)
        self.tend = self.tstart + self.duration
        if self.minStartTime is None:
            self.minStartTime = self.tstart
        if self.maxStartTime is None:
            self.maxStartTime = self.tend
        self.minStartTime = int(self.minStartTime)
        self.maxStartTime = int(self.maxStartTime)
        self.duration_days = (self.tend - self.tstart) / 86400

        self.data_duration = self.maxStartTime - self.minStartTime

        IFOs = self.detectors.split(",")
        if self.noiseSFTs:
            numSFTs = []
            for ifo in IFOs:
                constraint = lalpulsar.SFTConstraints()
                constraint.detector = ifo
                constraint.minStartTime = lal.LIGOTimeGPS(self.minStartTime)
                constraint.maxStartTime = lal.LIGOTimeGPS(self.maxStartTime)
                catalog = lalpulsar.SFTdataFind(self.noiseSFTs, constraint)
                numSFTs.append(catalog.length)
        else:
            numSFTs = len(IFOs) * [int(float(self.data_duration) / self.Tsft)]

        self.theta = np.array([self.phi, self.F0, self.F1, self.F2])

        if os.path.isdir(self.outdir) is False:
            os.makedirs(self.outdir)
        if self.tref is None:
            self.tref = self.tstart
        self.config_file_name = os.path.join(self.outdir, self.label + ".cff")
        self.sftfilenames = [
            lalpulsar.OfficialSFTFilename(
                dets[0],
                dets[1],
                numSFTs[ind],
                self.Tsft,
                self.minStartTime,
                self.data_duration,
                self.label,
            )
            for ind, dets in enumerate(IFOs)
        ]
        self.sftfilepath = ";".join(
            [os.path.join(self.outdir, fn) for fn in self.sftfilenames]
        )
        self.IFOs = ",".join(['"{}"'.format(d) for d in IFOs])

    def make_data(self):
        """ A convienience wrapper to generate a cff file then sfts """
        self.make_cff()
        self.run_makefakedata()

    def get_base_template(self, i, Alpha, Delta, h0, cosi, psi, phi, F0, F1, F2, tref):
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
        tstart,
        duration_days,
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
            tstart,
            duration_days * 86400,
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
        tstart,
        duration_days,
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
                tstart,
                duration_days,
            )

    def make_cff(self):
        """
        Generates a .cff file

        """

        content = self.get_single_config_line(
            0,
            self.Alpha,
            self.Delta,
            self.h0,
            self.cosi,
            self.psi,
            self.phi,
            self.F0,
            self.F1,
            self.F2,
            self.tref,
            self.transientWindowType,
            self.tstart,
            self.duration_days,
        )

        if self.check_if_cff_file_needs_rewritting(content):
            config_file = open(self.config_file_name, "w+")
            config_file.write(content)
            config_file.close()

    def calculate_fmin_Band(self):
        self.fmin = self.F0 - 0.5 * self.Band

    def check_cached_data_okay_to_use(self, cl_mfd):
        """ Check if cached data exists and, if it does, if it can be used """

        need_new = "Will create new SFT file."

        logging.info("Checking if cached data good to reuse...")
        if os.path.isfile(self.sftfilepath) is False:
            logging.info(
                "No SFT file matching {} found. {}".format(self.sftfilepath, need_new)
            )
            return False
        else:
            logging.info("OK: Matching SFT file found.")

        if "injectionSources" in cl_mfd:
            if os.path.isfile(self.config_file_name):
                if os.path.getmtime(self.sftfilepath) < os.path.getmtime(
                    self.config_file_name
                ):
                    logging.info(
                        (
                            "The config file {} has been modified since the SFT file {} "
                            + "was created. {}"
                        ).format(self.config_file_name, self.sftfilepath, need_new)
                    )
                    return False
                else:
                    logging.info(
                        "OK: The config file {} is older than the SFT file {}".format(
                            self.config_file_name, self.sftfilepath
                        )
                    )
                    # NOTE: at this point we assume it's safe to re-use, since
                    # check_if_cff_file_needs_rewritting()
                    # should have already been called before
            else:
                raise RuntimeError(
                    "Commandline requires file '{}' but it is missing.".format(
                        self.config_file_name
                    )
                )

        logging.info("Checking new commandline against existing SFT header...")
        catalog = lalpulsar.SFTdataFind(self.sftfilepath, None)
        cl_old = helper_functions.get_lalapps_commandline_from_SFTDescriptor(
            catalog.data[0]
        )
        if len(cl_old) == 0:
            logging.info(
                "Could not obtain comparison commandline from old SFT header. "
                + need_new
            )
            return False
        if not helper_functions.match_commandlines(cl_old, cl_mfd):
            logging.info("Commandlines unmatched. " + need_new)
            return False
        else:
            logging.info("OK: Commandline matched with old SFT header.")
        logging.info("Looks like cached data matches current options, will re-use it!")
        return True

    def check_if_cff_file_needs_rewritting(self, content):
        """ Check if the .cff file has changed

        Returns True if the file should be overwritten - where possible avoid
        overwriting to allow cached data to be used
        """
        logging.info("Checking if we can re-use injection config file...")
        if os.path.isfile(self.config_file_name) is False:
            logging.info("No config file {} found.".format(self.config_file_name))
            return True
        else:
            logging.info("Config file {} already exists.".format(self.config_file_name))

        with open(self.config_file_name, "r") as f:
            file_content = f.read()
            if file_content == content:
                logging.info(
                    "File contents match, no update of {} required.".format(
                        self.config_file_name
                    )
                )
                return False
            else:
                logging.info(
                    "File contents unmatched, updating {}.".format(
                        self.config_file_name
                    )
                )
                return True

    def run_makefakedata(self):
        """ Generate the sft data from the configuration file """

        # Remove old data:
        try:
            os.unlink(os.path.join(self.outdir, "*" + self.label + "*.sft"))
        except OSError:
            pass

        cl_mfd = []
        cl_mfd.append("lalapps_Makefakedata_v5")
        cl_mfd.append("--outSingleSFT=TRUE")
        cl_mfd.append('--outSFTdir="{}"'.format(self.outdir))
        cl_mfd.append('--outLabel="{}"'.format(self.label))

        if self.add_noise:
            if self.noiseSFTs is not None and self.SFTWindowType is None:
                raise ValueError(
                    "SFTWindowType is required when using noiseSFTs. "
                    "Please, make sure you understand the window function used "
                    "to produce noiseSFTs."
                )
            elif self.noiseSFTs is not None:
                if self.sqrtSX > 0.0:
                    logging.warning(
                        "In addition to using noiseSFTs, you are adding "
                        "Gaussian noise with sqrtSX={} "
                        "Please, make sure this is what you intend to do.".format(
                            self.sqrtSX
                        )
                    )
                cl_mfd.append('--noiseSFTs="{}"'.format(self.noiseSFTs))
            else:
                cl_mfd.append("--IFOs={}".format(self.IFOs))
            cl_mfd.append('--sqrtSX="{}"'.format(self.sqrtSX))
        else:
            cl_mfd.append("--IFOs={}".format(self.IFOs))

        if self.SFTWindowType is not None:
            cl_mfd.append('--SFTWindowType="{}"'.format(self.SFTWindowType))
            cl_mfd.append("--SFTWindowBeta={}".format(self.SFTWindowBeta))
        if self.minStartTime is None:
            cl_mfd.append("--startTime={:0.0f}".format(float(self.tstart)))
        else:
            cl_mfd.append("--startTime={:0.0f}".format(float(self.minStartTime)))
        if self.maxStartTime is None:
            cl_mfd.append("--duration={}".format(int(self.duration)))
        else:
            data_duration = self.maxStartTime - self.minStartTime
            cl_mfd.append("--duration={}".format(int(data_duration)))
        cl_mfd.append("--fmin={:.16g}".format(self.fmin))
        cl_mfd.append("--Band={:.16g}".format(self.Band))
        cl_mfd.append("--Tsft={}".format(self.Tsft))
        if self.h0 != 0:
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

    def predict_fstat(self, assumeSqrtSX=None):
        """ Wrapper to lalapps_PredictFstat """
        twoF_expected, twoF_sigma = predict_fstat(
            h0=self.h0,
            cosi=self.cosi,
            psi=self.psi,
            Alpha=self.Alpha,
            Delta=self.Delta,
            Freq=self.F0,
            sftfilepattern=self.sftfilepath,
            minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime,
            IFOs=self.detectors,
            assumeSqrtSX=(assumeSqrtSX or self.sqrtSX),
            tempory_filename=os.path.join(self.outdir, self.label + ".tmp"),
            earth_ephem=self.earth_ephem,
            sun_ephem=self.sun_ephem,
        )  # detectors OR IFO?
        return twoF_expected


class BinaryModulatedWriter(Writer):
    """ Instance object for generating SFTs containing a continuous wave signal for a source in a binary system """

    @helper_functions.initializer
    def __init__(
        self,
        label="Test",
        tstart=700000000,
        duration=100 * 86400,
        tref=None,
        F0=30,
        F1=1e-10,
        F2=0,
        Alpha=5e-3,
        Delta=6e-2,
        tp=0.0,
        argp=0.0,
        asini=0.0,
        ecc=0.0,
        period=0.0,
        h0=0.1,
        cosi=0.0,
        psi=0.0,
        phi=0,
        Tsft=1800,
        outdir=".",
        sqrtSX=0.0,
        noiseSFTs=None,
        SFTWindowType=None,
        SFTWindowBeta=0.0,
        Band=4,
        detectors="H1",
        minStartTime=None,
        maxStartTime=None,
        add_noise=True,
        transientWindowType="none",
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
        F0, F1, F2, Alpha, Delta, orbitTp, orbitArgp, orbitasini, orbitEcc, h0, cosi, psi, phi: float
            frequency, sky-position, binary orbit and amplitude parameters
        Tsft: float
            the sft duration
        minStartTime, maxStartTime: float
            if not None, the total span of data, this can be used to generate
            transient signals
        see `lalapps_Makefakedata_v5 --help` for help with the other paramaters
        """
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
            minStartTime=minStartTime,
            maxStartTime=maxStartTime,
            add_noise=add_noise,
            transientWindowType=transientWindowType,
        )

        self.parse_args_consistent_with_mfd()

    def parse_args_consistent_with_mfd(self):
        """ This will allow us to get rid of the get_single_config_* family
            Future rework may use a dictionary as the default input method,
        """
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
            "tp",
            "argp",
            "asini",
            "ecc",
            "period",
            "transientWindowType",
        ]

        signal_parameters = {
            key: self.__dict__.get(key, None) for key in signal_parameter_labels
        }
        self.signal_parameters = {
            key: value for key, value in signal_parameters.items() if value is not None
        }

        signal_parameter_formats = (len(signal_parameter_labels) - 1) * [":1.18e"] + [
            ":s"
        ]
        signal_formats = dict(zip(signal_parameter_labels, signal_parameter_formats))

        self.signal_parameters = translate_keys_to_lal(self.signal_parameters)
        signal_formats = translate_keys_to_lal(signal_formats)

        self.signal_formats = {
            key: signal_formats[key] for key in self.signal_parameters.keys()
        }

        if self.signal_parameters["transientWindowType"] != "none":
            self.signal_parameters["transientStartTime"] = self.tstart
            self.signal_formats["transientStartTime"] = ":10.0f"
            self.signal_parameters["transientTau"] = self.duration_days * 86400.0
            self.signal_formats["transientTau"] = ":10.0f"

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

    def make_cff(self):
        """
        Generates a .cff file

        """
        content = self.get_single_config_line(0)

        if self.check_if_cff_file_needs_rewritting(content):
            config_file = open(self.config_file_name, "w+")
            config_file.write(content)
            config_file.close()


class GlitchWriter(Writer):
    """ Instance object for generating SFTs containing glitch signals """

    @helper_functions.initializer
    def __init__(
        self,
        label="Test",
        tstart=700000000,
        duration=100 * 86400,
        dtglitch=None,
        delta_phi=0,
        delta_F0=0,
        delta_F1=0,
        delta_F2=0,
        tref=None,
        F0=30,
        F1=1e-10,
        F2=0,
        Alpha=5e-3,
        Delta=6e-2,
        h0=0.1,
        cosi=0.0,
        psi=0.0,
        phi=0,
        Tsft=1800,
        outdir=".",
        sqrtSX=0.0,
        noiseSFTs=None,
        SFTWindowType=None,
        SFTWindowBeta=0.0,
        Band=4,
        detectors="H1",
        minStartTime=None,
        maxStartTime=None,
        add_noise=True,
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
        minStartTime, maxStartTime: float
            if not None, the total span of data, this can be used to generate
            transient signals

        see `lalapps_Makefakedata_v5 --help` for help with the other paramaters
        """

        self.set_ephemeris_files()
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
            self.tbounds = [self.tstart, self.tend]
        else:
            self.dtglitch = np.atleast_1d(self.dtglitch)
            self.tglitch = self.tstart + self.dtglitch
            self.tbounds = np.concatenate(([self.tstart], self.tglitch, [self.tend]))
        logging.info("Using segment boundaries {}".format(self.tbounds))

        tbs = np.array(self.tbounds)
        self.durations_days = (tbs[1:] - tbs[:-1]) / 86400

        self.delta_thetas = np.atleast_2d(
            np.array([delta_phi, delta_F0, delta_F1, delta_F2]).T
        )

    def make_cff(self):
        """
        Generates an .cff file for a 'glitching' signal

        """

        thetas = self._calculate_thetas(self.theta, self.delta_thetas, self.tbounds)

        content = ""
        for i, (t, d, ts) in enumerate(
            zip(thetas, self.durations_days, self.tbounds[:-1])
        ):
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

        if self.check_if_cff_file_needs_rewritting(content):
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
        IFO="H1",
        minStartTime=None,
        maxStartTime=None,
        detectors="H1",
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
        self.set_ephemeris_files()
        self.tstart = int(tstart)
        self.duration = int(duration)

        if os.path.isdir(self.outdir) is False:
            os.makedirs(self.outdir)
        if tref is None:
            raise ValueError("Input `tref` not specified")

        self.nsfts = int(np.ceil(self.duration / self.Tsft))
        self.duration = self.duration / 86400.0
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

            if self.IFO == "H1":
                Lambda = lal.LHO_4K_DETECTOR_LATITUDE_RAD
            elif self.IFO == "L1":
                Lambda = lal.LLO_4K_DETECTOR_LATITUDE_RAD

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
            self.IFO[0],
            self.IFO[1],
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
        self.maxStartTime = None
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
        cl_mfd.append("--IFO={}".format(self.IFO))
        cl_mfd.append('--noiseSqrtSh="{}"'.format(self.sqrtSX))
        cl_mfd.append("--startTime={:0.0f}".format(mid_time - self.Tsft / 2.0))
        cl_mfd.append("--refTime={:0.0f}".format(mid_time))
        cl_mfd.append("--duration={}".format(int(self.duration)))
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
