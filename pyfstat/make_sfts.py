""" pyfstat tools to generate sfts """

import numpy as np
import logging
import os
import glob

import lal
import lalpulsar

from core import BaseSearchClass, tqdm
import helper_functions

earth_ephem, sun_ephem = helper_functions.set_up_ephemeris_configuration()


class Writer(BaseSearchClass):
    """ Instance object for generating SFTs """
    @helper_functions.initializer
    def __init__(self, label='Test', tstart=700000000, duration=100*86400,
                 tref=None, F0=30, F1=1e-10, F2=0, Alpha=5e-3,
                 Delta=6e-2, h0=0.1, cosi=0.0, psi=0.0, phi=0, Tsft=1800,
                 outdir=".", sqrtSX=1, Band=4, detectors='H1',
                 minStartTime=None, maxStartTime=None, add_noise=True):
        """
        Parameters
        ----------
        label: string
            a human-readable label to be used in naming the output files
        tstart, tend : float
            start and end times (in gps seconds) of the total observation span
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

        self.tend = self.tstart + self.duration
        if self.minStartTime is None:
            self.minStartTime = self.tstart
        if self.maxStartTime is None:
            self.maxStartTime = self.tend
        self.tbounds = [self.tstart, self.tend]
        logging.info('Using segment boundaries {}'.format(self.tbounds))

        self.check_inputs()

        if os.path.isdir(self.outdir) is False:
            os.makedirs(self.outdir)
        if self.tref is None:
            self.tref = self.tstart
        self.tend = self.tstart + self.duration
        self.duration_days = (self.tend-self.tstart) / 86400
        self.config_file_name = "{}/{}.cff".format(outdir, label)
        self.theta = np.array([phi, F0, F1, F2])
        self.data_duration = self.maxStartTime - self.minStartTime
        numSFTs = int(float(self.data_duration) / self.Tsft)
        self.sftfilenames = [
            lalpulsar.OfficialSFTFilename(
                dets[0], dets[1], numSFTs, self.Tsft, self.minStartTime,
                self.data_duration, self.label)
            for dets in self.detectors.split(',')]
        self.sftfilepath = ';'.join([
            '{}/{}'.format(self.outdir, fn) for fn in self.sftfilenames])
        self.calculate_fmin_Band()

    def check_inputs(self):
        self.minStartTime = int(self.minStartTime)
        self.maxStartTime = int(self.maxStartTime)

    def make_data(self):
        ''' A convienience wrapper to generate a cff file then sfts '''
        self.make_cff()
        self.run_makefakedata()

    def get_single_config_line(self, i, Alpha, Delta, h0, cosi, psi, phi, F0,
                               F1, F2, tref, tstart, duration_days):
        template = (
"""[TS{}]
Alpha = {:1.18e}
Delta = {:1.18e}
h0 = {:1.18e}
cosi = {:1.18e}
psi = {:1.18e}
phi0 = {:1.18e}
Freq = {:1.18e}
f1dot = {:1.18e}
f2dot = {:1.18e}
refTime = {:10.6f}
transientWindowType=rect
transientStartTime={:10.3f}
transientTauDays={:1.3f}\n""")
        return template.format(i, Alpha, Delta, h0, cosi, psi, phi, F0, F1,
                               F2, tref, tstart, duration_days)

    def make_cff(self):
        """
        Generates a .cff file

        """

        content = self.get_single_config_line(
            0, self.Alpha, self.Delta, self.h0, self.cosi, self.psi,
            self.phi, self.F0, self.F1, self.F2, self.tref, self.tstart,
            self.duration_days)

        if self.check_if_cff_file_needs_rewritting(content):
            config_file = open(self.config_file_name, "w+")
            config_file.write(content)
            config_file.close()

    def calculate_fmin_Band(self):
        self.fmin = self.F0 - .5 * self.Band

    def check_cached_data_okay_to_use(self, cl_mfd):
        """ Check if cached data exists and, if it does, if it can be used """

        getmtime = os.path.getmtime

        if os.path.isfile(self.sftfilepath) is False:
            logging.info('No SFT file matching {} found'.format(
                self.sftfilepath))
            return False
        else:
            logging.info('Matching SFT file found')

        if getmtime(self.sftfilepath) < getmtime(self.config_file_name):
            logging.info(
                ('The config file {} has been modified since the sft file {} '
                 + 'was created').format(
                    self.config_file_name, self.sftfilepath))
            return False

        logging.info(
            'The config file {} is older than the sft file {}'.format(
                self.config_file_name, self.sftfilepath))
        logging.info('Checking contents of cff file')
        cl_dump = 'lalapps_SFTdumpheader {} | head -n 20'.format(
            self.sftfilepath)
        output = helper_functions.run_commandline(cl_dump)
        calls = [line for line in output.split('\n') if line[:3] == 'lal']
        if calls[0] == cl_mfd:
            logging.info('Contents matched, use old sft file')
            return True
        else:
            logging.info('Contents unmatched, create new sft file')
            return False

    def check_if_cff_file_needs_rewritting(self, content):
        """ Check if the .cff file has changed

        Returns True if the file should be overwritten - where possible avoid
        overwriting to allow cached data to be used
        """
        if os.path.isfile(self.config_file_name) is False:
            logging.info('No config file {} found'.format(
                self.config_file_name))
            return True
        else:
            logging.info('Config file {} already exists'.format(
                self.config_file_name))

        with open(self.config_file_name, 'r') as f:
            file_content = f.read()
            if file_content == content:
                logging.info(
                    'File contents match, no update of {} required'.format(
                        self.config_file_name))
                return False
            else:
                logging.info(
                    'File contents unmatched, updating {}'.format(
                        self.config_file_name))
                return True

    def run_makefakedata(self):
        """ Generate the sft data from the configuration file """

        # Remove old data:
        try:
            os.unlink("{}/*{}*.sft".format(self.outdir, self.label))
        except OSError:
            pass

        cl_mfd = []
        cl_mfd.append('lalapps_Makefakedata_v5')
        cl_mfd.append('--outSingleSFT=TRUE')
        cl_mfd.append('--outSFTdir="{}"'.format(self.outdir))
        cl_mfd.append('--outLabel="{}"'.format(self.label))
        cl_mfd.append('--IFOs={}'.format(
            ",".join(['"{}"'.format(d) for d in self.detectors.split(",")])))
        if self.add_noise:
            cl_mfd.append('--sqrtSX="{}"'.format(self.sqrtSX))
        if self.minStartTime is None:
            cl_mfd.append('--startTime={:0.0f}'.format(float(self.tstart)))
        else:
            cl_mfd.append('--startTime={:0.0f}'.format(
                float(self.minStartTime)))
        if self.maxStartTime is None:
            cl_mfd.append('--duration={}'.format(int(self.duration)))
        else:
            data_duration = self.maxStartTime - self.minStartTime
            cl_mfd.append('--duration={}'.format(int(data_duration)))
        cl_mfd.append('--fmin={:.16g}'.format(self.fmin))
        cl_mfd.append('--Band={:.16g}'.format(self.Band))
        cl_mfd.append('--Tsft={}'.format(self.Tsft))
        if self.h0 != 0:
            cl_mfd.append('--injectionSources="{}"'.format(
                self.config_file_name))

        cl_mfd = " ".join(cl_mfd)

        if self.check_cached_data_okay_to_use(cl_mfd) is False:
            helper_functions.run_commandline(cl_mfd)

    def predict_fstat(self):
        """ Wrapper to lalapps_PredictFstat """
        cl_pfs = []
        cl_pfs.append("lalapps_PredictFstat")
        cl_pfs.append("--h0={}".format(self.h0))
        cl_pfs.append("--cosi={}".format(self.cosi))
        cl_pfs.append("--psi={}".format(self.psi))
        cl_pfs.append("--Alpha={}".format(self.Alpha))
        cl_pfs.append("--Delta={}".format(self.Delta))
        cl_pfs.append("--Freq={}".format(self.F0))

        cl_pfs.append("--DataFiles='{}'".format(
            self.outdir+"/*SFT_"+self.label+"*sft"))
        cl_pfs.append("--assumeSqrtSX={}".format(self.sqrtSX))

        cl_pfs.append("--minStartTime={}".format(int(self.minStartTime)))
        cl_pfs.append("--maxStartTime={}".format(int(self.maxStartTime)))

        cl_pfs = " ".join(cl_pfs)
        output = helper_functions.run_commandline(cl_pfs)
        twoF = float(output.split('\n')[-2])
        return float(twoF)


class GlitchWriter(Writer):
    """ Instance object for generating SFTs containing glitch signals """
    @helper_functions.initializer
    def __init__(self, label='Test', tstart=700000000, duration=100*86400,
                 dtglitch=None, delta_phi=0, delta_F0=0, delta_F1=0,
                 delta_F2=0, tref=None, F0=30, F1=1e-10, F2=0, Alpha=5e-3,
                 Delta=6e-2, h0=0.1, cosi=0.0, psi=0.0, phi=0, Tsft=1800,
                 outdir=".", sqrtSX=1, Band=4, detectors='H1',
                 minStartTime=None, maxStartTime=None, add_noise=True):
        """
        Parameters
        ----------
        label: string
            a human-readable label to be used in naming the output files
        tstart, tend : float
            start and end times (in gps seconds) of the total observation span
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

        for d in self.delta_phi, self.delta_F0, self.delta_F1, self.delta_F2:
            if np.size(d) == 1:
                d = np.atleast_1d(d)
        self.tend = self.tstart + self.duration
        if self.minStartTime is None:
            self.minStartTime = self.tstart
        if self.maxStartTime is None:
            self.maxStartTime = self.tend
        if self.dtglitch is None:
            self.tbounds = [self.tstart, self.tend]
        else:
            self.dtglitch = np.atleast_1d(self.dtglitch)
            self.tglitch = self.tstart + self.dtglitch
            self.tbounds = np.concatenate((
                [self.tstart], self.tglitch, [self.tend]))
        logging.info('Using segment boundaries {}'.format(self.tbounds))

        self.check_inputs()

        if os.path.isdir(self.outdir) is False:
            os.makedirs(self.outdir)
        if self.tref is None:
            self.tref = self.tstart
        self.tend = self.tstart + self.duration
        tbs = np.array(self.tbounds)
        self.durations_days = (tbs[1:] - tbs[:-1]) / 86400
        self.config_file_name = "{}/{}.cff".format(outdir, label)

        self.theta = np.array([phi, F0, F1, F2])
        self.delta_thetas = np.atleast_2d(
                np.array([delta_phi, delta_F0, delta_F1, delta_F2]).T)

        self.data_duration = self.maxStartTime - self.minStartTime
        numSFTs = int(float(self.data_duration) / self.Tsft)
        self.sftfilenames = [
            lalpulsar.OfficialSFTFilename(
                dets[0], dets[1], numSFTs, self.Tsft, self.minStartTime,
                self.data_duration, self.label)
            for dets in self.detectors.split(',')]
        self.sftfilepath = ';'.join([
            '{}/{}'.format(self.outdir, fn) for fn in self.sftfilenames])
        self.calculate_fmin_Band()

    def check_inputs(self):
        self.minStartTime = int(self.minStartTime)
        self.maxStartTime = int(self.maxStartTime)
        shapes = np.array([np.shape(x) for x in [self.delta_phi, self.delta_F0,
                                                 self.delta_F1, self.delta_F2]]
                          )
        if not np.all(shapes == shapes[0]):
            raise ValueError('all delta_* must be the same shape: {}'.format(
                shapes))

    def make_cff(self):
        """
        Generates an .cff file for a 'glitching' signal

        """

        thetas = self._calculate_thetas(self.theta, self.delta_thetas,
                                        self.tbounds)

        content = ''
        for i, (t, d, ts) in enumerate(zip(thetas, self.durations_days,
                                           self.tbounds[:-1])):
            line = self.get_single_config_line(
                i, self.Alpha, self.Delta, self.h0, self.cosi, self.psi,
                t[0], t[1], t[2], t[3], self.tref, ts, d)

            content += line

        if self.check_if_cff_file_needs_rewritting(content):
            config_file = open(self.config_file_name, "w+")
            config_file.write(content)
            config_file.close()


class FrequencyModulatedArtifactWriter(Writer):
    """ Instance object for generating SFTs containing artifacts """

    earth_ephem_default = earth_ephem
    sun_ephem_default = sun_ephem

    @helper_functions.initializer
    def __init__(self, outdir=".", tstart=700000000, data_duration=86400,
                 F0=30, tref=None, h0=10, Tsft=1800, sqrtSX=1, Band=4,
                 Pmod=lal.DAYSID_SI, phir0=0, Alpha=1.2, Delta=0.5, IFO='H1',
                 earth_ephem=None, sun_ephem=None):
        """
        Parameters
        ----------
        tstart, data_duration : float
            start and duration times (in gps seconds) of the total observation
        Pmod, Amp, F0, h0: float
            Modulation period, amplitude, frequency and h0 of the artifact
        Tsft: float
            the sft duration
        sqrtSX: float
            Background IFO noise

        see `lalapps_Makefakedata_v4 --help` for help with the other paramaters
        """

        if os.path.isdir(self.outdir) is False:
            os.makedirs(self.outdir)
        if tref is None:
            raise ValueError('Input `tref` not specified')

        self.nsfts = int(np.ceil(self.data_duration / self.Tsft))
        self.data_duration_days = self.data_duration / 86400.
        self.calculate_fmin_Band()

        self.cosi = 0
        self.Fmax = F0

        if self.earth_ephem is None:
            self.earth_ephem = self.earth_ephem_default
        if self.sun_ephem is None:
            self.sun_ephem = self.sun_ephem_default

        self.n = np.array([np.cos(Alpha)*np.cos(Delta),
                           np.sin(Alpha)*np.cos(Delta),
                           np.sin(Delta)])

    def get_frequency(self, t):
        spin_posvel = lalpulsar.PosVel3D_t()
        orbit_posvel = lalpulsar.PosVel3D_t()
        det = lal.CachedDetectors[4]
        ephems = lalpulsar.InitBarycenter(self.earth_ephem, self.sun_ephem)
        lalpulsar.DetectorPosVel(
            spin_posvel, orbit_posvel, lal.LIGOTimeGPS(t), det, ephems,
            lalpulsar.DETMOTION_ORBIT)

        # Pos and vel returned in units of c
        if self.IFO == 'H1':
            Lambda = lal.LHO_4K_DETECTOR_LATITUDE_RAD
        elif self.IFO == 'L1':
            Lambda = lal.LLO_4K_DETECTOR_LATITUDE_RAD
        phir = 2*np.pi*t/self.Pmod + self.phir0
        DeltaFSpin = lal.REARTH_SI/lal.C_SI * 2*np.pi/self.Pmod*(
            np.sin(self.Delta)*np.sin(Lambda)
            + np.cos(self.Alpha)*np.cos(self.Delta)*np.cos(Lambda)*np.cos(phir)
            + np.sin(self.Alpha)*np.cos(self.Delta)*np.sin(Lambda)*np.sin(phir)
            )
        f = self.F0+(np.dot(self.n, orbit_posvel.vel) + DeltaFSpin)*self.Fmax
        return f

    def get_h0(self, t):
        return self.h0

    def concatenate_sft_files(self, tmp_outdir):
        SFTFilename = lalpulsar.OfficialSFTFilename(
            self.IFO[0], self.IFO[1], self.nsfts, self.Tsft, self.tstart,
            int(self.data_duration), self.label)

        helper_functions.run_commandline(
            'rm {}/{}'.format(self.outdir, SFTFilename), raise_error=False)

        cl_splitSFTS = (
            'lalapps_splitSFTs -fs {} -fb {} -fe {} -o {}/{} -i {}/{}_tmp/*sft'
            .format(self.fmin, self.Band, self.fmin+self.Band, self.outdir,
                    SFTFilename, self.outdir, self.label))
        helper_functions.run_commandline(cl_splitSFTS)
        helper_functions.run_commandline('rm {} -r'.format(tmp_outdir))
        files = glob.glob('{}/{}*'.format(self.outdir, SFTFilename))
        if len(files) == 1:
            fn = files[0]
            fn_new = fn.split('.')[0] + '.sft'
            helper_functions.run_commandline('mv {} {}'.format(
                fn, fn_new))
        else:
            raise IOError(
                'Attempted to rename file, but multiple files found: {}'
                .format(files))

    def make_data(self):
        self.maxStartTime = None
        self.duration = self.Tsft
        linePhi = 0
        lineFreq_old = 0

        tmp_outdir = '{}/{}_tmp'.format(self.outdir, self.label)
        if os.path.isdir(tmp_outdir) is True:
            raise ValueError(
                'Temporary directory {} already exists, please rename'.format(
                    tmp_outdir))
        else:
            os.makedirs(tmp_outdir)

        for i in tqdm(range(self.nsfts)):
            self.minStartTime = self.tstart + i*self.Tsft
            mid_time = self.minStartTime + self.Tsft / 2.0
            lineFreq = self.get_frequency(mid_time)
            linePhi += np.pi*self.Tsft*(lineFreq_old+lineFreq)
            lineh0 = self.get_h0(mid_time)
            self.run_makefakedata_v4(mid_time, lineFreq, linePhi, lineh0,
                                     tmp_outdir)
            lineFreq_old = lineFreq

        SFTFilename = lalpulsar.OfficialSFTFilename(
            self.IFO[0], self.IFO[1], self.nsfts, self.Tsft, self.tstart,
            int(self.data_duration), self.label)

        helper_functions.run_commandline(
            'rm {}/{}'.format(self.outdir, SFTFilename), raise_error=False)

        cl_splitSFTS = (
            'lalapps_splitSFTs -fs {} -fb {} -fe {} -o {}/{} -i {}/{}_tmp/*sft'
            .format(self.fmin, self.Band, self.fmin+self.Band, self.outdir,
                    SFTFilename, self.outdir, self.label))
        helper_functions.run_commandline(cl_splitSFTS)
        helper_functions.run_commandline('rm {} -r'.format(tmp_outdir))
        files = glob.glob('{}/{}*'.format(self.outdir, SFTFilename))
        if len(files) == 1:
            fn = files[0]
            fn_new = fn.split('.')[0] + '.sft'
            helper_functions.run_commandline('mv {} {}'.format(
                fn, fn_new))
        else:
            raise IOError(
                'Attempted to rename file, but multiple files found: {}'
                .format(files))

    def run_makefakedata_v4(self, mid_time, lineFreq, linePhi, h0, tmp_outdir):
        """ Generate the sft data using the --lineFeature option """
        cl_mfd = []
        cl_mfd.append('lalapps_Makefakedata_v4')
        cl_mfd.append('--outSingleSFT=FALSE')
        cl_mfd.append('--outSFTbname="{}"'.format(tmp_outdir))
        cl_mfd.append('--IFO={}'.format(self.IFO))
        cl_mfd.append('--noiseSqrtSh="{}"'.format(self.sqrtSX))
        cl_mfd.append('--startTime={:0.0f}'.format(float(self.minStartTime)))
        cl_mfd.append('--refTime={:0.0f}'.format(mid_time))
        cl_mfd.append('--duration={}'.format(int(self.duration)))
        cl_mfd.append('--fmin={:.16g}'.format(self.fmin))
        cl_mfd.append('--Band={:.16g}'.format(self.Band))
        cl_mfd.append('--Tsft={}'.format(self.Tsft))
        cl_mfd.append('--Alpha={}'.format(self.Alpha))
        cl_mfd.append('--Delta={}'.format(self.Delta))
        cl_mfd.append('--Freq={}'.format(lineFreq))
        cl_mfd.append('--phi0={}'.format(linePhi))
        cl_mfd.append('--h0={}'.format(h0))
        cl_mfd.append('--cosi={}'.format(self.cosi))
        cl_mfd.append('--lineFeature=TRUE')
        cl_mfd = " ".join(cl_mfd)
        helper_functions.run_commandline(cl_mfd, log_level=10)


class FrequencyAmplitudeModulatedArtifactWriter(FrequencyModulatedArtifactWriter):
    """ Instance object for generating SFTs containing artifacts """
    def get_h0(self, t):
            return self.h0*np.sin(2*np.pi*t/self.Pmod+self.phir0)
