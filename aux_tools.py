""" Tools used to generate fake data containing glitches """

import os
import numpy as np
import logging
import subprocess
import lalpulsar
from glitch_searches import initializer, BaseSearchClass

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s: %(message)s',
                    datefmt='%H:%M')


class Writer(BaseSearchClass):
    """ Instance object for generating SFT containing glitch signals """
    @initializer
    def __init__(self, label='Test', tstart=700000000, duration=100*86400,
                 dtglitch=None,
                 delta_phi=0, delta_F0=0, delta_F1=0, delta_F2=0,
                 tref=None, phi=0, F0=30, F1=1e-10, F2=0, Alpha=5e-3,
                 Delta=6e-2, h0=0.1, cosi=0.0, psi=0.0, Tsft=1800, outdir=".",
                 sqrtSX=1, Band=2):
        """
        Parameters
        ----------
        label: string
            a human-readable label to be used in naming the output files
        tstart, tend : float
            start and end times (in gps seconds) of the total observation span
        dtglitch: float
            time (in gps seconds) of the glitch after tstart. To create data
            without a glitch, set dtglitch=tend-tstart or leave as None
        delta_phi, delta_F0, delta_F1: float
            instanteneous glitch magnitudes in rad, Hz, and Hz/s respectively
        tref: float or None
            reference time (default is None, which sets the reference time to
            tstart)
        phil, F0, F1, F2, Alpha, Delta, h0, cosi, psi: float
            pre-glitch phase, frequency, sky-position, and signal properties
        Tsft: float
            the sft duration

        see `lalapps_Makefakedata_v5 --help` for help with the other paramaters
        """

        for d in self.delta_phi, self.delta_F0, self.delta_F1, self.delta_F2:
            if np.size(d) == 1:
                d = [d]
        self.tend = self.tstart + self.duration
        if self.dtglitch is None or self.dtglitch == self.duration:
            self.tbounds = [self.tstart, self.tend]
        elif np.size(self.dtglitch) == 1:
            self.tbounds = [self.tstart, self.tstart+self.dtglitch, self.tend]
        else:
            self.tglitch = self.tstart + np.array(self.dtglitch)
            self.tbounds = [self.tstart] + list(self.tglitch) + [self.tend]

        if os.path.isdir(self.outdir) is False:
            os.makedirs(self.outdir)
        if self.tref is None:
            self.tref = self.tglitch
        self.tend = self.tstart + self.duration
        tbs = np.array(self.tbounds)
        self.durations_days = (tbs[1:] - tbs[:-1]) / 86400
        self.config_file_name = "{}/{}.cff".format(outdir, label)

        self.theta = np.array([phi, F0, F1, F2])
        self.delta_thetas = np.atleast_2d(
                np.array([delta_phi, delta_F0, delta_F1, delta_F2]).T)

        self.detector = 'H1'
        numSFTs = int(float(self.duration) / self.Tsft)
        self.sft_filename = lalpulsar.OfficialSFTFilename(
            'H', '1', numSFTs, self.Tsft, self.tstart, self.duration,
            self.label)
        self.sft_filepath = '{}/{}'.format(self.outdir, self.sft_filename)
        self.calculate_fmin_Band()

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
        Generates an .cff file for a 'glitching' signal

        """

        thetas = self.calculate_thetas(self.theta, self.delta_thetas,
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

    def calculate_fmin_Band(self):
        self.fmin = self.F0 - .5 * self.Band

    def check_cached_data_okay_to_use(self, cl):
        """ Check if cached data exists and, if it does, if it can be used """

        getmtime = os.path.getmtime

        if os.path.isfile(self.sft_filepath) is False:
            logging.info('No SFT file matching {} found'.format(
                self.sft_filepath))
            return False
        else:
            logging.info('Matching SFT file found')

        if getmtime(self.sft_filepath) < getmtime(self.config_file_name):
            logging.info(
                ('The config file {} has been modified since the sft file {} '
                 + 'was created').format(
                    self.config_file_name, self.sft_filepath))
            return False

        logging.info(
            'The config file {} is older than the sft file {}'.format(
                self.config_file_name, self.sft_filepath))
        logging.info('Checking contents of cff file')
        logging.info('Execute: {}'.format(
            'lalapps_SFTdumpheader {} | head -n 20'.format(self.sft_filepath)))
        output = subprocess.check_output(
            'lalapps_SFTdumpheader {} | head -n 20'.format(self.sft_filepath),
            shell=True)
        calls = [line for line in output.split('\n') if line[:3] == 'lal']
        if calls[0] == cl:
            logging.info('Contents matched, use old cff file')
            return True
        else:
            logging.info('Contents unmatched, create new cff file')
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

        cl = []
        cl.append('lalapps_Makefakedata_v5')
        cl.append('--outSingleSFT=TRUE')
        cl.append('--outSFTdir="{}"'.format(self.outdir))
        cl.append('--outLabel="{}"'.format(self.label))
        cl.append('--IFOs="{}"'.format(self.detector))
        cl.append('--sqrtSX="{}"'.format(self.sqrtSX))
        cl.append('--startTime={:10.9f}'.format(float(self.tstart)))
        cl.append('--duration={}'.format(int(self.duration)))
        cl.append('--fmin={}'.format(self.fmin))
        cl.append('--Band={}'.format(self.Band))
        cl.append('--Tsft={}'.format(self.Tsft))
        cl.append('--injectionSources="./{}"'.format(self.config_file_name))

        cl = " ".join(cl)

        if self.check_cached_data_okay_to_use(cl) is False:
            logging.info("Executing: " + cl)
            os.system(cl)
            os.system('\n')

    def predict_fstat(self):
        """ Wrapper to lalapps_PredictFstat """
        c_l = []
        c_l.append("lalapps_PredictFstat")
        c_l.append("--h0={}".format(self.h0))
        c_l.append("--cosi={}".format(self.cosi))
        c_l.append("--psi={}".format(self.psi))
        c_l.append("--Alpha={}".format(self.Alpha))
        c_l.append("--Delta={}".format(self.Delta))
        c_l.append("--Freq={}".format(self.F0))

        c_l.append("--DataFiles='{}'".format(
            self.outdir+"/*SFT_"+self.label+"*sft"))
        c_l.append("--assumeSqrtSX={}".format(self.sqrtSX))

        c_l.append("--minStartTime={}".format(self.tstart))
        c_l.append("--maxStartTime={}".format(self.tend))

        logging.info("Executing: " + " ".join(c_l) + "\n")
        output = subprocess.check_output(" ".join(c_l), shell=True)
        twoF = float(output.split('\n')[-2])
        return float(twoF)
