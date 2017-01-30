""" The core tools used in pyfstat """
import os
import logging
import copy
import glob
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.optimize
import lal
import lalpulsar

import helper_functions
helper_functions.set_up_matplotlib_defaults()
args, tqdm = helper_functions.set_up_command_line_arguments()
earth_ephem, sun_ephem = helper_functions.set_up_ephemeris_configuration()


def read_par(label, outdir):
    """ Read in a .par file, returns a dictionary of the values """
    filename = '{}/{}.par'.format(outdir, label)
    d = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.split('=')) > 1:
                key, val = line.rstrip('\n').split(' = ')
                key = key.strip()
                d[key] = np.float64(eval(val.rstrip('; ')))
    return d


class BaseSearchClass(object):
    """ The base search class, provides general functions """

    earth_ephem_default = earth_ephem
    sun_ephem_default = sun_ephem

    def add_log_file(self):
        """ Log output to a file, requires class to have outdir and label """
        logfilename = '{}/{}.log'.format(self.outdir, self.label)
        fh = logging.FileHandler(logfilename)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)-8s: %(message)s',
            datefmt='%y-%m-%d %H:%M'))
        logging.getLogger().addHandler(fh)

    def shift_matrix(self, n, dT):
        """ Generate the shift matrix

        Parameters
        ----------
        n: int
            The dimension of the shift-matrix to generate
        dT: float
            The time delta of the shift matrix

        Returns
        -------
        m: array (n, n)
            The shift matrix
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
                        m[i, j] = 2*np.pi*float(dT)**(j-i) / factorial(j-i)
                    else:
                        m[i, j] = float(dT)**(j-i) / factorial(j-i)
        return m

    def shift_coefficients(self, theta, dT):
        """ Shift a set of coefficients by dT

        Parameters
        ----------
        theta: array-like, shape (n,)
            vector of the expansion coefficients to transform starting from the
            lowest degree e.g [phi, F0, F1,...].
        dT: float
            difference between the two reference times as tref_new - tref_old.

        Returns
        -------
        theta_new: array-like shape (n,)
            vector of the coefficients as evaluate as the new reference time.
        """

        n = len(theta)
        m = self.shift_matrix(n, dT)
        return np.dot(m, theta)

    def calculate_thetas(self, theta, delta_thetas, tbounds, theta0_idx=0):
        """ Calculates the set of coefficients for the post-glitch signal """
        thetas = [theta]
        for i, dt in enumerate(delta_thetas):
            if i < theta0_idx:
                pre_theta_at_ith_glitch = self.shift_coefficients(
                    thetas[0], tbounds[i+1] - self.tref)
                post_theta_at_ith_glitch = pre_theta_at_ith_glitch - dt
                thetas.insert(0, self.shift_coefficients(
                    post_theta_at_ith_glitch, self.tref - tbounds[i+1]))

            elif i >= theta0_idx:
                pre_theta_at_ith_glitch = self.shift_coefficients(
                    thetas[i], tbounds[i+1] - self.tref)
                post_theta_at_ith_glitch = pre_theta_at_ith_glitch + dt
                thetas.append(self.shift_coefficients(
                    post_theta_at_ith_glitch, self.tref - tbounds[i+1]))
        return thetas

    def generate_loudest(self):
        params = read_par(self.label, self.outdir)
        for key in ['Alpha', 'Delta', 'F0', 'F1']:
            if key not in params:
                params[key] = self.theta_prior[key]
        cmd = ('lalapps_ComputeFstatistic_v2 -a {} -d {} -f {} -s {} -D "{}"'
               ' --refTime={} --outputLoudest="{}/{}.loudest" '
               '--minStartTime={} --maxStartTime={}').format(
                    params['Alpha'], params['Delta'], params['F0'],
                    params['F1'], self.sftfilepath, params['tref'],
                    self.outdir, self.label, self.minStartTime,
                    self.maxStartTime)
        subprocess.call([cmd], shell=True)

    def get_list_of_matching_sfts(self):
        matches = glob.glob(self.sftfilepath)
        if len(matches) > 0:
            return matches
        else:
            raise IOError('No sfts found matching {}'.format(
                self.sftfilepath))


class ComputeFstat(object):
    """ Base class providing interface to `lalpulsar.ComputeFstat` """

    earth_ephem_default = earth_ephem
    sun_ephem_default = sun_ephem

    @helper_functions.initializer
    def __init__(self, tref, sftfilepath=None, minStartTime=None,
                 maxStartTime=None, binary=False, transient=True, BSGL=False,
                 detector=None, minCoverFreq=None, maxCoverFreq=None,
                 earth_ephem=None, sun_ephem=None, injectSources=None
                 ):
        """
        Parameters
        ----------
        tref: int
            GPS seconds of the reference time.
        sftfilepath: str
            File patern to match SFTs
        minStartTime, maxStartTime: float GPStime
            Only use SFTs with timestemps starting from (including, excluding)
            this epoch
        binary: bool
            If true, search of binary parameters.
        transient: bool
            If true, allow for the Fstat to be computed over a transient range.
        BSGL: bool
            If true, compute the BSGL rather than the twoF value.
        detector: str
            Two character reference to the data to use, specify None for no
            contraint.
        minCoverFreq, maxCoverFreq: float
            The min and max cover frequency passed to CreateFstatInput, if
            either is None the range of frequencies in the SFT less 1Hz is
            used.
        earth_ephem, sun_ephem: str
            Paths of the two files containing positions of Earth and Sun,
            respectively at evenly spaced times, as passed to CreateFstatInput.
            If None defaults defined in BaseSearchClass will be used.

        """

        if earth_ephem is None:
            self.earth_ephem = self.earth_ephem_default
        if sun_ephem is None:
            self.sun_ephem = self.sun_ephem_default

        self.init_computefstatistic_single_point()

    def get_SFTCatalog(self):
        if hasattr(self, 'SFTCatalog'):
            return
        logging.info('Initialising SFTCatalog')
        constraints = lalpulsar.SFTConstraints()
        if self.detector:
            constraints.detector = self.detector
        if self.minStartTime:
            constraints.minStartTime = lal.LIGOTimeGPS(self.minStartTime)
        if self.maxStartTime:
            constraints.maxStartTime = lal.LIGOTimeGPS(self.maxStartTime)

        logging.info('Loading data matching pattern {}'.format(
                     self.sftfilepath))
        SFTCatalog = lalpulsar.SFTdataFind(self.sftfilepath, constraints)
        detector_names = list(set([d.header.name for d in SFTCatalog.data]))
        self.detector_names = detector_names
        SFT_timestamps = [d.header.epoch for d in SFTCatalog.data]
        if args.quite is False and args.no_interactive is False:
            try:
                from bashplotlib.histogram import plot_hist
                print('Data timestamps histogram:')
                plot_hist(SFT_timestamps, height=5, bincount=50)
            except IOError:
                pass
        if len(detector_names) == 0:
            raise ValueError('No data loaded.')
        logging.info('Loaded {} data files from detectors {}'.format(
            len(SFT_timestamps), detector_names))
        logging.info('Data spans from {} ({}) to {} ({})'.format(
            int(SFT_timestamps[0]),
            subprocess.check_output('lalapps_tconvert {}'.format(
                int(SFT_timestamps[0])), shell=True).rstrip('\n'),
            int(SFT_timestamps[-1]),
            subprocess.check_output('lalapps_tconvert {}'.format(
                int(SFT_timestamps[-1])), shell=True).rstrip('\n')))
        self.SFTCatalog = SFTCatalog

    def init_computefstatistic_single_point(self):
        """ Initilisation step of run_computefstatistic for a single point """

        self.get_SFTCatalog()

        logging.info('Initialising ephems')
        ephems = lalpulsar.InitBarycenter(self.earth_ephem, self.sun_ephem)

        logging.info('Initialising FstatInput')
        dFreq = 0
        if self.transient:
            self.whatToCompute = lalpulsar.FSTATQ_ATOMS_PER_DET
        else:
            self.whatToCompute = lalpulsar.FSTATQ_2F

        FstatOAs = lalpulsar.FstatOptionalArgs()
        FstatOAs.randSeed = lalpulsar.FstatOptionalArgsDefaults.randSeed
        FstatOAs.SSBprec = lalpulsar.FstatOptionalArgsDefaults.SSBprec
        FstatOAs.Dterms = lalpulsar.FstatOptionalArgsDefaults.Dterms
        FstatOAs.runningMedianWindow = lalpulsar.FstatOptionalArgsDefaults.runningMedianWindow
        FstatOAs.FstatMethod = lalpulsar.FstatOptionalArgsDefaults.FstatMethod
        FstatOAs.InjectSqrtSX = lalpulsar.FstatOptionalArgsDefaults.injectSqrtSX
        FstatOAs.assumeSqrtSX = lalpulsar.FstatOptionalArgsDefaults.assumeSqrtSX
        FstatOAs.prevInput = lalpulsar.FstatOptionalArgsDefaults.prevInput
        FstatOAs.collectTiming = lalpulsar.FstatOptionalArgsDefaults.collectTiming

        if hasattr(self, 'injectSource') and type(self.injectSources) == dict:
            logging.info('Injecting source with params: {}'.format(
                self.injectSources))
            PPV = lalpulsar.CreatePulsarParamsVector(1)
            PP = PPV.data[0]
            PP.Amp.h0 = self.injectSources['h0']
            PP.Amp.cosi = self.injectSources['cosi']
            PP.Amp.phi0 = self.injectSources['phi0']
            PP.Amp.psi = self.injectSources['psi']
            PP.Doppler.Alpha = self.injectSources['Alpha']
            PP.Doppler.Delta = self.injectSources['Delta']
            PP.Doppler.fkdot = np.array(self.injectSources['fkdot'])
            PP.Doppler.refTime = self.tref
            if 't0' not in self.injectSources:
                PP.Transient.type = lalpulsar.TRANSIENT_NONE
            FstatOAs.injectSources = PPV
        else:
            FstatOAs.injectSources = lalpulsar.FstatOptionalArgsDefaults.injectSources

        if self.minCoverFreq is None or self.maxCoverFreq is None:
            fAs = [d.header.f0 for d in self.SFTCatalog.data]
            fBs = [d.header.f0 + (d.numBins-1)*d.header.deltaF
                   for d in self.SFTCatalog.data]
            self.minCoverFreq = np.min(fAs) + 0.5
            self.maxCoverFreq = np.max(fBs) - 0.5
            logging.info('Min/max cover freqs not provided, using '
                         '{} and {}, est. from SFTs'.format(
                             self.minCoverFreq, self.maxCoverFreq))

        self.FstatInput = lalpulsar.CreateFstatInput(self.SFTCatalog,
                                                     self.minCoverFreq,
                                                     self.maxCoverFreq,
                                                     dFreq,
                                                     ephems,
                                                     FstatOAs
                                                     )

        logging.info('Initialising PulsarDoplerParams')
        PulsarDopplerParams = lalpulsar.PulsarDopplerParams()
        PulsarDopplerParams.refTime = self.tref
        PulsarDopplerParams.Alpha = 1
        PulsarDopplerParams.Delta = 1
        PulsarDopplerParams.fkdot = np.array([0, 0, 0, 0, 0, 0, 0])
        self.PulsarDopplerParams = PulsarDopplerParams

        logging.info('Initialising FstatResults')
        self.FstatResults = lalpulsar.FstatResults()

        if self.BSGL:
            if len(self.detector_names) < 2:
                raise ValueError("Can't use BSGL with single detector data")
            else:
                logging.info('Initialising BSGL')

            # Tuning parameters - to be reviewed
            numDetectors = 2
            if hasattr(self, 'nsegs'):
                p_val_threshold = 1e-6
                Fstar0s = np.linspace(0, 1000, 10000)
                p_vals = scipy.special.gammaincc(2*self.nsegs, Fstar0s)
                Fstar0 = Fstar0s[np.argmin(np.abs(p_vals - p_val_threshold))]
                if Fstar0 == Fstar0s[-1]:
                    raise ValueError('Max Fstar0 exceeded')
            else:
                Fstar0 = 15.
            logging.info('Using Fstar0 of {:1.2f}'.format(Fstar0))
            oLGX = np.zeros(10)
            oLGX[:numDetectors] = 1./numDetectors
            self.BSGLSetup = lalpulsar.CreateBSGLSetup(numDetectors,
                                                       Fstar0,
                                                       oLGX,
                                                       True,
                                                       1)
            self.twoFX = np.zeros(10)
            self.whatToCompute = (self.whatToCompute +
                                  lalpulsar.FSTATQ_2F_PER_DET)

        if self.transient:
            logging.info('Initialising transient parameters')
            self.windowRange = lalpulsar.transientWindowRange_t()
            self.windowRange.type = lalpulsar.TRANSIENT_RECTANGULAR
            self.windowRange.t0Band = 0
            self.windowRange.dt0 = 1
            self.windowRange.tauBand = 0
            self.windowRange.dtau = 1

    def compute_fullycoherent_det_stat_single_point(
            self, F0, F1, F2, Alpha, Delta, asini=None, period=None, ecc=None,
            tp=None, argp=None):
        """ Compute the fully-coherent det. statistic at a single point """

        return self.run_computefstatistic_single_point(
            self.minStartTime, self.maxStartTime, F0, F1, F2, Alpha, Delta,
            asini, period, ecc, tp, argp)

    def run_computefstatistic_single_point(self, tstart, tend, F0, F1,
                                           F2, Alpha, Delta, asini=None,
                                           period=None, ecc=None, tp=None,
                                           argp=None):
        """ Returns twoF or ln(BSGL) fully-coherently at a single point """

        self.PulsarDopplerParams.fkdot = np.array([F0, F1, F2, 0, 0, 0, 0])
        self.PulsarDopplerParams.Alpha = Alpha
        self.PulsarDopplerParams.Delta = Delta
        if self.binary:
            self.PulsarDopplerParams.asini = asini
            self.PulsarDopplerParams.period = period
            self.PulsarDopplerParams.ecc = ecc
            self.PulsarDopplerParams.tp = tp
            self.PulsarDopplerParams.argp = argp

        lalpulsar.ComputeFstat(self.FstatResults,
                               self.FstatInput,
                               self.PulsarDopplerParams,
                               1,
                               self.whatToCompute
                               )

        if self.transient is False:
            if self.BSGL is False:
                return self.FstatResults.twoF[0]

            twoF = np.float(self.FstatResults.twoF[0])
            self.twoFX[0] = self.FstatResults.twoFPerDet(0)
            self.twoFX[1] = self.FstatResults.twoFPerDet(1)
            log10_BSGL = lalpulsar.ComputeBSGL(twoF, self.twoFX,
                                               self.BSGLSetup)
            return log10_BSGL/np.log10(np.exp(1))

        self.windowRange.t0 = int(tstart)  # TYPE UINT4
        self.windowRange.tau = int(tend - tstart)  # TYPE UINT4

        FS = lalpulsar.ComputeTransientFstatMap(
            self.FstatResults.multiFatoms[0], self.windowRange, False)

        if self.BSGL is False:
            return 2*FS.F_mn.data[0][0]

        FstatResults_single = copy.copy(self.FstatResults)
        FstatResults_single.lenth = 1
        FstatResults_single.data = self.FstatResults.multiFatoms[0].data[0]
        FS0 = lalpulsar.ComputeTransientFstatMap(
            FstatResults_single.multiFatoms[0], self.windowRange, False)
        FstatResults_single.data = self.FstatResults.multiFatoms[0].data[1]
        FS1 = lalpulsar.ComputeTransientFstatMap(
            FstatResults_single.multiFatoms[0], self.windowRange, False)

        self.twoFX[0] = 2*FS0.F_mn.data[0][0]
        self.twoFX[1] = 2*FS1.F_mn.data[0][0]
        log10_BSGL = lalpulsar.ComputeBSGL(
                2*FS.F_mn.data[0][0], self.twoFX, self.BSGLSetup)

        return log10_BSGL/np.log10(np.exp(1))

    def calculate_twoF_cumulative(self, F0, F1, F2, Alpha, Delta, asini=None,
                                  period=None, ecc=None, tp=None, argp=None,
                                  tstart=None, tend=None, npoints=1000,
                                  minfraction=0.01, maxfraction=1):
        """ Calculate the cumulative twoF along the obseration span """
        duration = tend - tstart
        tstart = tstart + minfraction*duration
        taus = np.linspace(minfraction*duration, maxfraction*duration, npoints)
        twoFs = []
        if self.transient is False:
            self.transient = True
            self.init_computefstatistic_single_point()
        for tau in taus:
            twoFs.append(self.run_computefstatistic_single_point(
                tstart=tstart, tend=tstart+tau, F0=F0, F1=F1, F2=F2,
                Alpha=Alpha, Delta=Delta, asini=asini, period=period, ecc=ecc,
                tp=tp, argp=argp))

        return taus, np.array(twoFs)

    def plot_twoF_cumulative(self, label, outdir, ax=None, c='k', savefig=True,
                             title=None, **kwargs):

        taus, twoFs = self.calculate_twoF_cumulative(**kwargs)
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(taus/86400., twoFs, label=label, color=c)
        ax.set_xlabel(r'Days from $t_{{\rm start}}={:.0f}$'.format(
            kwargs['tstart']))
        if self.BSGL:
            ax.set_ylabel(r'$\log_{10}(\mathrm{BSGL})_{\rm cumulative}$')
        else:
            ax.set_ylabel(r'$\widetilde{2\mathcal{F}}_{\rm cumulative}$')
        ax.set_xlim(0, taus[-1]/86400)
        if title:
            ax.set_title(title)
        if savefig:
            plt.tight_layout()
            plt.savefig('{}/{}_twoFcumulative.png'.format(outdir, label))
            return taus, twoFs
        else:
            return ax


class SemiCoherentSearch(BaseSearchClass, ComputeFstat):
    """ A semi-coherent search """

    @helper_functions.initializer
    def __init__(self, label, outdir, tref, nsegs=None, sftfilepath=None,
                 binary=False, BSGL=False, minStartTime=None,
                 maxStartTime=None, minCoverFreq=None, maxCoverFreq=None,
                 detector=None, earth_ephem=None, sun_ephem=None,
                 injectSources=None):
        """
        Parameters
        ----------
        label, outdir: str
            A label and directory to read/write data from/to.
        tref, minStartTime, maxStartTime: int
            GPS seconds of the reference time, and start and end of the data.
        nsegs: int
            The (fixed) number of segments
        sftfilepath: str
            File patern to match SFTs

        For all other parameters, see pyfstat.ComputeFStat.
        """

        self.fs_file_name = "{}/{}_FS.dat".format(self.outdir, self.label)
        if self.earth_ephem is None:
            self.earth_ephem = self.earth_ephem_default
        if self.sun_ephem is None:
            self.sun_ephem = self.sun_ephem_default
        self.transient = True
        self.init_computefstatistic_single_point()
        self.init_semicoherent_parameters()

    def init_semicoherent_parameters(self):
        logging.info(('Initialising semicoherent parameters from {} to {} in'
                      ' {} segments').format(
            self.minStartTime, self.maxStartTime, self.nsegs))
        self.transient = True
        self.whatToCompute = lalpulsar.FSTATQ_2F+lalpulsar.FSTATQ_ATOMS_PER_DET
        self.tboundaries = np.linspace(self.minStartTime, self.maxStartTime,
                                       self.nsegs+1)

    def run_semi_coherent_computefstatistic_single_point(
            self, F0, F1, F2, Alpha, Delta, asini=None,
            period=None, ecc=None, tp=None, argp=None):
        """ Returns twoF or ln(BSGL) semi-coherently at a single point """

        self.PulsarDopplerParams.fkdot = np.array([F0, F1, F2, 0, 0, 0, 0])
        self.PulsarDopplerParams.Alpha = Alpha
        self.PulsarDopplerParams.Delta = Delta
        if self.binary:
            self.PulsarDopplerParams.asini = asini
            self.PulsarDopplerParams.period = period
            self.PulsarDopplerParams.ecc = ecc
            self.PulsarDopplerParams.tp = tp
            self.PulsarDopplerParams.argp = argp

        lalpulsar.ComputeFstat(self.FstatResults,
                               self.FstatInput,
                               self.PulsarDopplerParams,
                               1,
                               self.whatToCompute
                               )

        if self.transient is False:
            if self.BSGL is False:
                return self.FstatResults.twoF[0]

            twoF = np.float(self.FstatResults.twoF[0])
            self.twoFX[0] = self.FstatResults.twoFPerDet(0)
            self.twoFX[1] = self.FstatResults.twoFPerDet(1)
            log10_BSGL = lalpulsar.ComputeBSGL(twoF, self.twoFX,
                                               self.BSGLSetup)
            return log10_BSGL/np.log10(np.exp(1))

        detStat = 0
        for tstart, tend in zip(self.tboundaries[:-1], self.tboundaries[1:]):
            self.windowRange.t0 = int(tstart)  # TYPE UINT4
            self.windowRange.tau = int(tend - tstart)  # TYPE UINT4

            FS = lalpulsar.ComputeTransientFstatMap(
                self.FstatResults.multiFatoms[0], self.windowRange, False)

            if self.BSGL is False:
                detStat += 2*FS.F_mn.data[0][0]
                continue

            FstatResults_single = copy.copy(self.FstatResults)
            FstatResults_single.lenth = 1
            FstatResults_single.data = self.FstatResults.multiFatoms[0].data[0]
            FS0 = lalpulsar.ComputeTransientFstatMap(
                FstatResults_single.multiFatoms[0], self.windowRange, False)
            FstatResults_single.data = self.FstatResults.multiFatoms[0].data[1]
            FS1 = lalpulsar.ComputeTransientFstatMap(
                FstatResults_single.multiFatoms[0], self.windowRange, False)

            self.twoFX[0] = 2*FS0.F_mn.data[0][0]
            self.twoFX[1] = 2*FS1.F_mn.data[0][0]
            log10_BSGL = lalpulsar.ComputeBSGL(
                    2*FS.F_mn.data[0][0], self.twoFX, self.BSGLSetup)

            detStat += log10_BSGL/np.log10(np.exp(1))

        return detStat


class SemiCoherentGlitchSearch(BaseSearchClass, ComputeFstat):
    """ A semi-coherent glitch search

    This implements a basic `semi-coherent glitch F-stat in which the data
    is divided into segments either side of the proposed glitches and the
    fully-coherent F-stat in each segment is summed to give the semi-coherent
    F-stat
    """

    @helper_functions.initializer
    def __init__(self, label, outdir, tref, minStartTime, maxStartTime,
                 nglitch=0, sftfilepath=None, theta0_idx=0, BSGL=False,
                 minCoverFreq=None, maxCoverFreq=None,
                 detector=None, earth_ephem=None, sun_ephem=None):
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
        sftfilepath: str
            File patern to match SFTs
        theta0_idx, int
            Index (zero-based) of which segment the theta refers to - uyseful
            if providing a tight prior on theta to allow the signal to jump
            too theta (and not just from)

        For all other parameters, see pyfstat.ComputeFStat.
        """

        self.fs_file_name = "{}/{}_FS.dat".format(self.outdir, self.label)
        if self.earth_ephem is None:
            self.earth_ephem = self.earth_ephem_default
        if self.sun_ephem is None:
            self.sun_ephem = self.sun_ephem_default
        self.transient = True
        self.binary = False
        self.init_computefstatistic_single_point()

    def compute_nglitch_fstat(self, F0, F1, F2, Alpha, Delta, *args):
        """ Returns the semi-coherent glitch summed twoF """

        args = list(args)
        tboundaries = ([self.minStartTime] + args[-self.nglitch:]
                       + [self.maxStartTime])
        delta_F0s = args[-3*self.nglitch:-2*self.nglitch]
        delta_F1s = args[-2*self.nglitch:-self.nglitch]
        delta_F2 = np.zeros(len(delta_F0s))
        delta_phi = np.zeros(len(delta_F0s))
        theta = [0, F0, F1, F2]
        delta_thetas = np.atleast_2d(
                np.array([delta_phi, delta_F0s, delta_F1s, delta_F2]).T)

        thetas = self.calculate_thetas(theta, delta_thetas, tboundaries,
                                       theta0_idx=self.theta0_idx)

        twoFSum = 0
        for i, theta_i_at_tref in enumerate(thetas):
            ts, te = tboundaries[i], tboundaries[i+1]

            twoFVal = self.run_computefstatistic_single_point(
                ts, te, theta_i_at_tref[1], theta_i_at_tref[2],
                theta_i_at_tref[3], Alpha, Delta)
            twoFSum += twoFVal

        if np.isfinite(twoFSum):
            return twoFSum
        else:
            return -np.inf

    def compute_glitch_fstat_single(self, F0, F1, F2, Alpha, Delta, delta_F0,
                                    delta_F1, tglitch):
        """ Returns the semi-coherent glitch summed twoF for nglitch=1

        Note: OBSOLETE, used only for testing
        """

        theta = [F0, F1, F2]
        delta_theta = [delta_F0, delta_F1, 0]
        tref = self.tref

        theta_at_glitch = self.shift_coefficients(theta, tglitch - tref)
        theta_post_glitch_at_glitch = theta_at_glitch + delta_theta
        theta_post_glitch = self.shift_coefficients(
            theta_post_glitch_at_glitch, tref - tglitch)

        twoFsegA = self.run_computefstatistic_single_point(
            self.minStartTime, tglitch, theta[0], theta[1], theta[2], Alpha,
            Delta)

        if tglitch == self.maxStartTime:
            return twoFsegA

        twoFsegB = self.run_computefstatistic_single_point(
            tglitch, self.maxStartTime, theta_post_glitch[0],
            theta_post_glitch[1], theta_post_glitch[2], Alpha,
            Delta)

        return twoFsegA + twoFsegB


class Writer(BaseSearchClass):
    """ Instance object for generating SFTs containing glitch signals """
    @helper_functions.initializer
    def __init__(self, label='Test', tstart=700000000, duration=100*86400,
                 dtglitch=None, delta_phi=0, delta_F0=0, delta_F1=0,
                 delta_F2=0, tref=None, F0=30, F1=1e-10, F2=0, Alpha=5e-3,
                 Delta=6e-2, h0=0.1, cosi=0.0, psi=0.0, phi=0, Tsft=1800,
                 outdir=".", sqrtSX=1, Band=4, detector='H1',
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
        self.sftfilename = lalpulsar.OfficialSFTFilename(
            'H', '1', numSFTs, self.Tsft, self.minStartTime,
            self.data_duration, self.label)
        self.sftfilepath = '{}/{}'.format(self.outdir, self.sftfilename)
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
        logging.info('Execute: {}'.format(
            'lalapps_SFTdumpheader {} | head -n 20'.format(self.sftfilepath)))
        output = subprocess.check_output(
            'lalapps_SFTdumpheader {} | head -n 20'.format(self.sftfilepath),
            shell=True)
        calls = [line for line in output.split('\n') if line[:3] == 'lal']
        if calls[0] == cl:
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

        cl = []
        cl.append('lalapps_Makefakedata_v5')
        cl.append('--outSingleSFT=TRUE')
        cl.append('--outSFTdir="{}"'.format(self.outdir))
        cl.append('--outLabel="{}"'.format(self.label))
        cl.append('--IFOs="{}"'.format(self.detector))
        if self.add_noise:
            cl.append('--sqrtSX="{}"'.format(self.sqrtSX))
        if self.minStartTime is None:
            cl.append('--startTime={:10.9f}'.format(float(self.tstart)))
        else:
            cl.append('--startTime={:10.9f}'.format(float(self.minStartTime)))
        if self.maxStartTime is None:
            cl.append('--duration={}'.format(int(self.duration)))
        else:
            data_duration = self.maxStartTime - self.minStartTime
            cl.append('--duration={}'.format(int(data_duration)))
        cl.append('--fmin={}'.format(int(self.fmin)))
        cl.append('--Band={}'.format(self.Band))
        cl.append('--Tsft={}'.format(self.Tsft))
        if self.h0 != 0:
            cl.append('--injectionSources="{}"'.format(self.config_file_name))

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

        c_l.append("--minStartTime={}".format(int(self.minStartTime)))
        c_l.append("--maxStartTime={}".format(int(self.maxStartTime)))

        logging.info("Executing: " + " ".join(c_l) + "\n")
        output = subprocess.check_output(" ".join(c_l), shell=True)
        twoF = float(output.split('\n')[-2])
        return float(twoF)
