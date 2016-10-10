""" Classes for various types of searches using ComputeFstatistic """
import os
import sys
import itertools
import logging
import argparse
import copy
import glob
import inspect
from functools import wraps
import subprocess
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import emcee
import corner
import dill as pickle
import lal
import lalpulsar

plt.rcParams['text.usetex'] = True
plt.rcParams['axes.formatter.useoffset'] = False

config_file = os.path.expanduser('~')+'/.pyfstat.conf'
if os.path.isfile(config_file):
    d = {}
    with open(config_file, 'r') as f:
        for line in f:
            k, v = line.split('=')
            k = k.replace(' ', '')
            v = v.replace(' ', '').replace("'", "").replace('"', '').replace('\n', '')
            d[k] = v
    earth_ephem = d['earth_ephem']
    sun_ephem = d['sun_ephem']
else:
    logging.warning('No ~/.pyfstat.conf file found please provide the paths '
                    'when initialising searches')
    earth_ephem = None
    sun_ephem = None

parser = argparse.ArgumentParser()
parser.add_argument("-q", "--quite", help="Decrease output verbosity",
                    action="store_true")
parser.add_argument("-c", "--clean", help="Don't use cached data",
                    action="store_true")
parser.add_argument("-u", "--use-old-data", action="store_true")
parser.add_argument('unittest_args', nargs='*')
args, unknown = parser.parse_known_args()
sys.argv[1:] = args.unittest_args

if args.quite:
    log_level = logging.WARNING
else:
    log_level = logging.DEBUG

logging.basicConfig(level=log_level,
                    format='%(asctime)s %(levelname)-8s: %(message)s',
                    datefmt='%H:%M')


def initializer(func):
    """ Automatically assigns the parameters to self """
    names, varargs, keywords, defaults = inspect.getargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper


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
    """ The base search class, provides ephemeris and general utilities """

    earth_ephem_default = earth_ephem
    sun_ephem_default = sun_ephem

    def add_log_file(self):
        ' Log output to a log-file, requires class to have outdir and label '
        logfilename = '{}/{}.log'.format(self.outdir, self.label)
        fh = logging.FileHandler(logfilename)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)-8s: %(message)s',
            datefmt='%y-%m-%d %H:%M'))
        logging.getLogger().addHandler(fh)

    def shift_matrix(self, n, dT):
        """ Generate the shift matrix """
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


class ComputeFstat(object):
    """ Base class providing interface to lalpulsar.ComputeFstat """

    earth_ephem_default = earth_ephem
    sun_ephem_default = sun_ephem

    @initializer
    def __init__(self, tref, sftfilepath=None,
                 minStartTime=None, maxStartTime=None,
                 minCoverFreq=None, maxCoverFreq=None,
                 detector=None, earth_ephem=None, sun_ephem=None,
                 binary=False, transient=True, BSGL=False):
        """
        Parameters
        ----------
        tref: int
            GPS seconds of the reference time.
        sftfilepath: str
            File patern to match SFTs
        minCoverFreq, maxCoverFreq: float
            The min and max cover frequency passed to CreateFstatInput, if
            either is None the range of frequencies in the SFT less 1Hz is
            used.
        detector: str
            Two character reference to the data to use, specify None for no
            contraint.
        earth_ephem, sun_ephem: str
            Paths of the two files containing positions of Earth and Sun,
            respectively at evenly spaced times, as passed to CreateFstatInput.
            If None defaults defined in BaseSearchClass will be used.
        binary: bool
            If true, search of binary parameters.
        transient: bool
            If true, allow for the Fstat to be computed over a transient range.
        BSGL: bool
            If true, compute the BSGL rather than the twoF value.

        """

        if earth_ephem is None:
            self.earth_ephem = self.earth_ephem_default
        if sun_ephem is None:
            self.sun_ephem = self.sun_ephem_default

        self.init_computefstatistic_single_point()

    def init_computefstatistic_single_point(self):
        """ Initilisation step of run_computefstatistic for a single point """

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
        names = list(set([d.header.name for d in SFTCatalog.data]))
        epochs = [d.header.epoch for d in SFTCatalog.data]
        logging.info(
            'Loaded {} data files from detectors {} spanning {} to {}'.format(
                len(epochs), names, int(epochs[0]), int(epochs[-1])))

        logging.info('Initialising ephems')
        ephems = lalpulsar.InitBarycenter(self.earth_ephem, self.sun_ephem)

        logging.info('Initialising FstatInput')
        dFreq = 0
        if self.transient:
            self.whatToCompute = lalpulsar.FSTATQ_ATOMS_PER_DET
        else:
            self.whatToCompute = lalpulsar.FSTATQ_2F

        FstatOptionalArgs = lalpulsar.FstatOptionalArgsDefaults

        if self.minCoverFreq is None or self.maxCoverFreq is None:
            fA = SFTCatalog.data[0].header.f0
            numBins = SFTCatalog.data[0].numBins
            fB = fA + (numBins-1)*SFTCatalog.data[0].header.deltaF
            self.minCoverFreq = fA + 0.5
            self.maxCoverFreq = fB - 0.5
            logging.info('Min/max cover freqs not provided, using '
                         '{} and {}, est. from SFTs'.format(
                             self.minCoverFreq, self.maxCoverFreq))

        self.FstatInput = lalpulsar.CreateFstatInput(SFTCatalog,
                                                     self.minCoverFreq,
                                                     self.maxCoverFreq,
                                                     dFreq,
                                                     ephems,
                                                     FstatOptionalArgs
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
            logging.info('Initialising BSGL: this will fail if numDet < 2')
            # Tuning parameters - to be reviewed
            numDetectors = 2
            Fstar0sc = 15.
            oLGX = np.zeros(10)
            oLGX[:numDetectors] = 1./numDetectors
            self.BSGLSetup = lalpulsar.CreateBSGLSetup(numDetectors,
                                                       Fstar0sc,
                                                       oLGX,
                                                       False,
                                                       1)
            self.twoFX = np.zeros(10)
            self.whatToCompute = (lalpulsar.FSTATQ_2F +
                                  lalpulsar.FSTATQ_2F_PER_DET)

        if self.transient:
            logging.info('Initialising transient parameters')
            self.windowRange = lalpulsar.transientWindowRange_t()
            self.windowRange.type = lalpulsar.TRANSIENT_RECTANGULAR
            self.windowRange.t0Band = 0
            self.windowRange.dt0 = 1
            self.windowRange.tauBand = 0
            self.windowRange.dtau = 1

    def run_computefstatistic_single_point(self, tstart, tend, F0, F1,
                                           F2, Alpha, Delta, asini=None,
                                           period=None, ecc=None, tp=None,
                                           argp=None):
        """ Returns the twoF fully-coherently at a single point """

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
            BSGL = lalpulsar.ComputeBSGL(twoF, self.twoFX,
                                         self.BSGLSetup)
            return BSGL

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
        BSGL = lalpulsar.ComputeBSGL(2*FS.F_mn.data[0][0], self.twoFX,
                                     self.BSGLSetup)

        return BSGL


class SemiCoherentGlitchSearch(BaseSearchClass, ComputeFstat):
    """ A semi-coherent glitch search

    This implements a basic `semi-coherent glitch F-stat in which the data
    is divided into two segments either side of the proposed glitch and the
    fully-coherent F-stat in each segment is averaged to give the semi-coherent
    F-stat
    """

    @initializer
    def __init__(self, label, outdir, tref, tstart, tend, nglitch=0,
                 sftfilepath=None, theta0_idx=0, BSGL=False,
                 minCoverFreq=None, maxCoverFreq=None, minStartTime=None,
                 maxStartTime=None, detector=None, earth_ephem=None,
                 sun_ephem=None):
        """
        Parameters
        ----------
        label, outdir: str
            A label and directory to read/write data from/to.
        tref, tstart, tend: int
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
        minCoverFreq, maxCoverFreq: float
            The min and max cover frequency passed to CreateFstatInput, if
            either is None the range of frequencies in the SFT less 1Hz is
            used.
        detector: str
            Two character reference to the data to use, specify None for no
            contraint.
        earth_ephem, sun_ephem: str
            Paths of the two files containing positions of Earth and Sun,
            respectively at evenly spaced times, as passed to CreateFstatInput.
            If None defaults defined in BaseSearchClass will be used.
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
        tboundaries = [self.tstart] + args[-self.nglitch:] + [self.tend]
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

        Note: used for testing
        """

        theta = [F0, F1, F2]
        delta_theta = [delta_F0, delta_F1, 0]
        tref = self.tref

        theta_at_glitch = self.shift_coefficients(theta, tglitch - tref)
        theta_post_glitch_at_glitch = theta_at_glitch + delta_theta
        theta_post_glitch = self.shift_coefficients(
            theta_post_glitch_at_glitch, tref - tglitch)

        twoFsegA = self.run_computefstatistic_single_point(
            self.tstart, tglitch, theta[0], theta[1], theta[2], Alpha,
            Delta)

        if tglitch == self.tend:
            return twoFsegA

        twoFsegB = self.run_computefstatistic_single_point(
            tglitch, self.tend, theta_post_glitch[0],
            theta_post_glitch[1], theta_post_glitch[2], Alpha,
            Delta)

        return twoFsegA + twoFsegB


class MCMCSearch(BaseSearchClass):
    """ MCMC search using ComputeFstat"""
    @initializer
    def __init__(self, label, outdir, sftfilepath, theta_prior, tref,
                 tstart, tend, nsteps=[100, 100, 100], nwalkers=100, ntemps=1,
                 log10temperature_min=-5, theta_initial=None, scatter_val=1e-4,
                 binary=False, BSGL=False, minCoverFreq=None,
                 maxCoverFreq=None, detector=None, earth_ephem=None,
                 sun_ephem=None, theta0_idx=0):
        """
        Parameters
        label, outdir: str
            A label and directory to read/write data from/to
        sftfilepath: str
            File patern to match SFTs
        theta_prior: dict
            Dictionary of priors and fixed values for the search parameters.
            For each parameters (key of the dict), if it is to be held fixed
            the value should be the constant float, if it is be searched, the
            value should be a dictionary of the prior.
        theta_initial: dict, array, (None)
            Either a dictionary of distribution about which to distribute the
            initial walkers about, an array (from which the walkers will be
            scattered by scatter_val, or  None in which case the prior is used.
        tref, tstart, tend: int
            GPS seconds of the reference time, start time and end time
        nsteps: list (m,)
            List specifying the number of steps to take, the last two entries
            give the nburn and nprod of the 'production' run, all entries
            before are for iterative initialisation steps (usually just one)
            e.g. [1000, 1000, 500].
        nwalkers, ntemps: int,
            The number of walkers and temperates to use in the parallel
            tempered PTSampler.
        log10temperature_min float < 0
            The  log_10(tmin) value, the set of betas passed to PTSampler are
            generated from np.logspace(0, log10temperature_min, ntemps).
        binary: Bool
            If true, search over binary parameters
        detector: str
            Two character reference to the data to use, specify None for no
            contraint.
        minCoverFreq, maxCoverFreq: float
            Minimum and maximum instantaneous frequency which will be covered
            over the SFT time span as passed to CreateFstatInput
        earth_ephem, sun_ephem: str
            Paths of the two files containing positions of Earth and Sun,
            respectively at evenly spaced times, as passed to CreateFstatInput
            If None defaults defined in BaseSearchClass will be used

        """

        self.minStartTime = tstart
        self.maxStartTime = tend

        self.add_log_file()
        logging.info(
            'Set-up MCMC search for model {} on data {}'.format(
                self.label, self.sftfilepath))
        if os.path.isdir(outdir) is False:
            os.mkdir(outdir)
        self.pickle_path = '{}/{}_saved_data.p'.format(self.outdir, self.label)
        self.theta_prior['tstart'] = self.tstart
        self.theta_prior['tend'] = self.tend
        self.unpack_input_theta()
        self.ndim = len(self.theta_keys)
        if self.log10temperature_min:
            self.betas = np.logspace(0, self.log10temperature_min, self.ntemps)
        else:
            self.betas = None

        if earth_ephem is None:
            self.earth_ephem = self.earth_ephem_default
        if sun_ephem is None:
            self.sun_ephem = self.sun_ephem_default

        if args.clean and os.path.isfile(self.pickle_path):
            os.rename(self.pickle_path, self.pickle_path+".old")

        self.old_data_is_okay_to_use = self.check_old_data_is_okay_to_use()
        self.log_input()

    def log_input(self):
        logging.info('theta_prior = {}'.format(self.theta_prior))
        logging.info('nwalkers={}'.format(self.nwalkers))
        logging.info('scatter_val = {}'.format(self.scatter_val))
        logging.info('nsteps = {}'.format(self.nsteps))
        logging.info('ntemps = {}'.format(self.ntemps))
        logging.info('log10temperature_min = {}'.format(
            self.log10temperature_min))

    def inititate_search_object(self):
        logging.info('Setting up search object')
        self.search = ComputeFstat(
            tref=self.tref, sftfilepath=self.sftfilepath,
            minCoverFreq=self.minCoverFreq, maxCoverFreq=self.maxCoverFreq,
            earth_ephem=self.earth_ephem, sun_ephem=self.sun_ephem,
            detector=self.detector, BSGL=self.BSGL, transient=False,
            minStartTime=self.minStartTime, maxStartTime=self.maxStartTime)

    def logp(self, theta_vals, theta_prior, theta_keys, search):
        H = [self.generic_lnprior(**theta_prior[key])(p) for p, key in
             zip(theta_vals, theta_keys)]
        return np.sum(H)

    def logl(self, theta, search):
        for j, theta_i in enumerate(self.theta_idxs):
            self.fixed_theta[theta_i] = theta[j]
        FS = search.run_computefstatistic_single_point(*self.fixed_theta)
        return FS

    def unpack_input_theta(self):
        full_theta_keys = ['tstart', 'tend', 'F0', 'F1', 'F2', 'Alpha',
                           'Delta']
        if self.binary:
            full_theta_keys += [
                'asini', 'period', 'ecc', 'tp', 'argp']
        full_theta_keys_copy = copy.copy(full_theta_keys)

        full_theta_symbols = ['_', '_', '$f$', '$\dot{f}$', '$\ddot{f}$',
                              r'$\alpha$', r'$\delta$']
        if self.binary:
            full_theta_symbols += [
                'asini', 'period', 'period', 'ecc', 'tp', 'argp']

        self.theta_keys = []
        fixed_theta_dict = {}
        for key, val in self.theta_prior.iteritems():
            if type(val) is dict:
                fixed_theta_dict[key] = 0
                self.theta_keys.append(key)
            elif type(val) in [float, int, np.float64]:
                fixed_theta_dict[key] = val
            else:
                raise ValueError(
                    'Type {} of {} in theta not recognised'.format(
                        type(val), key))
            full_theta_keys_copy.pop(full_theta_keys_copy.index(key))

        if len(full_theta_keys_copy) > 0:
            raise ValueError(('Input dictionary `theta` is missing the'
                              'following keys: {}').format(
                                  full_theta_keys_copy))

        self.fixed_theta = [fixed_theta_dict[key] for key in full_theta_keys]
        self.theta_idxs = [full_theta_keys.index(k) for k in self.theta_keys]
        self.theta_symbols = [full_theta_symbols[i] for i in self.theta_idxs]

        idxs = np.argsort(self.theta_idxs)
        self.theta_idxs = [self.theta_idxs[i] for i in idxs]
        self.theta_symbols = [self.theta_symbols[i] for i in idxs]
        self.theta_keys = [self.theta_keys[i] for i in idxs]

    def check_initial_points(self, p0):
        for nt in range(self.ntemps):
            logging.info('Checking temperature {} chains'.format(nt))
            initial_priors = np.array([
                self.logp(p, self.theta_prior, self.theta_keys, self.search)
                for p in p0[nt]])
            number_of_initial_out_of_bounds = sum(initial_priors == -np.inf)

            if number_of_initial_out_of_bounds > 0:
                logging.warning(
                    'Of {} initial values, {} are -np.inf due to the prior'
                    .format(len(initial_priors),
                            number_of_initial_out_of_bounds))

                p0 = self.generate_new_p0_to_fix_initial_points(
                    p0, nt, initial_priors)

    def generate_new_p0_to_fix_initial_points(self, p0, nt, initial_priors):
        logging.info('Attempting to correct intial values')
        idxs = np.arange(self.nwalkers)[initial_priors == -np.inf]
        count = 0
        while sum(initial_priors == -np.inf) > 0 and count < 100:
            for j in idxs:
                p0[nt][j] = (p0[nt][np.random.randint(0, self.nwalkers)]*(
                             1+np.random.normal(0, 1e-10, self.ndim)))
            initial_priors = np.array([
                self.logp(p, self.theta_prior, self.theta_keys,
                          self.search)
                for p in p0[nt]])
            count += 1

        if sum(initial_priors == -np.inf) > 0:
            logging.info('Failed to fix initial priors')
        else:
            logging.info('Suceeded to fix initial priors')

        return p0

    def run_sampler_with_progress_bar(self, sampler, ns, p0):
        try:
            from tqdm import tqdm
            for result in tqdm(sampler.sample(p0, iterations=ns), total=ns):
                pass
        except ImportError:
            sampler.run_mcmc(p0, ns)
        return sampler

    def run(self, proposal_scale_factor=2):

        if self.old_data_is_okay_to_use is True:
            logging.warning('Using saved data from {}'.format(
                self.pickle_path))
            d = self.get_saved_data()
            self.sampler = d['sampler']
            self.samples = d['samples']
            self.lnprobs = d['lnprobs']
            self.lnlikes = d['lnlikes']
            return

        self.inititate_search_object()

        sampler = emcee.PTSampler(
            self.ntemps, self.nwalkers, self.ndim, self.logl, self.logp,
            logpargs=(self.theta_prior, self.theta_keys, self.search),
            loglargs=(self.search,), betas=self.betas, a=proposal_scale_factor)

        p0 = self.generate_initial_p0()
        p0 = self.apply_corrections_to_p0(p0)
        self.check_initial_points(p0)

        ninit_steps = len(self.nsteps) - 2
        for j, n in enumerate(self.nsteps[:-2]):
            logging.info('Running {}/{} initialisation with {} steps'.format(
                j+1, ninit_steps, n))
            sampler = self.run_sampler_with_progress_bar(sampler, n, p0)
            logging.info("Mean acceptance fraction: {}"
                         .format(np.mean(sampler.acceptance_fraction, axis=1)))
            if self.ntemps > 1:
                logging.info("Tswap acceptance fraction: {}"
                             .format(sampler.tswap_acceptance_fraction))
            fig, axes = self.plot_walkers(sampler, symbols=self.theta_symbols)
            fig.savefig('{}/{}_init_{}_walkers.png'.format(
                self.outdir, self.label, j))

            p0 = self.get_new_p0(sampler)
            p0 = self.apply_corrections_to_p0(p0)
            self.check_initial_points(p0)
            sampler.reset()

        if len(self.nsteps) > 1:
            nburn = self.nsteps[-2]
        else:
            nburn = 0
        nprod = self.nsteps[-1]
        logging.info('Running final burn and prod with {} steps'.format(
            nburn+nprod))
        sampler = self.run_sampler_with_progress_bar(sampler, nburn+nprod, p0)
        logging.info("Mean acceptance fraction: {}"
                     .format(np.mean(sampler.acceptance_fraction, axis=1)))
        if self.ntemps > 1:
            logging.info("Tswap acceptance fraction: {}"
                         .format(sampler.tswap_acceptance_fraction))

        fig, axes = self.plot_walkers(sampler, symbols=self.theta_symbols)
        fig.savefig('{}/{}_walkers.png'.format(self.outdir, self.label))

        samples = sampler.chain[0, :, nburn:, :].reshape((-1, self.ndim))
        lnprobs = sampler.lnprobability[0, :, nburn:].reshape((-1))
        lnlikes = sampler.lnlikelihood[0, :, nburn:].reshape((-1))
        self.sampler = sampler
        self.samples = samples
        self.lnprobs = lnprobs
        self.lnlikes = lnlikes
        self.save_data(sampler, samples, lnprobs, lnlikes)

    def plot_corner(self, figsize=(7, 7),  tglitch_ratio=False,
                    add_prior=False, nstds=None, label_offset=0.4,
                    dpi=300, rc_context={}, **kwargs):

        with plt.rc_context(rc_context):
            fig, axes = plt.subplots(self.ndim, self.ndim,
                                     figsize=figsize)

            samples_plt = copy.copy(self.samples)
            theta_symbols_plt = copy.copy(self.theta_symbols)
            theta_symbols_plt = [s.replace('_{glitch}', r'_\textrm{glitch}')
                                 for s in theta_symbols_plt]

            if tglitch_ratio:
                for j, k in enumerate(self.theta_keys):
                    if k == 'tglitch':
                        s = samples_plt[:, j]
                        samples_plt[:, j] = (s - self.tstart)/(
                                             self.tend - self.tstart)
                        theta_symbols_plt[j] = r'$R_{\textrm{glitch}}$'

            if type(nstds) is int and 'range' not in kwargs:
                _range = []
                for j, s in enumerate(samples_plt.T):
                    median = np.median(s)
                    std = np.std(s)
                    _range.append((median - nstds*std, median + nstds*std))
            else:
                _range = None

            fig_triangle = corner.corner(samples_plt,
                                         labels=theta_symbols_plt,
                                         fig=fig,
                                         bins=50,
                                         max_n_ticks=4,
                                         plot_contours=True,
                                         plot_datapoints=True,
                                         label_kwargs={'fontsize': 8},
                                         data_kwargs={'alpha': 0.1,
                                                      'ms': 0.5},
                                         range=_range,
                                         **kwargs)

            axes_list = fig_triangle.get_axes()
            axes = np.array(axes_list).reshape(self.ndim, self.ndim)
            plt.draw()
            for ax in axes[:, 0]:
                ax.yaxis.set_label_coords(-label_offset, 0.5)
            for ax in axes[-1, :]:
                ax.xaxis.set_label_coords(0.5, -label_offset)
            for ax in axes_list:
                ax.set_rasterized(True)
                ax.set_rasterization_zorder(-10)
            plt.tight_layout(h_pad=0.0, w_pad=0.0)
            fig.subplots_adjust(hspace=0.05, wspace=0.05)

            if add_prior:
                self.add_prior_to_corner(axes, samples_plt)

            fig_triangle.savefig('{}/{}_corner.png'.format(
                self.outdir, self.label), dpi=dpi)

    def add_prior_to_corner(self, axes, samples):
        for i, key in enumerate(self.theta_keys):
            ax = axes[i][i]
            xlim = ax.get_xlim()
            s = samples[:, i]
            prior = self.generic_lnprior(**self.theta_prior[key])
            x = np.linspace(s.min(), s.max(), 100)
            ax2 = ax.twinx()
            ax2.get_yaxis().set_visible(False)
            ax2.plot(x, [prior(xi) for xi in x], '-r')
            ax.set_xlim(xlim)

    def generic_lnprior(self, **kwargs):
        """ Return a lambda function of the pdf

        Parameters
        ----------
        kwargs: dict
            A dictionary containing 'type' of pdf and shape parameters

        """

        def logunif(x, a, b):
            above = x < b
            below = x > a
            if type(above) is not np.ndarray:
                if above and below:
                    return -np.log(b-a)
                else:
                    return -np.inf
            else:
                idxs = np.array([all(tup) for tup in zip(above, below)])
                p = np.zeros(len(x)) - np.inf
                p[idxs] = -np.log(b-a)
                return p

        def halfnorm(x, loc, scale):
            if x < 0:
                return -np.inf
            else:
                return -0.5*((x-loc)**2/scale**2+np.log(0.5*np.pi*scale**2))

        def cauchy(x, x0, gamma):
            return 1.0/(np.pi*gamma*(1+((x-x0)/gamma)**2))

        def exp(x, x0, gamma):
            if x > x0:
                return np.log(gamma) - gamma*(x - x0)
            else:
                return -np.inf

        if kwargs['type'] == 'unif':
            return lambda x: logunif(x, kwargs['lower'], kwargs['upper'])
        elif kwargs['type'] == 'halfnorm':
            return lambda x: halfnorm(x, kwargs['loc'], kwargs['scale'])
        elif kwargs['type'] == 'neghalfnorm':
            return lambda x: halfnorm(-x, kwargs['loc'], kwargs['scale'])
        elif kwargs['type'] == 'norm':
            return lambda x: -0.5*((x - kwargs['loc'])**2/kwargs['scale']**2
                                   + np.log(2*np.pi*kwargs['scale']**2))
        else:
            logging.info("kwargs:", kwargs)
            raise ValueError("Print unrecognise distribution")

    def generate_rv(self, **kwargs):
        dist_type = kwargs.pop('type')
        if dist_type == "unif":
            return np.random.uniform(low=kwargs['lower'], high=kwargs['upper'])
        if dist_type == "norm":
            return np.random.normal(loc=kwargs['loc'], scale=kwargs['scale'])
        if dist_type == "halfnorm":
            return np.abs(np.random.normal(loc=kwargs['loc'],
                                           scale=kwargs['scale']))
        if dist_type == "neghalfnorm":
            return -1 * np.abs(np.random.normal(loc=kwargs['loc'],
                                                scale=kwargs['scale']))
        if dist_type == "lognorm":
            return np.random.lognormal(
                mean=kwargs['loc'], sigma=kwargs['scale'])
        else:
            raise ValueError("dist_type {} unknown".format(dist_type))

    def plot_walkers(self, sampler, symbols=None, alpha=0.4, color="k", temp=0,
                     start=None, stop=None, draw_vline=None):
        """ Plot all the chains from a sampler """

        shape = sampler.chain.shape
        if len(shape) == 3:
            nwalkers, nsteps, ndim = shape
            chain = sampler.chain[:, :, :]
        if len(shape) == 4:
            ntemps, nwalkers, nsteps, ndim = shape
            if temp < ntemps:
                logging.info("Plotting temperature {} chains".format(temp))
            else:
                raise ValueError(("Requested temperature {} outside of"
                                  "available range").format(temp))
            chain = sampler.chain[temp, :, :, :]

        with plt.style.context(('classic')):
            fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8, 4*ndim))

            if ndim > 1:
                for i in range(ndim):
                    axes[i].ticklabel_format(useOffset=False, axis='y')
                    cs = chain[:, start:stop, i].T
                    axes[i].plot(cs, color="k", alpha=alpha)
                    if symbols:
                        axes[i].set_ylabel(symbols[i])
                    if draw_vline is not None:
                        axes[i].axvline(draw_vline, lw=2, ls="--")

            else:
                cs = chain[:, start:stop, 0].T
                axes.plot(cs, color='k', alpha=alpha)
                axes.ticklabel_format(useOffset=False, axis='y')

        return fig, axes

    def apply_corrections_to_p0(self, p0):
        """ Apply any correction to the initial p0 values """
        return p0

    def generate_scattered_p0(self, p):
        """ Generate a set of p0s scattered about p """
        p0 = [[p + self.scatter_val * p * np.random.randn(self.ndim)
               for i in xrange(self.nwalkers)]
              for j in xrange(self.ntemps)]
        return p0

    def generate_initial_p0(self):
        """ Generate a set of init vals for the walkers """

        if type(self.theta_initial) == dict:
            logging.info('Generate initial values from initial dictionary')
            if self.nglitch > 1:
                raise ValueError('Initial dict not implemented for nglitch>1')
            p0 = [[[self.generate_rv(**self.theta_initial[key])
                    for key in self.theta_keys]
                   for i in range(self.nwalkers)]
                  for j in range(self.ntemps)]
        elif type(self.theta_initial) == list:
            logging.info('Generate initial values from list of theta_initial')
            p0 = [[[self.generate_rv(**val)
                    for val in self.theta_initial]
                   for i in range(self.nwalkers)]
                  for j in range(self.ntemps)]
        elif self.theta_initial is None:
            logging.info('Generate initial values from prior dictionary')
            p0 = [[[self.generate_rv(**self.theta_prior[key])
                    for key in self.theta_keys]
                   for i in range(self.nwalkers)]
                  for j in range(self.ntemps)]
        elif len(self.theta_initial) == self.ndim:
            p0 = self.generate_scattered_p0(self.theta_initial)
        else:
            raise ValueError('theta_initial not understood')

        return p0

    def get_new_p0(self, sampler):
        """ Returns new initial positions for walkers are burn0 stage

        This returns new positions for all walkers by scattering points about
        the maximum posterior with scale `scatter_val`.

        """
        if sampler.chain[:, :, -1, :].shape[0] == 1:
            ntemps_temp = 1
        else:
            ntemps_temp = self.ntemps
        pF = sampler.chain[:, :, -1, :].reshape(
            ntemps_temp, self.nwalkers, self.ndim)[0, :, :]
        lnl = sampler.lnlikelihood[:, :, -1].reshape(
            self.ntemps, self.nwalkers)[0, :]
        lnp = sampler.lnprobability[:, :, -1].reshape(
            self.ntemps, self.nwalkers)[0, :]

        # General warnings about the state of lnp
        if any(np.isnan(lnp)):
            logging.warning(
                "Of {} lnprobs {} are nan".format(
                    len(lnp), np.sum(np.isnan(lnp))))
        if any(np.isposinf(lnp)):
            logging.warning(
                "Of {} lnprobs {} are +np.inf".format(
                    len(lnp), np.sum(np.isposinf(lnp))))
        if any(np.isneginf(lnp)):
            logging.warning(
                "Of {} lnprobs {} are -np.inf".format(
                    len(lnp), np.sum(np.isneginf(lnp))))

        lnp_finite = copy.copy(lnp)
        lnp_finite[np.isinf(lnp)] = np.nan
        p = pF[np.nanargmax(lnp_finite)]
        logging.info('Generating new p0 from max lnp which had twoF={}'
                     .format(lnl[np.nanargmax(lnp_finite)]))
        p0 = self.generate_scattered_p0(p)

        return p0

    def get_save_data_dictionary(self):
        d = dict(nsteps=self.nsteps, nwalkers=self.nwalkers,
                 ntemps=self.ntemps, theta_keys=self.theta_keys,
                 theta_prior=self.theta_prior, scatter_val=self.scatter_val,
                 log10temperature_min=self.log10temperature_min,
                 theta0_idx=self.theta0_idx)
        return d

    def save_data(self, sampler, samples, lnprobs, lnlikes):
        d = self.get_save_data_dictionary()
        d['sampler'] = sampler
        d['samples'] = samples
        d['lnprobs'] = lnprobs
        d['lnlikes'] = lnlikes

        if os.path.isfile(self.pickle_path):
            logging.info('Saving backup of {} as {}.old'.format(
                self.pickle_path, self.pickle_path))
            os.rename(self.pickle_path, self.pickle_path+".old")
        with open(self.pickle_path, "wb") as File:
            pickle.dump(d, File)

    def get_list_of_matching_sfts(self):
        matches = glob.glob(self.sftfilepath)
        if len(matches) > 0:
            return matches
        else:
            raise IOError('No sfts found matching {}'.format(
                self.sftfilepath))

    def get_saved_data(self):
        with open(self.pickle_path, "r") as File:
            d = pickle.load(File)
        return d

    def check_old_data_is_okay_to_use(self):
        if args.use_old_data:
            logging.info("Forcing use of old data")
            return True

        if os.path.isfile(self.pickle_path) is False:
            logging.info('No pickled data found')
            return False

        oldest_sft = min([os.path.getmtime(f) for f in
                          self.get_list_of_matching_sfts()])
        if os.path.getmtime(self.pickle_path) < oldest_sft:
            logging.info('Pickled data outdates sft files')
            return False

        old_d = self.get_saved_data().copy()
        new_d = self.get_save_data_dictionary().copy()

        old_d.pop('samples')
        old_d.pop('sampler')
        old_d.pop('lnprobs')
        old_d.pop('lnlikes')

        mod_keys = []
        for key in new_d.keys():
            if key in old_d:
                if new_d[key] != old_d[key]:
                    mod_keys.append((key, old_d[key], new_d[key]))
            else:
                raise ValueError('Keys {} not in old dictionary'.format(key))

        if len(mod_keys) == 0:
            return True
        else:
            logging.warning("Saved data differs from requested")
            logging.info("Differences found in following keys:")
            for key in mod_keys:
                if len(key) == 3:
                    if np.isscalar(key[1]) or key[0] == 'nsteps':
                        logging.info("    {} : {} -> {}".format(*key))
                    else:
                        logging.info("    " + key[0])
                else:
                    logging.info(key)
            return False

    def get_max_twoF(self, threshold=0.05):
        """ Returns the max 2F sample and the corresponding 2F value

        Note: the sample is returned as a dictionary along with an estimate of
        the standard deviation calculated from the std of all samples with a
        twoF within `threshold` (relative) to the max twoF

        """
        if any(np.isposinf(self.lnlikes)):
            logging.info('twoF values contain positive infinite values')
        if any(np.isneginf(self.lnlikes)):
            logging.info('twoF values contain negative infinite values')
        if any(np.isnan(self.lnlikes)):
            logging.info('twoF values contain nan')
        idxs = np.isfinite(self.lnlikes)
        jmax = np.nanargmax(self.lnlikes[idxs])
        maxtwoF = self.lnlikes[jmax]
        d = OrderedDict()

        repeats = []
        for i, k in enumerate(self.theta_keys):
            if k in d and k not in repeats:
                d[k+'_0'] = d[k]  # relabel the old key
                d.pop(k)
                repeats.append(k)
            if k in repeats:
                k = k + '_0'
                count = 1
                while k in d:
                    k = k.replace('_{}'.format(count-1), '_{}'.format(count))
                    count += 1
            d[k] = self.samples[jmax][i]
        return d, maxtwoF

    def get_median_stds(self):
        """ Returns a dict of the median and std of all production samples """
        d = OrderedDict()
        repeats = []
        for s, k in zip(self.samples.T, self.theta_keys):
            if k in d and k not in repeats:
                d[k+'_0'] = d[k]  # relabel the old key
                d[k+'_0_std'] = d[k+'_std']
                d.pop(k)
                d.pop(k+'_std')
                repeats.append(k)
            if k in repeats:
                k = k + '_0'
                count = 1
                while k in d:
                    k = k.replace('_{}'.format(count-1), '_{}'.format(count))
                    count += 1

            d[k] = np.median(s)
            d[k+'_std'] = np.std(s)
        return d

    def write_par(self, method='med'):
        """ Writes a .par of the best-fit params with an estimated std """
        logging.info('Writing {}/{}.par using the {} method'.format(
            self.outdir, self.label, method))

        median_std_d = self.get_median_stds()
        max_twoF_d, max_twoF = self.get_max_twoF()

        filename = '{}/{}.par'.format(self.outdir, self.label)
        with open(filename, 'w+') as f:
            f.write('MaxtwoF = {}\n'.format(max_twoF))
            f.write('theta0_index = {}\n'.format(self.theta0_idx))
            if method == 'med':
                for key, val in median_std_d.iteritems():
                    f.write('{} = {:1.16e}\n'.format(key, val))
            if method == 'twoFmax':
                for key, val in max_twoF_d.iteritems():
                    f.write('{} = {:1.16e}\n'.format(key, val))

    def print_summary(self):
        max_twoFd, max_twoF = self.get_max_twoF()
        median_std_d = self.get_median_stds()
        print('\nSummary:')
        print('theta0 index: {}'.format(self.theta0_idx))
        print('Max twoF: {} with parameters:'.format(max_twoF))
        for k in np.sort(max_twoFd.keys()):
            print('  {:10s} = {:1.9e}'.format(k, max_twoFd[k]))
        print('\nMedian +/- std for production values')
        for k in np.sort(median_std_d.keys()):
            if 'std' not in k:
                print('  {:10s} = {:1.9e} +/- {:1.9e}'.format(
                    k, median_std_d[k], median_std_d[k+'_std']))


class MCMCGlitchSearch(MCMCSearch):
    """ MCMC search using the SemiCoherentGlitchSearch """
    @initializer
    def __init__(self, label, outdir, sftfilepath, theta_prior, tref,
                 tstart, tend, nglitch=1, nsteps=[100, 100, 100], nwalkers=100,
                 ntemps=1, log10temperature_min=-5, theta_initial=None,
                 scatter_val=1e-4, dtglitchmin=1*86400, theta0_idx=0,
                 detector=None, BSGL=False,
                 minCoverFreq=None, maxCoverFreq=None, earth_ephem=None,
                 sun_ephem=None):
        """
        Parameters
        label, outdir: str
            A label and directory to read/write data from/to
_        sftfilepath: str
            File patern to match SFTs
        theta_prior: dict
            Dictionary of priors and fixed values for the search parameters.
            For each parameters (key of the dict), if it is to be held fixed
            the value should be the constant float, if it is be searched, the
            value should be a dictionary of the prior.
        theta_initial: dict, array, (None)
            Either a dictionary of distribution about which to distribute the
            initial walkers about, an array (from which the walkers will be
            scattered by scatter_val, or  None in which case the prior is used.
        scatter_val, float or ndim array
            Size of scatter to use about the initialisation step, if given as
            an array it must be of length ndim and the order is given by
            theta_keys
        nglitch: int
            The number of glitches to allow
        tref, tstart, tend: int
            GPS seconds of the reference time, start time and end time
        nsteps: list (m,)
            List specifying the number of steps to take, the last two entries
            give the nburn and nprod of the 'production' run, all entries
            before are for iterative initialisation steps (usually just one)
            e.g. [1000, 1000, 500].
        dtglitchmin: int
            The minimum duration (in seconds) of a segment between two glitches
            or a glitch and the start/end of the data
        nwalkers, ntemps: int,
            The number of walkers and temperates to use in the parallel
            tempered PTSampler.
        log10temperature_min float < 0
            The  log_10(tmin) value, the set of betas passed to PTSampler are
            generated from np.logspace(0, log10temperature_min, ntemps).
        theta0_idx, int
            Index (zero-based) of which segment the theta refers to - uyseful
            if providing a tight prior on theta to allow the signal to jump
            too theta (and not just from)
        detector: str
            Two character reference to the data to use, specify None for no
            contraint.
        minCoverFreq, maxCoverFreq: float
            Minimum and maximum instantaneous frequency which will be covered
            over the SFT time span as passed to CreateFstatInput
        earth_ephem, sun_ephem: str
            Paths of the two files containing positions of Earth and Sun,
            respectively at evenly spaced times, as passed to CreateFstatInput
            If None defaults defined in BaseSearchClass will be used

        """

        self.add_log_file()
        logging.info(('Set-up MCMC glitch search with {} glitches for model {}'
                      ' on data {}').format(self.nglitch, self.label,
                                            self.sftfilepath))
        if os.path.isdir(outdir) is False:
            os.mkdir(outdir)
        self.minStartTime = tstart
        self.maxStartTime = tend
        self.pickle_path = '{}/{}_saved_data.p'.format(self.outdir, self.label)
        self.unpack_input_theta()
        self.ndim = len(self.theta_keys)
        if self.log10temperature_min:
            self.betas = np.logspace(0, self.log10temperature_min, self.ntemps)
        else:
            self.betas = None
        if earth_ephem is None:
            self.earth_ephem = self.earth_ephem_default
        if sun_ephem is None:
            self.sun_ephem = self.sun_ephem_default

        if args.clean and os.path.isfile(self.pickle_path):
            os.rename(self.pickle_path, self.pickle_path+".old")

        self.old_data_is_okay_to_use = self.check_old_data_is_okay_to_use()
        self.log_input()

    def inititate_search_object(self):
        logging.info('Setting up search object')
        self.search = SemiCoherentGlitchSearch(
            label=self.label, outdir=self.outdir, sftfilepath=self.sftfilepath,
            tref=self.tref, tstart=self.tstart,
            tend=self.tend, minCoverFreq=self.minCoverFreq,
            maxCoverFreq=self.maxCoverFreq, earth_ephem=self.earth_ephem,
            sun_ephem=self.sun_ephem, detector=self.detector, BSGL=self.BSGL,
            nglitch=self.nglitch, theta0_idx=self.theta0_idx,
            minStartTime=self.minStartTime, maxStartTime=self.maxStartTime)

    def logp(self, theta_vals, theta_prior, theta_keys, search):
        if self.nglitch > 1:
            ts = [self.tstart] + theta_vals[-self.nglitch:] + [self.tend]
            if np.array_equal(ts, np.sort(ts)) is False:
                return -np.inf
            if any(np.diff(ts) < self.dtglitchmin):
                return -np.inf

        H = [self.generic_lnprior(**theta_prior[key])(p) for p, key in
             zip(theta_vals, theta_keys)]
        return np.sum(H)

    def logl(self, theta, search):
        for j, theta_i in enumerate(self.theta_idxs):
            self.fixed_theta[theta_i] = theta[j]
        FS = search.compute_nglitch_fstat(*self.fixed_theta)
        return FS

    def unpack_input_theta(self):
        glitch_keys = ['delta_F0', 'delta_F1', 'tglitch']
        full_glitch_keys = list(np.array(
            [[gk]*self.nglitch for gk in glitch_keys]).flatten())

        if 'tglitch_0' in self.theta_prior:
            full_glitch_keys[-self.nglitch:] = [
                'tglitch_{}'.format(i) for i in range(self.nglitch)]
        full_theta_keys = ['F0', 'F1', 'F2', 'Alpha', 'Delta']+full_glitch_keys
        full_theta_keys_copy = copy.copy(full_theta_keys)

        glitch_symbols = ['$\delta f$', '$\delta \dot{f}$', r'$t_{glitch}$']
        full_glitch_symbols = list(np.array(
            [[gs]*self.nglitch for gs in glitch_symbols]).flatten())
        full_theta_symbols = (['$f$', '$\dot{f}$', '$\ddot{f}$', r'$\alpha$',
                               r'$\delta$'] + full_glitch_symbols)
        self.theta_keys = []
        fixed_theta_dict = {}
        for key, val in self.theta_prior.iteritems():
            if type(val) is dict:
                fixed_theta_dict[key] = 0
                if key in glitch_keys:
                    for i in range(self.nglitch):
                        self.theta_keys.append(key)
                else:
                    self.theta_keys.append(key)
            elif type(val) in [float, int, np.float64]:
                fixed_theta_dict[key] = val
            else:
                raise ValueError(
                    'Type {} of {} in theta not recognised'.format(
                        type(val), key))
            if key in glitch_keys:
                for i in range(self.nglitch):
                    full_theta_keys_copy.pop(full_theta_keys_copy.index(key))
            else:
                full_theta_keys_copy.pop(full_theta_keys_copy.index(key))

        if len(full_theta_keys_copy) > 0:
            raise ValueError(('Input dictionary `theta` is missing the'
                              'following keys: {}').format(
                                  full_theta_keys_copy))

        self.fixed_theta = [fixed_theta_dict[key] for key in full_theta_keys]
        self.theta_idxs = [full_theta_keys.index(k) for k in self.theta_keys]
        self.theta_symbols = [full_theta_symbols[i] for i in self.theta_idxs]

        idxs = np.argsort(self.theta_idxs)
        self.theta_idxs = [self.theta_idxs[i] for i in idxs]
        self.theta_symbols = [self.theta_symbols[i] for i in idxs]
        self.theta_keys = [self.theta_keys[i] for i in idxs]

        # Correct for number of glitches in the idxs
        self.theta_idxs = np.array(self.theta_idxs)
        while np.sum(self.theta_idxs[:-1] == self.theta_idxs[1:]) > 0:
            for i, idx in enumerate(self.theta_idxs):
                if idx in self.theta_idxs[:i]:
                    self.theta_idxs[i] += 1

    def apply_corrections_to_p0(self, p0):
        p0 = np.array(p0)
        if self.nglitch > 1:
            p0[:, :, -self.nglitch:] = np.sort(p0[:, :, -self.nglitch:],
                                               axis=2)
        return p0


class GridSearch(BaseSearchClass):
    """ Gridded search using ComputeFstat """
    @initializer
    def __init__(self, label, outdir, sftfilepath, F0s=[0],
                 F1s=[0], F2s=[0], Alphas=[0], Deltas=[0], tref=None,
                 tstart=None, tend=None, minCoverFreq=None, maxCoverFreq=None,
                 earth_ephem=None, sun_ephem=None, detector=None, BSGL=False):
        """
        Parameters
        label, outdir: str
            A label and directory to read/write data from/to
        sftfilepath: str
            File patern to match SFTs
        F0s, F1s, F2s, delta_F0s, delta_F1s, tglitchs, Alphas, Deltas: tuple
            Length 3 tuple describing the grid for each parameter, e.g
            [F0min, F0max, dF0], for a fixed value simply give [F0].
        tref, tstart, tend: int
            GPS seconds of the reference time, start time and end time
        minCoverFreq, maxCoverFreq: float
            Minimum and maximum instantaneous frequency which will be covered
            over the SFT time span as passed to CreateFstatInput
        earth_ephem, sun_ephem: str
            Paths of the two files containing positions of Earth and Sun,
            respectively at evenly spaced times, as passed to CreateFstatInput
            If None defaults defined in BaseSearchClass will be used

        """

        self.minStartTime = tstart
        self.maxStartTime = tend

        if earth_ephem is None:
            self.earth_ephem = self.earth_ephem_default
        if sun_ephem is None:
            self.sun_ephem = self.sun_ephem_default

        if os.path.isdir(outdir) is False:
            os.mkdir(outdir)
        self.out_file = '{}/{}_gridFS.txt'.format(self.outdir, self.label)
        self.keys = ['_', '_', 'F0', 'F1', 'F2', 'Alpha', 'Delta']

    def inititate_search_object(self):
        logging.info('Setting up search object')
        self.search = ComputeFstat(
            tref=self.tref, sftfilepath=self.sftfilepath,
            minCoverFreq=self.minCoverFreq, maxCoverFreq=self.maxCoverFreq,
            earth_ephem=self.earth_ephem, sun_ephem=self.sun_ephem,
            detector=self.detector, transient=False,
            minStartTime=self.minStartTime, maxStartTime=self.maxStartTime,
            BSGL=self.BSGL)

    def get_array_from_tuple(self, x):
        if len(x) == 1:
            return np.array(x)
        else:
            return np.arange(x[0], x[1]*(1+1e-15), x[2])

    def get_input_data_array(self):
        arrays = []
        for tup in ([self.tstart], [self.tend], self.F0s, self.F1s, self.F2s,
                    self.Alphas, self.Deltas):
            arrays.append(self.get_array_from_tuple(tup))

        input_data = []
        for vals in itertools.product(*arrays):
            input_data.append(vals)

        self.arrays = arrays
        self.input_data = np.array(input_data)

    def check_old_data_is_okay_to_use(self):
        if os.path.isfile(self.out_file) is False:
            logging.info('No old data found, continuing with grid search')
            return False
        data = np.atleast_2d(np.genfromtxt(self.out_file, delimiter=' '))
        if np.all(data[:, 0:-1] == self.input_data):
            logging.info(
                'Old data found with matching input, no search performed')
            return data
        else:
            logging.info(
                'Old data found, input differs, continuing with grid search')
            return False

    def run(self, return_data=False):
        self.get_input_data_array()
        old_data = self.check_old_data_is_okay_to_use()
        if old_data is not False:
            self.data = old_data
            return

        self.inititate_search_object()

        logging.info('Total number of grid points is {}'.format(
            len(self.input_data)))

        data = []
        for vals in self.input_data:
            FS = self.search.run_computefstatistic_single_point(*vals)
            data.append(list(vals) + [FS])

        data = np.array(data)
        if return_data:
            return data
        else:
            logging.info('Saving data to {}'.format(self.out_file))
            np.savetxt(self.out_file, data, delimiter=' ')
            self.data = data

    def convert_F0_to_mismatch(self, F0, F0hat, Tseg):
        DeltaF0 = F0[1] - F0[0]
        m_spacing = (np.pi*Tseg*DeltaF0)**2 / 12.
        N = len(F0)
        return np.arange(-N*m_spacing/2., N*m_spacing/2., m_spacing)

    def convert_F1_to_mismatch(self, F1, F1hat, Tseg):
        DeltaF1 = F1[1] - F1[0]
        m_spacing = (np.pi*Tseg**2*DeltaF1)**2 / 720.
        N = len(F1)
        return np.arange(-N*m_spacing/2., N*m_spacing/2., m_spacing)

    def add_mismatch_to_ax(self, ax, x, y, xkey, ykey, xhat, yhat, Tseg):
        axX = ax.twiny()
        axX.zorder = -10
        axY = ax.twinx()
        axY.zorder = -10

        if xkey == 'F0':
            m = self.convert_F0_to_mismatch(x, xhat, Tseg)
            axX.set_xlim(m[0], m[-1])

        if ykey == 'F1':
            m = self.convert_F1_to_mismatch(y, yhat, Tseg)
            axY.set_ylim(m[0], m[-1])

    def plot_1D(self, xkey):
        fig, ax = plt.subplots()
        xidx = self.keys.index(xkey)
        x = np.unique(self.data[:, xidx])
        z = self.data[:, -1]
        plt.plot(x, z)
        fig.savefig('{}/{}_1D.png'.format(self.outdir, self.label))

    def plot_2D(self, xkey, ykey, ax=None, save=True, vmin=None, vmax=None,
                add_mismatch=None, xN=None, yN=None, flat_keys=[]):
        """ Plots a 2D grid of 2F values

        Parameters
        ----------
        add_mismatch: tuple (xhat, yhat, Tseg)
            If not None, add a secondary axis with the metric mismatch from the
            point xhat, yhat with duration Tseg
        """
        if ax is None:
            fig, ax = plt.subplots()
        xidx = self.keys.index(xkey)
        yidx = self.keys.index(ykey)
        flat_idxs = [self.keys.index(k) for k in flat_keys]

        x = np.unique(self.data[:, xidx])
        y = np.unique(self.data[:, yidx])
        flat_vals = [np.unique(self.data[:, j]) for j in flat_idxs]
        z = self.data[:, -1]

        Y, X = np.meshgrid(y, x)
        shape = [len(x), len(y)] + [len(v) for v in flat_vals]
        Z = z.reshape(shape)

        while Z.ndim > 2:
            Z = np.mean(Z, axis=-1)

        pax = ax.pcolormesh(X, Y, Z, cmap=plt.cm.viridis, vmin=vmin, vmax=vmax)
        plt.colorbar(pax, ax=ax)

        if add_mismatch:
            self.add_mismatch_to_ax(ax, x, y, xkey, ykey, *add_mismatch)

        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        ax.set_xlabel(xkey)
        ax.set_ylabel(ykey)

        if xN:
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(xN))
        if yN:
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(yN))

        if save:
            fig.tight_layout()
            fig.savefig('{}/{}_2D.png'.format(self.outdir, self.label))
        else:
            return ax

    def get_max_twoF(self):
        twoF = self.data[:, -1]
        return np.max(twoF)


class GridGlitchSearch(GridSearch):
    """ Gridded search using the SemiCoherentGlitchSearch """
    @initializer
    def __init__(self, label, outdir, sftfilepath=None, F0s=[0],
                 F1s=[0], F2s=[0], delta_F0s=[0], delta_F1s=[0], tglitchs=None,
                 Alphas=[0], Deltas=[0], tref=None, tstart=None, tend=None,
                 minCoverFreq=None, maxCoverFreq=None, write_after=1000,
                 earth_ephem=None, sun_ephem=None):
        """
        Parameters
        label, outdir: str
            A label and directory to read/write data from/to
        sftfilepath: str
            File patern to match SFTs
        F0s, F1s, F2s, delta_F0s, delta_F1s, tglitchs, Alphas, Deltas: tuple
            Length 3 tuple describing the grid for each parameter, e.g
            [F0min, F0max, dF0], for a fixed value simply give [F0].
        tref, tstart, tend: int
            GPS seconds of the reference time, start time and end time
        minCoverFreq, maxCoverFreq: float
            Minimum and maximum instantaneous frequency which will be covered
            over the SFT time span as passed to CreateFstatInput
        earth_ephem, sun_ephem: str
            Paths of the two files containing positions of Earth and Sun,
            respectively at evenly spaced times, as passed to CreateFstatInput
            If None defaults defined in BaseSearchClass will be used

        """
        if tglitchs is None:
            self.tglitchs = [self.tend]
        if earth_ephem is None:
            self.earth_ephem = self.earth_ephem_default
        if sun_ephem is None:
            self.sun_ephem = self.sun_ephem_default

        self.search = SemiCoherentGlitchSearch(
            label=label, outdir=outdir, sftfilepath=self.sftfilepath,
            tref=tref, tstart=tstart, tend=tend, minCoverFreq=minCoverFreq,
            maxCoverFreq=maxCoverFreq, earth_ephem=self.earth_ephem,
            sun_ephem=self.sun_ephem)

        if os.path.isdir(outdir) is False:
            os.mkdir(outdir)
        self.out_file = '{}/{}_gridFS.txt'.format(self.outdir, self.label)
        self.keys = ['F0', 'F1', 'F2', 'Alpha', 'Delta', 'delta_F0',
                     'delta_F1', 'tglitch']

    def get_input_data_array(self):
        arrays = []
        for tup in (self.F0s, self.F1s, self.F2s, self.Alphas, self.Deltas,
                    self.delta_F0s, self.delta_F1s, self.tglitchs):
            arrays.append(self.get_array_from_tuple(tup))

        input_data = []
        for vals in itertools.product(*arrays):
            input_data.append(vals)

        self.arrays = arrays
        self.input_data = np.array(input_data)


class Writer(BaseSearchClass):
    """ Instance object for generating SFTs containing glitch signals """
    @initializer
    def __init__(self, label='Test', tstart=700000000, duration=100*86400,
                 dtglitch=None,
                 delta_phi=0, delta_F0=0, delta_F1=0, delta_F2=0,
                 tref=None, phi=0, F0=30, F1=1e-10, F2=0, Alpha=5e-3,
                 Delta=6e-2, h0=0.1, cosi=0.0, psi=0.0, Tsft=1800, outdir=".",
                 sqrtSX=1, Band=4, detector='H1'):
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
            self.tref = self.tstart
        self.tend = self.tstart + self.duration
        tbs = np.array(self.tbounds)
        self.durations_days = (tbs[1:] - tbs[:-1]) / 86400
        self.config_file_name = "{}/{}.cff".format(outdir, label)

        self.theta = np.array([phi, F0, F1, F2])
        self.delta_thetas = np.atleast_2d(
                np.array([delta_phi, delta_F0, delta_F1, delta_F2]).T)

        numSFTs = int(float(self.duration) / self.Tsft)
        self.sftfilename = lalpulsar.OfficialSFTFilename(
            'H', '1', numSFTs, self.Tsft, self.tstart, self.duration,
            self.label)
        self.sftfilepath = '{}/{}'.format(self.outdir, self.sftfilename)
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
        cl.append('--sqrtSX="{}"'.format(self.sqrtSX))
        cl.append('--startTime={:10.9f}'.format(float(self.tstart)))
        cl.append('--duration={}'.format(int(self.duration)))
        cl.append('--fmin={}'.format(int(self.fmin)))
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
