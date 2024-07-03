"""PyFstat search & follow-up classes using MCMC-based methods

The general approach is described in
Ashton & Prix (PRD 97, 103020, 2018):
https://arxiv.org/abs/1802.05450
and we use the `ptemcee` sampler
described in Vousden et al. (MNRAS 455, 1919-1937, 2016):
https://arxiv.org/abs/1501.05823
and based on Foreman-Mackey et al. (PASP 125, 306, 2013):
https://arxiv.org/abs/1202.3665

Defining the prior
##################

The MCMC based searches (i.e. `pyfstat.MCMC*`) require a prior specification for each model parameter,
implemented via a `python dictionary <https://docs.python.org/tutorial/datastructures.html#dictionaries>`_.
This is best explained through a simple example, here is the prior for a *directed* search with a *uniform*
prior on the frequency and a *normal* prior on the frequency derivative:

.. code-block:: python

    theta_prior = {'F0': {'type': 'unif',
                          'lower': 29.9,
                          'upper': 30.1},
                   'F1': {'type': 'norm',
                          'loc': 0,
                          'scale': 1e-10},
                   'F2': 0,
                   'Alpha': 2.3,
                   'Delta': 1.8
                   }

For the sky positions ``Alpha`` and ``Delta``, we give the fixed values (i.e. they are considered *known* by
the MCMC simulation), the same is true for ``F2``, the second derivative of the frequency which we fix at ``0``.
Meanwhile, for the frequency ``F0`` and first frequency derivative ``F1`` we give a dictionary specifying their
prior distribution. This dictionary must contain three arguments: the ``type`` (in this case either ``unif`` or
``norm``) which specifies the type of distribution, then two shape arguments. The shape parameters will depend
on the ``type`` of distribution, but here we use ``lower`` and ``upper``, required for the ``unif`` prior while
``loc`` and ``scale`` are required for the ``norm`` prior.

Currently, two other types of prior are implemented: ``halfnorm``, ``neghalfnorm`` (both of which require ``loc``
and ``scale`` shape parameters). Further priors can be added by modifying ``pyfstat.MCMCSearch._generic_lnprior``.

"""

import copy
import logging
import os
import sys
from collections import OrderedDict

import corner
import dill as pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ptemcee

if ptemcee.__version__ == "1.0.0":
    # This is a very ugly hack to support numpy>=1.24
    ptemcee.sampler.np.float = float
from ptemcee import Sampler as PTSampler
from scipy.stats import lognorm
from tqdm import tqdm

import pyfstat.core as core
import pyfstat.optimal_setup_functions as optimal_setup_functions
import pyfstat.utils as utils
from pyfstat.core import BaseSearchClass

logger = logging.getLogger(__name__)


class MCMCSearch(BaseSearchClass):
    """
    MCMC search using ComputeFstat.

    Evaluates the coherent F-statistic across a parameter space region
    corresponding to an isolated/binary-modulated CW signal.
    """

    symbol_dictionary = dict(
        F0=r"$f$",
        F1=r"$\dot{f}$",
        F2=r"$\ddot{f}$",
        Alpha=r"$\alpha$",
        Delta=r"$\delta$",
        asini=r"asini",
        period=r"P",
        ecc=r"ecc",
        tp=r"tp",
        argp=r"argp",
    )
    """
        Key, val pairs of the parameters (`F0`, `F1`, ...), to LaTeX math
        symbols for plots
    """
    unit_dictionary = dict(
        F0=r"Hz",
        F1=r"Hz/s",
        F2=r"Hz/s$^2$",
        Alpha=r"rad",
        Delta=r"rad",
        asini="",
        period=r"s",
        ecc="",
        tp=r"s",
        argp="",
    )
    """
        Key, val pairs of the parameters (i.e. `F0`, `F1`), and the
        units (i.e. `Hz`)
    """
    transform_dictionary = {}
    """
        Key, val pairs of the parameters (i.e. `F0`, `F1`), where the key is
        itself a dictionary which can item `multiplier`, `subtractor`, or
        `unit` by which to transform by and update the units.
    """

    def __init__(
        self,
        theta_prior,
        tref,
        label,
        outdir="data",
        minStartTime=None,
        maxStartTime=None,
        sftfilepattern=None,
        detectors=None,
        nsteps=[100, 100],
        nwalkers=100,
        ntemps=1,
        log10beta_min=-5,
        theta_initial=None,
        rhohatmax=1000,
        binary=False,
        BSGL=False,
        BtSG=False,
        SSBprec=None,
        RngMedWindow=None,
        minCoverFreq=None,
        maxCoverFreq=None,
        injectSources=None,
        assumeSqrtSX=None,
        transientWindowType=None,
        tCWFstatMapVersion="lal",
        earth_ephem=None,
        sun_ephem=None,
        allowedMismatchFromSFTLength=None,
        clean=False,
    ):
        """
        Parameters
        ----------
        theta_prior: dict
            Dictionary of priors and fixed values for the search parameters.
            For each parameters (key of the dict), if it is to be held fixed
            the value should be the constant float, if it is be searched, the
            value should be a dictionary of the prior.
        tref, minStartTime, maxStartTime: int
            GPS seconds of the reference time, start time and end time. While tref
            is requirede, minStartTime and maxStartTime default to None in which
            case all available data is used.
        label, outdir: str
            A label and output directory (optional, default is `data`) to
            name files
        sftfilepattern: str, optional
            Pattern to match SFTs using wildcards (`*?`) and ranges [0-9];
            mutiple patterns can be given separated by colons.
        detectors: str, optional
            Two character reference to the detectors to use, specify None for no
            contraint and comma separated strings for multiple references.
        nsteps: list (2,), optional
            Number of burn-in and production steps to take, [nburn, nprod]. See
            `pyfstat.MCMCSearch.setup_initialisation()` for details on adding
            initialisation steps.
        nwalkers, ntemps: int, optional
            The number of walkers and temperates to use in the parallel
            tempered PTSampler.
        log10beta_min: float < 0, optional
            The log_10(beta) value. If given, the set of betas passed to PTSampler
            are generated from `np.logspace(0, log10beta_min, ntemps)` (given
            in descending order to ptemcee).
        theta_initial: dict, array, optional
            A dictionary of distribution about which to distribute the
            initial walkers about.
        rhohatmax: float, optional
            Upper bound for the SNR scale parameter (required to normalise the
            Bayes factor) - this needs to be carefully set when using the
            evidence.
        binary: bool, optional
            If true, search over binary orbital parameters.
        BSGL: bool, optional
            If true, use the BSGL statistic.
        BtSG: bool, optional
            If true, use the transient lnBtSG statistic.
            (Only for transient searches.)
        SSBPrec: int, optional
            SSBPrec (SSB precision) to use when calling ComputeFstat. See `core.ComputeFstat`.
        RngMedWindow: int, optional
            Running-Median window size (number of bins) for ComputeFstat. See `core.ComputeFstat`.
        minCoverFreq, maxCoverFreq: float, optional
            Minimum and maximum instantaneous frequency which will be covered
            over the SFT time span as passed to CreateFstatInput. See `core.ComputeFstat`.
        injectSources: dict, optional
            If given, inject these properties into the SFT files before running
            the search. See `core.ComputeFstat`.
        assumeSqrtSX: float or list or str
            Don't estimate noise-floors, but assume (stationary) per-IFO sqrt{SX}.
            See `core.ComputeFstat`.
        transientWindowType: str
            If 'rect' or 'exp',
            compute atoms so that a transient (t0,tau) map can later be computed.
            ('none' instead of None explicitly calls the transient-window function,
            but with the full range, for debugging). See `core.ComputeFstat`.
            Currently only supported for nsegs=1.
        tCWFstatMapVersion: str
            Choose between standard 'lal' implementation,
            'pycuda' for gpu, and some others for devel/debug.
        allowedMismatchFromSFTLength: float
            Maximum allowed mismatch from SFTs being too long
            [Default: what's hardcoded in XLALFstatMaximumSFTLength].
        clean: bool
            If true, ignore existing data and overwrite.
            Otherwise, re-use existing data if no inconsistencies are found.
        """
        self._set_init_params_dict(locals())
        self.theta_prior = theta_prior
        self.tref = tref
        self.label = label
        self.outdir = outdir
        self.minStartTime = minStartTime
        self.maxStartTime = maxStartTime
        self.sftfilepattern = sftfilepattern
        self.detectors = detectors
        self.nsteps = nsteps
        self.nwalkers = nwalkers
        self.ntemps = ntemps
        self.log10beta_min = log10beta_min
        self.theta_initial = theta_initial
        self.rhohatmax = rhohatmax
        self.binary = binary
        self.BSGL = BSGL
        self.BtSG = BtSG
        self.SSBprec = SSBprec
        self.RngMedWindow = RngMedWindow
        self.minCoverFreq = minCoverFreq
        self.maxCoverFreq = maxCoverFreq
        self.injectSources = injectSources
        self.assumeSqrtSX = assumeSqrtSX
        self.transientWindowType = transientWindowType
        self.tCWFstatMapVersion = tCWFstatMapVersion
        self.set_ephemeris_files(earth_ephem, sun_ephem)
        self.allowedMismatchFromSFTLength = allowedMismatchFromSFTLength
        self.clean = clean

        os.makedirs(outdir, exist_ok=True)
        self.output_file_header = self.get_output_file_header()
        logger.info("Set-up MCMC search for model {}".format(self.label))
        if sftfilepattern:
            logger.info("Using data {}".format(self.sftfilepattern))
        else:
            logger.info("No sftfilepattern given")
        if injectSources:
            logger.info("Inject sources: {}".format(injectSources))
        self.pickle_path = os.path.join(self.outdir, self.label + "_saved_data.p")
        self._unpack_input_theta()
        self.ndim = len(self.theta_keys)
        if self.log10beta_min:
            self.betas = np.logspace(0, self.log10beta_min, self.ntemps)
        else:
            self.betas = None

        if self.clean and os.path.isfile(self.pickle_path):
            os.rename(self.pickle_path, self.pickle_path + ".old")

        self._set_likelihoodcoef()
        self._log_input()

    def _set_likelihoodcoef(self):
        """Additional constant terms to turn a detection statistic into a likelihood.

        In general, the (log-)likelihood can be obtained from the signal-to-noise
        (log-)Bayes factor
        (omitting the overall Gaussian-noise normalization term)
        but the detection statistic may only be a monotonic function of the
        Bayes factor, not the full thing.
        E.g. this is the case for the standard CW F-statistic!
        """
        if self.BSGL:
            # In this case, the corresponding term is already included
            # in the detection statistic itself.
            # See Eq. (36) in Keitel et al (PRD 89, 064023, 2014):
            # https://arxiv.org/abs/1311.5738
            # where Fstar0 = ln(cstar) = ln(rhohatmax**4/70).
            # We just need to switch to natural log basis.
            self.likelihooddetstatmultiplier = np.log(10)
            self.likelihoodcoef = 0
        else:
            # If assuming only Gaussian noise + signal,
            # the likelihood is essentially the F-statistic,
            # but with an extra constant term depending on the amplitude prior.
            # See Eq. (9) of Ashton & Prix (PRD 97, 103020, 2018):
            # https://arxiv.org/abs/1802.05450
            # Also need to go from twoF to F.
            self.likelihooddetstatmultiplier = 0.5
            self.likelihoodcoef = np.log(70.0 / self.rhohatmax**4)

    def _log_input(self):
        logger.info("theta_prior = {}".format(self.theta_prior))
        logger.info("nwalkers={}".format(self.nwalkers))
        logger.info("nsteps = {}".format(self.nsteps))
        logger.info("ntemps = {}".format(self.ntemps))
        logger.info("log10beta_min = {}".format(self.log10beta_min))

    def _get_search_ranges(self):
        """take prior widths as proxy "search ranges" to allow covering band estimate"""
        if (self.minCoverFreq is None) or (self.maxCoverFreq is None):
            normal_stds = 3  # this might not always be enough
            prior_bounds, norm_trunc_warn = self._get_prior_bounds(normal_stds)
            if norm_trunc_warn:
                logger.warning(
                    "Gaussian priors (normal / half-normal) have been truncated"
                    " at {:f} standard deviations for estimating the coverage"
                    " frequency band. If sampling fails at any point, please"
                    " consider manually setting [minCoverFreq,maxCoverFreq] to"
                    " more generous values.".format(normal_stds)
                )
            # first start with parameters that have non-delta prior ranges
            search_ranges = {
                key: [bound["lower"], bound["upper"]]
                for key, bound in prior_bounds.items()
            }
            # then add fixed-point (delta prior) parameters
            for key in self.theta_prior:
                if key not in self.theta_keys:
                    search_ranges[key] = [self.theta_prior[key]]
            return search_ranges
        else:
            return None

    def _initiate_search_object(self):
        logger.info("Setting up search object")
        search_ranges = self._get_search_ranges()
        self.search = core.ComputeFstat(
            tref=self.tref,
            sftfilepattern=self.sftfilepattern,
            minCoverFreq=self.minCoverFreq,
            maxCoverFreq=self.maxCoverFreq,
            search_ranges=search_ranges,
            detectors=self.detectors,
            BSGL=self.BSGL,
            transientWindowType=self.transientWindowType,
            minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime,
            binary=self.binary,
            injectSources=self.injectSources,
            assumeSqrtSX=self.assumeSqrtSX,
            SSBprec=self.SSBprec,
            RngMedWindow=self.RngMedWindow,
            tCWFstatMapVersion=self.tCWFstatMapVersion,
            earth_ephem=self.earth_ephem,
            sun_ephem=self.sun_ephem,
            allowedMismatchFromSFTLength=self.allowedMismatchFromSFTLength,
        )
        if self.minStartTime is None:
            self.minStartTime = self.search.minStartTime
        if self.maxStartTime is None:
            self.maxStartTime = self.search.maxStartTime

    def _logp(self, theta_vals, theta_prior, theta_keys, search):
        H = [
            self._generic_lnprior(**theta_prior[key])(p)
            for p, key in zip(theta_vals, theta_keys)
        ]
        return np.sum(H)

    def _set_point_for_evaluation(self, theta):
        """Combines fixed and variable parameters to form a valid evaluation point.

        Parameters
        ----------
        theta: list or np.ndarray
            The sampled (variable) parameters.
        Returns
        -------
        p: list
            The full parameter space point as a list.
        """
        p = copy.copy(self.fixed_theta)
        for j, theta_i in enumerate(self.theta_idxs):
            p[theta_i] = theta[j]
        return p

    def _logl(self, theta, search):
        in_theta = self._set_point_for_evaluation(theta)
        detstat = search.get_det_stat(*in_theta)
        return detstat * self.likelihooddetstatmultiplier + self.likelihoodcoef

    def _unpack_input_theta(self):
        self.full_theta_keys = ["F0", "F1", "F2", "Alpha", "Delta"]
        if self.binary:
            self.full_theta_keys += ["asini", "period", "ecc", "tp", "argp"]
        full_theta_keys_copy = copy.copy(self.full_theta_keys)

        self.theta_keys = []
        fixed_theta_dict = {}
        for key, val in self.theta_prior.items():
            if type(val) is dict:
                fixed_theta_dict[key] = 0
                self.theta_keys.append(key)
            elif type(val) in [float, int, np.float64]:
                fixed_theta_dict[key] = val
            else:
                raise ValueError(
                    "Type {} of {} in theta not recognised".format(type(val), key)
                )
            full_theta_keys_copy.pop(full_theta_keys_copy.index(key))

        if len(full_theta_keys_copy) > 0:
            raise ValueError(
                ("Input dictionary `theta` is missing the" "following keys: {}").format(
                    full_theta_keys_copy
                )
            )

        self.fixed_theta = [fixed_theta_dict[key] for key in self.full_theta_keys]
        self.theta_idxs = [self.full_theta_keys.index(k) for k in self.theta_keys]
        self.theta_symbols = [self.symbol_dictionary[k] for k in self.theta_keys]

        idxs = np.argsort(self.theta_idxs)
        self.theta_idxs = [self.theta_idxs[i] for i in idxs]
        self.theta_symbols = [self.theta_symbols[i] for i in idxs]
        self.theta_keys = [self.theta_keys[i] for i in idxs]

        self.output_keys = self.theta_keys.copy()
        self.output_keys.append("twoF")
        if self.BSGL:
            self.output_keys.append("log10BSGL")

    def _evaluate_logpost(self, p0vec):
        init_logp = np.array(
            [
                self._logp(p, self.theta_prior, self.theta_keys, self.search)
                for p in p0vec
            ]
        )
        init_logl = np.array([self._logl(p, self.search) for p in p0vec])
        return init_logl + init_logp

    def _check_initial_points(self, p0):
        for nt in range(self.ntemps):
            logger.info("Checking temperature {} chains".format(nt))
            num = sum(self._evaluate_logpost(p0[nt]) == -np.inf)
            if num > 0:
                logger.warning(
                    "Of {} initial values, {} are -np.inf due to the prior".format(
                        len(p0[0]), num
                    )
                )
                p0 = self._generate_new_p0_to_fix_initial_points(p0, nt)

    def _generate_new_p0_to_fix_initial_points(self, p0, nt):
        logger.info("Attempting to correct intial values")
        init_logpost = self._evaluate_logpost(p0[nt])
        idxs = np.arange(self.nwalkers)[init_logpost == -np.inf]
        count = 0
        while sum(init_logpost == -np.inf) > 0 and count < 100:
            for j in idxs:
                p0[nt][j] = p0[nt][np.random.randint(0, self.nwalkers)] * (
                    1 + np.random.normal(0, 1e-10, self.ndim)
                )
            init_logpost = self._evaluate_logpost(p0[nt])
            count += 1

        if sum(init_logpost == -np.inf) > 0:
            logger.info("Failed to fix initial priors")
        else:
            logger.info("Suceeded to fix initial priors")

        return p0

    def setup_initialisation(self, nburn0, scatter_val=1e-10):
        """Add an initialisation step to the MCMC run

        If called prior to `run()`, adds an intial step in which the MCMC
        simulation is run for `nburn0` steps. After this, the MCMC simulation
        continues in the usual manner (i.e. for nburn and nprod steps), but the
        walkers are reset scattered around the maximum likelihood position
        of the initialisation step.

        Parameters
        ----------
        nburn0: int
            Number of initialisation steps to take.
        scatter_val: float
            Relative number to scatter walkers around the maximum likelihood
            position after the initialisation step. If the maximum likelihood
            point is located at `p`, the new walkers are randomly drawn from a
            multivariate gaussian distribution centered at `p` with standard
            deviation `diag(scatter_val * p)`.
        """

        logger.info(
            "Setting up initialisation with nburn0={}, scatter_val={}".format(
                nburn0, scatter_val
            )
        )
        self.nsteps = [nburn0] + self.nsteps
        self.scatter_val = scatter_val

    def _run_sampler(self, p0, nprod=0, nburn=0, window=50):
        for result in tqdm(
            self.sampler.sample(p0, iterations=nburn + nprod), total=nburn + nprod
        ):
            pass

        self.mean_acceptance_fraction = np.mean(
            self.sampler.acceptance_fraction, axis=1
        )
        logger.info(
            "Mean acceptance fraction: {}".format(self.mean_acceptance_fraction)
        )
        if self.ntemps > 1:
            self.tswap_acceptance_fraction = self.sampler.tswap_acceptance_fraction
            logger.info(
                "Tswap acceptance fraction: {}".format(
                    self.sampler.tswap_acceptance_fraction
                )
            )
        self.autocorr_time = self._get_autocorr_time(
            sampler=self.sampler, window=window
        )
        logger.info("Autocorrelation length: {}".format(self.autocorr_time))

    def _get_autocorr_time(self, sampler, window=50):
        """
        Returns a matrix of autocorrelation lengths for each
        parameter in each temperature of shape ``(Ntemps, Ndim)``.

        This is copied from sampler.py of ptemcee-1.0.0
        [(c) Daniel Foreman-Mackey & contributors],
        to allow us to call the locally fixed _autocorr_function().

        :param window: (optional)
            The size of the windowing function. This is equivalent to the
            maximum number of lags to use. (default: 50)

        """
        acors = np.zeros((sampler.ntemps, sampler.dim))

        for i in range(sampler.ntemps):
            x = np.mean(sampler._chain[i, :, :, :], axis=0)
            acors[i, :] = self._autocorr_integrated_time(x=x, window=window)
        return acors

    def _autocorr_integrated_time(self, x, axis=0, window=50, fast=False):
        """
        Estimate the integrated autocorrelation time of a time series.

        See `Sokal's notes <http://www.stat.unc.edu/faculty/cji/Sokal.pdf>`_ on
        MCMC and sample estimators for autocorrelation times.

        This version of this function is copied from util.py of ptemcee-1.0.0
        [(c) Daniel Foreman-Mackey & contributors],
        and fixed up to be compatible with numpy>=1.23.0.

        :param x:
            The time series. If multidimensional, set the time axis using the
            ``axis`` keyword argument and the function will be computed for every
            other axis.

        :param axis: (optional)
            The time axis of ``x``. Assumed to be the first axis if not specified.

        :param window: (optional)
            The size of the window to use. (default: 50)

        :param fast: (optional)
            If ``True``, only use the largest ``2^n`` entries for efficiency.
            (default: False)

        """
        # Compute the autocorrelation function.
        f = self._autocorr_function(x, axis=axis, fast=fast)

        # Special case 1D for simplicity.
        if len(f.shape) == 1:
            return 1 + 2 * np.sum(f[1:window])

        # N-dimensional case.
        m = [
            slice(None),
        ] * len(f.shape)
        m[axis] = slice(1, window)
        m = tuple(m)  # fix for numpy>=1.23.0
        tau = 1 + 2 * np.sum(f[m], axis=axis)

        return tau

    def _autocorr_function(self, x, axis=0, fast=False):
        """
        Estimate the autocorrelation function of a time series using the FFT.

        This version of this function is copied from util.py of ptemcee-1.0.0
        [(c) Daniel Foreman-Mackey & contributors],
        and fixed up to be compatible with numpy>=1.23.0.

        :param x:
            The time series. If multidimensional, set the time axis using the
            ``axis`` keyword argument and the function will be computed for every
            other axis.

        :param axis: (optional)
            The time axis of ``x``. Assumed to be the first axis if not specified.

        :param fast: (optional)
            If ``True``, only use the largest ``2^n`` entries for efficiency.
            (default: False)

        """
        x = np.atleast_1d(x)
        m = [
            slice(None),
        ] * len(x.shape)

        # For computational efficiency, crop the chain to the largest power of
        # two if requested.
        if fast:
            n = int(2 ** np.floor(np.log2(x.shape[axis])))
            m[axis] = slice(0, n)
            x = x
        else:
            n = x.shape[axis]

        # Compute the FFT and then (from that) the auto-correlation function.
        f = np.fft.fft(x - np.mean(x, axis=axis), n=2 * n, axis=axis)
        m[axis] = slice(0, n)
        m_tuple = tuple(m)  # fix for numpy>=1.23.0
        acf = np.fft.ifft(f * np.conjugate(f), axis=axis)[m_tuple].real
        m[axis] = 0
        m_tuple = tuple(m)  # fix for numpy>=1.23.0
        return acf / acf[m_tuple]

    def _estimate_run_time(self):
        """Print the estimated run time

        Uses timing coefficients based on a Lenovo T460p Intel(R)
        Core(TM) i5-6300HQ CPU @ 2.30GHz.

        """
        # Todo: add option to time on a machine, and move coefficients to
        # ~/.pyfstat.conf
        if isinstance(self.theta_prior["Alpha"], dict) or isinstance(
            self.theta_prior["Delta"], dict
        ):
            tau0LD = 5.2e-7
            tau0T = 1.5e-8
            tau0S = 1.2e-4
            tau0C = 5.8e-6
        else:
            tau0LD = 1.3e-7
            tau0T = 1.5e-8
            tau0S = 9.1e-5
            tau0C = 5.5e-6
        Nsfts = (self.maxStartTime - self.minStartTime) / 1800.0
        if hasattr(self, "run_setup"):
            ts = []
            for row in self.run_setup:
                nsteps = row[0]
                nsegs = row[1]
                numb_evals = np.sum(nsteps) * self.nwalkers * self.ntemps
                t = (tau0S + tau0LD * Nsfts) * numb_evals
                if nsegs > 1:
                    t += (tau0C + tau0T * Nsfts) * nsegs * numb_evals
                ts.append(t)
            time = np.sum(ts)
        else:
            numb_evals = np.sum(self.nsteps) * self.nwalkers * self.ntemps
            time = (tau0S + tau0LD * Nsfts) * numb_evals
            if getattr(self, "nsegs", 1) > 1:
                time += (tau0C + tau0T * Nsfts) * self.nsegs * numb_evals

        logger.info(
            "Estimated run-time = {} s = {:1.0f}:{:1.0f} m".format(
                time, *divmod(time, 60)
            )
        )

    def run(
        self,
        proposal_scale_factor=2,
        save_pickle=True,
        export_samples=True,
        save_loudest=True,
        plot_walkers=True,
        walker_plot_args=None,
        window=50,
    ):
        """Run the MCMC simulatation

        Parameters
        ----------
        proposal_scale_factor: float
            The proposal scale factor `a > 1` used by the sampler.
            See Goodman & Weare (Comm App Math Comp Sci, Vol 5, No. 1, 2010): 10.2140/camcos.2010.5.65.
            The bigger the value, the wider the range to draw proposals from.
            If the acceptance fraction is too low, you can raise it by
            decreasing the `a` parameter; and if it is too high, you can reduce
            it by increasing the `a` parameter.
            See Foreman-Mackay et al. (PASP 125 306, 2013): https://arxiv.org/abs/1202.3665.
        save_pickle: bool
            If true, save a pickle file of the full sampler state.
        export_samples: bool
            If true, save ASCII samples file to disk. See `MCMCSearch.export_samples_to_disk`.
        save_loudest: bool
            If true, save a CFSv2 .loudest file to disk. See `MCMCSearch.generate_loudest`.
        plot_walkers: bool
            If true, save trace plots of the walkers.
        walker_plot_args:
            Dictionary passed as kwargs to _plot_walkers to control the plotting.
            Histogram of sampled detection statistic values can be retrieved setting "plot_det_stat" to `True`.
            Parameters corresponding to an injected signal can be passed through "injection_parameters"
            as a dictionary containing the parameters of said signal. All parameters being searched for must
            be present, otherwise this option is ignored.
            If both "fig" and "axes" entries are set, the plot is not saved to disk
            directly, but (fig, axes) are returned.
        window: int
            The minimum number of autocorrelation times needed to trust the
            result when estimating the autocorrelation time (see
            ptemcee.Sampler.get_autocorr_time for further details.

        """

        self._initiate_search_object()

        self.old_data_is_okay_to_use = self._check_old_data_is_okay_to_use()
        if self.old_data_is_okay_to_use is True:
            logger.warning("Using saved data from {}".format(self.pickle_path))
            d = self.get_saved_data_dictionary()
            self.samples = d["samples"]
            self.lnprobs = d["lnprobs"]
            self.lnlikes = d["lnlikes"]
            self.all_lnlikelihood = d["all_lnlikelihood"]
            self.chain = d["chain"]
            return

        self._estimate_run_time()

        walker_plot_args = walker_plot_args or {}

        self.sampler = PTSampler(
            ntemps=self.ntemps,
            nwalkers=self.nwalkers,
            dim=self.ndim,
            logl=self._logl,
            logp=self._logp,
            logpargs=(self.theta_prior, self.theta_keys, self.search),
            loglargs=(self.search,),
            betas=self.betas,
            a=proposal_scale_factor,
        )

        p0 = self._generate_initial_p0()
        p0 = self._apply_corrections_to_p0(p0)
        self._check_initial_points(p0)

        # Run initialisation steps if required
        ninit_steps = len(self.nsteps) - 2
        for j, n in enumerate(self.nsteps[:-2]):
            logger.info(
                "Running {}/{} initialisation with {} steps".format(j, ninit_steps, n)
            )
            self._run_sampler(p0, nburn=n, window=window)
            if plot_walkers:
                # For now, this plot will always be saved to disk,
                # never returned as fig/axes.
                try:
                    walker_fig, walker_axes = self._plot_walkers(**walker_plot_args)
                    walker_fig.tight_layout()
                    walker_fig.savefig(
                        os.path.join(
                            self.outdir, "{}_init_{}_walkers.png".format(self.label, j)
                        )
                    )
                    plt.close(walker_fig)
                except Exception as e:
                    logger.warning(
                        "Failed to plot initialisation walkers due to Error {}".format(
                            e
                        )
                    )

            p0 = self._get_new_p0()
            p0 = self._apply_corrections_to_p0(p0)
            self._check_initial_points(p0)
            self.sampler.reset()

        if len(self.nsteps) > 1:
            nburn = self.nsteps[-2]
        else:
            nburn = 0
        nprod = self.nsteps[-1]
        logger.info("Running final burn and prod with {} steps".format(nburn + nprod))
        self._run_sampler(p0, nburn=nburn, nprod=nprod)

        samples = self.sampler.chain[0, :, nburn:, :].reshape((-1, self.ndim))
        lnprobs = self.sampler.logprobability[0, :, nburn:].reshape((-1))
        lnlikes = self.sampler.loglikelihood[0, :, nburn:].reshape((-1))
        all_lnlikelihood = self.sampler.loglikelihood[:, :, nburn:]
        self.samples = samples
        self.chain = self.sampler.chain
        self.lnprobs = lnprobs
        self.lnlikes = lnlikes
        self.all_lnlikelihood = all_lnlikelihood
        if save_pickle:
            self._pickle_data(samples, lnprobs, lnlikes, all_lnlikelihood)
        if export_samples:
            self.export_samples_to_disk()
        if save_loudest:
            self.generate_loudest()

        if plot_walkers:
            try:
                walker_fig, walker_axes = self._plot_walkers(
                    nprod=nprod, **walker_plot_args
                )
                walker_fig.tight_layout()
            except Exception as e:
                logger.warning("Failed to plot walkers due to Error {}".format(e))
                return
            if (walker_plot_args.get("fig") is not None) and (
                walker_plot_args.get("axes") is not None
            ):
                self.walker_fig = walker_fig
                self.walker_axes = walker_axes
            else:
                try:
                    walker_fig.savefig(
                        os.path.join(self.outdir, self.label + "_walkers.png")
                    )
                    plt.close(walker_fig)
                except Exception as e:
                    logger.warning(
                        "Failed to save walker plots due to Error {}".format(e)
                    )

    def _get_rescale_multiplier_for_key(self, key):
        """Get the rescale multiplier from the transform_dictionary

        Can either be a float, a string (in which case it is interpretted as
        a attribute of the MCMCSearch class, e.g. minStartTime, or non-existent
        in which case 0 is returned
        """
        if key not in self.transform_dictionary:
            return 1

        if "multiplier" in self.transform_dictionary[key]:
            val = self.transform_dictionary[key]["multiplier"]
            if isinstance(val, str):
                if hasattr(self, val):
                    multiplier = getattr(
                        self, self.transform_dictionary[key]["multiplier"]
                    )
                else:
                    raise ValueError("multiplier {} not a class attribute".format(val))
            else:
                multiplier = val
        else:
            multiplier = 1
        return multiplier

    def _get_rescale_subtractor_for_key(self, key):
        """Get the rescale subtractor from the transform_dictionary

        Can either be a float, a string (in which case it is interpretted as
        a attribute of the MCMCSearch class, e.g. minStartTime, or non-existent
        in which case 0 is returned
        """
        if key not in self.transform_dictionary:
            return 0

        if "subtractor" in self.transform_dictionary[key]:
            val = self.transform_dictionary[key]["subtractor"]
            if isinstance(val, str):
                if hasattr(self, val):
                    subtractor = getattr(
                        self, self.transform_dictionary[key]["subtractor"]
                    )
                else:
                    raise ValueError("subtractor {} not a class attribute".format(val))
            else:
                subtractor = val
        else:
            subtractor = 0
        return subtractor

    def _scale_samples(self, samples, theta_keys):
        """Scale the samples using the transform_dictionary"""
        for key in theta_keys:
            if key in self.transform_dictionary:
                idx = theta_keys.index(key)
                s = samples[:, idx]
                subtractor = self._get_rescale_subtractor_for_key(key)
                s = s - subtractor
                multiplier = self._get_rescale_multiplier_for_key(key)
                s *= multiplier
                samples[:, idx] = s

        return samples

    def _get_labels(self, newline_units=False):
        """Combine the units, symbols and rescaling to give labels"""

        labels = []
        for key in self.theta_keys:
            values = self.transform_dictionary.get(key, {})
            s, label, u = [
                values.get(slu_key, None) for slu_key in ["symbol", "label", "unit"]
            ]

            if label is None:
                s = s or self.symbol_dictionary[key].replace(
                    "_{glitch}", r"_\mathrm{glitch}"
                )
                u = u or self.unit_dictionary[key]
                label = (
                    f"{s}"
                    + ("\n" if newline_units else " ")
                    + (f"[{u}]" if u != "" else "")
                )

            labels.append(label)

        return labels

    def plot_corner(
        self,
        figsize=(10, 10),
        add_prior=False,
        nstds=None,
        label_offset=0.4,
        dpi=300,
        rc_context={},
        tglitch_ratio=False,
        fig_and_axes=None,
        save_fig=True,
        **kwargs,
    ):
        """Generate a corner plot of the posterior

        Using the `corner` package (https://pypi.python.org/pypi/corner/),
        generate estimates of the posterior from the production samples.

        Parameters
        ----------
        figsize: tuple (7, 7)
            Figure size in inches (passed to plt.subplots)
        add_prior: bool, str
            If true, plot the prior as a red line. If 'full' then for uniform
            priors plot the full extent of the prior.
        nstds: float
            The number of standard deviations to plot centered on the median.
            Standard deviation is computed from the samples using `numpy.std`.
        label_offset: float
            Offset the labels from the plot: useful to prevent overlapping the
            tick labels with the axis labels. This option is passed to `ax.[x|y]axis.set_label_coords`.
        dpi: int
            Passed to plt.savefig.
        rc_context: dict
            Dictionary of rc values to set while generating the figure (see
            matplotlib rc for more details).
        tglitch_ratio: bool
            If true, and tglitch is a parameter, plot posteriors as the
            fractional time at which the glitch occurs instead of the actual
            time.
        fig_and_axes: tuple
            (fig, axes) tuple to plot on. The axes must be of the right shape,
            namely (ndim, ndim)
        save_fig: bool
            If true, save the figure, else return the fig, axes.
        **kwargs:
            Passed to corner.corner. Use "truths" to plot the true parameters of a signal.

        Returns
        -------
        fig, axes:
            The matplotlib figure and axes, only returned if save_fig = False.

        """

        if "truths" in kwargs:
            if not isinstance(kwargs["truths"], dict):
                raise ValueError("'truths' must be a dictionary.")

            missing_keys = set(self.theta_keys) - kwargs["truths"].keys()
            if missing_keys:
                logger.warning(
                    f"plot_corner(): Missing keys {missing_keys} in 'truths' dictionary,"
                    " argument will be ignored."
                )
                kwargs["truths"] = None
            else:
                kwargs["truths"] = [kwargs["truths"][key] for key in self.theta_keys]
                kwargs["truths"] = self._scale_samples(
                    np.reshape(kwargs["truths"], (1, -1)), self.theta_keys
                ).ravel()

                if "truth_color" not in kwargs:
                    kwargs["truth_color"] = "black"

        if self.ndim < 2:
            with plt.rc_context(rc_context):
                if fig_and_axes is None:
                    fig, ax = plt.subplots(figsize=figsize)
                else:
                    fig, ax = fig_and_axes
                ax.hist(self.samples, bins=50, histtype="stepfilled")
                ax.set_xlabel(self.theta_symbols[0])

            fig.savefig(os.path.join(self.outdir, self.label + "_corner.png"), dpi=dpi)
            plt.close(fig)
            return

        with plt.rc_context(rc_context):
            if fig_and_axes is None:
                fig, axes = plt.subplots(self.ndim, self.ndim, figsize=figsize)
            else:
                fig, axes = fig_and_axes

            samples_plt = copy.copy(self.samples)
            labels = self._get_labels(newline_units=False)

            samples_plt = self._scale_samples(samples_plt, self.theta_keys)

            if tglitch_ratio:
                for j, k in enumerate(self.theta_keys):
                    if k == "tglitch":
                        s = samples_plt[:, j]
                        samples_plt[:, j] = (s - self.minStartTime) / (
                            self.maxStartTime - self.minStartTime
                        )
                        labels[j] = r"$R_{\mathrm{glitch}}$"

            if type(nstds) is int and "range" not in kwargs:
                _range = []
                for j, s in enumerate(samples_plt.T):
                    median = np.median(s)
                    std = np.std(s)
                    _range.append((median - nstds * std, median + nstds * std))
            elif "range" in kwargs:
                _range = kwargs.pop("range")
            else:
                _range = None

            hist_kwargs = kwargs.pop("hist_kwargs", dict())
            if "density" not in hist_kwargs:
                hist_kwargs["density"] = True

            fig_triangle = corner.corner(
                samples_plt,
                labels=labels,
                fig=fig,
                bins=50,
                max_n_ticks=4,
                plot_contours=True,
                plot_datapoints=True,
                label_kwargs={"fontsize": 12},
                data_kwargs={"alpha": 0.1, "ms": 0.5},
                range=_range,
                hist_kwargs=hist_kwargs,
                show_titles=True,
                fill_contours=True,
                quantiles=(
                    [0.05, 0.5, 0.95]  # [lower, central, upper]
                    if "quantiles" not in kwargs
                    else kwargs.pop("quantiles")
                ),
                verbose=True if "verbose" not in kwargs else kwargs.pop("verbose"),
                **kwargs,
            )

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

                for tick in ax.xaxis.get_major_ticks():
                    # tick.label1.set_fontsize(8)
                    tick.label1.set_rotation(30)
                for tick in ax.yaxis.get_major_ticks():
                    # tick.label1.set_fontsize(8)
                    tick.label1.set_rotation(30)

            plt.tight_layout()
            fig.subplots_adjust(hspace=0.1, wspace=0.1)

            if add_prior:
                self._add_prior_to_corner(axes, self.samples, add_prior)

            if save_fig:
                fig_triangle.savefig(
                    os.path.join(self.outdir, self.label + "_corner.png"), dpi=dpi
                )
                plt.close(fig_triangle)
            else:
                return fig, axes

    def plot_chainconsumer(self, save_fig=True, label_offset=0.25, dpi=300, **kwargs):
        """Generate a corner plot of the posterior using the `chaniconsumer` package.

        `chainconsumer` is an optional dependency of PyFstat. See https://samreay.github.io/ChainConsumer/.

        Parameters are akin to the ones described in MCMCSearch.plot_corner.
        Only the differing parameters are explicitly described.

        Parameters
        ----------
        **kwargs:
            Passed to chainconsumer.plotter.plot. Use "truths" to plot the true parameters of a signal.
        """
        try:
            import chainconsumer
        except ImportError:
            logger.warning(
                "Could not import 'chainconsumer' package, please install it to use this method."
            )
            return

        samples_plt = copy.copy(self.samples)
        labels = self._get_labels(newline_units=True)

        samples_plt = self._scale_samples(samples_plt, self.theta_keys)
        if "truth" in kwargs:
            if not isinstance(kwargs["truth"], dict):
                raise ValueError("'truth' must be a dictionary.")
            missing_keys = np.setdiff1d(self.theta_keys, list(kwargs["truth"].keys()))
            if len(missing_keys) > 0:
                logger.warning(
                    "plot_chainconsumer(): Missing keys {} in 'truth' dictionary,"
                    " argument will be ignored.".format(missing_keys)
                )
                kwargs["truth"] = None
            else:
                parameters_in_order = np.array(
                    [kwargs["truth"][key] for key in self.theta_keys]
                ).reshape((1, -1))
                kwargs["truth"] = self._scale_samples(
                    parameters_in_order, self.theta_keys
                ).ravel()

        c = chainconsumer.ChainConsumer()
        c.add_chain(samples_plt, parameters=labels)
        # We set usetex=False to avoid dependency on 'kpsewhich' TeX tool
        c.configure(smooth=0, summary=False, sigma2d=True, usetex=False)
        fig = c.plotter.plot(**kwargs)

        axes_list = fig.get_axes()
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

        if save_fig:
            fig.savefig(
                os.path.join(self.outdir, self.label + "_chainconsumer_corner.png"),
                dpi=dpi,
            )
            plt.close(fig)
        else:
            return fig, axes

    def _add_prior_to_corner(self, axes, samples, add_prior):
        for i, key in enumerate(self.theta_keys):
            ax = axes[i][i]
            s = samples[:, i]
            lnprior = self._generic_lnprior(**self.theta_prior[key])
            if add_prior == "full" and self.theta_prior[key]["type"] == "unif":
                lower = self.theta_prior[key]["lower"]
                upper = self.theta_prior[key]["upper"]
                r = upper - lower
                xlim = [lower - 0.05 * r, upper + 0.05 * r]
                x = np.linspace(xlim[0], xlim[1], 1000)
            else:
                xlim = ax.get_xlim()
                x = np.linspace(s.min(), s.max(), 1000)
            multiplier = self._get_rescale_multiplier_for_key(key)
            subtractor = self._get_rescale_subtractor_for_key(key)
            ax.plot(
                (x - subtractor) * multiplier,
                [np.exp(lnprior(xi)) for xi in x],
                "-C3",
                label="prior",
            )

            for j in range(i, self.ndim):
                axes[j][i].set_xlim(xlim[0], xlim[1])
            for k in range(0, i):
                axes[i][k].set_ylim(xlim[0], xlim[1])

    def _get_prior_bounds(self, normal_stds=2):
        """Get the lower/upper bounds of all priors

        Parameters
        ----------
        normal_stds: float
            Number of standard deviations to cut normal (Gaussian) or half-norm
            distributions at.

        Returns
        -------
        prior_bounds: dict
            Dictionary of ["lower","upper"] pairs for each parameter
        norm_warning: bool
            A flag that is true if any parameter has a norm or half-norm prior.
            Caller functions may wish to warn the user that the prior has
            been truncated at normal_stds.
        """
        prior_bounds = {}
        norm_trunc_warning = False
        for key in self.theta_keys:
            prior_bounds[key] = {}
            prior_dict = self.theta_prior[key]
            norm_trunc_warning = "norm" in prior_dict["type"] or norm_trunc_warning

            if prior_dict["type"] == "unif":
                prior_bounds[key]["lower"] = prior_dict["lower"]
                prior_bounds[key]["upper"] = prior_dict["upper"]
            elif prior_dict["type"] == "log10unif":
                prior_bounds[key]["lower"] = 10 ** prior_dict["log10lower"]
                prior_bounds[key]["upper"] = 10 ** prior_dict["log10upper"]
            elif prior_dict["type"] == "norm":
                prior_bounds[key]["lower"] = (
                    prior_dict["loc"] - normal_stds * prior_dict["scale"]
                )
                prior_bounds[key]["upper"] = (
                    prior_dict["loc"] + normal_stds * prior_dict["scale"]
                )
            elif prior_dict["type"] == "halfnorm":
                prior_bounds[key]["lower"] = prior_dict["loc"]
                prior_bounds[key]["upper"] = (
                    prior_dict["loc"] + normal_stds * prior_dict["scale"]
                )
            elif prior_dict["type"] == "neghalfnorm":
                prior_bounds[key]["upper"] = prior_dict["loc"]
                prior_bounds[key]["lower"] = (
                    prior_dict["loc"] - normal_stds * prior_dict["scale"]
                )
            elif prior_dict["type"] == "lognorm":
                prior_bounds[key]["lower"] = np.exp(
                    prior_dict["loc"] - normal_stds * prior_dict["scale"]
                )
                prior_bounds[key]["upper"] = np.exp(
                    prior_dict["loc"] + normal_stds * prior_dict["scale"]
                )
            else:
                raise ValueError(
                    "Not implemented for prior type {}".format(prior_dict["type"])
                )
        return prior_bounds, norm_trunc_warning

    def plot_prior_posterior(
        self,
        normal_stds=2,
        injection_parameters=None,
        fig_and_axes=None,
        save_fig=True,
    ):
        """Plot the prior and posterior probability distributions in the same figure

        Parameters
        ----------
        normal_stds: int
           Bounds of priors in terms of their standard deviation. Only used if
           `norm`, `halfnorm`, `neghalfnorm` or `lognorm` priors are given, otherwise ignored.
        injection_parameters: dict
            Dictionary containing the parameters of a signal. All parameters being searched must be
            present as dictionary keys, otherwise this option is ignored.
        fig_and_axes: tuple
            (fig, axes) tuple to plot on.
        save_fig: bool
            If true, save the figure, else return the fig, axes.

        Returns
        -------
        (fig, ax): (matplotlib.pyplot.figure, matplotlib.pyplot.axes)
            If `save_fig` evaluates to `False`, return figure and axes.
        """

        # Check injection parameters first
        injection_parameters = injection_parameters or {}
        missing_keys = set(self.theta_keys) - injection_parameters.keys()
        if missing_keys:
            logger.warning(
                f"plot_prior_posterior(): Missing keys {missing_keys} in 'injection_parameters',"
                " no injection parameters will be highlighted."
            )
            injection_parameters = None

        if fig_and_axes is None:
            fig, axes = plt.subplots(nrows=self.ndim, figsize=(8, 4 * self.ndim))
        else:
            fig, axes = fig_and_axes
        if self.ndim == 1:
            axes = [axes]
        N = 1000
        from scipy.stats import gaussian_kde

        prior_bounds, _ = self._get_prior_bounds(normal_stds)
        for i, (ax, key) in enumerate(zip(axes, self.theta_keys)):
            prior_dict = self.theta_prior[key]
            ln_prior_func = self._generic_lnprior(**prior_dict)
            x = np.linspace(prior_bounds[key]["lower"], prior_bounds[key]["upper"], N)
            prior = np.exp([ln_prior_func(xi) for xi in x])  # may not be vectorized

            priorln = ax.plot(x, prior, "C3", label="prior")
            ax.set(xlabel=self.theta_symbols[i], yticks=[])

            s = self.samples[:, i]
            while len(s) > 10**4:
                # random downsample to avoid slow calculation of kde
                s = np.random.choice(s, size=int(len(s) / 2.0))
            kde = gaussian_kde(s)
            ax2 = ax.twinx()
            postln = ax2.plot(x, kde.pdf(x), "k", label="posterior")
            ax2.set(yticks=[], yticklabels=[])

            if injection_parameters is not None:
                injection = ax.axvline(
                    injection_parameters[key],
                    label="Injection",
                    color="purple",
                    ls="--",
                )

        plotlines = priorln + postln
        labs = [plotline.get_label() for plotline in plotlines]
        if injection_parameters is not None:
            plotlines.append(injection)
            labs.append("injection")
        axes[0].legend(plotlines, labs, loc=1, framealpha=0.8)

        if save_fig:
            fig.savefig(os.path.join(self.outdir, self.label + "_prior_posterior.png"))
            plt.close(fig)
        else:
            return fig, axes

    def plot_cumulative_max(self, **kwargs):
        """Plot the cumulative twoF for the maximum posterior estimate.

        This method accepts the same arguments as `pyfstat.core.ComputeFstat.plot_twoF_cumulative`,
        except for `CFS_input`, which is taken from the loudest candidate; and `label` and `outdir`,
        which are taken from the instance of this class.

        For example, one can pass signal arguments to predic_twoF_cumulative through `PFS_kwargs`, or
        set the number of segments using `num_segments_(CFS|PFS)`. The same applies for other options
        such as `tstart`, `tend` or `savefig`. Every single of these arguments will be passed to
        `pyfstat.core.ComputeFstat.plot_twoF_cumulative` as they are, using their default argument
        otherwise.

        See `pyfstat.core.ComputeFstat.plot_twoF_cumulative` for a comprehensive list of accepted
        arguments and their default values.

        Unlike the core function, here savefig=True is the default,
        for consistency with other MCMC plotting functions.
        """
        logger.info("Getting cumulative 2F")
        d, maxtwoF = self.get_max_twoF()
        for key, val in self.theta_prior.items():
            if key not in d:
                d[key] = val

        if kwargs.get("savefig") is None:
            kwargs["savefig"] = True

        self.search.plot_twoF_cumulative(
            CFS_input=d, label=self.label, outdir=self.outdir, **kwargs
        )

    def _generic_lnprior(self, **kwargs):
        """Return a lambda function of the pdf

        Parameters
        ----------
        **kwargs:
            A dictionary containing 'type' of pdf and shape parameters

        """

        def log_of_unif(x, a, b):
            above = x < b
            below = x > a
            if type(above) is not np.ndarray:
                if above and below:
                    return -np.log(b - a)
                else:
                    return -np.inf
            else:
                idxs = np.array([all(tup) for tup in zip(above, below)])
                p = np.zeros(len(x)) - np.inf
                p[idxs] = -np.log(b - a)
                return p

        def log_of_log10unif(x, log10lower, log10upper):
            log10x = np.log10(x)
            above = log10x < log10upper
            below = log10x > log10lower
            if type(above) is not np.ndarray:
                if above and below:
                    return -np.log(x * np.log(10) * (log10upper - log10lower))
                else:
                    return -np.inf
            else:
                idxs = np.array([all(tup) for tup in zip(above, below)])
                p = np.zeros(len(x)) - np.inf
                p[idxs] = -np.log(x * np.log(10) * (log10upper - log10lower))
                return p

        def log_of_halfnorm(x, loc, scale):
            if x < loc:
                return -np.inf
            else:
                return -0.5 * (
                    (x - loc) ** 2 / scale**2 + np.log(0.5 * np.pi * scale**2)
                )

        def cauchy(x, x0, gamma):
            return 1.0 / (np.pi * gamma * (1 + ((x - x0) / gamma) ** 2))

        def exp(x, x0, gamma):
            if x > x0:
                return np.log(gamma) - gamma * (x - x0)
            else:
                return -np.inf

        if kwargs["type"] == "unif":
            return lambda x: log_of_unif(x, kwargs["lower"], kwargs["upper"])
        if kwargs["type"] == "log10unif":
            return lambda x: log_of_log10unif(
                x, kwargs["log10lower"], kwargs["log10upper"]
            )
        elif kwargs["type"] == "halfnorm":
            return lambda x: log_of_halfnorm(x, kwargs["loc"], kwargs["scale"])
        elif kwargs["type"] == "neghalfnorm":
            return lambda x: log_of_halfnorm(-x, kwargs["loc"], kwargs["scale"])
        elif kwargs["type"] == "norm":
            return lambda x: -0.5 * (
                (x - kwargs["loc"]) ** 2 / kwargs["scale"] ** 2
                + np.log(2 * np.pi * kwargs["scale"] ** 2)
            )
        elif kwargs["type"] == "lognorm":
            # as of scipy 1.4.1 and numpy 1.18.1 the following parametrisation
            # should be consistent with np.random.lognormal in _generate_rv()
            return lambda x: lognorm.pdf(
                x, s=kwargs["scale"], scale=np.exp(kwargs["loc"])
            )
        else:
            logger.info("kwargs:", kwargs)
            raise ValueError("Prior pdf type {:s} unknown.".format(kwargs["type"]))

    def _generate_rv(self, **kwargs):
        dist_type = kwargs.pop("type")
        if dist_type == "unif":
            return np.random.uniform(low=kwargs["lower"], high=kwargs["upper"])
        if dist_type == "log10unif":
            return 10 ** (
                np.random.uniform(low=kwargs["log10lower"], high=kwargs["log10upper"])
            )
        if dist_type == "norm":
            return np.random.normal(loc=kwargs["loc"], scale=kwargs["scale"])
        if dist_type == "halfnorm":
            return np.abs(np.random.normal(loc=kwargs["loc"], scale=kwargs["scale"]))
        if dist_type == "neghalfnorm":
            return -1 * np.abs(
                np.random.normal(loc=kwargs["loc"], scale=kwargs["scale"])
            )
        if dist_type == "lognorm":
            return np.random.lognormal(mean=kwargs["loc"], sigma=kwargs["scale"])
        else:
            raise ValueError("dist_type {} unknown".format(dist_type))

    def _plot_walkers(
        self,
        symbols=None,
        alpha=0.8,
        color="k",
        temp=0,
        lw=0.1,
        nprod=0,
        add_det_stat_burnin=False,
        fig=None,
        axes=None,
        xoffset=0,
        injection_parameters=None,
        plot_det_stat=False,
        context="ggplot",
        labelpad=5,
    ):
        """Plot all the chains from a sampler"""
        if injection_parameters is not None:
            if not isinstance(injection_parameters, dict):
                raise ValueError("injection_parameters is not a dictionary")

            missing_keys = set(self.theta_keys) - injection_parameters.keys()
            if missing_keys:
                logger.warning(
                    f"plot_walkers(): Missing keys {missing_keys} in 'injection_parameters',"
                    " argument will be ignored."
                )
                injection_parameters = None
            else:
                scaled_injection_parameters = {
                    key: (
                        injection_parameters[key]
                        - self._get_rescale_subtractor_for_key(key)
                    )
                    * self._get_rescale_multiplier_for_key(key)
                    for key in injection_parameters.keys()
                }

        if symbols is None:
            symbols = self._get_labels()
        if context not in plt.style.available:
            raise ValueError(
                (
                    "The requested context {} is not available; please select a"
                    " context from `plt.style.available`"
                ).format(context)
            )

        if np.ndim(axes) > 1:
            axes = axes.flatten()

        shape = self.sampler.chain.shape
        if len(shape) == 3:
            nwalkers, nsteps, ndim = shape
            chain = self.sampler.chain[:, :, :].copy()
        if len(shape) == 4:
            ntemps, nwalkers, nsteps, ndim = shape
            if temp < ntemps:
                logger.info("Plotting temperature {} chains".format(temp))
            else:
                raise ValueError(
                    ("Requested temperature {} outside of" "available range").format(
                        temp
                    )
                )
            chain = self.sampler.chain[temp, :, :, :].copy()

        samples = chain.reshape((nwalkers * nsteps, ndim))
        samples = self._scale_samples(samples, self.theta_keys)
        chain = chain.reshape((nwalkers, nsteps, ndim))

        if plot_det_stat:
            extra_subplots = 1
        else:
            extra_subplots = 0
        with plt.style.context((context)):
            if fig is None and axes is None:
                fig = plt.figure(figsize=(4, 3.0 * ndim))
                ax = fig.add_subplot(ndim + extra_subplots, 1, 1)
                axes = [ax] + [
                    fig.add_subplot(ndim + extra_subplots, 1, i)
                    for i in range(2, ndim + 1)
                ]

            idxs = np.arange(chain.shape[1])
            burnin_idx = chain.shape[1] - nprod
            last_idx = burnin_idx
            if ndim > 1:
                for i in range(ndim):
                    axes[i].ticklabel_format(useOffset=False, axis="y")
                    cs = chain[:, :, i].T
                    if burnin_idx > 0:
                        axes[i].plot(
                            xoffset + idxs[: last_idx + 1],
                            cs[: last_idx + 1],
                            color="C3",
                            alpha=alpha,
                            lw=lw,
                        )
                        axes[i].axvline(xoffset + last_idx, color="k", ls="--", lw=0.5)
                    axes[i].plot(
                        xoffset + idxs[burnin_idx:],
                        cs[burnin_idx:],
                        color="k",
                        alpha=alpha,
                        lw=lw,
                    )
                    if injection_parameters is not None:
                        axes[i].axhline(
                            scaled_injection_parameters[self.theta_keys[i]],
                            ls="--",
                            lw=2.0,
                            color="orange",
                        )
                    axes[i].set_xlim(0, xoffset + idxs[-1])
                    if symbols:
                        axes[i].set_ylabel(symbols[i], labelpad=labelpad)
            else:
                axes[0].ticklabel_format(useOffset=False, axis="y")
                cs = chain[:, :, temp].T
                if burnin_idx:
                    axes[0].plot(
                        idxs[:burnin_idx],
                        cs[:burnin_idx],
                        color="C3",
                        alpha=alpha,
                        lw=lw,
                    )
                axes[0].plot(
                    idxs[burnin_idx:], cs[burnin_idx:], color="k", alpha=alpha, lw=lw
                )
                if injection_parameters is not None:
                    axes[0].axhline(
                        scaled_injection_parameters[self.theta_keys[0]],
                        ls="--",
                        lw=5.0,
                        color="orange",
                    )
                if symbols:
                    axes[0].set_ylabel(symbols[0], labelpad=labelpad)

            axes[-1].set_xlabel(r"Number of steps", labelpad=0.2)

            if plot_det_stat:
                if len(axes) == ndim:
                    axes.append(fig.add_subplot(ndim + 1, 1, ndim + 1))

                lnl = self.sampler.loglikelihood[temp, :, :]
                if burnin_idx and add_det_stat_burnin:
                    burn_in_vals = lnl[:, :burnin_idx].flatten()
                    try:
                        detstat_burnin = (
                            burn_in_vals[~np.isnan(burn_in_vals)] - self.likelihoodcoef
                        ) / self.likelihooddetstatmultiplier
                        axes[-1].hist(
                            detstat_burnin, bins=50, histtype="step", color="C3"
                        )
                    except ValueError:
                        logger.info(
                            "Histogram of detection statistic failed, "
                            "most likely all values were the same."
                        )
                        pass
                else:
                    detstat_burnin = []
                prod_vals = lnl[:, burnin_idx:].flatten()
                try:
                    detstat = (
                        prod_vals[~np.isnan(prod_vals)] - self.likelihoodcoef
                    ) / self.likelihooddetstatmultiplier
                    axes[-1].hist(detstat, bins=50, histtype="step", color="k")
                except ValueError:
                    logger.info(
                        "Histogram of detection statistic failed, "
                        "most likely all values were the same."
                    )
                    pass
                if self.BSGL:
                    axes[-1].set_xlabel(r"$\log_{10}\mathcal{B}_\mathrm{S/GL}$")
                else:
                    axes[-1].set_xlabel(r"$\widetilde{2\mathcal{F}}$")
                axes[-1].set_ylabel(r"$\mathrm{Counts}$")
                combined_vals = np.append(detstat_burnin, detstat)
                if len(combined_vals) > 0:
                    minv = np.min(combined_vals)
                    maxv = np.max(combined_vals)
                    Range = abs(maxv - minv)
                    axes[-1].set_xlim(minv - 0.1 * Range, maxv + 0.1 * Range)

                xfmt = matplotlib.ticker.ScalarFormatter()
                xfmt.set_powerlimits((-4, 4))
                axes[-1].xaxis.set_major_formatter(xfmt)

        return fig, axes

    def _apply_corrections_to_p0(self, p0):
        """Apply any correction to the initial p0 values"""
        return p0

    def _generate_scattered_p0(self, p):
        """Generate a set of p0s scattered about p"""
        p0 = [
            [
                p + self.scatter_val * p * np.random.randn(self.ndim)
                for i in range(self.nwalkers)
            ]
            for j in range(self.ntemps)
        ]
        return p0

    def _generate_initial_p0(self):
        """Generate a set of init vals for the walkers"""

        if isinstance(self.theta_initial, dict):
            logger.info("Generate initial values from initial dictionary")
            if hasattr(self, "nglitch") and self.nglitch > 1:
                raise ValueError("Initial dict not implemented for nglitch>1")
            p0 = [
                [
                    [
                        self._generate_rv(**self.theta_initial[key])
                        for key in self.theta_keys
                    ]
                    for i in range(self.nwalkers)
                ]
                for j in range(self.ntemps)
            ]
        elif self.theta_initial is None:
            logger.info("Generate initial values from prior dictionary")
            p0 = [
                [
                    [
                        self._generate_rv(**self.theta_prior[key])
                        for key in self.theta_keys
                    ]
                    for i in range(self.nwalkers)
                ]
                for j in range(self.ntemps)
            ]
        else:
            raise ValueError("theta_initial not understood")

        return p0

    def _get_new_p0(self):
        """Returns new initial positions for walkers are burn0 stage

        This returns new positions for all walkers by scattering points about
        the maximum posterior with scale `scatter_val`.

        """
        temp_idx = 0
        pF = self.sampler.chain[temp_idx, :, :, :]
        lnl = self.sampler.loglikelihood[temp_idx, :, :]
        lnp = self.sampler.logprobability[temp_idx, :, :]

        # General warnings about the state of lnp
        if np.any(np.isnan(lnp)):
            logger.warning(
                "Of {} lnprobs {} are nan".format(np.shape(lnp), np.sum(np.isnan(lnp)))
            )
        if np.any(np.isposinf(lnp)):
            logger.warning(
                "Of {} lnprobs {} are +np.inf".format(
                    np.shape(lnp), np.sum(np.isposinf(lnp))
                )
            )
        if np.any(np.isneginf(lnp)):
            logger.warning(
                "Of {} lnprobs {} are -np.inf".format(
                    np.shape(lnp), np.sum(np.isneginf(lnp))
                )
            )

        lnp_finite = copy.copy(lnp)
        lnp_finite[np.isinf(lnp)] = np.nan
        idx = np.unravel_index(np.nanargmax(lnp_finite), lnp_finite.shape)
        p = pF[idx]
        p0 = self._generate_scattered_p0(p)

        logger.info(
            (
                "Gen. new p0 from pos {} which had det. stat.={:2.1f}"
                " and lnp={:2.1f}"
            ).format(idx[1], lnl[idx], lnp_finite[idx])
        )

        return p0

    def _get_data_dictionary_to_save(self):
        d = dict(
            nsteps=self.nsteps,
            nwalkers=self.nwalkers,
            ntemps=self.ntemps,
            theta_keys=self.theta_keys,
            theta_prior=self.theta_prior,
            log10beta_min=self.log10beta_min,
            BSGL=self.BSGL,
            minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime,
        )
        return d

    def _pickle_data(self, samples, lnprobs, lnlikes, all_lnlikelihood):
        d = self._get_data_dictionary_to_save()
        d["samples"] = samples
        d["lnprobs"] = lnprobs
        d["lnlikes"] = lnlikes
        d["chain"] = self.sampler.chain
        d["all_lnlikelihood"] = all_lnlikelihood

        if os.path.isfile(self.pickle_path):
            logger.info(
                "Saving backup of {} as {}.old".format(
                    self.pickle_path, self.pickle_path
                )
            )
            os.rename(self.pickle_path, self.pickle_path + ".old")
        with open(self.pickle_path, "wb") as File:
            pickle.dump(d, File)

    def get_saved_data_dictionary(self):
        """Read the data saved in `self.pickel_path` and return it as a dictionary.

        Returns
        --------
        d: dict
            Dictionary containing the data saved in the pickle `self.pickle_path`.
        """
        with open(self.pickle_path, "rb") as File:
            d = pickle.load(File)
        return d

    def _check_old_data_is_okay_to_use(self):
        if os.path.isfile(self.pickle_path) is False:
            logger.info("No pickled data found")
            return False

        if self.sftfilepattern is not None:
            oldest_sft = min(
                [os.path.getmtime(f) for f in self._get_list_of_matching_sfts()]
            )
            if os.path.getmtime(self.pickle_path) < oldest_sft:
                logger.info("Pickled data outdates sft files")
                return False

        old_d = self.get_saved_data_dictionary().copy()
        new_d = self._get_data_dictionary_to_save().copy()

        old_d.pop("samples")
        old_d.pop("lnprobs")
        old_d.pop("lnlikes")
        old_d.pop("all_lnlikelihood")
        old_d.pop("chain")

        for key in "minStartTime", "maxStartTime":
            if new_d[key] is None:
                new_d[key] = old_d[key]
                setattr(self, key, new_d[key])

        mod_keys = []
        for key in list(new_d.keys()):
            if key in old_d:
                if new_d[key] != old_d[key]:
                    mod_keys.append((key, old_d[key], new_d[key]))
            else:
                raise ValueError("Keys {} not in old dictionary".format(key))

        if len(mod_keys) == 0:
            return True
        else:
            logger.warning("Saved data differs from requested")
            logger.info("Differences found in following keys:")
            for key in mod_keys:
                if len(key) == 3:
                    if np.isscalar(key[1]) or key[0] == "nsteps":
                        logger.info("    {} : {} -> {}".format(*key))
                    else:
                        logger.info("    " + key[0])
                else:
                    logger.info(key)
            return False

    def _get_savetxt_fmt_dict(self):
        fmt_dict = utils.get_doppler_params_output_format(self.theta_keys)
        fmt_dict["twoF"] = "%.9g"
        if self.BSGL:
            fmt_dict["log10BSGL"] = "%.9g"
        return fmt_dict

    def _get_savetxt_gmt_list(self):
        """Returns a list of output format specifiers, ordered like the samples

        This is required because the output of _get_savetxt_fmt_dict()
        will depend on the order in which those entries have been coded up.
        """
        fmt_dict = self._get_savetxt_fmt_dict()
        fmt_list = [fmt_dict[key] for key in self.output_keys]
        return fmt_list

    def export_samples_to_disk(self):
        """
        Export MCMC samples into a text file using `numpy.savetxt`.
        """
        self.samples_file = os.path.join(self.outdir, self.label + "_samples.dat")
        logger.info("Exporting samples to {}".format(self.samples_file))
        header = "\n".join(self.output_file_header)
        header += "\n" + " ".join(self.output_keys)
        outfmt = self._get_savetxt_gmt_list()
        samples_out = copy.copy(self.samples)
        # For convenience, we always save a twoF column,
        # even if log10BSGL was used for the likelihood.
        detstat = np.atleast_2d(self._get_detstat_from_loglikelihood()).T
        if self.BSGL:
            twoF = np.zeros_like(detstat)
            self.search.BSGL = False
            for idx, samp in enumerate(self.samples):
                p = self._set_point_for_evaluation(samp)
                if isinstance(p, dict):
                    twoF[idx] = self.search.get_det_stat(**p)
                else:
                    twoF[idx] = self.search.get_det_stat(*p)
            self.search.BSGL = self.BSGL
            samples_out = np.concatenate((samples_out, twoF), axis=1)
        # TODO: add single-IFO F-stats?
        samples_out = np.concatenate((samples_out, detstat), axis=1)
        Ncols = np.shape(samples_out)[1]
        if len(outfmt) != Ncols:
            raise RuntimeError(
                "Lengths of data rows ({:d})"
                " and output format ({:d})"
                " do not match."
                " If your search class uses different"
                " keys than the base MCMCSearch class,"
                " override the _get_savetxt_fmt_dict"
                " method.".format(Ncols, len(outfmt))
            )
        np.savetxt(
            self.samples_file,
            samples_out,
            delimiter=" ",
            header=header,
            fmt=outfmt,
        )

    def _get_detstat_from_loglikelihood(self, idx=None):
        """Inverts the extra terms applied in logl()."""
        return (
            self.lnlikes[idx if idx is not None else ...] - self.likelihoodcoef
        ) / self.likelihooddetstatmultiplier

    def get_max_twoF(self):
        """Get the max. likelihood (loudest) sample and the compute
        its corresponding detection statistic.

        The employed detection statistic depends on `self.BSGL`
        (i.e. 2F if `self.BSGL` evaluates to `False`, log10BSGL otherwise).

        Returns
        -------
        d: dict
            Parameters of the loudest sample.

        maxtwoF: float
            Detection statistic (2F or log10BSGL) corresponding to the loudest sample.
        """
        if not hasattr(self, "search"):
            raise RuntimeError(
                "Object has no self.lnlikes attribute, please execute .run() first."
            )
        if any(np.isposinf(self.lnlikes)):
            logger.info("lnlike values contain positive infinite values")
        if any(np.isneginf(self.lnlikes)):
            logger.info("lnlike values contain negative infinite values")
        if any(np.isnan(self.lnlikes)):
            logger.info("lnlike values contain nan")
        idxs = np.isfinite(self.lnlikes)
        jmax = np.nanargmax(self.lnlikes[idxs])
        d = OrderedDict()

        if self.BSGL:
            # need to recompute twoF at the max likelihood
            if hasattr(self, "search") is False:
                self._initiate_search_object()
            p = self._set_point_for_evaluation(self.samples[jmax])
            self.search.BSGL = False
            if isinstance(p, dict):
                maxtwoF = self.search.get_det_stat(**p)
            else:
                maxtwoF = self.search.get_det_stat(*p)
            self.search.BSGL = self.BSGL
        else:
            # can just reuse the logl value
            maxtwoF = self._get_detstat_from_loglikelihood(jmax)

        repeats = []
        for i, k in enumerate(self.theta_keys):
            if k in d and k not in repeats:
                d[k + "_0"] = d[k]  # relabel the old key
                d.pop(k)
                repeats.append(k)
            if k in repeats:
                k = k + "_0"
                count = 1
                while k in d:
                    k = k.replace("_{}".format(count - 1), "_{}".format(count))
                    count += 1
            d[k] = self.samples[jmax][i]
        return d, maxtwoF

    def get_summary_stats(self):
        """Returns a dict of point estimates for all production samples.

        Point estimates are computed on the MCMC samples using `numpy.mean`,
        `numpy.std` and `numpy.quantiles` with q=[0.005, 0.05, 0.25, 0.5, 0.75, 0.95, 0.995].

        Returns
        -------
        d: dict
            Dictionary containing point estimates corresponding to ["mean", "std", "lower99",
            "lower90", "lower50", "median", "upper50", "upper90", "upper99"].
        """
        d = OrderedDict()
        repeats = []  # taken from old get_median_stds(), not sure why necessary
        for s, k in zip(self.samples.T, self.theta_keys):
            if k in d and k not in repeats:
                d[k + "_0"] = d[k]  # relabel the old key
                d.pop(k)
                repeats.append(k)
            if k in repeats:
                k = k + "_0"
                count = 1
                while k in d:
                    k = k.replace("_{}".format(count - 1), "_{}".format(count))
                    count += 1

            d[k] = {}
            d[k]["mean"] = np.mean(s)
            d[k]["std"] = np.std(s)
            (
                d[k]["lower99"],
                d[k]["lower90"],
                d[k]["lower50"],
                d[k]["median"],
                d[k]["upper50"],
                d[k]["upper90"],
                d[k]["upper99"],
            ) = np.quantile(s, [0.005, 0.05, 0.25, 0.5, 0.75, 0.95, 0.995])

        return d

    def check_if_samples_are_railing(self, threshold=0.01):
        """Returns a boolean estimate of if the samples are railing

        Parameters
        ----------
        threshold: float [0, 1]
            Fraction of the uniform prior to test (at upper and lower bound)

        Returns
        -------
        return_flag: bool
            IF true, the samples are railing

        """
        return_flag = False
        for s, k in zip(self.samples.T, self.theta_keys):
            prior = self.theta_prior[k]
            if prior["type"] == "unif":
                prior_range = prior["upper"] - prior["lower"]
                edges = []
                fracs = []
                for bound in ["lower", "upper"]:
                    bools = np.abs(s - prior[bound]) / prior_range < threshold
                    if np.any(bools):
                        edges.append(bound)
                        fracs.append(str(100 * float(np.sum(bools)) / len(bools)))
                if len(edges) > 0:
                    logger.warning(
                        "{}% of the {} posterior is railing on the {} edges".format(
                            "% & ".join(fracs), k, " & ".join(edges)
                        )
                    )
                    return_flag = True
        return return_flag

    def write_par(self, method="median"):
        """Writes a .par of the best-fit params with an estimated std

        Parameters
        ----------
        method: str
            How to select the `best-fit` params. Available methods: "median", "mean", "twoFmax".
        """

        if method == "med":
            method = "median"
        if method in ["median", "mean"]:
            summary_stats = self.get_summary_stats()
            filename = os.path.join(self.outdir, self.label + "_" + method + ".par")
            logger.info("Writing {} using {} parameters.".format(filename, method))
        elif method == "twoFmax":
            max_twoF_d, max_twoF = self.get_max_twoF()
            filename = os.path.join(self.outdir, self.label + "_max2F.par")
            logger.info("Writing {} at max twoF = {}.".format(filename, max_twoF))
        else:
            raise ValueError("Method '{}' not supported.".format(method))

        with open(filename, "w+") as f:
            for hline in self.output_file_header:
                f.write("# {:s}\n".format(hline))
            if method == "twoFmax":
                f.write("MaxtwoF = {}\n".format(max_twoF))
            f.write("tref = {}\n".format(self.tref))
            if hasattr(self, "theta0_index"):
                f.write("theta0_index = {}\n".format(self.theta0_idx))
            if method in ["median", "mean"]:
                for key, stat_d in summary_stats.items():
                    f.write(
                        "{} = {:1.16e}\n".format(
                            key,
                            stat_d[method],
                        )
                    )
            elif method == "twoFmax":
                for key, val in max_twoF_d.items():
                    f.write("{} = {:1.16e}\n".format(key, val))

    def generate_loudest(self):
        """Use ComputeFstatistic_v2 executable to produce a .loudest file"""
        max_params, max_twoF = self.get_max_twoF()
        for key in self.theta_prior:
            if key not in max_params:
                max_params[key] = self.theta_prior[key]
        max_params = self.translate_keys_to_lal(max_params)
        for key in ["transient-t0Epoch", "transient-t0Offset", "transient-tau"]:
            if key in max_params and not int(max_params[key]) == max_params[key]:
                rounded = int(round(max_params[key]))
                logger.warning(
                    "Rounding {:s}={:f} to {:d} for CFSv2 call.".format(
                        key, max_params[key], rounded
                    )
                )
                max_params[key] = rounded
        signal_parameter_keys = list(
            self.translate_keys_to_lal(self.theta_prior).keys()
        )
        par_keys = list(max_params.keys())
        pardiff = np.setdiff1d(par_keys, signal_parameter_keys)
        if len(pardiff) > 0:
            raise RuntimeError(
                f"Dictionary for parameters at max2F point {par_keys}"
                " did include keys"
                # " (other than refTime)"
                " not expected from signal parameters being searched over:"
                f" {pardiff} not in {signal_parameter_keys}."
            )
        self.loudest_file = utils.generate_loudest_file(
            max_params=max_params,
            tref=self.tref,
            outdir=self.outdir,
            label=self.label,
            sftfilepattern=self.sftfilepattern,
            minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime,
            transientWindowType=getattr(self, "transientWindowType", None),
            earth_ephem=self.earth_ephem,
            sun_ephem=self.sun_ephem,
        )

    def write_prior_table(self):
        """Generate a .tex file of the prior"""
        with open(os.path.join(self.outdir, self.label + "_prior.tex"), "w") as f:
            f.write(
                r"\begin{tabular}{c l c} \hline" + "\n"
                r"Parameter & & &  \\ \hhline{====}"
            )

            for key, prior in self.theta_prior.items():
                if type(prior) is dict:
                    Type = prior["type"]
                    if Type == "unif":
                        a = prior["lower"]
                        b = prior["upper"]
                        line = r"{} & $\mathrm{{Unif}}$({}, {}) & {}\\"
                    elif Type == "norm":
                        a = prior["loc"]
                        b = prior["scale"]
                        line = r"{} & $\mathcal{{N}}$({}, {}) & {}\\"
                    elif Type == "halfnorm":
                        a = prior["loc"]
                        b = prior["scale"]
                        line = r"{} & $|\mathcal{{N}}$({}, {})| & {}\\"

                    u = self.unit_dictionary[key]
                    s = self.symbol_dictionary[key]
                    f.write("\n")
                    a = utils.texify_float(a)
                    b = utils.texify_float(b)
                    f.write(" " + line.format(s, a, b, u) + r" \\")
            f.write("\n\\end{tabular}\n")

    def print_summary(self):
        """Prints a summary of the max twoF found to the terminal"""
        max_twoFd, max_twoF = self.get_max_twoF()
        summary_stats = self.get_summary_stats()
        logger.info("Summary:")
        if hasattr(self, "theta0_idx"):
            logger.info("theta0 index: {}".format(self.theta0_idx))
        logger.info("Max twoF: {} with parameters:".format(max_twoF))
        for k in np.sort(list(max_twoFd.keys())):
            logger.info("  {:10s} = {:1.9e}".format(k, max_twoFd[k]))
        logger.info("Mean +- std for production values:")
        for k in np.sort(list(summary_stats.keys())):
            logger.info(
                "  {:10s} = {:1.9e} +/- {:1.9e}".format(
                    k, summary_stats[k]["mean"], summary_stats[k]["std"]
                )
            )
        logger.info("Median and 90% quantiles for production values:")
        for k in np.sort(list(summary_stats.keys())):
            logger.info(
                "  {:10s} = {:1.9e} - {:1.9e} + {:1.9e}".format(
                    k,
                    summary_stats[k]["median"],
                    summary_stats[k]["median"] - summary_stats[k]["lower90"],
                    summary_stats[k]["upper90"] - summary_stats[k]["median"],
                )
            )

    def _CF_twoFmax(self, theta, twoFmax, ntrials):
        Fmax = twoFmax / 2.0
        return (
            np.exp(1j * theta * twoFmax)
            * ntrials
            / 2.0
            * Fmax
            * np.exp(-Fmax)
            * (1 - (1 + Fmax) * np.exp(-Fmax)) ** (ntrials - 1)
        )

    def _pdf_twoFhat(self, twoFhat, nglitch, ntrials, twoFmax=100, dtwoF=0.1):
        if np.ndim(ntrials) == 0:
            ntrials = np.zeros(nglitch + 1) + ntrials
        twoFmax_int = np.arange(0, twoFmax, dtwoF)
        theta_int = np.arange(-1 / dtwoF, 1.0 / dtwoF, 1.0 / twoFmax)
        CF_twoFmax_theta = np.array(
            [
                [
                    np.trapz(self._CF_twoFmax(t, twoFmax_int, ntrial), twoFmax_int)
                    for t in theta_int
                ]
                for ntrial in ntrials
            ]
        )
        CF_twoFhat_theta = np.prod(CF_twoFmax_theta, axis=0)
        pdf = (1 / (2 * np.pi)) * np.array(
            [
                np.trapz(
                    np.exp(-1j * theta_int * twoFhat_val) * CF_twoFhat_theta, theta_int
                )
                for twoFhat_val in twoFhat
            ]
        )
        return pdf.real

    def _p_val_twoFhat(self, twoFhat, ntrials, twoFhatmax=500, Npoints=1000):
        """Caluculate the p-value for the given twoFhat in Gaussian noise

        Parameters
        ----------
        twoFhat: float
            The observed twoFhat value
        ntrials: int, array of len Nglitch+1
            The number of trials for each glitch+1
        """
        twoFhats = np.linspace(twoFhat, twoFhatmax, Npoints)
        pdf = self._pdf_twoFhat(twoFhats, self.nglitch, ntrials)
        return np.trapz(pdf, twoFhats)

    def get_p_value(self, delta_F0=0, time_trials=0):
        """Gets the p-value for the maximum twoFhat value assuming Gaussian noise

        Parameters
        ----------
        delta_F0: float
            Frequency variation due to a glitch.
        time_trials: int, optional
            Number of trials in each glitch + 1.
        """
        d, max_twoF = self.get_max_twoF()
        if self.nglitch == 1:
            tglitches = [d["tglitch"]]
        else:
            tglitches = [d["tglitch_{}".format(i)] for i in range(self.nglitch)]
        tboundaries = [self.minStartTime] + tglitches + [self.maxStartTime]
        deltaTs = np.diff(tboundaries)
        ntrials = [time_trials + delta_F0 * dT for dT in deltaTs]
        p_val = self._p_val_twoFhat(max_twoF, ntrials)
        logger.info("p-value = {}".format(p_val))
        return p_val

    def compute_evidence(self, make_plots=False, write_to_file=None):
        """Computes the evidence/marginal likelihood for the model.

        Parameters
        ----------
        make_plots: bool
           Plot the results and save them to os.path.join(self.outdir, self.label + "_beta_lnl.png")
        write_to_file: str
           If given, dump evidence and uncertainty estimation to the specified path.

        Returns
        -------
        log10evidence: float
            Estimation of the log10 evidence.
        log10evidence_err: float
            Log10 uncertainty of the evidence estimation.
        """
        betas = self.betas
        mean_lnlikes = np.mean(np.mean(self.all_lnlikelihood, axis=1), axis=1)

        mean_lnlikes = mean_lnlikes[::-1]
        betas = betas[::-1]

        if any(np.isinf(mean_lnlikes)):
            logger.warning(
                "mean_lnlikes contains inf: recalculating without"
                " the {} infs".format(len(betas[np.isinf(mean_lnlikes)]))
            )
            idxs = np.isinf(mean_lnlikes)
            mean_lnlikes = mean_lnlikes[~idxs]
            betas = betas[~idxs]

        log10evidence = np.trapz(mean_lnlikes, betas) / np.log(10)
        z1 = np.trapz(mean_lnlikes, betas)
        z2 = np.trapz(mean_lnlikes[::-1][::2][::-1], betas[::-1][::2][::-1])
        log10evidence_err = np.abs(z1 - z2) / np.log(10)

        logger.info(
            "log10 evidence for {} = {} +/- {}".format(
                self.label, log10evidence, log10evidence_err
            )
        )

        if write_to_file:
            EvidenceDict = self.read_evidence_file_to_dict(write_to_file)
            EvidenceDict[self.label] = [log10evidence, log10evidence_err]
            self.write_evidence_file_from_dict(EvidenceDict, write_to_file)

        if make_plots:
            fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(6, 8))
            ax1.semilogx(betas, mean_lnlikes, "-o")
            ax1.set_xlabel(r"$\beta$")
            ax1.set_ylabel(r"$\langle \log(\mathcal{L}) \rangle$")
            min_betas = []
            evidence = []
            for i in range(int(len(betas) / 2.0)):
                min_betas.append(betas[i])
                lnZ = np.trapz(mean_lnlikes[i:], betas[i:])
                evidence.append(lnZ / np.log(10))

            ax2.semilogx(min_betas, evidence, "-o")
            ax2.set_ylabel(
                r"$\int_{\beta_{\mathrm{Min}}}^{\beta=1}"
                + r"\langle \log(\mathcal{L})\rangle d\beta$",
                size=16,
            )
            ax2.set_xlabel(r"$\beta_{\mathrm{min}}$")
            plt.tight_layout()
            fig.savefig(os.path.join(self.outdir, self.label + "_beta_lnl.png"))
            plt.close(fig)

        return log10evidence, log10evidence_err

    @staticmethod
    def read_evidence_file_to_dict(evidence_file_name="Evidences.txt"):
        """Read evidence file and put it into an OrderedDict

        An evidence file contains paris (log10evidence, log10evidence_err) for each
        considered model. These pairs are prepended by the `self.label` variable.

        Parameters
        ----------
        evidence_file_name: str
            Filename to read.

        Returns
        -------
        EvidenceDict: dict
            Dictionary with the contents of `evidence_file_name`
        """
        EvidenceDict = OrderedDict()
        if os.path.isfile(evidence_file_name):
            with open(evidence_file_name, "r") as f:
                for line in f:
                    key, log10evidence, log10evidence_err = line.split(" ")
                    EvidenceDict[key] = [float(log10evidence), float(log10evidence_err)]
        return EvidenceDict

    def write_evidence_file_from_dict(self, EvidenceDict, evidence_file_name):
        """Write evidence dict to a file

        Parameters
        ----------
        EvidenceDict: dict
            Dictionary to dump into a file.
        evidence_file_name: str
            File name to dump dict into.
        """
        with open(evidence_file_name, "w+") as f:
            for key, val in EvidenceDict.items():
                f.write("{} {} {}\n".format(key, val[0], val[1]))


class MCMCGlitchSearch(MCMCSearch):
    """MCMC search using the SemiCoherentGlitchSearch

    See parent MCMCSearch for a list of all additional parameters, here we list
    only the additional init parameters of this class.
    """

    symbol_dictionary = dict(
        F0=r"$f$",
        F1=r"$\dot{f}$",
        F2=r"$\ddot{f}$",
        Alpha=r"$\alpha$",
        Delta=r"$\delta$",
    )
    """
        Key, val pairs of the parameters (`F0`, `F1`, ...), to LaTeX math
        symbols for plots
    """
    glitch_symbol_dictionary = dict(
        delta_F0=r"$\delta f$",
        delta_F1=r"$\delta \dot{f}$",
        tglitch=r"$t_\mathrm{glitch}$",
    )
    """
        Key, val pairs of glitch parameters (`dF0`, `dF1`, `tglitch`), to LaTeX math
        symbols for plots. This dictionary included within `self.symbol_dictionary`.
    """
    symbol_dictionary.update(glitch_symbol_dictionary)
    unit_dictionary = dict(
        F0=r"Hz",
        F1=r"Hz/s",
        F2=r"Hz/s$^2$",
        Alpha=r"rad",
        Delta=r"rad",
        delta_F0=r"Hz",
        delta_F1=r"Hz/s",
        tglitch=r"s",
    )
    """
        Key, val pairs of the parameters (`F0`, `F1`, ..., including glitch parameters),
        and the units (`Hz`, `Hz/s`, ...).
    """
    transform_dictionary = dict(
        tglitch={
            "multiplier": 1 / 86400.0,
            "subtractor": "minStartTime",
            "unit": "day",
            "label": r"$t^{g}_0$ \n [d]",
        }
    )
    """
        Key, val pairs of the parameters (`F0`, `F1`, ...), where the key is
        itself a dictionary which can item `multiplier`, `subtractor`, or
        `unit` by which to transform by and update the units.
    """

    @utils.initializer
    def __init__(
        self,
        theta_prior,
        tref,
        label,
        outdir="data",
        minStartTime=None,
        maxStartTime=None,
        sftfilepattern=None,
        detectors=None,
        nsteps=[100, 100],
        nwalkers=100,
        ntemps=1,
        log10beta_min=-5,
        theta_initial=None,
        rhohatmax=1000,
        binary=False,
        BSGL=False,
        SSBprec=None,
        RngMedWindow=None,
        minCoverFreq=None,
        maxCoverFreq=None,
        injectSources=None,
        assumeSqrtSX=None,
        dtglitchmin=1 * 86400,
        theta0_idx=0,
        nglitch=1,
        earth_ephem=None,
        sun_ephem=None,
        allowedMismatchFromSFTLength=None,
        clean=False,
    ):
        """
        Parameters
        ----------
        nglitch: int
            The number of glitches to allow
        dtglitchmin: int
            The minimum duration (in seconds) of a segment between two glitches
            or a glitch and the start/end of the data
        theta0_idx, int
            Index (zero-based) of which segment the theta refers to - useful
            if providing a tight prior on theta to allow the signal to jump
            too theta (and not just from)
        """
        self._set_init_params_dict(locals())
        os.makedirs(outdir, exist_ok=True)
        self.output_file_header = self.get_output_file_header()
        logger.info(
            (
                "Set-up MCMC glitch search with {} glitches for model {}" " on data {}"
            ).format(self.nglitch, self.label, self.sftfilepattern)
        )
        self.pickle_path = os.path.join(self.outdir, self.label + "_saved_data.p")
        self._unpack_input_theta()
        self.ndim = len(self.theta_keys)
        if self.log10beta_min:
            self.betas = np.logspace(0, self.log10beta_min, self.ntemps)
        else:
            self.betas = None
        if self.clean and os.path.isfile(self.pickle_path):
            os.rename(self.pickle_path, self.pickle_path + ".old")

        self.old_data_is_okay_to_use = self._check_old_data_is_okay_to_use()
        self._log_input()
        self._set_likelihoodcoef()
        self.set_ephemeris_files(earth_ephem, sun_ephem)
        self.allowedMismatchFromSFTLength = allowedMismatchFromSFTLength

    def _set_likelihoodcoef(self):
        """Additional constant terms to turn a detection statistic into a likelihood.

        See MCMCSearch._set_likelihoodcoef for the base implementation.
        This method simply extends it in order to account for the increased number
        of segments due to the presence of glitches.
        """
        super()._set_likelihoodcoef()
        self.likelihoodcoef *= self.nglitch + 1

    def _initiate_search_object(self):
        logger.info("Setting up search object")
        search_ranges = self._get_search_ranges()
        self.search = core.SemiCoherentGlitchSearch(
            label=self.label,
            outdir=self.outdir,
            sftfilepattern=self.sftfilepattern,
            tref=self.tref,
            minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime,
            minCoverFreq=self.minCoverFreq,
            maxCoverFreq=self.maxCoverFreq,
            search_ranges=search_ranges,
            detectors=self.detectors,
            BSGL=self.BSGL,
            nglitch=self.nglitch,
            theta0_idx=self.theta0_idx,
            injectSources=self.injectSources,
            earth_ephem=self.earth_ephem,
            sun_ephem=self.sun_ephem,
            allowedMismatchFromSFTLength=self.allowedMismatchFromSFTLength,
        )
        if self.minStartTime is None:
            self.minStartTime = self.search.minStartTime
        if self.maxStartTime is None:
            self.maxStartTime = self.search.maxStartTime

    def _logp(self, theta_vals, theta_prior, theta_keys, search):
        if self.nglitch > 1:
            ts = (
                [self.minStartTime]
                + list(theta_vals[-self.nglitch :])
                + [self.maxStartTime]
            )
            if np.array_equal(ts, np.sort(ts)) is False:
                return -np.inf
            if any(np.diff(ts) < self.dtglitchmin):
                return -np.inf

        H = [
            self._generic_lnprior(**theta_prior[key])(p)
            for p, key in zip(theta_vals, theta_keys)
        ]
        return np.sum(H)

    def _logl(self, theta, search):
        in_theta = self._set_point_for_evaluation(theta)
        if self.nglitch > 1:
            ts = (
                [self.minStartTime] + list(theta[-self.nglitch :]) + [self.maxStartTime]
            )
            if np.array_equal(ts, np.sort(ts)) is False:
                return -np.inf
        # FIXME: BSGL case?
        twoF = search.get_semicoherent_nglitch_twoF(*in_theta)
        return twoF * self.likelihooddetstatmultiplier + self.likelihoodcoef

    def _unpack_input_theta(self):
        base_keys = ["F0", "F1", "F2", "Alpha", "Delta"]
        glitch_keys = ["delta_F0", "delta_F1", "tglitch"]
        full_glitch_keys = list(
            np.array([[gk] * self.nglitch for gk in glitch_keys]).flatten()
        )

        if "tglitch_0" in self.theta_prior:
            full_glitch_keys[-self.nglitch :] = [
                "tglitch_{}".format(i) for i in range(self.nglitch)
            ]
            full_glitch_keys[-2 * self.nglitch : -1 * self.nglitch] = [
                "delta_F1_{}".format(i) for i in range(self.nglitch)
            ]
            full_glitch_keys[-4 * self.nglitch : -2 * self.nglitch] = [
                "delta_F0_{}".format(i) for i in range(self.nglitch)
            ]
        full_theta_keys = base_keys + full_glitch_keys
        full_theta_keys_copy = copy.copy(full_theta_keys)

        full_glitch_symbols = list(
            np.array(
                [[gs] * self.nglitch for gs in self.glitch_symbol_dictionary]
            ).flatten()
        )
        full_theta_symbols = [
            self.symbol_dictionary[key] for key in base_keys
        ] + full_glitch_symbols
        self.theta_keys = []
        fixed_theta_dict = {}
        for key, val in self.theta_prior.items():
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
                    "Type {} of {} in theta not recognised".format(type(val), key)
                )
            if key in glitch_keys:
                for i in range(self.nglitch):
                    full_theta_keys_copy.pop(full_theta_keys_copy.index(key))
            else:
                full_theta_keys_copy.pop(full_theta_keys_copy.index(key))

        if len(full_theta_keys_copy) > 0:
            raise ValueError(
                ("Input dictionary `theta` is missing the" "following keys: {}").format(
                    full_theta_keys_copy
                )
            )

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

        self.output_keys = self.theta_keys.copy()
        self.output_keys.append("twoF")
        if self.BSGL:
            self.output_keys.append("log10BSGL")

    def _get_data_dictionary_to_save(self):
        d = dict(
            nsteps=self.nsteps,
            nwalkers=self.nwalkers,
            ntemps=self.ntemps,
            theta_keys=self.theta_keys,
            theta_prior=self.theta_prior,
            log10beta_min=self.log10beta_min,
            theta0_idx=self.theta0_idx,
            BSGL=self.BSGL,
            minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime,
        )
        return d

    def _apply_corrections_to_p0(self, p0):
        p0 = np.array(p0)
        if self.nglitch > 1:
            p0[:, :, -self.nglitch :] = np.sort(p0[:, :, -self.nglitch :], axis=2)
        return p0

    def plot_cumulative_max(self, savefig=True):
        """
        Override MCMCSearch.plot_cumulative_max implementation to deal with the
        split at glitches.

        Parameters
        ----------
        savefig: boolean
            included for consistency with core plot_twoF_cumulative() function.
            If true, save the figure in outdir.
            If false, return an axis object.
        """

        logger.info("Getting cumulative 2F")
        fig, ax = plt.subplots()
        d, maxtwoF = self.get_max_twoF()
        for key, val in self.theta_prior.items():
            if key not in d:
                d[key] = val

        if self.nglitch > 1:
            delta_F0s = [d["delta_F0_{}".format(i)] for i in range(self.nglitch)]
            delta_F0s.insert(self.theta0_idx, 0)
            delta_F0s = np.array(delta_F0s)
            delta_F0s[: self.theta0_idx] *= -1
            tglitches = [d["tglitch_{}".format(i)] for i in range(self.nglitch)]
        elif self.nglitch == 1:
            delta_F0s = [d["delta_F0"]]
            delta_F0s.insert(self.theta0_idx, 0)
            delta_F0s = np.array(delta_F0s)
            delta_F0s[: self.theta0_idx] *= -1
            tglitches = [d["tglitch"]]

        tboundaries = [self.minStartTime] + tglitches + [self.maxStartTime]

        for j in range(self.nglitch + 1):
            ts = tboundaries[j]
            te = tboundaries[j + 1]
            if (te - ts) / 86400 < 5:
                logger.info("Period too short to perform cumulative search")
                continue
            if j < self.theta0_idx:
                summed_deltaF0 = np.sum(delta_F0s[j : self.theta0_idx])
                F0_j = d["F0"] - summed_deltaF0
                actual_ts, taus, twoFs = self.search.calculate_twoF_cumulative(
                    F0_j,
                    F1=d["F1"],
                    F2=d["F2"],
                    Alpha=d["Alpha"],
                    Delta=d["Delta"],
                    tstart=ts,
                    tend=te,
                )

            elif j >= self.theta0_idx:
                summed_deltaF0 = np.sum(delta_F0s[self.theta0_idx : j + 1])
                F0_j = d["F0"] + summed_deltaF0
                actual_ts, taus, twoFs = self.search.calculate_twoF_cumulative(
                    F0_j,
                    F1=d["F1"],
                    F2=d["F2"],
                    Alpha=d["Alpha"],
                    Delta=d["Delta"],
                    tstart=ts,
                    tend=te,
                )
            ax.plot(actual_ts + taus, twoFs)

        ax.set_xlabel("GPS time")
        if savefig:
            fig.savefig(os.path.join(self.outdir, self.label + "_twoFcumulative.png"))
            plt.close(fig)
        return ax

    def _get_savetxt_fmt_dict(self):
        fmt_dict = utils.get_doppler_params_output_format(self.theta_keys)
        if "tglitch" in self.theta_keys:
            fmt_dict["tglitch"] = "%d"
        if "delta_F0" in self.theta_keys:
            fmt_dict["delta_F0"] = "%.16g"
        if "delta_F1" in self.theta_keys:
            fmt_dict["delta_F1"] = "%.16g"
        fmt_dict["twoF"] = "%.9g"
        if self.BSGL:
            fmt_dict["log10BSGL"] = "%.9g"
        return fmt_dict


class MCMCSemiCoherentSearch(MCMCSearch):
    """MCMC search for a signal using the semicoherent ComputeFstat.

    Evaluates the semicoherent F-statistic acros a parameter space region
    corresponding to an isolated/binary-modulated CW signal.

    See MCMCSearch for a list of additional parameters, here we list only the additional
    init parameters of this class.
    """

    def __init__(
        self,
        theta_prior,
        tref,
        label,
        outdir="data",
        minStartTime=None,
        maxStartTime=None,
        sftfilepattern=None,
        detectors=None,
        nsteps=[100, 100],
        nwalkers=100,
        ntemps=1,
        log10beta_min=-5,
        theta_initial=None,
        rhohatmax=1000,
        binary=False,
        BSGL=False,
        SSBprec=None,
        RngMedWindow=None,
        minCoverFreq=None,
        maxCoverFreq=None,
        injectSources=None,
        assumeSqrtSX=None,
        nsegs=None,
        earth_ephem=None,
        sun_ephem=None,
        allowedMismatchFromSFTLength=None,
        clean=False,
    ):
        """
        Parameters
        ----------
        nsegs: int
            The number of segments into which the input datastream will be devided.
            Coherence time is computed internally as (maxStartTime - minStarTime) / nsegs.
        """
        self._set_init_params_dict(locals())
        self.theta_prior = theta_prior
        self.tref = tref
        self.label = label
        self.outdir = outdir
        self.minStartTime = minStartTime
        self.maxStartTime = maxStartTime
        self.sftfilepattern = sftfilepattern
        self.detectors = detectors
        self.nsteps = nsteps
        self.nwalkers = nwalkers
        self.ntemps = ntemps
        self.log10beta_min = log10beta_min
        self.theta_initial = theta_initial
        self.rhohatmax = rhohatmax
        self.binary = binary
        self.BSGL = BSGL
        self.SSBprec = SSBprec
        self.RngMedWindow = RngMedWindow
        self.minCoverFreq = minCoverFreq
        self.maxCoverFreq = maxCoverFreq
        self.injectSources = injectSources
        self.assumeSqrtSX = assumeSqrtSX
        self.nsegs = nsegs
        self.set_ephemeris_files(earth_ephem, sun_ephem)
        self.allowedMismatchFromSFTLength = allowedMismatchFromSFTLength
        self.clean = clean

        os.makedirs(outdir, exist_ok=True)
        self.output_file_header = self.get_output_file_header()
        logger.info(
            ("Set-up MCMC semi-coherent search for model {} on data" "{}").format(
                self.label, self.sftfilepattern
            )
        )
        self.pickle_path = os.path.join(self.outdir, self.label + "_saved_data.p")
        self._unpack_input_theta()
        self.ndim = len(self.theta_keys)
        if self.log10beta_min:
            self.betas = np.logspace(0, self.log10beta_min, self.ntemps)
        else:
            self.betas = None
        if self.clean and os.path.isfile(self.pickle_path):
            os.rename(self.pickle_path, self.pickle_path + ".old")

        self._log_input()

        if self.nsegs:
            self._set_likelihoodcoef()
        else:
            logger.info("Value `nsegs` not yet provided")

    def _set_likelihoodcoef(self):
        """Additional constant terms to turn a detection statistic into a likelihood.

        See MCMCSearch._set_likelihoodcoef for the base implementation.
        This method simply extends it in order to account for the increased number
        of segments a semicoherent search works with.
        """
        super()._set_likelihoodcoef()
        self.likelihoodcoef *= self.nsegs

    def _get_data_dictionary_to_save(self):
        d = dict(
            nsteps=self.nsteps,
            nwalkers=self.nwalkers,
            ntemps=self.ntemps,
            theta_keys=self.theta_keys,
            theta_prior=self.theta_prior,
            log10beta_min=self.log10beta_min,
            BSGL=self.BSGL,
            nsegs=self.nsegs,
            minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime,
        )
        return d

    def _initiate_search_object(self):
        logger.info("Setting up search object")
        search_ranges = self._get_search_ranges()
        self.search = core.SemiCoherentSearch(
            label=self.label,
            outdir=self.outdir,
            tref=self.tref,
            nsegs=self.nsegs,
            sftfilepattern=self.sftfilepattern,
            binary=self.binary,
            BSGL=self.BSGL,
            minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime,
            minCoverFreq=self.minCoverFreq,
            maxCoverFreq=self.maxCoverFreq,
            search_ranges=search_ranges,
            detectors=self.detectors,
            injectSources=self.injectSources,
            assumeSqrtSX=self.assumeSqrtSX,
            earth_ephem=self.earth_ephem,
            sun_ephem=self.sun_ephem,
            allowedMismatchFromSFTLength=self.allowedMismatchFromSFTLength,
        )
        if self.minStartTime is None:
            self.minStartTime = self.search.minStartTime
        if self.maxStartTime is None:
            self.maxStartTime = self.search.maxStartTime

    def _logp(self, theta_vals, theta_prior, theta_keys, search):
        H = [
            self._generic_lnprior(**theta_prior[key])(p)
            for p, key in zip(theta_vals, theta_keys)
        ]
        return np.sum(H)


class MCMCFollowUpSearch(MCMCSemiCoherentSearch, core.DeprecatedClass):
    """Hierarchical follow-up procedure

    Executes MCMC runs with increasing coherence times in order to follow up a parameter space
    region. The main idea is to use an MCMC run to identify an interesting parameter space region
    to then zoom-in said region using a finer "effective resolution" by increasing the coherence time.
    See Ashton & Prix (PRD 97, 103020, 2018): https://arxiv.org/abs/1802.05450

    See MCMCSemiCoherentSearch for a list of additional parameters, here we list only the additional
    init parameters of this class.
    """

    def __init__(
        self,
        theta_prior,
        tref,
        label,
        outdir="data",
        minStartTime=None,
        maxStartTime=None,
        sftfilepattern=None,
        detectors=None,
        nsteps=[100, 100],
        nwalkers=100,
        ntemps=1,
        log10beta_min=-5,
        theta_initial=None,
        rhohatmax=1000,
        binary=False,
        BSGL=False,
        SSBprec=None,
        RngMedWindow=None,
        minCoverFreq=None,
        maxCoverFreq=None,
        injectSources=None,
        assumeSqrtSX=None,
        earth_ephem=None,
        sun_ephem=None,
        allowedMismatchFromSFTLength=None,
        clean=False,
    ):
        self._set_init_params_dict(locals())
        self.theta_prior = theta_prior
        self.tref = tref
        self.label = label
        self.outdir = outdir
        self.minStartTime = minStartTime
        self.maxStartTime = maxStartTime
        self.sftfilepattern = sftfilepattern
        self.detectors = detectors
        self.nsteps = nsteps
        self.nwalkers = nwalkers
        self.ntemps = ntemps
        self.log10beta_min = log10beta_min
        self.theta_initial = theta_initial
        self.rhohatmax = rhohatmax
        self.binary = binary
        self.BSGL = BSGL
        self.SSBprec = SSBprec
        self.RngMedWindow = RngMedWindow
        self.minCoverFreq = minCoverFreq
        self.maxCoverFreq = maxCoverFreq
        self.injectSources = injectSources
        self.assumeSqrtSX = assumeSqrtSX
        self.nsegs = None
        self.set_ephemeris_files(earth_ephem, sun_ephem)
        self.allowedMismatchFromSFTLength = allowedMismatchFromSFTLength
        self.clean = clean

        os.makedirs(outdir, exist_ok=True)
        self.output_file_header = self.get_output_file_header()
        logger.info(
            ("Set-up MCMC semi-coherent search for model {} on data" "{}").format(
                self.label, self.sftfilepattern
            )
        )
        self.pickle_path = os.path.join(self.outdir, self.label + "_saved_data.p")
        self._unpack_input_theta()
        self.ndim = len(self.theta_keys)
        if self.log10beta_min:
            self.betas = np.logspace(0, self.log10beta_min, self.ntemps)
        else:
            self.betas = None
        if self.clean and os.path.isfile(self.pickle_path):
            os.rename(self.pickle_path, self.pickle_path + ".old")

        self._log_input()

        if self.nsegs:
            self._set_likelihoodcoef()
        else:
            logger.info("Value `nsegs` not yet provided")

    def _get_data_dictionary_to_save(self):
        d = dict(
            nwalkers=self.nwalkers,
            ntemps=self.ntemps,
            theta_keys=self.theta_keys,
            theta_prior=self.theta_prior,
            log10beta_min=self.log10beta_min,
            BSGL=self.BSGL,
            minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime,
            run_setup=self.run_setup,
        )
        return d

    def run(
        self,
        run_setup=None,
        proposal_scale_factor=2,
        NstarMax=10,
        Nsegs0=None,
        save_pickle=True,
        export_samples=True,
        save_loudest=True,
        plot_walkers=True,
        walker_plot_args=None,
        log_table=True,
        gen_tex_table=True,
        window=50,
    ):
        """Run the follow-up with the given run_setup.

        See MCMCSearch.run's docstring for a description of the remainder arguments.

        Parameters
        ----------
        run_setup, log_table, gen_tex_table:
            See `MCMCFollowUpSearch.init_run_setup`.
        NstarMax, Nsegs0:
            See `pyfstat.optimal_setup_functions.get_optimal_setup`.
        """

        self.nsegs = 1
        self._set_likelihoodcoef()
        self._initiate_search_object()
        run_setup = self.init_run_setup(
            run_setup,
            NstarMax=NstarMax,
            Nsegs0=Nsegs0,
            log_table=log_table,
            gen_tex_table=gen_tex_table,
        )
        self.run_setup = run_setup
        self._estimate_run_time()

        walker_plot_args = walker_plot_args or {}

        self.old_data_is_okay_to_use = self._check_old_data_is_okay_to_use()
        if self.old_data_is_okay_to_use is True:
            logger.warning("Using saved data from {}".format(self.pickle_path))
            d = self.get_saved_data_dictionary()
            self.samples = d["samples"]
            self.lnprobs = d["lnprobs"]
            self.lnlikes = d["lnlikes"]
            self.all_lnlikelihood = d["all_lnlikelihood"]
            self.chain = d["chain"]
            self.nsegs = run_setup[-1][1]
            return

        nsteps_total = 0
        for j, ((nburn, nprod), nseg, reset_p0) in enumerate(run_setup):
            p0 = self._get_p0_per_stage(reset_p0)

            self.nsegs = nseg
            self._set_likelihoodcoef()
            self.search.nsegs = nseg
            self._update_search_object()
            self.search.init_semicoherent_parameters()
            self.sampler = PTSampler(
                ntemps=self.ntemps,
                nwalkers=self.nwalkers,
                dim=self.ndim,
                logl=self._logl,
                logp=self._logp,
                logpargs=(self.theta_prior, self.theta_keys, self.search),
                loglargs=(self.search,),
                betas=self.betas,
                a=proposal_scale_factor,
            )

            Tcoh = (self.maxStartTime - self.minStartTime) / nseg / 86400.0
            logger.info(
                (
                    "Running {}/{} with {} steps and {} nsegs " "(Tcoh={:1.2f} days)"
                ).format(j + 1, len(run_setup), (nburn, nprod), nseg, Tcoh)
            )
            self._run_sampler(p0, nburn=nburn, nprod=nprod, window=window)
            logger.info(
                "Max detection statistic of run was {}".format(
                    np.max(self.sampler.loglikelihood)
                )
            )

            if plot_walkers:
                try:
                    walker_fig, walker_axes = self._plot_walkers(
                        nprod=nprod, xoffset=nsteps_total, **walker_plot_args
                    )
                    for ax in walker_axes[: self.ndim]:
                        ax.axvline(nsteps_total, color="k", ls="--", lw=0.25)
                except Exception as e:
                    logger.warning("Failed to plot walkers due to Error {}".format(e))

            nsteps_total += nburn + nprod

        if plot_walkers:
            nstep_list = np.array(
                [el[0][0] for el in run_setup] + [run_setup[-1][0][1]]
            )
            mids = np.cumsum(nstep_list) - nstep_list / 2
            mid_labels = ["{:1.0f}".format(i) for i in np.arange(0, len(mids) - 1)]
            mid_labels += ["Production"]
            for ax in walker_axes[: self.ndim]:
                axy = ax.twiny()
                axy.tick_params(pad=-10, direction="in", axis="x", which="major")
                axy.minorticks_off()
                axy.set_xlim(ax.get_xlim())
                axy.set_xticks(mids)
                axy.set_xticklabels(mid_labels)

        samples = self.sampler.chain[0, :, nburn:, :].reshape((-1, self.ndim))
        lnprobs = self.sampler.logprobability[0, :, nburn:].reshape((-1))
        lnlikes = self.sampler.loglikelihood[0, :, nburn:].reshape((-1))
        all_lnlikelihood = self.sampler.loglikelihood
        self.samples = samples
        self.lnprobs = lnprobs
        self.lnlikes = lnlikes
        self.all_lnlikelihood = all_lnlikelihood
        if save_pickle:
            self._pickle_data(samples, lnprobs, lnlikes, all_lnlikelihood)
        if export_samples:
            self.export_samples_to_disk()
        if save_loudest:
            self.generate_loudest()

        if plot_walkers:
            try:
                walker_fig.tight_layout()
            except Exception as e:
                logger.warning(
                    "Failed to set tight layout for walkers plot due to Error {}".format(
                        e
                    )
                )
            if (walker_plot_args.get("fig") is not None) and (
                walker_plot_args.get("axes") is not None
            ):
                self.walker_fig = walker_fig
                self.walker_axes = walker_axes
            else:
                try:
                    walker_fig.savefig(
                        os.path.join(self.outdir, self.label + "_walkers.png")
                    )
                    plt.close(walker_fig)
                except Exception as e:
                    logger.warning(
                        "Failed to save walker plots due to Error {}".format(e)
                    )

    def _update_search_object(self):
        logger.info("Update search object")
        self.search.init_computefstatistic()

    def init_run_setup(
        self,
        run_setup=None,
        NstarMax=1000,
        Nsegs0=None,
        log_table=True,
        gen_tex_table=True,
        setup_only=False,
        no_template_counting=True,
    ):
        """
        Initialize the setup of the follow-up run computing the required quantities fro, NstarMax
        and Nsegs0.

        Parameters
        ----------
        NstarMax, Nsegs0: int
            Required parameters to create a new follow-up setup.
            See `pyfstat.optimal_setup_functions.get_optimal_setup`.
        run_setup: optional
            If None, a new setup will be created from NstarMax and Nsegs0.
            Use `MCMCFollowUpSearch.read_setup_input_file` to read a previous
            setup file.
        log_table: bool
            Log follow-up setup using `logger.info` as a table.
        gen_tex_table: bool
            Dump follow-up setup into a text file as a tex table.
            File is constructed as `os.path.join(self.outdir, self.label + "_run_setup.tex")`.

        Returns
        -------
        run_setup: list
           List containing the setup of the follow-up run.
        """
        if run_setup is None and Nsegs0 is None:
            raise ValueError(
                "You must either specify the run_setup, or Nsegs0 and NStarMax"
                " from which the optimal run_setup can be estimated"
            )
        if run_setup is None:
            logger.info("No run_setup provided")

            run_setup_input_file = os.path.join(
                self.outdir, self.label + "_run_setup.p"
            )

            if os.path.isfile(run_setup_input_file):
                logger.info(
                    "Checking old setup input file {}".format(run_setup_input_file)
                )
                old_setup = self.read_setup_input_file(run_setup_input_file)
                if self._check_old_run_setup(
                    old_setup,
                    NstarMax=NstarMax,
                    Nsegs0=Nsegs0,
                    theta_prior=self.theta_prior,
                ):
                    logger.info(
                        "Using old setup with NstarMax={}, Nsegs0={}".format(
                            NstarMax, Nsegs0
                        )
                    )
                    nsegs_vals = old_setup["nsegs_vals"]
                    Nstar_vals = old_setup["Nstar_vals"]
                    generate_setup = False
                else:
                    generate_setup = True
            else:
                generate_setup = True

            if generate_setup:
                nsegs_vals, Nstar_vals = optimal_setup_functions.get_optimal_setup(
                    NstarMax,
                    Nsegs0,
                    self.tref,
                    self.minStartTime,
                    self.maxStartTime,
                    self.theta_prior,
                    self.search.detector_names,
                )
                self._write_setup_input_file(
                    run_setup_input_file,
                    NstarMax,
                    Nsegs0,
                    nsegs_vals,
                    Nstar_vals,
                    self.theta_prior,
                )

            run_setup = [
                ((self.nsteps[0], 0), nsegs, False) for nsegs in nsegs_vals[:-1]
            ]
            run_setup.append(((self.nsteps[0], self.nsteps[1]), nsegs_vals[-1], False))

        else:
            logger.info("Calculating the number of templates for this setup")
            Nstar_vals = []
            for i, rs in enumerate(run_setup):
                rs = list(rs)
                if len(rs) == 2:
                    rs.append(False)
                if np.shape(rs[0]) == ():
                    rs[0] = (rs[0], 0)
                run_setup[i] = rs

                if no_template_counting:
                    Nstar_vals.append([1, 1, 1])
                else:
                    Nstar = optimal_setup_functions.get_Nstar_estimate(
                        rs[1],
                        self.tref,
                        self.minStartTime,
                        self.maxStartTime,
                        self.theta_prior,
                        self.search.detector_names,
                    )
                    Nstar_vals.append(Nstar)

        if log_table:
            logger.info("Using run-setup as follows:")
            logger.info("Stage | nburn | nprod | nsegs | Tcoh d | resetp0 | Nstar")
            for i, rs in enumerate(run_setup):
                Tcoh = (self.maxStartTime - self.minStartTime) / rs[1] / 86400
                if Nstar_vals[i] is None:
                    vtext = "N/A"
                else:
                    vtext = "{:0.3e}".format(int(Nstar_vals[i]))
                logger.info(
                    "{} | {} | {} | {} | {} | {} | {}".format(
                        str(i).ljust(5),
                        str(rs[0][0]).ljust(5),
                        str(rs[0][1]).ljust(5),
                        str(rs[1]).ljust(5),
                        "{:6.1f}".format(Tcoh),
                        str(rs[2]).ljust(7),
                        vtext,
                    )
                )

        if gen_tex_table:
            filename = os.path.join(self.outdir, self.label + "_run_setup.tex")
            with open(filename, "w+") as f:
                f.write(r"\begin{tabular}{c|ccc}" + "\n")
                f.write(
                    r"Stage & $N_\mathrm{seg}$ &"
                    r"$T_\mathrm{coh}^{\rm days}$ &"
                    r"$\mathcal{N}^*(\Nseg^{(\ell)}, \Delta\mathbf{\lambda}^{(0)})$ \\ \hline"
                    "\n"
                )
                for i, rs in enumerate(run_setup):
                    Tcoh = float(self.maxStartTime - self.minStartTime) / rs[1] / 86400
                    line = r"{} & {} & {} & {} \\" + "\n"
                    if Nstar_vals[i] is None:
                        Nstar = "N/A"
                    else:
                        Nstar = Nstar_vals[i]
                    line = line.format(
                        i,
                        rs[1],
                        "{:1.1f}".format(Tcoh),
                        utils.texify_float(Nstar),
                    )
                    f.write(line)
                f.write(r"\end{tabular}" + "\n")

        if setup_only:
            logger.info("Exit as requested by setup_only flag")
            sys.exit()
        else:
            return run_setup

    def read_setup_input_file(self, run_setup_input_file):
        with open(run_setup_input_file, "rb+") as f:
            d = pickle.load(f)
        return d

    def _write_setup_input_file(
        self,
        run_setup_input_file,
        NstarMax,
        Nsegs0,
        nsegs_vals,
        Nstar_vals,
        theta_prior,
    ):
        d = dict(
            NstarMax=NstarMax,
            Nsegs0=Nsegs0,
            nsegs_vals=nsegs_vals,
            theta_prior=theta_prior,
            Nstar_vals=Nstar_vals,
        )
        with open(run_setup_input_file, "wb+") as f:
            pickle.dump(d, f)

    def _check_old_run_setup(self, old_setup, **kwargs):
        try:
            truths = [val == old_setup[key] for key, val in kwargs.items()]
            if all(truths):
                return True
            else:
                logger.info("Old setup doesn't match one of NstarMax, Nsegs0 or prior")
        except KeyError as e:
            logger.info("Error found when comparing with old setup: {}".format(e))
            return False

    def _get_p0_per_stage(self, reset_p0=False):
        """Returns new initial positions for walkers at each stage of the ladder"""
        if not hasattr(self, "sampler"):  # must be stage 0
            p0 = self._generate_initial_p0()
            p0 = self._apply_corrections_to_p0(p0)
        elif reset_p0:
            p0 = self._get_new_p0(self.sampler)
            p0 = self._apply_corrections_to_p0(p0)
            # self._check_initial_points(p0)
        else:
            p0 = self.sampler.chain[:, :, -1, :]
        return p0


class MCMCTransientSearch(MCMCSearch):
    """MCMC search for a transient signal using ComputeFstat"""

    symbol_dictionary = dict(
        F0=r"$f$",
        F1=r"$\dot{f}$",
        F2=r"$\ddot{f}$",
        Alpha=r"$\alpha$",
        Delta=r"$\delta$",
        transient_tstart=r"$t_\mathrm{start}$",
        transient_duration=r"$\Delta T$",
    )
    """
        Key, val pairs of the parameters (`F0`, `F1`, ...), to LaTeX math
        symbols for plots
    """
    unit_dictionary = dict(
        F0=r"Hz",
        F1=r"Hz/s",
        F2=r"Hz/s$^2$",
        Alpha=r"rad",
        Delta=r"rad",
        transient_tstart=r"s",
        transient_duration=r"s",
    )
    """
        Key, val pairs of the parameters (`F0`, `F1`, ..., including glitch parameters),
        and the units (`Hz`, `Hz/s`, ...).
    """
    transform_dictionary = dict(
        transient_duration={
            "multiplier": 1 / 86400.0,
            "unit": "day",
            "symbol": "Transient duration",
        },
        transient_tstart={
            "multiplier": 1 / 86400.0,
            "subtractor": "minStartTime",
            "unit": "day",
            "label": "Transient start-time \n days after minStartTime",
        },
    )
    """
        Key, val pairs of the parameters (`F0`, `F1`, ...), where the key is
        itself a dictionary which can item `multiplier`, `subtractor`, or
        `unit` by which to transform by and update the units.
    """

    def _initiate_search_object(self):
        logger.info("Setting up search object")
        if not self.transientWindowType:
            self.transientWindowType = "rect"
        search_ranges = self._get_search_ranges()
        self.search = core.ComputeFstat(
            tref=self.tref,
            sftfilepattern=self.sftfilepattern,
            minCoverFreq=self.minCoverFreq,
            maxCoverFreq=self.maxCoverFreq,
            search_ranges=search_ranges,
            detectors=self.detectors,
            transientWindowType=self.transientWindowType,
            minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime,
            BSGL=self.BSGL,
            BtSG=self.BtSG,
            binary=self.binary,
            injectSources=self.injectSources,
            tCWFstatMapVersion=self.tCWFstatMapVersion,
            earth_ephem=self.earth_ephem,
            sun_ephem=self.sun_ephem,
            allowedMismatchFromSFTLength=self.allowedMismatchFromSFTLength,
        )
        if self.minStartTime is None:
            self.minStartTime = self.search.minStartTime
        if self.maxStartTime is None:
            self.maxStartTime = self.search.maxStartTime

    def _set_point_for_evaluation(self, theta):
        """Combines fixed and variable parameters to form a valid evaluation point.

        Parameters
        ----------
        theta: list or np.ndarray
            The sampled (variable) parameters.
        Returns
        -------
        p: dict
            The full parameter space point as a dictionary.
            (different from base MCMCSearch class!)
        """
        p = {
            key: self.fixed_theta[k]
            for k, key in enumerate(self.full_theta_keys)
            if "transient" not in key
        }
        p.update(
            {
                key: theta[k]
                for k, key in enumerate(self.theta_keys)
                if "transient" not in key
            }
        )
        # FIXME: this can be simplified when changing all theta lists to dicts
        if "transient_tstart" in self.theta_keys:
            p["tstart"] = theta[self.theta_keys.index("transient_tstart")]
        else:
            p["tstart"] = self.fixed_theta[
                self.full_theta_keys.index("transient_tstart")
            ]
        if "transient_duration" in self.theta_keys:
            tau = theta[self.theta_keys.index("transient_duration")]
        else:
            tau = self.fixed_theta[self.full_theta_keys.index("transient_duration")]
        p["tend"] = p["tstart"] + tau
        return p

    def _logl(self, theta, search):
        in_theta = self._set_point_for_evaluation(theta)
        if in_theta["tend"] > self.maxStartTime:
            return -np.inf
        detstat = search.get_det_stat(**in_theta)
        return detstat * self.likelihooddetstatmultiplier + self.likelihoodcoef

    def _unpack_input_theta(self):
        self.full_theta_keys = ["F0", "F1", "F2", "Alpha", "Delta"]
        if self.binary:
            self.full_theta_keys += ["asini", "period", "ecc", "tp", "argp"]
        self.full_theta_keys += ["transient_tstart", "transient_duration"]
        full_theta_keys_copy = copy.copy(self.full_theta_keys)

        self.theta_keys = []
        fixed_theta_dict = {}
        for key, val in self.theta_prior.items():
            if type(val) is dict:
                fixed_theta_dict[key] = 0
                self.theta_keys.append(key)
            elif type(val) in [float, int, np.float64]:
                fixed_theta_dict[key] = val
            else:
                raise ValueError(
                    "Type {} of {} in theta not recognised".format(type(val), key)
                )
            full_theta_keys_copy.pop(full_theta_keys_copy.index(key))

        if len(full_theta_keys_copy) > 0:
            raise ValueError(
                ("Input dictionary `theta` is missing the" "following keys: {}").format(
                    full_theta_keys_copy
                )
            )

        self.fixed_theta = [fixed_theta_dict[key] for key in self.full_theta_keys]
        self.theta_idxs = [self.full_theta_keys.index(k) for k in self.theta_keys]
        self.theta_symbols = [self.symbol_dictionary[k] for k in self.theta_keys]

        idxs = np.argsort(self.theta_idxs)
        self.theta_idxs = [self.theta_idxs[i] for i in idxs]
        self.theta_symbols = [self.theta_symbols[i] for i in idxs]
        self.theta_keys = [self.theta_keys[i] for i in idxs]

        self.output_keys = self.theta_keys.copy()
        self.output_keys.append("twoF")
        if self.BSGL:
            self.output_keys.append("log10BSGL")

    def _get_savetxt_fmt_dict(self):
        fmt_dict = utils.get_doppler_params_output_format(self.theta_keys)
        if "transient_tstart" in self.theta_keys:
            fmt_dict["transient_tstart"] = "%d"
        if "transient_duration" in self.theta_keys:
            fmt_dict["transient_duration"] = "%d"
        fmt_dict["twoF"] = "%.9g"
        if self.BSGL:
            fmt_dict["log10BSGL"] = "%.9g"
        return fmt_dict
