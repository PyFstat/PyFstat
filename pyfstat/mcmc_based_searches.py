""" Searches using MCMC-based methods """

import sys
import os
import copy
import logging
from collections import OrderedDict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import emcee
import corner
import dill as pickle

import core
from core import tqdm, args, earth_ephem, sun_ephem
from optimal_setup_functions import get_V_estimate
from optimal_setup_functions import get_optimal_setup
import helper_functions


class MCMCSearch(core.BaseSearchClass):
    """ MCMC search using ComputeFstat"""

    symbol_dictionary = dict(
        F0='$f$', F1='$\dot{f}$', F2='$\ddot{f}$', alpha=r'$\alpha$',
        delta='$\delta$')
    unit_dictionary = dict(
        F0='Hz', F1='Hz/s', F2='Hz/s$^2$', alpha=r'rad', delta='rad')
    rescale_dictionary = {}


    @helper_functions.initializer
    def __init__(self, label, outdir, theta_prior, tref, minStartTime,
                 maxStartTime, sftfilepath=None, nsteps=[100, 100],
                 nwalkers=100, ntemps=1, log10temperature_min=-5,
                 theta_initial=None, scatter_val=1e-10,
                 binary=False, BSGL=False, minCoverFreq=None,
                 maxCoverFreq=None, detectors=None, earth_ephem=None,
                 sun_ephem=None, injectSources=None, assumeSqrtSX=None):
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
        tref, minStartTime, maxStartTime: int
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
        detectors: str
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

        if os.path.isdir(outdir) is False:
            os.mkdir(outdir)
        self.add_log_file()
        logging.info(
            'Set-up MCMC search for model {} on data {}'.format(
                self.label, self.sftfilepath))
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

        self.log_input()

    def log_input(self):
        logging.info('theta_prior = {}'.format(self.theta_prior))
        logging.info('nwalkers={}'.format(self.nwalkers))
        logging.info('scatter_val = {}'.format(self.scatter_val))
        logging.info('nsteps = {}'.format(self.nsteps))
        logging.info('ntemps = {}'.format(self.ntemps))
        logging.info('log10temperature_min = {}'.format(
            self.log10temperature_min))

    def initiate_search_object(self):
        logging.info('Setting up search object')
        self.search = core.ComputeFstat(
            tref=self.tref, sftfilepath=self.sftfilepath,
            minCoverFreq=self.minCoverFreq, maxCoverFreq=self.maxCoverFreq,
            earth_ephem=self.earth_ephem, sun_ephem=self.sun_ephem,
            detectors=self.detectors, BSGL=self.BSGL, transient=False,
            minStartTime=self.minStartTime, maxStartTime=self.maxStartTime,
            binary=self.binary, injectSources=self.injectSources,
            assumeSqrtSX=self.assumeSqrtSX)

    def logp(self, theta_vals, theta_prior, theta_keys, search):
        H = [self.generic_lnprior(**theta_prior[key])(p) for p, key in
             zip(theta_vals, theta_keys)]
        return np.sum(H)

    def logl(self, theta, search):
        for j, theta_i in enumerate(self.theta_idxs):
            self.fixed_theta[theta_i] = theta[j]
        FS = search.compute_fullycoherent_det_stat_single_point(
            *self.fixed_theta)
        return FS

    def unpack_input_theta(self):
        full_theta_keys = ['F0', 'F1', 'F2', 'Alpha', 'Delta']
        if self.binary:
            full_theta_keys += [
                'asini', 'period', 'ecc', 'tp', 'argp']
        full_theta_keys_copy = copy.copy(full_theta_keys)

        full_theta_symbols = ['$f$', '$\dot{f}$', '$\ddot{f}$', r'$\alpha$',
                              r'$\delta$']
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

    def OLD_run_sampler_with_progress_bar(self, sampler, ns, p0):
        for result in tqdm(sampler.sample(p0, iterations=ns), total=ns):
            pass
        return sampler

    def setup_convergence_testing(
            self, convergence_period=10, convergence_length=10,
            convergence_burnin_fraction=0.25, convergence_threshold_number=10,
            convergence_threshold=1.2, convergence_prod_threshold=2,
            convergence_plot_upper_lim=2):
        """
        If called, convergence testing is used during the MCMC simulation

        This uses the Gelmanr-Rubin statistic based on the ratio of between and
        within walkers variance. The original statistic was developed for
        multiple (independent) MCMC simulations, in this context we simply use
        the walkers

        Parameters
        ----------
        convergence_period: int
            period (in number of steps) at which to test convergence
        convergence_length: int
            number of steps to use in testing convergence - this should be
            large enough to measure the variance, but if it is too long
            this will result in incorect early convergence tests
        convergence_burnin_fraction: float [0, 1]
            the fraction of the burn-in period after which to start testing
        convergence_threshold_number: int
            the number of consecutive times where the test passes after which
            to break the burn-in and go to production
        convergence_threshold: float
            the threshold to use in diagnosing convergence. Gelman & Rubin
            recomend a value of 1.2, 1.1 for strict convergence
        convergence_prod_threshold: float
            the threshold to test the production values with
        convergence_plot_upper_lim: float
            the upper limit to use in the diagnostic plot
        """

        if convergence_length > convergence_period:
            raise ValueError('convergence_length must be < convergence_period')
        logging.info('Setting up convergence testing')
        self.convergence_length = convergence_length
        self.convergence_period = convergence_period
        self.convergence_burnin_fraction = convergence_burnin_fraction
        self.convergence_prod_threshold = convergence_prod_threshold
        self.convergence_diagnostic = []
        self.convergence_diagnosticx = []
        self.convergence_threshold_number = convergence_threshold_number
        self.convergence_threshold = convergence_threshold
        self.convergence_number = 0
        self.convergence_plot_upper_lim = convergence_plot_upper_lim

    def get_convergence_statistic(self, i, sampler):
        s = sampler.chain[0, :, i-self.convergence_length+1:i+1, :]
        within_std = np.mean(np.var(s, axis=1), axis=0)
        per_walker_mean = np.mean(s, axis=1)
        mean = np.mean(per_walker_mean, axis=0)
        between_std = np.sqrt(np.mean((per_walker_mean-mean)**2, axis=0))
        W = within_std
        B_over_n = between_std**2 / self.convergence_period
        Vhat = ((self.convergence_period-1.)/self.convergence_period * W
                + B_over_n + B_over_n / float(self.nwalkers))
        c = np.sqrt(Vhat/W)
        self.convergence_diagnostic.append(c)
        self.convergence_diagnosticx.append(i - self.convergence_length/2)
        return c

    def burnin_convergence_test(self, i, sampler, nburn):
        if i < self.convergence_burnin_fraction*nburn:
            return False
        if np.mod(i+1, self.convergence_period) != 0:
            return False
        c = self.get_convergence_statistic(i, sampler)
        if np.all(c < self.convergence_threshold):
            self.convergence_number += 1
        else:
            self.convergence_number = 0
        return self.convergence_number > self.convergence_threshold_number

    def prod_convergence_test(self, i, sampler, nburn):
        testA = i > nburn + self.convergence_length
        testB = np.mod(i+1, self.convergence_period) == 0
        if testA and testB:
            self.get_convergence_statistic(i, sampler)

    def check_production_convergence(self, k):
        bools = np.any(
            np.array(self.convergence_diagnostic)[k:, :]
            > self.convergence_prod_threshold, axis=1)
        if np.any(bools):
            logging.warning(
                '{} convergence tests in the production run of {} failed'
                .format(np.sum(bools), len(bools)))

    def run_sampler(self, sampler, p0, nprod=0, nburn=0):
        if hasattr(self, 'convergence_period'):
            logging.info('Running {} burn-in steps with convergence testing'
                         .format(nburn))
            iterator = tqdm(sampler.sample(p0, iterations=nburn), total=nburn)
            for i, output in enumerate(iterator):
                if self.burnin_convergence_test(i, sampler, nburn):
                    logging.info(
                        'Converged at {} before max number {} of steps reached'
                        .format(i, nburn))
                    self.convergence_idx = i
                    break
            iterator.close()
            logging.info('Running {} production steps'.format(nprod))
            j = nburn
            k = len(self.convergence_diagnostic)
            for result in tqdm(sampler.sample(output[0], iterations=nprod),
                               total=nprod):
                self.prod_convergence_test(j, sampler, nburn)
                j += 1
            self.check_production_convergence(k)
            return sampler
        else:
            for result in tqdm(sampler.sample(p0, iterations=nburn+nprod),
                               total=nburn+nprod):
                pass
            return sampler

    def run(self, proposal_scale_factor=2, create_plots=True, **kwargs):

        self.old_data_is_okay_to_use = self.check_old_data_is_okay_to_use()
        if self.old_data_is_okay_to_use is True:
            logging.warning('Using saved data from {}'.format(
                self.pickle_path))
            d = self.get_saved_data()
            self.sampler = d['sampler']
            self.samples = d['samples']
            self.lnprobs = d['lnprobs']
            self.lnlikes = d['lnlikes']
            return

        self.initiate_search_object()

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
                j, ninit_steps, n))
            sampler = self.run_sampler(sampler, p0, nburn=n)
            logging.info("Mean acceptance fraction: {}"
                         .format(np.mean(sampler.acceptance_fraction, axis=1)))
            if self.ntemps > 1:
                logging.info("Tswap acceptance fraction: {}"
                             .format(sampler.tswap_acceptance_fraction))
            if create_plots:
                fig, axes = self.plot_walkers(sampler,
                                              symbols=self.theta_symbols,
                                              **kwargs)
                fig.tight_layout()
                fig.savefig('{}/{}_init_{}_walkers.png'.format(
                    self.outdir, self.label, j), dpi=400)

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
        sampler = self.run_sampler(sampler, p0, nburn=nburn, nprod=nprod)
        logging.info("Mean acceptance fraction: {}"
                     .format(np.mean(sampler.acceptance_fraction, axis=1)))
        if self.ntemps > 1:
            logging.info("Tswap acceptance fraction: {}"
                         .format(sampler.tswap_acceptance_fraction))

        if create_plots:
            fig, axes = self.plot_walkers(sampler, symbols=self.theta_symbols,
                                          nprod=nprod, **kwargs)
            fig.tight_layout()
            fig.savefig('{}/{}_walkers.png'.format(self.outdir, self.label),
                        dpi=200)

        samples = sampler.chain[0, :, nburn:, :].reshape((-1, self.ndim))
        lnprobs = sampler.lnprobability[0, :, nburn:].reshape((-1))
        lnlikes = sampler.lnlikelihood[0, :, nburn:].reshape((-1))
        self.sampler = sampler
        self.samples = samples
        self.lnprobs = lnprobs
        self.lnlikes = lnlikes
        self.save_data(sampler, samples, lnprobs, lnlikes)

    def get_rescale_multiplier_for_key(self, key):
        """ Get the rescale multiplier from the rescale_dictionary

        Can either be a float, a string (in which case it is interpretted as
        a attribute of the MCMCSearch class, e.g. minStartTime, or non-existent
        in which case 0 is returned
        """
        if key not in self.rescale_dictionary:
            return 1

        if 'multiplier' in self.rescale_dictionary[key]:
            val = self.rescale_dictionary[key]['multiplier']
            if type(val) == str:
                if hasattr(self, val):
                    multiplier = getattr(
                        self, self.rescale_dictionary[key]['multiplier'])
                else:
                    raise ValueError(
                        "multiplier {} not a class attribute".format(val))
            else:
                multiplier = val
        else:
            multiplier = 1
        return multiplier

    def get_rescale_subtractor_for_key(self, key):
        """ Get the rescale subtractor from the rescale_dictionary

        Can either be a float, a string (in which case it is interpretted as
        a attribute of the MCMCSearch class, e.g. minStartTime, or non-existent
        in which case 0 is returned
        """
        if key not in self.rescale_dictionary:
            return 0

        if 'subtractor' in self.rescale_dictionary[key]:
            val = self.rescale_dictionary[key]['subtractor']
            if type(val) == str:
                if hasattr(self, val):
                    subtractor = getattr(
                        self, self.rescale_dictionary[key]['subtractor'])
                else:
                    raise ValueError(
                        "subtractor {} not a class attribute".format(val))
            else:
                subtractor = val
        else:
            subtractor = 0
        return subtractor

    def scale_samples(self, samples, theta_keys):
        """ Scale the samples using the rescale_dictionary """
        for key in theta_keys:
            if key in self.rescale_dictionary:
                idx = theta_keys.index(key)
                s = samples[:, idx]
                subtractor = self.get_rescale_subtractor_for_key(key)
                s = s - subtractor
                multiplier = self.get_rescale_multiplier_for_key(key)
                s *= multiplier
                samples[:, idx] = s

        return samples

    def get_labels(self):
        """ Combine the units, symbols and rescaling to give labels """

        labels = []
        for key in self.theta_keys:
            label = None
            s = self.symbol_dictionary[key]
            s.replace('_{glitch}', r'_\textrm{glitch}')
            u = self.unit_dictionary[key]
            if key in self.rescale_dictionary:
                if 'symbol' in self.rescale_dictionary[key]:
                    s = self.rescale_dictionary[key]['symbol']
                if 'label' in self.rescale_dictionary[key]:
                    label = self.rescale_dictionary[key]['label']
                if 'unit' in self.rescale_dictionary[key]:
                    u = self.rescale_dictionary[key]['unit']
            if label is None:
                label = '{} \n [{}]'.format(s, u)
            labels.append(label)
        return labels

    def plot_corner(self, figsize=(7, 7),  tglitch_ratio=False,
                    add_prior=False, nstds=None, label_offset=0.4,
                    dpi=300, rc_context={}, **kwargs):

        if self.ndim < 2:
            with plt.rc_context(rc_context):
                fig, ax = plt.subplots(figsize=figsize)
                ax.hist(self.samples, bins=50, histtype='stepfilled')
                ax.set_xlabel(self.theta_symbols[0])

            fig.savefig('{}/{}_corner.png'.format(
                self.outdir, self.label), dpi=dpi)
            return

        with plt.rc_context(rc_context):
            fig, axes = plt.subplots(self.ndim, self.ndim,
                                     figsize=figsize)

            samples_plt = copy.copy(self.samples)
            labels = self.get_labels()

            samples_plt = self.scale_samples(samples_plt, self.theta_keys)

            if tglitch_ratio:
                for j, k in enumerate(self.theta_keys):
                    if k == 'tglitch':
                        s = samples_plt[:, j]
                        samples_plt[:, j] = (
                            s - self.minStartTime)/(
                                self.maxStartTime - self.minStartTime)
                        labels[j] = r'$R_{\textrm{glitch}}$'

            if type(nstds) is int and 'range' not in kwargs:
                _range = []
                for j, s in enumerate(samples_plt.T):
                    median = np.median(s)
                    std = np.std(s)
                    _range.append((median - nstds*std, median + nstds*std))
            else:
                _range = None

            fig_triangle = corner.corner(samples_plt,
                                         labels=labels,
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
                self.add_prior_to_corner(axes, self.samples)

            fig_triangle.savefig('{}/{}_corner.png'.format(
                self.outdir, self.label), dpi=dpi)

    def add_prior_to_corner(self, axes, samples):
        for i, key in enumerate(self.theta_keys):
            ax = axes[i][i]
            xlim = ax.get_xlim()
            s = samples[:, i]
            prior = self.generic_lnprior(**self.theta_prior[key])
            x = np.linspace(s.min(), s.max(), 100)
            multiplier = self.get_rescale_multiplier_for_key(key)
            subtractor = self.get_rescale_subtractor_for_key(key)
            ax2 = ax.twinx()
            ax2.get_yaxis().set_visible(False)
            ax2.plot((x-subtractor)*multiplier, [prior(xi) for xi in x], '-r')
            ax2.set_xlim(xlim)

    def plot_prior_posterior(self, normal_stds=2):
        """ Plot the posterior in the context of the prior """
        fig, axes = plt.subplots(nrows=self.ndim, figsize=(8, 4*self.ndim))
        N = 1000
        from scipy.stats import gaussian_kde

        for i, (ax, key) in enumerate(zip(axes, self.theta_keys)):
            prior_dict = self.theta_prior[key]
            prior_func = self.generic_lnprior(**prior_dict)
            if prior_dict['type'] == 'unif':
                x = np.linspace(prior_dict['lower'], prior_dict['upper'], N)
                prior = prior_func(x)
                prior[0] = 0
                prior[-1] = 0
            elif prior_dict['type'] == 'norm':
                lower = prior_dict['loc'] - normal_stds * prior_dict['scale']
                upper = prior_dict['loc'] + normal_stds * prior_dict['scale']
                x = np.linspace(lower, upper, N)
                prior = prior_func(x)
            elif prior_dict['type'] == 'halfnorm':
                lower = prior_dict['loc']
                upper = prior_dict['loc'] + normal_stds * prior_dict['scale']
                x = np.linspace(lower, upper, N)
                prior = [prior_func(xi) for xi in x]
            elif prior_dict['type'] == 'neghalfnorm':
                upper = prior_dict['loc']
                lower = prior_dict['loc'] - normal_stds * prior_dict['scale']
                x = np.linspace(lower, upper, N)
                prior = [prior_func(xi) for xi in x]
            else:
                raise ValueError('Not implemented for prior type {}'.format(
                    prior_dict['type']))
            priorln = ax.plot(x, prior, 'r', label='prior')
            ax.set_xlabel(self.theta_symbols[i])

            s = self.samples[:, i]
            while len(s) > 10**4:
                # random downsample to avoid slow calculation of kde
                s = np.random.choice(s, size=int(len(s)/2.))
            kde = gaussian_kde(s)
            ax2 = ax.twinx()
            postln = ax2.plot(x, kde.pdf(x), 'k', label='posterior')
            ax2.set_yticklabels([])
            ax.set_yticklabels([])

        lns = priorln + postln
        labs = [l.get_label() for l in lns]
        axes[0].legend(lns, labs, loc=1, framealpha=0.8)

        fig.savefig('{}/{}_prior_posterior.png'.format(
            self.outdir, self.label))

    def plot_cumulative_max(self, **kwargs):
        d, maxtwoF = self.get_max_twoF()
        for key, val in self.theta_prior.iteritems():
            if key not in d:
                d[key] = val

        if hasattr(self, 'search') is False:
            self.initiate_search_object()
        if self.binary is False:
            self.search.plot_twoF_cumulative(
                self.label, self.outdir, F0=d['F0'], F1=d['F1'], F2=d['F2'],
                Alpha=d['Alpha'], Delta=d['Delta'],
                tstart=self.minStartTime, tend=self.maxStartTime,
                **kwargs)
        else:
            self.search.plot_twoF_cumulative(
                self.label, self.outdir, F0=d['F0'], F1=d['F1'], F2=d['F2'],
                Alpha=d['Alpha'], Delta=d['Delta'], asini=d['asini'],
                period=d['period'], ecc=d['ecc'], argp=d['argp'], tp=d['argp'],
                tstart=self.minStartTime, tend=self.maxStartTime, **kwargs)

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
            if x < loc:
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
                     lw=0.1, nprod=0, add_det_stat_burnin=False,
                     fig=None, axes=None, xoffset=0, plot_det_stat=False,
                     context='classic', subtractions=None, labelpad=0.05):
        """ Plot all the chains from a sampler """

        if np.ndim(axes) > 1:
            axes = axes.flatten()

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

        if subtractions is None:
            subtractions = [0 for i in range(ndim)]
        else:
            if len(subtractions) != self.ndim:
                raise ValueError('subtractions must be of length ndim')

        if plot_det_stat:
            extra_subplots = 1
        else:
            extra_subplots = 0
        with plt.style.context((context)):
            plt.rcParams['text.usetex'] = True
            if fig is None and axes is None:
                fig = plt.figure(figsize=(4, 3.0*ndim))
                ax = fig.add_subplot(ndim+extra_subplots, 1, 1)
                axes = [ax] + [fig.add_subplot(ndim+extra_subplots, 1, i)
                               for i in range(2, ndim+1)]

            idxs = np.arange(chain.shape[1])
            burnin_idx = chain.shape[1] - nprod
            if hasattr(self, 'convergence_idx'):
                convergence_idx = self.convergence_idx
            else:
                convergence_idx = burnin_idx
            if ndim > 1:
                for i in range(ndim):
                    axes[i].ticklabel_format(useOffset=False, axis='y')
                    cs = chain[:, :, i].T
                    if burnin_idx > 0:
                        axes[i].plot(xoffset+idxs[:convergence_idx],
                                     cs[:convergence_idx]-subtractions[i],
                                     color="r", alpha=alpha,
                                     lw=lw)
                    axes[i].plot(xoffset+idxs[burnin_idx:],
                                 cs[burnin_idx:]-subtractions[i],
                                 color="k", alpha=alpha, lw=lw)
                    if symbols:
                        if subtractions[i] == 0:
                            axes[i].set_ylabel(symbols[i], labelpad=labelpad)
                        else:
                            axes[i].set_ylabel(
                                symbols[i]+'$-$'+symbols[i]+'$_0$',
                                labelpad=labelpad)

                    if hasattr(self, 'convergence_diagnostic'):
                        ax = axes[i].twinx()
                        c_x = np.array(self.convergence_diagnosticx)
                        c_y = np.array(self.convergence_diagnostic)
                        break_idx = np.argmin(np.abs(c_x - burnin_idx))
                        ax.plot(c_x[:break_idx], c_y[:break_idx, i], '-b')
                        ax.plot(c_x[break_idx:], c_y[break_idx:, i], '-b')
                        ax.set_ylabel('PSRF')
                        ax.ticklabel_format(useOffset=False)
                        ax.set_ylim(1, self.convergence_plot_upper_lim)
            else:
                axes[0].ticklabel_format(useOffset=False, axis='y')
                cs = chain[:, :, temp].T
                if burnin_idx:
                    axes[0].plot(idxs[:burnin_idx], cs[:burnin_idx],
                                 color="r", alpha=alpha, lw=lw)
                axes[0].plot(idxs[burnin_idx:], cs[burnin_idx:], color="k",
                             alpha=alpha, lw=lw)
                if symbols:
                    axes[0].set_ylabel(symbols[0], labelpad=labelpad)

            if plot_det_stat:
                if len(axes) == ndim:
                    axes.append(fig.add_subplot(ndim+1, 1, ndim+1))

                lnl = sampler.lnlikelihood[temp, :, :]
                if burnin_idx and add_det_stat_burnin:
                    burn_in_vals = lnl[:, :burnin_idx].flatten()
                    try:
                        axes[-1].hist(burn_in_vals[~np.isnan(burn_in_vals)],
                                      bins=50, histtype='step', color='r')
                    except ValueError:
                        logging.info('Det. Stat. hist failed, most likely all '
                                     'values where the same')
                        pass
                else:
                    burn_in_vals = []
                prod_vals = lnl[:, burnin_idx:].flatten()
                try:
                    axes[-1].hist(prod_vals[~np.isnan(prod_vals)], bins=50,
                                  histtype='step', color='k')
                except ValueError:
                    logging.info('Det. Stat. hist failed, most likely all '
                                 'values where the same')
                    pass
                if self.BSGL:
                    axes[-1].set_xlabel(r'$\mathcal{B}_\mathrm{S/GL}$')
                else:
                    axes[-1].set_xlabel(r'$\widetilde{2\mathcal{F}}$')
                axes[-1].set_ylabel(r'$\textrm{Counts}$')
                combined_vals = np.append(burn_in_vals, prod_vals)
                if len(combined_vals) > 0:
                    minv = np.min(combined_vals)
                    maxv = np.max(combined_vals)
                    Range = abs(maxv-minv)
                    axes[-1].set_xlim(minv-0.1*Range, maxv+0.1*Range)

                xfmt = matplotlib.ticker.ScalarFormatter()
                xfmt.set_powerlimits((-4, 4))
                axes[-1].xaxis.set_major_formatter(xfmt)

            axes[-2].set_xlabel(r'$\textrm{Number of steps}$', labelpad=0.2)
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
            if hasattr(self, 'nglitch') and self.nglitch > 1:
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
        temp_idx = 0
        pF = sampler.chain[temp_idx, :, :, :]
        lnl = sampler.lnlikelihood[temp_idx, :, :]
        lnp = sampler.lnprobability[temp_idx, :, :]

        # General warnings about the state of lnp
        if np.any(np.isnan(lnp)):
            logging.warning(
                "Of {} lnprobs {} are nan".format(
                    np.shape(lnp), np.sum(np.isnan(lnp))))
        if np.any(np.isposinf(lnp)):
            logging.warning(
                "Of {} lnprobs {} are +np.inf".format(
                    np.shape(lnp), np.sum(np.isposinf(lnp))))
        if np.any(np.isneginf(lnp)):
            logging.warning(
                "Of {} lnprobs {} are -np.inf".format(
                    np.shape(lnp), np.sum(np.isneginf(lnp))))

        lnp_finite = copy.copy(lnp)
        lnp_finite[np.isinf(lnp)] = np.nan
        idx = np.unravel_index(np.nanargmax(lnp_finite), lnp_finite.shape)
        p = pF[idx]
        p0 = self.generate_scattered_p0(p)

        self.search.BSGL = False
        twoF = self.logl(p, self.search)
        self.search.BSGL = self.BSGL

        logging.info(('Gen. new p0 from pos {} which had det. stat.={:2.1f},'
                      ' twoF={:2.1f} and lnp={:2.1f}')
                     .format(idx[1], lnl[idx], twoF, lnp_finite[idx]))

        return p0

    def get_save_data_dictionary(self):
        d = dict(nsteps=self.nsteps, nwalkers=self.nwalkers,
                 ntemps=self.ntemps, theta_keys=self.theta_keys,
                 theta_prior=self.theta_prior, scatter_val=self.scatter_val,
                 log10temperature_min=self.log10temperature_min,
                 BSGL=self.BSGL)
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

        if self.sftfilepath is not None:
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
        """ Returns the max likelihood sample and the corresponding 2F value

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
        maxlogl = self.lnlikes[jmax]
        d = OrderedDict()

        if self.BSGL:
            if hasattr(self, 'search') is False:
                self.initiate_search_object()
            p = self.samples[jmax]
            self.search.BSGL = False
            maxtwoF = self.logl(p, self.search)
            self.search.BSGL = self.BSGL
        else:
            maxtwoF = maxlogl

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

    def check_if_samples_are_railing(self, threshold=0.01):
        return_flag = False
        for s, k in zip(self.samples.T, self.theta_keys):
            prior = self.theta_prior[k]
            if prior['type'] == 'unif':
                prior_range = prior['upper'] - prior['lower']
                edges = []
                fracs = []
                for l in ['lower', 'upper']:
                    bools = np.abs(s - prior[l])/prior_range < threshold
                    if np.any(bools):
                        edges.append(l)
                        fracs.append(str(100*float(np.sum(bools))/len(bools)))
                if len(edges) > 0:
                    logging.warning(
                        '{}% of the {} posterior is railing on the {} edges'
                        .format('% & '.join(fracs), k, ' & '.join(edges)))
                    return_flag = True
        return return_flag

    def write_par(self, method='med'):
        """ Writes a .par of the best-fit params with an estimated std """
        logging.info('Writing {}/{}.par using the {} method'.format(
            self.outdir, self.label, method))

        median_std_d = self.get_median_stds()
        max_twoF_d, max_twoF = self.get_max_twoF()

        logging.info('Writing par file with max twoF = {}'.format(max_twoF))
        filename = '{}/{}.par'.format(self.outdir, self.label)
        with open(filename, 'w+') as f:
            f.write('MaxtwoF = {}\n'.format(max_twoF))
            f.write('tref = {}\n'.format(self.tref))
            if hasattr(self, 'theta0_index'):
                f.write('theta0_index = {}\n'.format(self.theta0_idx))
            if method == 'med':
                for key, val in median_std_d.iteritems():
                    f.write('{} = {:1.16e}\n'.format(key, val))
            if method == 'twoFmax':
                for key, val in max_twoF_d.iteritems():
                    f.write('{} = {:1.16e}\n'.format(key, val))

    def write_prior_table(self):
        with open('{}/{}_prior.tex'.format(self.outdir, self.label), 'w') as f:
            f.write(r"\begin{tabular}{c l c} \hline" + '\n'
                    r"Parameter & & &  \\ \hhline{====}")

            for key, prior in self.theta_prior.iteritems():
                if type(prior) is dict:
                    Type = prior['type']
                    if Type == "unif":
                        a = prior['lower']
                        b = prior['upper']
                        line = r"{} & $\mathrm{{Unif}}$({}, {}) & {}\\"
                    elif Type == "norm":
                        a = prior['loc']
                        b = prior['scale']
                        line = r"{} & $\mathcal{{N}}$({}, {}) & {}\\"
                    elif Type == "halfnorm":
                        a = prior['loc']
                        b = prior['scale']
                        line = r"{} & $|\mathcal{{N}}$({}, {})| & {}\\"

                    u = self.unit_dictionary[key]
                    s = self.symbol_dictionary[key]
                    f.write("\n")
                    a = helper_functions.texify_float(a)
                    b = helper_functions.texify_float(b)
                    f.write(" " + line.format(s, a, b, u) + r" \\")
            f.write("\n\end{tabular}\n")

    def print_summary(self):
        max_twoFd, max_twoF = self.get_max_twoF()
        median_std_d = self.get_median_stds()
        logging.info('Summary:')
        if hasattr(self, 'theta0_idx'):
            logging.info('theta0 index: {}'.format(self.theta0_idx))
        logging.info('Max twoF: {} with parameters:'.format(max_twoF))
        for k in np.sort(max_twoFd.keys()):
            print('  {:10s} = {:1.9e}'.format(k, max_twoFd[k]))
        logging.info('Median +/- std for production values')
        for k in np.sort(median_std_d.keys()):
            if 'std' not in k:
                logging.info('  {:10s} = {:1.9e} +/- {:1.9e}'.format(
                    k, median_std_d[k], median_std_d[k+'_std']))
        logging.info('\n')

    def CF_twoFmax(self, theta, twoFmax, ntrials):
        Fmax = twoFmax/2.0
        return (np.exp(1j*theta*twoFmax)*ntrials/2.0
                * Fmax*np.exp(-Fmax)*(1-(1+Fmax)*np.exp(-Fmax))**(ntrials-1))

    def pdf_twoFhat(self, twoFhat, nglitch, ntrials, twoFmax=100, dtwoF=0.1):
        if np.ndim(ntrials) == 0:
            ntrials = np.zeros(nglitch+1) + ntrials
        twoFmax_int = np.arange(0, twoFmax, dtwoF)
        theta_int = np.arange(-1/dtwoF, 1./dtwoF, 1./twoFmax)
        CF_twoFmax_theta = np.array(
            [[np.trapz(self.CF_twoFmax(t, twoFmax_int, ntrial), twoFmax_int)
              for t in theta_int]
             for ntrial in ntrials])
        CF_twoFhat_theta = np.prod(CF_twoFmax_theta, axis=0)
        pdf = (1/(2*np.pi)) * np.array(
            [np.trapz(np.exp(-1j*theta_int*twoFhat_val)
             * CF_twoFhat_theta, theta_int) for twoFhat_val in twoFhat])
        return pdf.real

    def p_val_twoFhat(self, twoFhat, ntrials, twoFhatmax=500, Npoints=1000):
        """ Caluculate the p-value for the given twoFhat in Gaussian noise

        Parameters
        ----------
        twoFhat: float
            The observed twoFhat value
        ntrials: int, array of len Nglitch+1
            The number of trials for each glitch+1
        """
        twoFhats = np.linspace(twoFhat, twoFhatmax, Npoints)
        pdf = self.pdf_twoFhat(twoFhats, self.nglitch, ntrials)
        return np.trapz(pdf, twoFhats)

    def get_p_value(self, delta_F0, time_trials=0):
        """ Get's the p-value for the maximum twoFhat value """
        d, max_twoF = self.get_max_twoF()
        if self.nglitch == 1:
            tglitches = [d['tglitch']]
        else:
            tglitches = [d['tglitch_{}'.format(i)]
                         for i in range(self.nglitch)]
        tboundaries = [self.minStartTime] + tglitches + [self.maxStartTime]
        deltaTs = np.diff(tboundaries)
        ntrials = [time_trials + delta_F0 * dT for dT in deltaTs]
        p_val = self.p_val_twoFhat(max_twoF, ntrials)
        print('p-value = {}'.format(p_val))
        return p_val

    def get_evidence(self):
        fburnin = float(self.nsteps[-2])/np.sum(self.nsteps[-2:])
        lnev, lnev_err = self.sampler.thermodynamic_integration_log_evidence(
            fburnin=fburnin)

        log10evidence = lnev/np.log(10)
        log10evidence_err = lnev_err/np.log(10)
        return log10evidence, log10evidence_err

    def compute_evidence_long(self):
        """ Computes the evidence/marginal likelihood for the model """
        betas = self.betas
        alllnlikes = self.sampler.lnlikelihood[:, :, self.nsteps[-2]:]
        mean_lnlikes = np.mean(np.mean(alllnlikes, axis=1), axis=1)

        mean_lnlikes = mean_lnlikes[::-1]
        betas = betas[::-1]

        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(6, 8))

        if any(np.isinf(mean_lnlikes)):
            print("WARNING mean_lnlikes contains inf: recalculating without"
                  " the {} infs".format(len(betas[np.isinf(mean_lnlikes)])))
            idxs = np.isinf(mean_lnlikes)
            mean_lnlikes = mean_lnlikes[~idxs]
            betas = betas[~idxs]
            log10evidence = np.trapz(mean_lnlikes, betas)/np.log(10)
            z1 = np.trapz(mean_lnlikes, betas)
            z2 = np.trapz(mean_lnlikes[::-1][::2][::-1],
                          betas[::-1][::2][::-1])
            log10evidence_err = np.abs(z1 - z2) / np.log(10)

        ax1.semilogx(betas, mean_lnlikes, "-o")
        ax1.set_xlabel(r"$\beta$")
        ax1.set_ylabel(r"$\langle \log(\mathcal{L}) \rangle$")
        print("log10 evidence for {} = {} +/- {}".format(
              self.label, log10evidence, log10evidence_err))
        min_betas = []
        evidence = []
        for i in range(len(betas)/2):
            min_betas.append(betas[i])
            lnZ = np.trapz(mean_lnlikes[i:], betas[i:])
            evidence.append(lnZ/np.log(10))

        ax2.semilogx(min_betas, evidence, "-o")
        ax2.set_ylabel(r"$\int_{\beta_{\textrm{Min}}}^{\beta=1}" +
                       r"\langle \log(\mathcal{L})\rangle d\beta$", size=16)
        ax2.set_xlabel(r"$\beta_{\textrm{min}}$")
        plt.tight_layout()
        fig.savefig("{}/{}_beta_lnl.png".format(self.outdir, self.label))


class MCMCGlitchSearch(MCMCSearch):
    """ MCMC search using the SemiCoherentGlitchSearch """

    symbol_dictionary = dict(
        F0='$f$', F1='$\dot{f}$', F2='$\ddot{f}$', alpha=r'$\alpha$',
        delta='$\delta$', delta_F0='$\delta f$',
        delta_F1='$\delta \dot{f}$', tglitch='$t_\mathrm{glitch}$')
    unit_dictionary = dict(
        F0='Hz', F1='Hz/s', F2='Hz/s$^2$', alpha=r'rad', delta='rad',
        delta_F0='Hz', delta_F1='Hz/s', tglitch='s')
    rescale_dictionary = dict()

    @helper_functions.initializer
    def __init__(self, label, outdir, sftfilepath, theta_prior, tref,
                 minStartTime, maxStartTime, nglitch=1, nsteps=[100, 100],
                 nwalkers=100, ntemps=1, log10temperature_min=-5,
                 theta_initial=None, scatter_val=1e-10, dtglitchmin=1*86400,
                 theta0_idx=0, detectors=None, BSGL=False, minCoverFreq=None,
                 maxCoverFreq=None, earth_ephem=None, sun_ephem=None):
        """
        Parameters
        ----------
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
            scattered by scatter_val), or None in which case the prior is used.
        scatter_val, float or ndim array
            Size of scatter to use about the initialisation step, if given as
            an array it must be of length ndim and the order is given by
            theta_keys
        nglitch: int
            The number of glitches to allow
        tref, minStartTime, maxStartTime: int
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
        detectors: str
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

        if os.path.isdir(outdir) is False:
            os.mkdir(outdir)
        self.add_log_file()
        logging.info(('Set-up MCMC glitch search with {} glitches for model {}'
                      ' on data {}').format(self.nglitch, self.label,
                                            self.sftfilepath))
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

    def initiate_search_object(self):
        logging.info('Setting up search object')
        self.search = core.SemiCoherentGlitchSearch(
            label=self.label, outdir=self.outdir, sftfilepath=self.sftfilepath,
            tref=self.tref, minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime, minCoverFreq=self.minCoverFreq,
            maxCoverFreq=self.maxCoverFreq, earth_ephem=self.earth_ephem,
            sun_ephem=self.sun_ephem, detectors=self.detectors, BSGL=self.BSGL,
            nglitch=self.nglitch, theta0_idx=self.theta0_idx)

    def logp(self, theta_vals, theta_prior, theta_keys, search):
        if self.nglitch > 1:
            ts = ([self.minStartTime] + list(theta_vals[-self.nglitch:])
                  + [self.maxStartTime])
            if np.array_equal(ts, np.sort(ts)) is False:
                return -np.inf
            if any(np.diff(ts) < self.dtglitchmin):
                return -np.inf

        H = [self.generic_lnprior(**theta_prior[key])(p) for p, key in
             zip(theta_vals, theta_keys)]
        return np.sum(H)

    def logl(self, theta, search):
        if self.nglitch > 1:
            ts = ([self.minStartTime] + list(theta[-self.nglitch:])
                  + [self.maxStartTime])
            if np.array_equal(ts, np.sort(ts)) is False:
                return -np.inf

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
            full_glitch_keys[-2*self.nglitch:-1*self.nglitch] = [
                'delta_F1_{}'.format(i) for i in range(self.nglitch)]
            full_glitch_keys[-4*self.nglitch:-2*self.nglitch] = [
                'delta_F0_{}'.format(i) for i in range(self.nglitch)]
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

    def get_save_data_dictionary(self):
        d = dict(nsteps=self.nsteps, nwalkers=self.nwalkers,
                 ntemps=self.ntemps, theta_keys=self.theta_keys,
                 theta_prior=self.theta_prior, scatter_val=self.scatter_val,
                 log10temperature_min=self.log10temperature_min,
                 theta0_idx=self.theta0_idx, BSGL=self.BSGL)
        return d

    def apply_corrections_to_p0(self, p0):
        p0 = np.array(p0)
        if self.nglitch > 1:
            p0[:, :, -self.nglitch:] = np.sort(p0[:, :, -self.nglitch:],
                                               axis=2)
        return p0

    def plot_cumulative_max(self):

        fig, ax = plt.subplots()
        d, maxtwoF = self.get_max_twoF()
        for key, val in self.theta_prior.iteritems():
            if key not in d:
                d[key] = val

        if self.nglitch > 1:
            delta_F0s = [d['delta_F0_{}'.format(i)] for i in
                         range(self.nglitch)]
            delta_F0s.insert(self.theta0_idx, 0)
            delta_F0s = np.array(delta_F0s)
            delta_F0s[:self.theta0_idx] *= -1
            tglitches = [d['tglitch_{}'.format(i)] for i in
                         range(self.nglitch)]
        elif self.nglitch == 1:
            delta_F0s = [d['delta_F0']]
            delta_F0s.insert(self.theta0_idx, 0)
            delta_F0s = np.array(delta_F0s)
            delta_F0s[:self.theta0_idx] *= -1
            tglitches = [d['tglitch']]

        tboundaries = [self.minStartTime] + tglitches + [self.maxStartTime]

        for j in range(self.nglitch+1):
            ts = tboundaries[j]
            te = tboundaries[j+1]
            if (te - ts)/86400 < 5:
                logging.info('Period too short to perform cumulative search')
                continue
            if j < self.theta0_idx:
                summed_deltaF0 = np.sum(delta_F0s[j:self.theta0_idx])
                F0_j = d['F0'] - summed_deltaF0
                taus, twoFs = self.search.calculate_twoF_cumulative(
                    F0_j, F1=d['F1'], F2=d['F2'], Alpha=d['Alpha'],
                    Delta=d['Delta'], tstart=ts, tend=te)

            elif j >= self.theta0_idx:
                summed_deltaF0 = np.sum(delta_F0s[self.theta0_idx:j+1])
                F0_j = d['F0'] + summed_deltaF0
                taus, twoFs = self.search.calculate_twoF_cumulative(
                    F0_j, F1=d['F1'], F2=d['F2'], Alpha=d['Alpha'],
                    Delta=d['Delta'], tstart=ts, tend=te)
            ax.plot(ts+taus, twoFs)

        ax.set_xlabel('GPS time')
        fig.savefig('{}/{}_twoFcumulative.png'.format(self.outdir, self.label))


class MCMCSemiCoherentSearch(MCMCSearch):
    """ MCMC search for a signal using the semi-coherent ComputeFstat """
    @helper_functions.initializer
    def __init__(self, label, outdir, theta_prior, tref, sftfilepath=None,
                 nsegs=None, nsteps=[100, 100, 100], nwalkers=100,
                 binary=False, ntemps=1, log10temperature_min=-5,
                 theta_initial=None, scatter_val=1e-10, detectors=None,
                 BSGL=False, minStartTime=None, maxStartTime=None,
                 minCoverFreq=None, maxCoverFreq=None, earth_ephem=None,
                 sun_ephem=None, injectSources=None, assumeSqrtSX=None):
        """

        """

        if os.path.isdir(outdir) is False:
            os.mkdir(outdir)
        self.add_log_file()
        logging.info(('Set-up MCMC semi-coherent search for model {} on data'
                      '{}').format(
            self.label, self.sftfilepath))
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

        self.log_input()

    def get_save_data_dictionary(self):
        d = dict(nsteps=self.nsteps, nwalkers=self.nwalkers,
                 ntemps=self.ntemps, theta_keys=self.theta_keys,
                 theta_prior=self.theta_prior, scatter_val=self.scatter_val,
                 log10temperature_min=self.log10temperature_min,
                 BSGL=self.BSGL, nsegs=self.nsegs)
        return d

    def initiate_search_object(self):
        logging.info('Setting up search object')
        self.search = core.SemiCoherentSearch(
            label=self.label, outdir=self.outdir, tref=self.tref,
            nsegs=self.nsegs, sftfilepath=self.sftfilepath, binary=self.binary,
            BSGL=self.BSGL, minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime, minCoverFreq=self.minCoverFreq,
            maxCoverFreq=self.maxCoverFreq, detectors=self.detectors,
            earth_ephem=self.earth_ephem, sun_ephem=self.sun_ephem,
            injectSources=self.injectSources, assumeSqrtSX=self.assumeSqrtSX)

    def logp(self, theta_vals, theta_prior, theta_keys, search):
        H = [self.generic_lnprior(**theta_prior[key])(p) for p, key in
             zip(theta_vals, theta_keys)]
        return np.sum(H)

    def logl(self, theta, search):
        for j, theta_i in enumerate(self.theta_idxs):
            self.fixed_theta[theta_i] = theta[j]
        FS = search.run_semi_coherent_computefstatistic_single_point(
            *self.fixed_theta)
        return FS


class MCMCFollowUpSearch(MCMCSemiCoherentSearch):
    """ A follow up procudure increasing the coherence time in a zoom """
    def get_save_data_dictionary(self):
        d = dict(nwalkers=self.nwalkers, ntemps=self.ntemps,
                 theta_keys=self.theta_keys, theta_prior=self.theta_prior,
                 scatter_val=self.scatter_val,
                 log10temperature_min=self.log10temperature_min,
                 BSGL=self.BSGL, run_setup=self.run_setup)
        return d

    def update_search_object(self):
        logging.info('Update search object')
        self.search.init_computefstatistic_single_point()

    def get_width_from_prior(self, prior, key):
        if prior[key]['type'] == 'unif':
            return prior[key]['upper'] - prior[key]['lower']

    def get_mid_from_prior(self, prior, key):
        if prior[key]['type'] == 'unif':
            return .5*(prior[key]['upper'] + prior[key]['lower'])

    def init_V_estimate_parameters(self):
        if 'Alpha' in self.theta_keys:
            DeltaAlpha = self.get_width_from_prior(self.theta_prior, 'Alpha')
            DeltaDelta = self.get_width_from_prior(self.theta_prior, 'Delta')
            DeltaMid = self.get_mid_from_prior(self.theta_prior, 'Delta')
            DeltaOmega = np.sin(np.pi/2 - DeltaMid) * DeltaDelta * DeltaAlpha
            logging.info('Search over Alpha and Delta')
        else:
            logging.info('No sky search requested')
            DeltaOmega = 0
        if 'F0' in self.theta_keys:
            DeltaF0 = self.get_width_from_prior(self.theta_prior, 'F0')
        else:
            raise ValueError("You aren't searching over F0?")
        DeltaFs = [DeltaF0]
        if 'F1' in self.theta_keys:
            DeltaF1 = self.get_width_from_prior(self.theta_prior, 'F1')
            DeltaFs.append(DeltaF1)
            if 'F2' in self.theta_keys:
                DeltaF2 = self.get_width_from_prior(self.theta_prior, 'F2')
                DeltaFs.append(DeltaF2)
        logging.info('Searching over Frequency and {} spin-down components'
                     .format(len(DeltaFs)-1))

        if type(self.theta_prior['F0']) == dict:
            fiducial_freq = self.get_mid_from_prior(self.theta_prior, 'F0')
        else:
            fiducial_freq = self.theta_prior['F0']

        return fiducial_freq, DeltaOmega, DeltaFs

    def read_setup_input_file(self, run_setup_input_file):
        with open(run_setup_input_file, 'r+') as f:
            d = pickle.load(f)
        return d

    def write_setup_input_file(self, run_setup_input_file, R, Nsegs0,
                               nsegs_vals, V_vals, DeltaOmega, DeltaFs):
        d = dict(R=R, Nsegs0=Nsegs0, nsegs_vals=nsegs_vals, V_vals=V_vals,
                 DeltaOmega=DeltaOmega, DeltaFs=DeltaFs)
        with open(run_setup_input_file, 'w+') as f:
            pickle.dump(d, f)

    def check_old_run_setup(self, old_setup, **kwargs):
        try:
            truths = [val == old_setup[key] for key, val in kwargs.iteritems()]
            return all(truths)
        except KeyError:
            return False

    def init_run_setup(self, run_setup=None, R=10, Nsegs0=None, log_table=True,
                       gen_tex_table=True):

        if run_setup is None and Nsegs0 is None:
            raise ValueError(
                'You must either specify the run_setup, or Nsegs0 from which '
                'the optimial run_setup given R can be estimated')
        fiducial_freq, DeltaOmega, DeltaFs = self.init_V_estimate_parameters()
        if run_setup is None:
            logging.info('No run_setup provided')

            run_setup_input_file = '{}/{}_run_setup.p'.format(
                self.outdir, self.label)

            if os.path.isfile(run_setup_input_file):
                logging.info('Checking old setup input file {}'.format(
                    run_setup_input_file))
                old_setup = self.read_setup_input_file(run_setup_input_file)
                if self.check_old_run_setup(old_setup, R=R,
                                            Nsegs0=Nsegs0,
                                            DeltaOmega=DeltaOmega,
                                            DeltaFs=DeltaFs):
                    logging.info('Using old setup with R={}, Nsegs0={}'.format(
                        R, Nsegs0))
                    nsegs_vals = old_setup['nsegs_vals']
                    V_vals = old_setup['V_vals']
                    generate_setup = False
                else:
                    logging.info(
                        'Old setup does not match requested R, Nsegs0')
                    generate_setup = True
            else:
                generate_setup = True

            if generate_setup:
                nsegs_vals, V_vals = get_optimal_setup(
                    R, Nsegs0, self.tref, self.minStartTime,
                    self.maxStartTime, DeltaOmega, DeltaFs, fiducial_freq,
                    self.search.detector_names, self.earth_ephem,
                    self.sun_ephem)
                self.write_setup_input_file(run_setup_input_file, R, Nsegs0,
                                            nsegs_vals, V_vals, DeltaOmega,
                                            DeltaFs)

            run_setup = [((self.nsteps[0], 0),  nsegs, False)
                         for nsegs in nsegs_vals[:-1]]
            run_setup.append(
                ((self.nsteps[0], self.nsteps[1]), nsegs_vals[-1], False))

        else:
            logging.info('Calculating the number of templates for this setup')
            V_vals = []
            for i, rs in enumerate(run_setup):
                rs = list(rs)
                if len(rs) == 2:
                    rs.append(False)
                if np.shape(rs[0]) == ():
                    rs[0] = (rs[0], 0)
                run_setup[i] = rs

                if args.no_template_counting:
                    V_vals.append([1, 1, 1])
                else:
                    V, Vsky, Vpe = get_V_estimate(
                        rs[1], self.tref, self.minStartTime, self.maxStartTime,
                        DeltaOmega, DeltaFs, fiducial_freq,
                        self.search.detector_names, self.earth_ephem,
                        self.sun_ephem)
                    V_vals.append([V, Vsky, Vpe])

        if log_table:
            logging.info('Using run-setup as follows:')
            logging.info('Stage | nburn | nprod | nsegs | Tcoh d | resetp0 |'
                         ' V = Vsky x Vpe')
            for i, rs in enumerate(run_setup):
                Tcoh = (self.maxStartTime - self.minStartTime) / rs[1] / 86400
                if V_vals[i] is None:
                    vtext = 'N/A'
                else:
                    vtext = '{:1.0e} = {:1.0e} x {:1.0e}'.format(
                            V_vals[i][0], V_vals[i][1], V_vals[i][2])
                logging.info('{} | {} | {} | {} | {} | {} | {}'.format(
                    str(i).ljust(5), str(rs[0][0]).ljust(5),
                    str(rs[0][1]).ljust(5), str(rs[1]).ljust(5),
                    '{:6.1f}'.format(Tcoh), str(rs[2]).ljust(7),
                    vtext))

        if gen_tex_table:
            filename = '{}/{}_run_setup.tex'.format(self.outdir, self.label)
            if DeltaOmega > 0:
                with open(filename, 'w+') as f:
                    f.write(r'\begin{tabular}{c|cccccc}' + '\n')
                    f.write(r'Stage & $\Nseg$ & $\Tcoh^{\rm days}$ &'
                            r'$\Nsteps$ & $\V$ & $\Vsky$ & $\Vpe$ \\ \hline'
                            '\n')
                    for i, rs in enumerate(run_setup):
                        Tcoh = float(
                            self.maxStartTime - self.minStartTime)/rs[1]/86400
                        line = r'{} & {} & {} & {} & {} & {} & {} \\' + '\n'
                        if V_vals[i][0] is None:
                            V = Vsky = Vpe = 'N/A'
                        else:
                            V, Vsky, Vpe = V_vals[i]
                        if rs[0][-1] == 0:
                            nsteps = rs[0][0]
                        else:
                            nsteps = '{},{}'.format(*rs[0])
                        line = line.format(i, rs[1], '{:1.1f}'.format(Tcoh),
                                           nsteps,
                                           helper_functions.texify_float(V),
                                           helper_functions.texify_float(Vsky),
                                           helper_functions.texify_float(Vpe))
                        f.write(line)
                    f.write(r'\end{tabular}' + '\n')
            else:
                with open(filename, 'w+') as f:
                    f.write(r'\begin{tabular}{c|cccc}' + '\n')
                    f.write(r'Stage & $\Nseg$ & $\Tcoh^{\rm days}$ &'
                            r'$\Nsteps$ & $\Vpe$ \\ \hline'
                            '\n')
                    for i, rs in enumerate(run_setup):
                        Tcoh = float(
                            self.maxStartTime - self.minStartTime)/rs[1]/86400
                        line = r'{} & {} & {} & {} & {} \\' + '\n'
                        if V_vals[i] is None:
                            V = Vsky = Vpe = 'N/A'
                        else:
                            V, Vsky, Vpe = V_vals[i]
                        if rs[0][-1] == 0:
                            nsteps = rs[0][0]
                        else:
                            nsteps = '{},{}'.format(*rs[0])
                        line = line.format(i, rs[1], '{:1.1f}'.format(Tcoh),
                                           nsteps,
                                           helper_functions.texify_float(Vpe))
                        f.write(line)
                    f.write(r'\end{tabular}' + '\n')

        if args.setup_only:
            logging.info("Exit as requested by setup_only flag")
            sys.exit()
        else:
            return run_setup

    def run(self, run_setup=None, proposal_scale_factor=2, R=10, Nsegs0=None,
            create_plots=True, log_table=True, gen_tex_table=True, fig=None,
            axes=None, return_fig=False, **kwargs):
        """ Run the follow-up with the given run_setup

        Parameters
        ----------
        run_setup: list of tuples

        """

        self.nsegs = 1
        self.initiate_search_object()
        run_setup = self.init_run_setup(
            run_setup, R=R, Nsegs0=Nsegs0, log_table=log_table,
            gen_tex_table=gen_tex_table)
        self.run_setup = run_setup

        self.old_data_is_okay_to_use = self.check_old_data_is_okay_to_use()
        if self.old_data_is_okay_to_use is True:
            logging.warning('Using saved data from {}'.format(
                self.pickle_path))
            d = self.get_saved_data()
            self.sampler = d['sampler']
            self.samples = d['samples']
            self.lnprobs = d['lnprobs']
            self.lnlikes = d['lnlikes']
            self.nsegs = run_setup[-1][1]
            return

        nsteps_total = 0
        for j, ((nburn, nprod), nseg, reset_p0) in enumerate(run_setup):
            if j == 0:
                p0 = self.generate_initial_p0()
                p0 = self.apply_corrections_to_p0(p0)
            elif reset_p0:
                p0 = self.get_new_p0(sampler)
                p0 = self.apply_corrections_to_p0(p0)
                # self.check_initial_points(p0)
            else:
                p0 = sampler.chain[:, :, -1, :]

            self.nsegs = nseg
            self.search.nsegs = nseg
            self.update_search_object()
            self.search.init_semicoherent_parameters()
            sampler = emcee.PTSampler(
                self.ntemps, self.nwalkers, self.ndim, self.logl, self.logp,
                logpargs=(self.theta_prior, self.theta_keys, self.search),
                loglargs=(self.search,), betas=self.betas,
                a=proposal_scale_factor)

            Tcoh = (self.maxStartTime-self.minStartTime)/nseg/86400.
            logging.info(('Running {}/{} with {} steps and {} nsegs '
                          '(Tcoh={:1.2f} days)').format(
                j+1, len(run_setup), (nburn, nprod), nseg, Tcoh))
            sampler = self.run_sampler(sampler, p0, nburn=nburn, nprod=nprod)
            logging.info("Mean acceptance fraction: {}"
                         .format(np.mean(sampler.acceptance_fraction, axis=1)))
            if self.ntemps > 1:
                logging.info("Tswap acceptance fraction: {}"
                             .format(sampler.tswap_acceptance_fraction))
            logging.info('Max detection statistic of run was {}'.format(
                np.max(sampler.lnlikelihood)))

            if create_plots:
                fig, axes = self.plot_walkers(
                    sampler, symbols=self.theta_symbols, fig=fig, axes=axes,
                    nprod=nprod, xoffset=nsteps_total, **kwargs)
                for ax in axes[:self.ndim]:
                    ax.axvline(nsteps_total, color='k', ls='--', lw=0.25)

            nsteps_total += nburn+nprod

        samples = sampler.chain[0, :, nburn:, :].reshape((-1, self.ndim))
        lnprobs = sampler.lnprobability[0, :, nburn:].reshape((-1))
        lnlikes = sampler.lnlikelihood[0, :, nburn:].reshape((-1))
        self.sampler = sampler
        self.samples = samples
        self.lnprobs = lnprobs
        self.lnlikes = lnlikes
        self.save_data(sampler, samples, lnprobs, lnlikes)

        if create_plots:
            try:
                fig.tight_layout()
            except (ValueError, RuntimeError) as e:
                logging.warning('Tight layout encountered {}'.format(e))
            if return_fig:
                return fig, axes
            else:
                fig.savefig('{}/{}_walkers.png'.format(
                    self.outdir, self.label), dpi=200)


class MCMCTransientSearch(MCMCSearch):
    """ MCMC search for a transient signal using the ComputeFstat """

    symbol_dictionary = dict(
        F0='$f$', F1='$\dot{f}$', F2='$\ddot{f}$',
        alpha=r'$\alpha$', delta='$\delta$',
        transient_tstart='$t_\mathrm{start}$', transient_duration='$\Delta T$')
    unit_dictionary = dict(
        F0='Hz', F1='Hz/s', F2='Hz/s$^2$', alpha=r'rad', delta='rad',
        transient_tstart='s', transient_duration='s')

    rescale_dictionary = dict(
        transient_duration={'multiplier': 1/86400.,
                            'unit': 'day',
                            'symbol': 'Transient duration'},
        transient_tstart={
            'multiplier': 1/86400.,
            'subtractor': 'minStartTime',
            'unit': 'day',
            'label': 'Transient start-time \n days after minStartTime'}
            )

    def initiate_search_object(self):
        logging.info('Setting up search object')
        self.search = core.ComputeFstat(
            tref=self.tref, sftfilepath=self.sftfilepath,
            minCoverFreq=self.minCoverFreq, maxCoverFreq=self.maxCoverFreq,
            earth_ephem=self.earth_ephem, sun_ephem=self.sun_ephem,
            detectors=self.detectors, transient=True,
            minStartTime=self.minStartTime, maxStartTime=self.maxStartTime,
            BSGL=self.BSGL, binary=self.binary)

    def logl(self, theta, search):
        for j, theta_i in enumerate(self.theta_idxs):
            self.fixed_theta[theta_i] = theta[j]
        in_theta = copy.copy(self.fixed_theta)
        in_theta[1] = in_theta[0] + in_theta[1]
        if in_theta[1] > self.maxStartTime:
            return -np.inf
        FS = search.run_computefstatistic_single_point(*in_theta)
        return FS

    def unpack_input_theta(self):
        full_theta_keys = ['transient_tstart',
                           'transient_duration', 'F0', 'F1', 'F2', 'Alpha',
                           'Delta']
        if self.binary:
            full_theta_keys += [
                'asini', 'period', 'ecc', 'tp', 'argp']
        full_theta_keys_copy = copy.copy(full_theta_keys)

        full_theta_symbols = [r'$t_{\rm start}$', r'$\Delta T$',
                              '$f$', '$\dot{f}$', '$\ddot{f}$',
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
