"""Bilby support for PyFstat MCMC-based searches."""

import logging
import os
import sys
from contextlib import contextmanager
from types import MethodType
from unittest.mock import patch

import bilby
import lal
import lalpulsar
import numpy as np
from bilby.core.sampler import ptemcee as bilby_ptemcee

import pyfstat.utils as utils

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Bilby likelihood interface
# -----------------------------------------------------------------------------


class PyFstatBilbyLikelihood(bilby.Likelihood):
    """Bilby likelihood wrapper for a PyFstat MCMC search.

    The parameter-dependent part is the existing PyFstat MCMC likelihood.
    If ``include_noise_log_likelihood`` is true, the cached data-only Gaussian
    noise likelihood is added following the standalone Bilby pipeline.
    If it is false, the noise likelihood is defined as zero and Bilby samples the
    likelihood ratio directly. The default selects the absolute likelihood for
    file-backed SFTs, except for BSGL searches, and the likelihood ratio
    otherwise. BSGL is a signal-versus-Gaussian-or-line-noise Bayes factor, so
    adding a Gaussian-only noise likelihood would not produce a valid absolute
    signal likelihood.
    """

    def __init__(self, pyfstat_search, include_noise_log_likelihood=None):
        self.pyfstat_search = pyfstat_search
        has_no_file_backed_sfts = pyfstat_search.sftfilepattern is None
        uses_bsgl = getattr(pyfstat_search, "BSGL", False)
        if uses_bsgl:
            if include_noise_log_likelihood is True:
                logger.warning(
                    "Ignoring include_noise_log_likelihood=True because "
                    "BSGL uses a signal-versus-Gaussian-or-line-noise Bayes "
                    "factor, for which Gaussian-only normalization is "
                    "inconsistent."
                )
            include_noise_log_likelihood = False
        elif include_noise_log_likelihood is None:
            include_noise_log_likelihood = not has_no_file_backed_sfts
        elif include_noise_log_likelihood and has_no_file_backed_sfts:
            raise ValueError(
                "The absolute noise likelihood cannot be computed for "
                "data without file-backed SFT amplitudes. "
                "Set include_noise_log_likelihood=False to use the likelihood "
                "ratio."
            )
        self.include_noise_log_likelihood = include_noise_log_likelihood
        logger.info(
            "Bilby likelihood normalization: the Gaussian noise likelihood "
            "is included by default for file-backed, non-BSGL searches so "
            "Bilby can report absolute signal evidence and the corresponding "
            "signal-versus-Gaussian-noise Bayes factor. It is disabled for "
            "BSGL searches because BSGL already uses a composite "
            "Gaussian-or-line-noise denominator, and for on-the-fly data "
            "whose SFT amplitudes are unavailable. This parameter-independent "
            "normalization does not affect posterior parameter estimation. "
            "For this search, include_noise_log_likelihood=%s.",
            include_noise_log_likelihood,
        )
        self.bilby_parameter_names = get_bilby_parameter_names(
            pyfstat_search.theta_keys
        )
        self._cached_noise_log_likelihood = None
        self._sft_periodograms_by_detector = None
        parameters = {name: None for name in self.bilby_parameter_names}
        super().__init__(parameters=parameters)

    def log_likelihood_ratio(self, parameters=None):
        if parameters is not None:
            self.parameters.update(parameters)
        theta = np.array(
            [self.parameters[name] for name in self.bilby_parameter_names],
            dtype=float,
        )
        return self.pyfstat_search._logl(theta, self.pyfstat_search.search)

    def log_likelihood(self, parameters=None):
        logl = self.log_likelihood_ratio(parameters=parameters)
        if self.include_noise_log_likelihood:
            logl += self.noise_log_likelihood()
        return logl

    def noise_log_likelihood(self):
        if not self.include_noise_log_likelihood:
            return 0.0
        if self._cached_noise_log_likelihood is None:
            self._cached_noise_log_likelihood = self._compute_noise_log_likelihood()
        return self._cached_noise_log_likelihood

    def _compute_noise_log_likelihood(self):
        """Compute the data-only likelihood for complex positive-frequency SFTs.

        If ``assumeSqrtSX`` is set, use its stationary one-sided PSD so that
        this normalization matches the F-statistic's noise model. Otherwise,
        ``PeriodoToRngmed`` estimates the expected SFT-bin power
        ``Q = E[|X|^2] = Tsft * S / 2``, which is converted to the one-sided
        PSD ``S``. Then evaluate the normalized complex-Gaussian likelihood
        for every detector, SFT, and frequency bin.
        """
        search = self.pyfstat_search.search
        constraints = _get_sft_constraints(
            minStartTime=self.pyfstat_search.minStartTime,
            maxStartTime=self.pyfstat_search.maxStartTime,
        )
        _, _, data = utils.sft.get_sft_as_arrays(
            self.pyfstat_search.sftfilepattern,
            fMin=search.minCoverFreq,
            fMax=search.maxCoverFreq,
            constraints=constraints,
        )
        assumed_psds = _get_assumed_psds_by_detector(search)
        periodograms = (
            self._get_running_median_periodograms() if assumed_psds is None else None
        )
        logl = 0.0

        for det_idx, det_name in enumerate(data):
            power = np.abs(data[det_name]) ** 2
            if assumed_psds is None:
                psd = 2.0 * periodograms[det_idx].T / search.Tsft
                psd = _match_frequency_bins(psd, power)
            else:
                psd = assumed_psds[det_name]
            psd = np.maximum(psd, np.finfo(float).tiny)

            quadratic = 2.0 * power / (search.Tsft * psd)
            log_normalization = np.log(np.pi * search.Tsft * psd / 2.0)
            logl -= np.sum(quadratic + log_normalization)

        return logl

    def _get_running_median_periodograms(self):
        if self._sft_periodograms_by_detector is not None:
            return self._sft_periodograms_by_detector

        periodograms = {}
        multi_sfts = lalpulsar.LoadMultiSFTs(
            self.pyfstat_search.search.SFTCatalog,
            fMin=_get_wing_fmin(self.pyfstat_search.search),
            fMax=_get_wing_fmax(self.pyfstat_search.search),
        )
        for det_idx in range(multi_sfts.length):
            sfts = multi_sfts.data[det_idx]
            periodograms[det_idx] = np.vstack(
                [
                    _running_median_periodogram(sfts.data[idx])
                    for idx in range(sfts.length)
                ]
            )

        self._sft_periodograms_by_detector = periodograms
        return periodograms


# -----------------------------------------------------------------------------
# Likelihood numerical helpers
# -----------------------------------------------------------------------------


def _get_sft_constraints(minStartTime=None, maxStartTime=None):
    if minStartTime is None and maxStartTime is None:
        return None

    constraints = lalpulsar.SFTConstraints()
    if minStartTime is not None:
        constraints.minStartTime = lal.LIGOTimeGPS(minStartTime)
    if maxStartTime is not None:
        constraints.maxStartTime = lal.LIGOTimeGPS(maxStartTime)
    return constraints


def _get_assumed_psds_by_detector(search):
    if search.assumeSqrtSX is None:
        return None

    sqrt_sx = np.asarray(utils.parse_list_of_numbers(search.assumeSqrtSX))
    if sqrt_sx.size == 1:
        sqrt_sx = np.repeat(sqrt_sx, search.numDetectors)
    elif sqrt_sx.size != search.numDetectors:
        raise ValueError(
            "assumeSqrtSX must contain either one value or one value per "
            f"detector ({sqrt_sx.size}!={search.numDetectors})."
        )

    return dict(zip(search.detector_names, sqrt_sx**2))


def _get_wing_fmin(search, wing_bins=50):
    return max(0.0, search.minCoverFreq - wing_bins / search.Tsft)


def _get_wing_fmax(search, wing_bins=50):
    return search.maxCoverFreq + wing_bins / search.Tsft


def _running_median_periodogram(sft, block_size=101):
    periodogram = lal.CreateREAL8FrequencySeries(
        name="",
        epoch=sft.epoch,
        f0=sft.f0,
        deltaF=sft.deltaF,
        sampleUnits=lal.DimensionlessUnit,
        length=sft.data.length,
    )
    periodogram.data = lal.CreateREAL8Vector(sft.data.length)
    lalpulsar.SFTtoPeriodogram(periodogram, sft)

    running_median = lal.CreateREAL8FrequencySeries(
        name="",
        epoch=sft.epoch,
        f0=sft.f0,
        deltaF=sft.deltaF,
        sampleUnits=lal.DimensionlessUnit,
        length=sft.data.length,
    )
    running_median.data = lal.CreateREAL8Vector(sft.data.length)
    lalpulsar.PeriodoToRngmed(running_median, periodogram, block_size)
    return running_median.data.data.copy()


def _match_frequency_bins(psd, power):
    if psd.shape[0] == power.shape[0]:
        return psd
    if psd.shape[0] > power.shape[0] + 100:
        return psd[50:-50, :]
    if psd.shape[0] > power.shape[0]:
        start = (psd.shape[0] - power.shape[0]) // 2
        return psd[start : start + power.shape[0], :]
    raise ValueError(
        f"Periodogram has fewer frequency bins ({psd.shape[0]}) than "
        f"the SFT data ({power.shape[0]})."
    )


# -----------------------------------------------------------------------------
# Parameter and prior conversion
# -----------------------------------------------------------------------------


def get_bilby_parameter_names(theta_keys):
    """Return unique Bilby parameter names preserving PyFstat theta order.

    This is necessary in glitch searches where multiple parameters of the same
    type are sampled. PyFstat uses arrays, so duplicated labels are acceptable,
    while Bilby uses dictionaries, which require unique names.
    """
    counts = {}
    names = []
    for key in theta_keys:
        if theta_keys.count(key) == 1:
            names.append(key)
        else:
            counts[key] = counts.get(key, 0)
            names.append(f"{key}_{counts[key]}")
            counts[key] += 1
    return names


def get_bilby_prior_dict(pyfstat_search, bilby_priors=None):
    """Return Bilby priors for the sampled PyFstat parameters.

    If ``bilby_priors`` is supplied, use it directly after checking that its
    keys exactly match the sampled Bilby parameter names. Otherwise, convert
    the corresponding PyFstat priors.
    """
    if bilby_priors is not None:
        expected_names = set(get_bilby_parameter_names(pyfstat_search.theta_keys))
        supplied_names = set(bilby_priors)
        missing_names = sorted(expected_names - supplied_names)
        unexpected_names = sorted(supplied_names - expected_names)
        if missing_names or unexpected_names:
            raise ValueError(
                "Bilby prior keys must exactly match the sampled parameter "
                f"names. Missing: {missing_names}; unexpected: "
                f"{unexpected_names}."
            )
        return bilby_priors

    priors = bilby.core.prior.PriorDict()
    for pyfstat_name, bilby_name in zip(
        pyfstat_search.theta_keys,
        get_bilby_parameter_names(pyfstat_search.theta_keys),
    ):
        priors[bilby_name] = pyfstat_prior_to_bilby_prior(
            name=bilby_name,
            prior_dict=pyfstat_search.theta_prior[pyfstat_name],
        )
    return priors


def pyfstat_prior_to_bilby_prior(name, prior_dict):
    """Convert one PyFstat prior specification into a Bilby prior."""
    prior_type = prior_dict["type"]
    bilby_prior = bilby.core.prior

    if prior_type == "unif":
        return bilby_prior.Uniform(
            minimum=prior_dict["lower"], maximum=prior_dict["upper"], name=name
        )
    if prior_type == "log10unif":
        # PyFstat stores these bounds in log10(parameter); Bilby expects the
        # corresponding physical lower/upper parameter values.
        return bilby_prior.LogUniform(
            minimum=10 ** prior_dict["log10lower"],
            maximum=10 ** prior_dict["log10upper"],
            name=name,
        )
    if prior_type == "norm":
        return bilby_prior.Gaussian(
            mu=prior_dict["loc"], sigma=prior_dict["scale"], name=name
        )
    if prior_type == "halfnorm":
        return bilby_prior.TruncatedGaussian(
            mu=prior_dict["loc"],
            sigma=prior_dict["scale"],
            minimum=prior_dict["loc"],
            maximum=np.inf,
            name=name,
        )
    if prior_type == "neghalfnorm":
        return bilby_prior.TruncatedGaussian(
            mu=prior_dict["loc"],
            sigma=prior_dict["scale"],
            minimum=-np.inf,
            maximum=prior_dict["loc"],
            name=name,
        )
    if prior_type == "lognorm":
        return bilby_prior.LogNormal(
            mu=prior_dict["loc"], sigma=prior_dict["scale"], name=name
        )

    raise ValueError(
        f"Cannot convert PyFstat prior type '{prior_type}' for '{name}' "
        "to a Bilby prior."
    )


# -----------------------------------------------------------------------------
# Result conversion
# -----------------------------------------------------------------------------


def _store_bilby_result(pyfstat_search, result):
    posterior = result.posterior
    samples = posterior[get_bilby_parameter_names(pyfstat_search.theta_keys)].to_numpy(
        dtype=float
    )
    lnlikes = _get_posterior_column(posterior, "log_likelihood")
    use_ratio = getattr(result, "use_ratio", None)
    noise_log_likelihood = getattr(result, "log_noise_evidence", None)
    can_reuse = (
        # Saved values must be present and finite, with a known convention.
        lnlikes is not None
        and np.all(np.isfinite(lnlikes))
        and use_ratio is not None
        and (
            # Ratios need no correction; absolute values need a noise constant.
            use_ratio
            or (noise_log_likelihood is not None and np.isfinite(noise_log_likelihood))
        )
    )
    if can_reuse:
        logger.debug("Reusing %d saved Bilby posterior likelihoods", len(samples))
        # Copy so normalization leaves Bilby's posterior unchanged.
        lnlikes = lnlikes.copy()
        # PyFstat expects likelihood ratios; absolute likelihoods include
        # the parameter-independent Gaussian-noise normalization.
        if not use_ratio:
            lnlikes -= noise_log_likelihood
    else:
        # Recompute when saved values cannot be safely converted to ratios.
        logger.info(
            "Recomputing %d posterior likelihoods: saved values are missing, "
            "nonfinite, or their normalization is unknown (use_ratio=%s, "
            "log_noise_evidence=%s)",
            len(samples),
            use_ratio,
            noise_log_likelihood,
        )
        lnlikes = np.array(
            [pyfstat_search._logl(sample, pyfstat_search.search) for sample in samples],
            dtype=float,
        )
    lnpriors = _get_posterior_column(posterior, "log_prior")

    pyfstat_search.samples = samples
    pyfstat_search.lnlikes = lnlikes
    pyfstat_search.lnprobs = lnlikes if lnpriors is None else lnlikes + lnpriors
    pyfstat_search.all_lnlikelihood = lnlikes.reshape((1, 1, -1))
    pyfstat_search.chain = samples.reshape((1, 1, len(samples), pyfstat_search.ndim))
    pyfstat_search.bilby_result = result


def _get_posterior_column(posterior, name):
    if name in posterior:
        return np.asarray(posterior[name], dtype=float)
    return None


# -----------------------------------------------------------------------------
# Logging and stream management
# -----------------------------------------------------------------------------


@contextmanager
def _preserve_pyfstat_logging_around_bilby():
    """Keep Bilby logging setup from duplicating PyFstat log records.

    Bilby configures the root logger while running. If PyFstat records also
    propagate to root during and after that setup, they can appear twice on the
    console: once through PyFstat's own handlers and once through root. This
    context manager limits the workaround to Bilby-backed searches and restores
    the previous logging state for ordinary PyFstat runs.
    """
    pyfstat_logger = logging.getLogger("pyfstat")
    root_logger = logging.getLogger()
    previous_propagate = pyfstat_logger.propagate
    previous_root_handlers = tuple(root_logger.handlers)
    previous_root_level = root_logger.level

    pyfstat_logger.propagate = False
    try:
        yield
    finally:
        for handler in list(root_logger.handlers):
            if handler not in previous_root_handlers:
                root_logger.removeHandler(handler)
        root_logger.setLevel(previous_root_level)
        pyfstat_logger.propagate = previous_propagate


class _TeeStream:
    """Write text to both an original stream and a file."""

    def __init__(self, stream, file):
        self.stream = stream
        self.file = file

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


@contextmanager
def _tee_stdout_stderr(path, mode="w"):
    """Mirror stdout and stderr to ``path`` while preserving console output."""
    if path is None:
        yield
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(path, mode) as file:
        sys.stdout = _TeeStream(original_stdout, file)
        sys.stderr = _TeeStream(original_stderr, file)
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


@contextmanager
def _capture_bilby_logger(path, mode="a", level=logging.INFO):
    """Attach a temporary file handler to Bilby's logger."""
    if path is None:
        yield
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    bilby_logger = bilby.core.utils.logger
    previous_level = bilby_logger.level
    handler = logging.FileHandler(path, mode=mode)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(name)s %(levelname)-8s: %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
        )
    )
    bilby_logger.addHandler(handler)
    bilby_logger.setLevel(level)
    try:
        yield
    finally:
        bilby_logger.removeHandler(handler)
        handler.close()
        bilby_logger.setLevel(previous_level)


# -----------------------------------------------------------------------------
# Fixed-step ptemcee adapter
# -----------------------------------------------------------------------------


def _ptemcee_sample_with_reused_state(sampler, p0=None, **kwargs):
    """Continue FixedStepsPtemcee's underlying sampler from evaluated state."""
    if sampler._pyfstat_reuse_state and sampler._pyfstat_state_initialized:
        p0 = None
    # setup_sampler replaces this object's sample method with this wrapper.
    # The class still holds ptemcee's original method, which we call here.
    original_sample = type(sampler).sample
    sampling_results = original_sample(sampler, p0=p0, **kwargs)

    for state in sampling_results:
        positions = state[0]
        log_posteriors = state[1]
        log_likelihoods = state[2]

        # Save the current positions and their likelihoods for the next sampling call
        sampler._p0 = positions
        sampler._logposterior0 = log_posteriors
        sampler._loglikelihood0 = log_likelihoods
        sampler._pyfstat_state_initialized = True
        # Pass the state to Bilby, which requests the next sampling result.
        yield state


class FixedStepsPtemcee(bilby_ptemcee.Ptemcee):
    """Legacy Bilby ptemcee with explicit burn-in and production step counts.

    Retains every production step at the cold temperature.
    Run searches sequentially or in separate processes to avoid interference
    between their shared stopping rules.
    A fixed step budget does not certify convergence.
    """

    def __init__(self, likelihood, priors, nsteps, **kwargs):
        if (
            not isinstance(nsteps, (list, tuple, np.ndarray))
            or np.ndim(nsteps) != 1
            or len(nsteps) != 2
            or any(
                isinstance(n, bool) or not isinstance(n, (int, np.integer))
                for n in nsteps
            )
            or nsteps[0] < 0
            or nsteps[1] <= 0
        ):
            raise ValueError(
                "nsteps must be [nonnegative burn-in, positive production]"
            )
        self.fixed_nburn, self.fixed_nprod = map(int, nsteps)
        if kwargs.get("resume", False):
            raise ValueError("Fixed-step ptemcee requires resume=False")
        kwargs.update(niterations_per_check=1, resume=False)
        super().__init__(likelihood, priors=priors, **kwargs)

    def _verify_external_sampler(self):
        __import__("ptemcee")

    def setup_sampler(self):
        sampler = super().setup_sampler()
        # Disable state reuse for periodic parameters because Bilby may wrap their
        # positions after the likelihood was computed.
        sampler._pyfstat_reuse_state = not np.any(self._periodic)
        sampler._pyfstat_state_initialized = False
        # Replace sample only on this object; its class method stays unchanged.
        sampler.sample = MethodType(_ptemcee_sample_with_reused_state, sampler)
        return sampler

    def _check_fixed_steps(self, iteration, *unused, **unused_kwargs):
        # Stop after the fixed number of burn-in and production steps
        # ignoring all other arguments
        self.discard = self.fixed_nburn
        retained = max(0, iteration - self.fixed_nburn) * self.nwalkers
        stop = iteration >= self.fixed_nburn + self.fixed_nprod
        return stop, 0, 1, np.nan, retained

    def run_sampler(self):
        with patch.object(bilby_ptemcee, "check_iteration", self._check_fixed_steps):
            return super().run_sampler()


# -----------------------------------------------------------------------------
# Top-level search orchestration
# -----------------------------------------------------------------------------


def run_bilby_search(
    pyfstat_search,
    sampler="dynesty",
    sampler_kwargs=None,
    bilby_priors=None,
    include_noise_log_likelihood=None,
    save_pickle=True,
    export_samples=True,
    save_loudest=True,
    save_bilby_logs=False,
    save_bilby_progress=False,
    bilby_log_file=None,
    bilby_progress_file=None,
    nsteps=None,
    **run_sampler_kwargs,
):
    """Run a PyFstat MCMC-based search through ``bilby.run_sampler``."""
    with _preserve_pyfstat_logging_around_bilby():
        pyfstat_search._initiate_search_object()

        priors = get_bilby_prior_dict(
            pyfstat_search,
            bilby_priors=bilby_priors,
        )
        likelihood = PyFstatBilbyLikelihood(
            pyfstat_search,
            include_noise_log_likelihood=include_noise_log_likelihood,
        )
        sampler_kwargs = sampler_kwargs or {}
        kwargs = {**run_sampler_kwargs, **sampler_kwargs}
        steps = kwargs.pop("nsteps", nsteps)
        if steps is not None:
            if sampler != "ptemcee":
                raise ValueError("Fixed nsteps is supported only for sampler='ptemcee'")
            kwargs.setdefault("nwalkers", pyfstat_search.nwalkers)
            kwargs.setdefault("ntemps", pyfstat_search.ntemps)
            kwargs.setdefault("log10beta_min", pyfstat_search.log10beta_min)
            sampler = FixedStepsPtemcee(
                likelihood,
                priors,
                steps,
                outdir=pyfstat_search.outdir,
                label=pyfstat_search.label,
                **kwargs,
            )
            kwargs = {}
        if save_bilby_logs and bilby_log_file is None:
            bilby_log_file = os.path.join(
                pyfstat_search.outdir, f"{pyfstat_search.label}.log"
            )
        if save_bilby_progress and bilby_progress_file is None:
            bilby_progress_file = os.path.join(
                pyfstat_search.outdir,
                f"{pyfstat_search.label}_bilby_progress.log",
            )

        with _capture_bilby_logger(bilby_log_file):
            with _tee_stdout_stderr(bilby_progress_file):
                result = bilby.run_sampler(
                    likelihood=likelihood,
                    priors=priors,
                    sampler=sampler,
                    outdir=pyfstat_search.outdir,
                    label=pyfstat_search.label,
                    **kwargs,
                )
        _store_bilby_result(pyfstat_search, result)

        pyfstat_search.bilby_pickle_path = os.path.join(
            pyfstat_search.outdir,
            f"{pyfstat_search.label}_bilby_saved_data.p",
        )
        if save_pickle:
            pyfstat_search._pickle_data(
                pyfstat_search.samples,
                pyfstat_search.lnprobs,
                pyfstat_search.lnlikes,
                pyfstat_search.all_lnlikelihood,
                pickle_path=pyfstat_search.bilby_pickle_path,
            )
        if export_samples:
            pyfstat_search.export_samples_to_disk()
        if save_loudest:
            pyfstat_search.generate_loudest()

    return result
