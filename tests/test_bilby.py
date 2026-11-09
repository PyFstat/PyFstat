"""Tests for Bilby fixed-step sampling and saved likelihood reuse."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

bilby = pytest.importorskip("bilby")

from bilby.core.sampler import ptemcee as bilby_ptemcee  # noqa: E402

from pyfstat.bilby_based_searches import (  # noqa: E402
    FixedStepsPtemcee,
    _store_bilby_result,
)

# Fixed-step sampling


class GaussianLikelihood(bilby.Likelihood):
    def __init__(self):
        super().__init__({"x": 0.0})

    def log_likelihood(self, parameters=None):
        parameters = self.parameters if parameters is None else parameters
        return -0.5 * parameters["x"] ** 2


@pytest.mark.parametrize("nsteps", [[5, 7], [0, 7]])
def test_fixed_steps_ptemcee_runs(tmp_path, nsteps):
    """Smoke test: the state-reusing adapter runs and returns finite samples."""
    pytest.importorskip("ptemcee")
    original_callback = bilby_ptemcee.check_iteration
    sampler = FixedStepsPtemcee(
        GaussianLikelihood(),
        priors={"x": bilby.core.prior.Uniform(-5, 5)},
        nsteps=nsteps,
        outdir=str(tmp_path),
        label="fixed",
        nwalkers=4,
        ntemps=2,
        check_point_plot=False,
        verbose=False,
        random=np.random.RandomState(123),
    )
    result = sampler.run_sampler()
    assert sampler.iteration == sum(nsteps)
    assert np.all(sampler.sampler.nprop == sum(nsteps))
    assert result.discard == nsteps[0]
    assert result.samples.shape == (4 * nsteps[1], 1)
    assert np.all(np.isfinite(result.samples))
    assert np.all(np.isfinite(result.log_likelihood_evaluations))
    assert bilby_ptemcee.check_iteration is original_callback


@pytest.mark.parametrize("nsteps", [3, [1], [-1, 5], [1, 0], [True, 5], [1.5, 5]])
def test_fixed_steps_rejects_invalid_budget(nsteps):
    with pytest.raises(ValueError, match="nsteps must"):
        FixedStepsPtemcee(GaussianLikelihood(), priors={}, nsteps=nsteps)


# Saved likelihood reuse and normalization


@pytest.mark.parametrize("use_ratio", [True, False])
def test_reuses_saved_likelihoods(use_ratio):
    ratios = np.array([3.0, 4.0])
    noise = -100.0
    saved = ratios if use_ratio else ratios + noise
    posterior = pd.DataFrame(
        {"F0": [30.0, 30.1], "log_likelihood": saved, "log_prior": [-2.0, -2.0]}
    )
    result = SimpleNamespace(
        posterior=posterior, use_ratio=use_ratio, log_noise_evidence=noise
    )
    search = SimpleNamespace(theta_keys=["F0"], ndim=1, _logl=Mock())

    _store_bilby_result(search, result)

    search._logl.assert_not_called()
    np.testing.assert_array_equal(search.lnlikes, ratios)
    np.testing.assert_array_equal(search.lnprobs, ratios - 2.0)
    np.testing.assert_array_equal(posterior["log_likelihood"], saved)
    assert search.all_lnlikelihood.shape == (1, 1, 2)
    assert search.chain.shape == (1, 1, 2, 1)


@pytest.mark.parametrize(
    "saved, metadata",
    [
        (None, {"use_ratio": True}),
        ([np.nan, 4.0], {"use_ratio": True}),
        ([3.0, 4.0], {}),
        ([3.0, 4.0], {"use_ratio": False}),
        ([3.0, 4.0], {"use_ratio": False, "log_noise_evidence": np.nan}),
    ],
)
def test_recomputes_when_saved_likelihoods_cannot_be_normalized(saved, metadata):
    posterior = pd.DataFrame({"F0": [30.0, 30.1]})
    if saved is not None:
        posterior["log_likelihood"] = saved
    result = SimpleNamespace(posterior=posterior, **metadata)
    search = SimpleNamespace(
        theta_keys=["F0"], ndim=1, search=object(), _logl=Mock(side_effect=[3.0, 4.0])
    )

    _store_bilby_result(search, result)

    assert search._logl.call_count == 2
    np.testing.assert_array_equal(search.lnlikes, [3.0, 4.0])
    np.testing.assert_array_equal(search.lnprobs, search.lnlikes)
