import logging

import numpy as np
import pytest
from commons_for_tests import is_flaky

from pyfstat import AllSkyInjectionParametersGenerator, InjectionParametersGenerator
from pyfstat.injection_parameters import _pyfstat_custom_priors, custom_prior


@custom_prior
def my_custom_prior(generator, shift):
    # Mean 0 so tests are simple
    return -2 * generator.uniform() + shift


@pytest.fixture(
    params=[
        {
            "gaussian_parameter": {"stats.norm": {"loc": 0.0, "scale": 1.0}},
            "fixed_parameter": 0.0,
        },
        {
            "function_prior": {"my_custom_prior": {"shift": 1.0}},
            "now_from_module": {"uniform_sky_declination": {}},
        },
    ],
    ids=["dictionary", "functions"],
)
def input_priors(request):
    return request.param


@pytest.fixture()
def seed():
    return 150914


@pytest.fixture()
def rng_object(seed):
    return np.random.default_rng(seed)


def test_custom_prior_decorator():
    @custom_prior
    def dummy_prior(generator):
        # For testing purposes
        return

    assert dummy_prior.__name__ in _pyfstat_custom_priors

    with pytest.raises(ValueError):
        custom_prior(dummy_prior)


def test_prior_parsing(input_priors, rng_object):
    ipg = InjectionParametersGenerator(priors=input_priors, generator=rng_object)

    for key in input_priors:
        assert key in ipg.priors


def test_seed_and_generator_init(caplog, input_priors, seed, rng_object):

    with pytest.raises(ValueError):
        InjectionParametersGenerator(
            priors=input_priors, seed=seed, generator=rng_object
        )

    with caplog.at_level(logging.WARNING):
        InjectionParametersGenerator(priors=input_priors, seed=None, generator=None)
    assert "uninitialized" in caplog.text


def test_seed_and_generator_compatibility(input_priors, seed, rng_object):
    ipg_seed = InjectionParametersGenerator(priors=input_priors, seed=seed)
    ipg_gen = InjectionParametersGenerator(priors=input_priors, generator=rng_object)

    for i in range(5):
        assert ipg_gen.draw() == ipg_seed.draw()


@pytest.mark.flaky(max_runs=3, min_passes=1, rerun_filter=is_flaky)
def test_rng_sampling(input_priors, rng_object):
    ipg = InjectionParametersGenerator(priors=input_priors, generator=rng_object)

    samples = [ipg.draw() for i in range(10000)]
    for key in ipg.priors:
        np.testing.assert_allclose(np.mean([s[key] for s in samples]), 0, atol=5e-2)


@pytest.mark.flaky(max_runs=3, min_passes=1, rerun_filter=is_flaky)
def test_all_sky_generation(rng_object):
    all_sky = AllSkyInjectionParametersGenerator(generator=rng_object)

    samples = [all_sky.draw() for i in range(10000)]

    np.testing.assert_allclose(np.mean([s["Alpha"] for s in samples]), np.pi, atol=5e-2)
    np.testing.assert_allclose(np.mean([s["Delta"] for s in samples]), 0, atol=5e-2)
