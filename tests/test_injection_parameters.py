import logging

import numpy as np
import pytest
from commons_for_tests import is_flaky

from pyfstat import AllSkyInjectionParametersGenerator, InjectionParametersGenerator
from pyfstat.injection_parameters import _pyfstat_custom_priors, custom_prior


@custom_prior
def my_custom_prior(generator, size, shift):
    # Mean 0 so tests are simple
    return -2 * generator.uniform(size=size) + shift


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
    def dummy_prior(generator, size):
        # For testing purposes
        return

    def dummier_prior(generator):
        # For testing purposes
        return

    assert dummy_prior.__name__ in _pyfstat_custom_priors

    with pytest.raises(ValueError):
        custom_prior(dummy_prior)

    with pytest.raises(TypeError):
        custom_prior(dummier_prior)


def test_prior_parsing(input_priors, rng_object):
    ipg = InjectionParametersGenerator(priors=input_priors, generator=rng_object)

    for key in input_priors:
        assert key in ipg.priors

    faulty_prior = {"faulty_parameter": {"distro_one": {}, "distro_two": {}}}
    with pytest.raises(ValueError):
        InjectionParametersGenerator(priors=faulty_prior, generator=rng_object)

    faulty_prior = {"faulty_parameter": {0: {}}}
    with pytest.raises(ValueError):
        InjectionParametersGenerator(priors=faulty_prior, generator=rng_object)


def test_old_api_raise(rng_object):
    faulty_prior = {"a_callable": lambda: 42}
    with pytest.raises(ValueError):
        InjectionParametersGenerator(priors=faulty_prior, generator=rng_object)

    faulty_prior = {"a_numpy_function": {"uniform": {"low": 0, "high": 1}}}
    with pytest.raises(ValueError):
        InjectionParametersGenerator(priors=faulty_prior, generator=rng_object)


def test_seed_and_generator_init(caplog, input_priors, seed, rng_object):
    with pytest.raises(ValueError):
        InjectionParametersGenerator(
            priors=input_priors, seed=seed, generator=rng_object
        )

    with caplog.at_level(logging.INFO):
        InjectionParametersGenerator(priors=input_priors, seed=None, generator=None)
    assert "use default" in caplog.text


def test_seed_and_generator_compatibility(input_priors, seed, rng_object):
    ipg_seed = InjectionParametersGenerator(priors=input_priors, seed=seed)
    ipg_gen = InjectionParametersGenerator(priors=input_priors, generator=rng_object)

    gen_draw = ipg_gen.draw_many(size=5)
    seed_draw = ipg_seed.draw_many(size=5)
    for key in input_priors:
        assert np.all(gen_draw[key] == seed_draw[key])


@pytest.mark.flaky(max_runs=3, min_passes=1, rerun_filter=is_flaky)
def test_rng_sampling(input_priors, rng_object):
    ipg = InjectionParametersGenerator(priors=input_priors, generator=rng_object)

    samples = ipg.draw_many(size=10000)
    for key in ipg.priors:
        np.testing.assert_allclose(samples[key].mean(), 0, atol=5e-2)


@pytest.mark.flaky(max_runs=3, min_passes=1, rerun_filter=is_flaky)
def test_all_sky_generation(rng_object):
    all_sky = AllSkyInjectionParametersGenerator(generator=rng_object)

    samples = all_sky.draw_many(size=10000)

    np.testing.assert_allclose(samples["Alpha"].mean(), np.pi, atol=5e-2)
    np.testing.assert_allclose(samples["Delta"].mean(), 0, atol=5e-2)
