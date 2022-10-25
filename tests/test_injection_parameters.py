import numpy as np
import pytest
from commons_for_tests import is_flaky

from pyfstat import AllSkyInjectionParametersGenerator, InjectionParametersGenerator


def my_custom_prior(generator, shift):
    # Mean 0 so tests are simple
    return -2 * generator.uniform() + shift


@pytest.fixture()
def input_priors():
    return {
        "gaussian_parameter": {"stats.norm": {"loc": 0.0, "scale": 1.0}},
        # "custom_parameter": {"my_custom_prior": {"shift": 1.}},
    }


@pytest.fixture()
def seed():
    return 150914


@pytest.fixture()
def rng_object(seed):
    return np.random.default_rng(seed)


def test_prior_parsing(input_priors, rng_object):
    ipg = InjectionParametersGenerator(priors=input_priors, generator=rng_object)

    for key in input_priors:
        assert key in ipg.priors


def test_seed_and_generator_consistency(input_priors, seed, rng_object):
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
