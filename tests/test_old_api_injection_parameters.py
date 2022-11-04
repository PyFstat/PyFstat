import numpy as np
import pytest
from commons_for_tests import is_flaky

import pyfstat


@pytest.fixture(
    params=[
        {
            f"parameter_{key}": {"uniform": {"low": key, "high": key}}
            for key in [0.0, 1.0]
        },
        {
            "parameter_0.0": lambda: 0.0,
            "parameter_1.0": lambda: 1.0,
        },
        {f"parameter_{key}": key for key in [0.0, 1.0]},
    ],
    ids=["Numpy-priors", "callable-priors", "fixed-priors"],
)
def empty_priors(request):
    return request.param


@pytest.fixture(
    params=[
        pyfstat.InjectionParametersGenerator,
        pyfstat.AllSkyInjectionParametersGenerator,
    ]
)
def parameters_generator(request):
    return request.param


@pytest.fixture()
def seed():
    return 150914


def test_prior_parsing(parameters_generator, empty_priors, seed):
    parameters = parameters_generator(priors=empty_priors, seed=seed).draw()
    print(parameters)
    for key in parameters:
        if key not in ["Alpha", "Delta"]:
            assert parameters[key] == float(key[-3:])


def test_seed_and_generator_consistency(parameters_generator, seed):
    priors = {"parameter": {"normal": {"loc": 0, "scale": 1}}}

    seed_generator = parameters_generator(priors=priors, seed=seed)
    generator_generator = parameters_generator(
        priors=priors, generator=np.random.default_rng(seed)
    )
    for i in range(5):
        assert (
            seed_generator.draw()["parameter"]
            == generator_generator.draw()["parameter"]
        )


def test_rng_seed(parameters_generator, seed):

    samples = []
    for i in range(2):
        injection_generator = parameters_generator(
            priors={"ParameterA": {"normal": {"loc": 0, "scale": 1}}}, seed=seed
        )
        samples.append(injection_generator.draw())

    for key in injection_generator.priors:
        assert samples[0][key] == samples[1][key]


@pytest.mark.flaky(max_runs=5, min_passes=1, rerun_filter=is_flaky)
def test_rng_generation(parameters_generator, seed):
    injection_generator = parameters_generator(
        priors={"ParameterA": {"normal": {"loc": 0, "scale": 0.01}}}, seed=seed
    )

    samples = [injection_generator.draw() for i in range(1000)]

    for key, distro in injection_generator.priors.items():
        mean = np.mean([s[key] for s in samples])

        if key == "Alpha":
            np.testing.assert_allclose(mean, np.pi, atol=1e-1)
        else:
            np.testing.assert_allclose(mean, 0, atol=1e-1)
