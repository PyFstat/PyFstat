import numpy as np
import pytest
from commons_for_tests import is_flaky

import pyfstat


@pytest.fixture(
    params=[
        {key: {"uniform": {"low": key, "high": key}} for key in [0.0, 1.0]},
        {
            key: (lambda value=key: value) for key in [0.0, 1.0]
        },  # Lambda + Comprehension
        {key: key for key in [0.0, 1.0]},
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


def test_prior_parsing(parameters_generator, empty_priors):
    parameters = parameters_generator(priors=empty_priors).draw()
    print(empty_priors)
    for key in empty_priors:
        assert parameters[key] == key


def test_rng_seed(parameters_generator):
    seed = 150914

    samples = []
    for i in range(2):
        injection_generator = parameters_generator(
            priors={"ParameterA": {"normal": {"loc": 0, "scale": 1}}}, seed=seed
        )
        samples.append(injection_generator.draw())

    for key in injection_generator.priors:
        print(key)
        assert samples[0][key] == samples[1][key]


@pytest.mark.flaky(max_runs=5, min_passes=1, rerun_filter=is_flaky)
def test_rng_generation(parameters_generator):
    injection_generator = parameters_generator(
        priors={"ParameterA": {"normal": {"loc": 0, "scale": 0.01}}}
    )

    samples = [injection_generator.draw() for i in range(1000)]

    for key, distro in injection_generator.priors.items():
        mean = np.mean([s[key] for s in samples])

        if key == "Alpha":
            np.testing.assert_allclose(mean, np.pi, atol=1e-1)
        else:
            np.testing.assert_allclose(mean, 0, atol=1e-1)
