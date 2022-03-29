import numpy as np

# FIXME this should be made cleaner with fixtures
from commons_for_tests import BaseForTestsWithOutdir

import pyfstat


class TestInjectionParametersGenerator(BaseForTestsWithOutdir):
    label = "TestInjectionParametersGenerator"
    class_to_test = pyfstat.InjectionParametersGenerator

    def test_numpy_priors(self):
        numpy_priors = {
            "ParameterA": {"uniform": {"low": 0.0, "high": 0.0}},
            "ParameterB": {"uniform": {"low": 1.0, "high": 1.0}},
        }
        InjectionGenerator = self.class_to_test(priors=numpy_priors)

        parameters = InjectionGenerator.draw()
        self.assertTrue(parameters["ParameterA"] == 0.0)
        self.assertTrue(parameters["ParameterB"] == 1.0)

    def test_callable_priors(self):
        callable_priors = {"ParameterA": lambda: 0.0, "ParameterB": lambda: 1.0}
        InjectionGenerator = self.class_to_test(priors=callable_priors)

        parameters = InjectionGenerator.draw()
        self.assertTrue(parameters["ParameterA"] == 0.0)
        self.assertTrue(parameters["ParameterB"] == 1.0)

    def test_constant_priors(self):
        constant_priors = {"ParameterA": 0.0, "ParameterB": 1.0}
        InjectionGenerator = self.class_to_test(priors=constant_priors)

        parameters = InjectionGenerator.draw()
        self.assertTrue(parameters["ParameterA"] == 0.0)
        self.assertTrue(parameters["ParameterB"] == 1.0)

    def test_rng_seed(self):
        a_seed = 420

        samples = []
        for i in range(2):
            InjectionGenerator = self.class_to_test(
                priors={"ParameterA": {"normal": {"loc": 0, "scale": 1}}}, seed=a_seed
            )
            samples.append(InjectionGenerator.draw())
        self.assertTrue(samples[0]["ParameterA"] == samples[1]["ParameterA"])

    def test_rng_generation(self):
        InjectionGenerator = self.class_to_test(
            priors={"ParameterA": {"normal": {"loc": 0, "scale": 0.01}}}
        )
        samples = [InjectionGenerator.draw()["ParameterA"] for i in range(100)]
        mean = np.mean(samples)
        self.assertTrue(np.abs(mean) < 0.1)


class TestAllSkyInjectionParametersGenerator(TestInjectionParametersGenerator):
    label = "TestAllSkyInjectionParametersGenerator"

    class_to_test = pyfstat.AllSkyInjectionParametersGenerator

    def test_rng_seed(self):
        a_seed = 420

        samples = []
        for i in range(2):
            InjectionGenerator = self.class_to_test(seed=a_seed)
            samples.append(InjectionGenerator.draw())
        self.assertTrue(samples[0]["Alpha"] == samples[1]["Alpha"])
        self.assertTrue(samples[0]["Delta"] == samples[1]["Delta"])

    def test_rng_generation(self):
        InjectionGenerator = self.class_to_test()
        ra_samples = [
            InjectionGenerator.draw()["Alpha"] / np.pi - 1.0 for i in range(500)
        ]
        dec_samples = [
            InjectionGenerator.draw()["Delta"] * 2.0 / np.pi for i in range(500)
        ]
        self.assertTrue(np.abs(np.mean(ra_samples)) < 0.1)
        self.assertTrue(np.abs(np.mean(dec_samples)) < 0.1)
