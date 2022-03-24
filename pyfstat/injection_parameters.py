"""Generate injection parameters drawn from different prior populations"""
import functools
import logging
from typing import Union

import numpy as np
from attrs import Factory, define, field

isotropic_amplitude_priors = {
    "cosi": {"uniform": {"low": -1.0, "high": 1.0}},
    "psi": {"uniform": {"low": -0.25 * np.pi, "high": 0.25 * np.pi}},
    "phi": {"uniform": {"low": 0, "high": 2 * np.pi}},
}


@define(kw_only=True, slots=False)
class InjectionParametersGenerator:
    """
    Draw injection parameter samples from priors and return in dictionary format.

    Parameters
    ----------
    priors: dict
        Each key refers to one of the signal's parameters
        (following the PyFstat convention).
        Priors can be given as values in three formats
        (by order of evaluation):

        1. Callable without required arguments:
        `{"ParameterA": np.random.uniform}`.

        2. Dict containing numpy.random distribution as key and kwargs in a dict as value:
        `{"ParameterA": {"uniform": {"low": 0, "high":1}}}`.

        3. Constant value to be returned as is:
        `{"ParameterA": 1.0}`.

    seed: None, int
        Argument to be fed to numpy.random.default_rng,
        with all of its accepted types.

    _rng: np.random.Generator
        Alternatively, this class accepts an already-initialized np.Generator,
        in which case the `seed` argument will be ignored.
    """

    seed: Union[None, int] = field(default=None)
    _rng: np.random.Generator = field(
        default=Factory(lambda self: np.random.default_rng(self.seed), takes_self=True)
    )
    priors: dict = field(factory=dict)

    def __attrs_post_init__(self):
        self.priors = self._parse_priors(self.priors)

    def _parse_priors(self, priors_input_format):
        """Internal method to do the actual prior setup."""
        priors = {}

        for parameter_name, parameter_prior in priors_input_format.items():
            if callable(parameter_prior):
                priors[parameter_name] = parameter_prior
            elif isinstance(parameter_prior, dict):
                rng_function_name = next(iter(parameter_prior))
                rng_function = getattr(self._rng, rng_function_name)
                rng_kwargs = parameter_prior[rng_function_name]
                priors[parameter_name] = functools.partial(rng_function, **rng_kwargs)
            else:  # Assume it is something to be returned as is
                priors[parameter_name] = functools.partial(lambda x: x, parameter_prior)

        return priors

    def draw(self):
        """Draw a single multi-dimensional parameter space point from the given priors.

        Returns
        ----------
        injection_parameters: dict
            Dictionary with parameter names as keys and their numeric values.
        """
        injection_parameters = {
            parameter_name: parameter_prior()
            for parameter_name, parameter_prior in self.priors.items()
        }
        return injection_parameters


class AllSkyInjectionParametersGenerator(InjectionParametersGenerator):
    """
    Draw injection parameter samples from priors and return in dictionary format.
    This class works in exactly the same way as `InjectionParametersGenerator`,
    but including by default two extra keys, `Alpha` and `Delta` (sky position's
    right ascension and declination in radians), which are sample isotropically
    across the celesetial sphere.

    `Alpha`'s distribution is Uniform(0, 2 pi), and
    `sin(Delta)`'s distribution is Uniform(-1, 1).

    The only reason this class exists is because, using the notation we specified
    in the base class, there is no way to generate arcsine distributed numbers using
    a specific seed, as numpy does not have such a number generator and hence has to
    be constructed by applying a function to a uniform number.
    """

    def _parse_priors(self, priors_input_format):
        sky_priors = {
            "Alpha": lambda: self._rng.uniform(low=0.0, high=2 * np.pi),
            "Delta": lambda: np.arcsin(self._rng.uniform(low=-1.0, high=1.0)),
        }

        for key in sky_priors:
            if key in priors_input_format:
                logging.warning(
                    f"Found {key} key in input priors with value {priors_input_format[key]}.\n"
                    "Overwritting to produce uniform samples across the sky."
                )
            priors_input_format[key] = sky_priors[key]

        return super()._parse_priors(priors_input_format)
