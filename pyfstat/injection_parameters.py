"""Generate injection parameters drawn from different prior populations"""
import functools
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

isotropic_amplitude_priors = {
    "cosi": {"uniform": {"low": -1.0, "high": 1.0}},
    "psi": {"uniform": {"low": -0.25 * np.pi, "high": 0.25 * np.pi}},
    "phi": {"uniform": {"low": 0, "high": 2 * np.pi}},
}


class InjectionParametersGenerator:
    """
    Draw injection parameter samples from priors and return in dictionary format.

    Parameters
    ----------
    priors:
        Each key refers to one of the signal's parameters
        (following the PyFstat convention).

        Priors should be given as String matching one of scipy.stats's distributions with corresponding kwargs.

        `{"ParameterA": {"uniform": {"loc": 0, "scale": 1}}}`

        Alternatively, the following three options, which were recommended on a previous release,
        are still a valid input:

            1. Callable without required arguments:
            `{"ParameterA": np.random.uniform}`.

            2. Dict containing numpy.random distribution as key and kwargs in a dict as value:
            `{"ParameterA": {"uniform": {"low": 0, "high":1}}}`.

            3. Constant value to be returned as is:
            `{"ParameterA": 1.0}`.

        Note, however, that these options are deprecated and will be removed
        in a future PyFstat release. These old options do not follow completely the current way of
        specifying seeds in this class.

    generator:
        Numpy random number generator (e.g. `np.random.default_rng`) which will be used to draw
        random numbers from. Conflicts with `seed`.

    seed:
        Random seed to create instantiate `np.random.default_rng` from scratch. Conflicts
        with `generator`. If neither `seed` nor `generator` are given, a random number generator
        will be instantiated using `seed=None` and a warning will be thrown.
    """

    def __init__(
        self,
        priors: dict,
        seed: Optional[int] = None,
        generator: Optional[np.random.Generator] = None,
    ):
        self._parse_generator(seed=seed, generator=generator)
        self._parse_priors(priors)

    def _parse_generator(
        self, seed: Optional[int], generator: Optional[np.random.Generator]
    ):

        if (seed is not None) and (generator is not None):
            raise ValueError(
                "Incompatible inputs: please, "
                "use either a `seed` or an already initialized `np.random.Generator`"
            )
        if (not seed) and (not generator):
            logger.warning(
                f"No `generator` was provided and `seed` was set to {seed}, "
                "which looks uninitialized. Please, make sure you are aware of your seed choice"
            )

        self._rng = generator or np.random.default_rng(seed)

    def _deprecated_prior_parsing(self, parameter_prior) -> dict:
        if callable(parameter_prior):
            return parameter_prior
        elif isinstance(parameter_prior, dict):
            rng_function_name = next(iter(parameter_prior))
            rng_function = getattr(self._rng, rng_function_name)
            rng_kwargs = parameter_prior[rng_function_name]
            return functools.partial(rng_function, **rng_kwargs)
        else:  # Assume it is something to be returned as is
            return functools.partial(lambda x: x, parameter_prior)

    def _parse_priors(self, priors_input_format: dict):
        """Internal method to do the actual prior setup."""
        self.priors = {}

        for parameter_name, parameter_prior in priors_input_format.items():
            logging.warning(
                f"Parsing parameter `{parameter_name}` using a deprecated API"
            )
            self.priors[parameter_name] = self._deprecated_prior_parsing(
                parameter_prior
            )

    def draw(self) -> dict:
        """Draw a single multi-dimensional parameter space point from the given priors.

        Returns
        ----------
        injection_parameters:
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
                logger.warning(
                    f"Found {key} key in input priors with value {priors_input_format[key]}.\n"
                    "Overwritting to produce uniform samples across the sky."
                )

        return super()._parse_priors({**priors_input_format, **sky_priors})
