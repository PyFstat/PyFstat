"""Generate injection parameters drawn from different prior populations"""
import functools
import logging
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

isotropic_amplitude_priors = {
    "cosi": {"stats.uniform": {"loc": -1.0, "scale": 2.0}},
    "psi": {"stats.uniform": {"loc": -0.25 * np.pi, "scale": 0.5 * np.pi}},
    "phi": {"stats.uniform": {"loc": 0, "scale": 2 * np.pi}},
}


def uniform_sky_declination(generator: np.random.Generator) -> float:
    """
    Return declination values such that, when paired with right ascension
    values sampled uniformly along [0, 2*pi], the resulting pairs of samples
    are uniformly distributed on the 2-sphere.

    Parameters
    ----------
    generator:
        As required by InjectionParametersGenerator.

    Returns
    -------
    declination:
        Declination value distributed
    """
    return np.arcsin(generator.uniform(low=-1.0, high=1.0))


class InjectionParametersGenerator:
    """
    Draw injection parameter samples from priors and return in dictionary format.

    Parameters
    ----------
    priors:
        Each key refers to one of the signal's parameters
        (following the PyFstat convention).

        Each parameter's prior should be given as a dictionary entry as follows:
        `{"parameter": {"<function>": {**kwargs}}}`
        where <function> refers may be (exclusively) either a `scipy.stats` function
        (e.g. `stats.uniform`) or a user-defined function in the global scope.

        - | If a `scipy.stats` function is used, it *must* be given as `stats.*`
          | (i.e. the `stats` namespace should be explicitly included).
          |  For example, a uniform prior between 3 and 5 would be written as
          | `{"parameter": {"stats.uniform": {"loc": 3, "scale": 5}}}`.

        - | If a user-defined function is used, such a function *must* take
          | a `generator` kwarg as one of its arguments and use such a generator
          | to generate any required random number within the function.
          | The `generator` kwarg is *required* regardless of whether this is a
          | deterministic or random function.
          | For example, a negative log-distributed random number could be constructed as

        ```
        def negative_log_uniform(generator):
            return -10**(generator.uniform())
        ```

        Alternatively, the following three options, which were recommended
        on a previous release, are still a valid input as long as the `_old` substring
        is appended to the parameter's name:

            1. Callable without required arguments:
            `{"ParameterA_old": np.random.uniform}`.

            2. Dict containing numpy.random distribution as key and kwargs
            in a dict as value:
            `{"ParameterA_old": {"uniform": {"low": 0, "high":1}}}`.

            3. Constant value to be returned as is:
            `{"ParameterA_old": 1.0}`.

        Note, however, that these options are deprecated and will be removed
        in a future PyFstat release.
        These old options do not follow completely the current way of
        specifying seeds in this class.

    generator:
        Numpy random number generator (e.g. `np.random.default_rng`)
        which will be used to draw random numbers from. Conflicts with `seed`.

    seed:
        Random seed to create instantiate `np.random.default_rng` from scratch.
        Conflicts with `generator`. If neither `seed` nor `generator` are given,
        a random number generator will be instantiated using `seed=None`
        and a warning will be thrown.
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
                "which looks uninitialized. "
                "Please, make sure you are aware of your seed choice"
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

            # FIXME to be removed once deprecation is effective
            if "_old" == parameter_name[-4:]:
                actual_name = parameter_name[:-4]
                logger.warning(
                    f"Parsing parameter `{actual_name}` using a deprecated API."
                )
                self.priors[actual_name] = self._deprecated_prior_parsing(
                    parameter_prior
                )
                print(self.priors)
                continue

            prior_distribution = next(iter(parameter_prior))

            if "stats." in prior_distribution:
                _, stats_function = prior_distribution.split(".")
                logger.debug(
                    f"Setting {stats_function} distribution from `scipy.stats` "
                    f"to parameter {parameter_name}"
                )
                rv_object = getattr(stats, stats_function)(
                    **parameter_prior[prior_distribution]
                )
                rv_object.random_state = self._rng
                self.priors[parameter_name] = rv_object.rvs
            else:
                logger.debug(
                    f"Setting {prior_distribution} distribution from local namespace "
                    f"to parameter {parameter_name}"
                )
                # FIXME: Any alternatives?
                custom_function = locals().get(
                    prior_distribution, None
                ) or globals().get(prior_distribution, None)
                self.priors[parameter_name] = functools.partial(
                    custom_function,
                    generator=self._rng,
                    **parameter_prior[prior_distribution],
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

    def __init__(
        self,
        priors: Optional[dict] = None,
        seed: Optional[int] = None,
        generator: Optional[np.random.Generator] = None,
    ):

        priors = priors or {}

        sky_priors = {
            "Alpha": {"stats.uniform": {"loc": 0.0, "scale": 2 * np.pi}},
            "Delta": {"uniform_sky_declination": {}},
        }

        for key in sky_priors:
            if key in priors:
                logger.warning(
                    f"Found input key {key} with value {priors[key]}.\n"
                    "Overwritting to produce uniform samples across the sky."
                )

        super().__init__(
            priors={**priors, **sky_priors}, seed=seed, generator=generator
        )
