"""Generate injection parameters drawn from different prior populations"""

import functools
import logging
from collections.abc import Mapping
from inspect import signature
from typing import Callable, Dict, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

_pyfstat_custom_priors = {}


def custom_prior(prior_function: Callable) -> Callable:
    """
    Intended to be used as a decorator to add custom functions to
    the list of available priors for ``InjectionParametersGenerator``.

    For example::

        @pyfstat.custom_prior
        def negative_log_uniform(generator, size):
            return -10**(generator.uniform(size=size))

    will add the key ``negative_log_uniform`` to ``_pyfstat_custom_priors``
    with said function as the corresponding value.

    A function decorated with ``custom_prior`` *must* take ``generator`` and ``size`` as
    keyword arguments; otherwise, a ``TypeError`` will be raised.
    Additional arguments can be provided as needed.

    See docstring of :func:`~pyfstat.injection_parameters.InjectionParametersGenerator`
    for an example on how to draw samples from a custom prior.

    Parameters
    ----------
    prior_function:
        Function to be added into ``_pyfstat_custom_priors`` with a key
        corresponding *exactly* to the name it was given at definition time.

    Returns
    -------
    prior_function: Callable
        Same function as the input function.
    """

    function_name = f"{prior_function.__name__}"
    if function_name in _pyfstat_custom_priors:
        raise ValueError(
            f"Custom prior `{function_name}` already defined in "
            "`pyfstat._pyfstat_custom_priors.` "
            "Please, use a different function name"
        )

    function_signature = signature(prior_function)
    if any(
        required_kwarg not in function_signature.parameters
        for required_kwarg in ["generator", "size"]
    ):
        raise TypeError(
            f"Custom prior function `{function_name}` must accept"
            " `generator` and `size` as keyword arguments."
            " Please, make sure your custom prior follows"
            " the signature specified in this functions docstring."
        )

    _pyfstat_custom_priors[function_name] = prior_function

    return prior_function


"""
standard choices for angle ranges

We are following here R. Prix: https://dcc.ligo.org/T0900149-v6/public
"""
isotropic_amplitude_distribution = {
    "cosi": {"stats.uniform": {"loc": -1.0, "scale": 2.0}},
    "psi": {"stats.uniform": {"loc": -0.25 * np.pi, "scale": 0.5 * np.pi}},
    "phi": {"stats.uniform": {"loc": 0, "scale": 2 * np.pi}},
}


@custom_prior
def uniform_sky_declination(generator: np.random.Generator, size: int) -> np.ndarray:
    """
    Return declination values such that, when paired with right ascension
    values sampled uniformly along [0, 2*pi], the resulting pairs of samples
    are uniformly distributed on the 2-sphere.

    Parameters
    ----------
    generator:
        As required by InjectionParametersGenerator.
    size:
        As required by InjectionParameterGenerator. Gets passed directly
        as a kwargs to ``generator``'s methods.

    Returns
    -------
    declination: np.ndarray
        Declination value distributed
    """
    return np.arcsin(generator.uniform(low=-1.0, high=1.0, size=size))


class InjectionParametersGenerator:
    """
    Draw injection parameter samples from priors and return in dictionary format.

    Parameters
    ----------
    priors:
        Each key refers to one of the signal's parameters
        (following the PyFstat convention).

        Each parameter's prior should be given as a dictionary entry as follows:
        ``{"parameter": {"<function>": {**kwargs}}}``
        where <function> may be (exclusively) either a user-defined function decorated
        with ``@custom_prior`` or the name of a ``scipy.stats`` random variable.

        * | If a user-defined function is used, such a function *must* take
           | a ``generator`` kwarg as one of its arguments and use such a generator
           | (``np.random.Generator`` type)
           | to generate any required random number within the function.
           | (A deterministic function could then simply ignore this kwarg.)
           | For example, a negative log-distributed random number could be constructed as

        ::

            import pyfstat

            @pyfstat.injection_parameters.custom_prior
            def negative_log_uniform(generator, size):
                return -10**(generator.uniform(size=size))

            priors = {"my_parameter": {"negative_log_uniform": {}}}
            ipg = pyfstat.InjectionParametersGenerator(priors=priors, seed=42)
            ipg.draw()

        * | If a ``scipy.stats`` function is used, it *must* be given as ``stats.*``
           | (i.e. the ``stats`` namespace should be explicitly included).
           |  For example, a uniform prior between 3 and 5 would be written as
           | ``{"parameter": {"stats.uniform": {"loc": 3, "scale": 2}}}``.

        ::

            import pyfstat

            priors = {"my_parameter": "stats.uniform": {"loc": 3, "scale": 2}}
            ipg = pyfstat.InjectionParametersGenerator(priors=priors, seed=42)
            ipg.draw()

        Delta-priors (i.e. priors for a deterministic output) can also be specified by
        giving the fixed value to be returned as-is. For example, specifying a fixed
        value of 1 for the parameter ``A`` would be ``{"A": 1}``.

    generator:
        Numpy random number generator (e.g. ``np.random.default_rng``)
        which will be used to draw random numbers from. Conflicts with ``seed``.

    seed:
        Random seed to create instantiate ``np.random.default_rng`` from scratch.
        Conflicts with ``generator``. If neither ``seed`` nor ``generator`` are given,
        a random number generator will be instantiated using ``seed=None``
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
            logger.info(
                "No `generator` or `seed` was provided."
                f" Will use default `np.random.default_rng({seed})`"
            )

        self._rng = generator or np.random.default_rng(seed)

    def _parse_priors(self, priors_input_format: dict):
        """
        Internal method to do the actual prior setup.

        Order of checks:
            0. If the old API is used (no `stats` in the name or
               a callable as a value, raise ValueError exception.
            1. If prior is not a dictionary, consider it a delta prior.
            2. Otherwise, raise ValueError if format is not consistent.
        """
        self.priors = {}

        for parameter_name, parameter_prior in priors_input_format.items():
            if callable(parameter_prior):
                raise ValueError(
                    f"Attempted to use bare callable prior for `{parameter_name}`"
                    " Please, make sure your priors are either"
                    " a `stats` distribution or the *name* of a"
                    " function decorated with `@custom_prior`."
                )

            if not isinstance(parameter_prior, Mapping):
                # If not dictionary, then return as is (delta prior)
                self.priors[parameter_name] = (
                    lambda size, val=parameter_prior: np.repeat(val, repeats=size)
                )
                continue

            if len(parameter_prior) > 1:
                raise ValueError(
                    f"{parameter_prior} does not follow"
                    " the required format to specify a prior"
                    " as the inner dictionary contains more than one key."
                    " Please, specify a parameter's distribution as"
                    " `{'parameter_name': {'parameter_distro': {**distro_kwargs}}}`."
                )

            distribution = next(iter(parameter_prior))
            if not isinstance(distribution, str):
                raise ValueError(
                    "Prior distribution's key should be a string."
                    f" Got {type(distribution)}`"
                )

            if distribution in _pyfstat_custom_priors:
                logger.debug(
                    f"Setting {distribution} distribution from `_pyfstat_custom_priors` "
                    f"with kwargs `{parameter_prior[distribution]}` "
                    f"to parameter {parameter_name}"
                )
                custom_function = _pyfstat_custom_priors[distribution]
                self.priors[parameter_name] = functools.partial(
                    custom_function,
                    generator=self._rng,
                    **parameter_prior[distribution],
                )
            elif "stats." in distribution:
                _, stats_function = distribution.split(".")
                logger.debug(
                    f"Setting {stats_function} distribution from `scipy.stats` "
                    f"with kwargs `{parameter_prior[distribution]}` "
                    f"to parameter {parameter_name}"
                )
                rv_object = getattr(stats, stats_function)(
                    **parameter_prior[distribution]
                )
                rv_object.random_state = self._rng
                self.priors[parameter_name] = rv_object.rvs
            else:
                raise ValueError(
                    f"Distribution `{distribution}` not found in "
                    "`_pyfstat_custom_priors` or `scipy.stats` namespace. "
                    "Current custom distributions are "
                    f"`{list(_pyfstat_custom_priors.keys())}`. "
                    "Please, make sure to follow the proper rules to specify "
                    "a prior distribution."
                )

    def draw(self) -> Dict:
        """Draw a single multi-dimensional parameter space point from the given priors.

        Returns
        -------
        parameters: Dict
            Dictionary of parameter values (one value each).
            Each key corresponds to that found in ``self.priors``.
        """

        return {
            parameter_name: parameter_prior(size=1)[0]
            for parameter_name, parameter_prior in self.priors.items()
        }

    def draw_many(self, size) -> Dict:
        """Draw ``size`` multi-dimensional parameter space points from the given priors.

        Parameters
        ----------
        size:
            Number of samples to return.

        Returns
        -------
        parameters: Dict
            Dictionary of arrays. Each key corresponds to that found in ``self.priors``.
            Values are numpy arrays of shape ``size`` as returned by their corresponding method.
        """
        return {
            parameter_name: parameter_prior(size=size)
            for parameter_name, parameter_prior in self.priors.items()
        }


class AllSkyInjectionParametersGenerator(InjectionParametersGenerator):
    """
    Draw injection parameter samples from priors and return in dictionary format.
    This class works in exactly the same way as ``InjectionParametersGenerator``,
    but including by default two extra keys, ``Alpha`` and ``Delta`` (sky position's
    right ascension and declination in radians), which are sample isotropically
    across the celestial sphere.

    `Alpha`'s distribution is Uniform(0, 2 pi), and
    `sin(Delta)`'s distribution is Uniform(-1, 1).
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
                    f"Found input key `{key}` with value {priors[key]}.\n"
                    "Overwriting to produce uniform samples across the sky."
                )

        super().__init__(
            priors={**priors, **sky_priors}, seed=seed, generator=generator
        )
