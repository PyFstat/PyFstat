"""Generate injection parameters drawn from different prior populations"""

import logging
import numpy as np
import functools


class InjectionParametersGenerator:
    """
    Draw injection parameter samples from priors and return in dictionary format.
    """

    def __init__(self, priors=None, seed=None):
        """
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

        seed:
            Argument to be fed to numpy.random.default_rng,
            with all of its accepted types.
        """
        self.set_seed(seed)
        self.set_priors(priors or {})

    def set_priors(self, new_priors):
        """Set priors to draw parameter space points from.

        Parameters
        ----------
        new_priors: dict
            The new set of priors to update the object with.
        """
        if type(new_priors) is not dict:
            raise ValueError(
                "new_priors is not a dict type.\nPlease, check "
                "this class' docstring to learn about the expected format: "
                f"{self.__init__.__doc__}"
            )

        self.priors = {}
        self._update_priors(new_priors)

    def set_seed(self, seed):
        """Set the random seed for subsequent draws.

        Parameters
        ----------
        seed:
            Argument to be fed to numpy.random.default_rng,
            with all of its accepted types.
        """
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)

    def _update_priors(self, new_priors):
        """Internal method to do the actual prior setup."""
        for parameter_name, parameter_prior in new_priors.items():
            if callable(parameter_prior):
                self.priors[parameter_name] = parameter_prior
            elif isinstance(parameter_prior, dict):
                rng_function_name = next(iter(parameter_prior))
                rng_function = getattr(self._rng, rng_function_name)
                rng_kwargs = parameter_prior[rng_function_name]
                self.priors[parameter_name] = functools.partial(
                    rng_function, **rng_kwargs
                )
            else:  # Assume it is something to be returned as is
                self.priors[parameter_name] = functools.partial(
                    lambda x: x, parameter_prior
                )

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

    def __call__(self):
        return self.draw()


class AllSkyInjectionParametersGenerator(InjectionParametersGenerator):
    """Like InjectionParametersGenerator, but with hardcoded all-sky priors.

    This ensures uniform coverage of the 2D celestial sphere:
    uniform distribution in `Alpha` and sine distribution for `Delta`.

    It assumes 1) PyFstat notation and 2) equatorial coordinates.

    `Alpha` and `Delta` are given 'restricted' status to stop the user from
    changing them as long as using this special class.
    """

    def set_priors(self, new_priors):
        """Set priors to draw parameter space points from.

        Parameters
        ----------
        new_priors: dict
            The new set of priors to update the object with.
        """
        self._check_if_updating_sky_priors(new_priors)
        super().set_priors({**new_priors, **self.restricted_priors})

    def set_seed(self, seed):
        """Set the random seed for subsequent draws.

        Parameters
        ----------
        seed:
            Argument to be fed to numpy.random.default_rng,
            with all of its accepted types.
        """
        super().set_seed(seed)
        self.restricted_priors = {
            # This is required because numpy has no arcsin distro
            "Alpha": lambda: self._rng.uniform(low=0.0, high=2 * np.pi),
            "Delta": lambda: np.arcsin(self._rng.uniform(low=-1.0, high=1.0)),
        }

    def _check_if_updating_sky_priors(self, new_priors):
        """Internal method to stop the user from changing the sky prior."""
        if any(
            restricted_key in new_priors
            for restricted_key in self.restricted_priors.keys()
        ):
            logging.warning(
                "Ignoring specified sky priors (Alpha, Delta)."
                "This class is explicitly coded to prevent that from happening. "
                "Please instead use InjectionParametersGenerator if that's really what you want to do."
            )
