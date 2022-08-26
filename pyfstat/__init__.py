import logging

from ._version import get_versions
from .core import (
    BaseSearchClass,
    ComputeFstat,
    SearchForSignalWithJumps,
    SemiCoherentGlitchSearch,
    SemiCoherentSearch,
)
from .grid_based_searches import (
    DMoff_NO_SPIN,
    FrequencySlidingWindow,
    GridGlitchSearch,
    GridSearch,
    GridUniformPriorSearch,
    SliceGridSearch,
    TransientGridSearch,
)
from .gridcorner import gridcorner
from .helper_functions import set_up_logger
from .injection_parameters import (
    AllSkyInjectionParametersGenerator,
    InjectionParametersGenerator,
)
from .make_sfts import (
    BinaryModulatedWriter,
    FrequencyAmplitudeModulatedArtifactWriter,
    FrequencyModulatedArtifactWriter,
    GlitchWriter,
    LineWriter,
    Writer,
)
from .mcmc_based_searches import (
    MCMCFollowUpSearch,
    MCMCGlitchSearch,
    MCMCSearch,
    MCMCSemiCoherentSearch,
    MCMCTransientSearch,
)
from .snr import DetectorStates, SignalToNoiseRatio
from .tcw_fstat_map_funcs import pyTransientFstatMap

__version__ = get_versions()["version"]
del get_versions

# fallback logging setup
root_logger = logging.getLogger()
if len(root_logger.handlers) == 0:
    logger = set_up_logger()
    logger.info(f"Running PyFstat version {__version__}")
    logger.warning(
        "No logging handler found."
        " We've set up some nicely formatted stdout logging at INFO level."
        " Feel free to override from the calling application level if you don't like it,"
        " and we'll respect that!"
    )
