from ._version import get_versions
from .logging import _get_default_logger, set_up_logger

__version__ = get_versions()["version"]
del get_versions

try:
    logger = _get_default_logger()
    logger.info(f"Running PyFstat version {__version__}")
except Exception as e:  # pragma: no cover
    print(
        f"Logging setup failed with exception: {e}\n"
        "Proceeding without default logging."
    )

from . import _version
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
from .injection_parameters import (
    AllSkyInjectionParametersGenerator,
    InjectionParametersGenerator,
    isotropic_amplitude_distribution,
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

__version__ = _version.get_versions()["version"]
