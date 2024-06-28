"""
Randomly sampling parameter space points
========================================

Application of dedicated classes to sample software injection
parameters according to the specified parameter space priors.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from pyfstat import (
    AllSkyInjectionParametersGenerator,
    InjectionParametersGenerator,
    Writer,
    isotropic_amplitude_distribution,
    set_up_logger,
)

label = "PyFstatExampleInjectionParametersGenerator"
outdir = os.path.join("PyFstat_example_data", label)
logger = set_up_logger(label=label, outdir=outdir)

# Properties of the GW data
gw_data = {
    "sqrtSX": 1e-23,
    "tstart": 1000000000,
    "duration": 86400,
    "detectors": "H1,L1",
    "Band": 1,
    "Tsft": 1800,
}

logger.info("Drawing random signal parameters...")

# Draw random signal phase parameters.
# The AllSkyInjectionParametersGenerator covers [Alpha,Delta] priors automatically.
# The rest can be a mix of nontrivial prior distributions and fixed values.
phase_params_generator = AllSkyInjectionParametersGenerator(
    priors={
        "F0": {"stats.uniform": {"loc": 29.0, "scale": 2.0}},
        "F1": -1e-10,
        "F2": 0,
    },
    seed=23,
)
phase_parameters = phase_params_generator.draw()
phase_parameters["tref"] = gw_data["tstart"]

# Draw random signal amplitude parameters.
# Here we use the plain InjectionParametersGenerator class.
amplitude_params_generator = InjectionParametersGenerator(
    priors={
        "h0": {"stats.norm": {"loc": 1e-24, "scale": 1e-26}},
        **isotropic_amplitude_distribution,
    },
    seed=42,
)
amplitude_parameters = amplitude_params_generator.draw()

# Now we can pass the parameter dictionaries to the Writer class and make SFTs.
data = Writer(
    label=label,
    outdir=outdir,
    **gw_data,
    **phase_parameters,
    **amplitude_parameters,
)
data.make_data()

# Now we draw many phase parameters and check the sky distribution
Ndraws = 10000
phase_parameters = phase_params_generator.draw_many(size=Ndraws)
Alphas = phase_parameters["Alpha"]
Deltas = phase_parameters["Delta"]
plotfile = os.path.join(outdir, label + "_allsky.png")
logger.info(f"Plotting sky distribution of {Ndraws} points to file: {plotfile}")
plt.subplot(111, projection="aitoff")
plt.plot(Alphas - np.pi, Deltas, ".", markersize=1)
plt.savefig(plotfile, dpi=300)
plt.close()
plotfile = os.path.join(outdir, label + "_alpha_hist.png")
logger.info(f"Plotting Alpha distribution of {Ndraws} points to file: {plotfile}")
plt.hist(Alphas, 50)
plt.xlabel("Alpha")
plt.ylabel("draws")
plt.savefig(plotfile, dpi=100)
plt.close()
plotfile = os.path.join(outdir, label + "_delta_hist.png")
logger.info(f"Plotting Delta distribution of {Ndraws} points to file: {plotfile}")
plt.hist(Deltas, 50)
plt.xlabel("Delta")
plt.ylabel("draws")
plt.savefig(plotfile, dpi=100)
plt.close()
plotfile = os.path.join(outdir, label + "_sindelta_hist.png")
logger.info(f"Plotting sin(Delta) distribution of {Ndraws} points to file: {plotfile}")
plt.hist(np.sin(Deltas), 50)
plt.xlabel("sin(Delta)")
plt.ylabel("draws")
plt.savefig(plotfile, dpi=100)
plt.close()
