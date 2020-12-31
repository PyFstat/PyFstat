"""
Randomly sampling parameter space points
========================================

Application of dedicated classes to sample software injection
parameters according to the specified parameter space priors.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pyfstat import (
    InjectionParametersGenerator,
    AllSkyInjectionParametersGenerator,
    Writer,
)

label = "PyFstat_example_InjectionParametersGenerator"
outdir = os.path.join("PyFstat_example_data", label)

# Properties of the GW data
gw_data = {
    "sqrtSX": 1e-23,
    "tstart": 1000000000,
    "duration": 86400,
    "detectors": "H1,L1",
    "Band": 1,
    "Tsft": 1800,
}

print("Drawing random signal parameters...")

# Draw random signal phase parameters.
# The AllSkyInjectionParametersGenerator covers [Alpha,Delta] priors automatically.
# The rest can be a mix of nontrivial prior distributions and fixed values.
phase_params_generator = AllSkyInjectionParametersGenerator(
    priors={
        "F0": {"uniform": {"low": 29.0, "high": 31.0}},
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
        "h0": {"normal": {"loc": 1e-24, "scale": 1e-24}},
        "cosi": {"uniform": {"low": 0.0, "high": 1.0}},
        "phi": {"uniform": {"low": 0.0, "high": 2 * np.pi}},
        "psi": {"uniform": {"low": 0.0, "high": np.pi}},
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
phase_parameters = [phase_params_generator.draw() for n in range(Ndraws)]
Alphas = np.array([p["Alpha"] for p in phase_parameters])
Deltas = np.array([p["Delta"] for p in phase_parameters])
plotfile = os.path.join(outdir, label + "_allsky.png")
print(f"Plotting sky distribution of {Ndraws} points to file: {plotfile}")
plt.subplot(111, projection="aitoff")
plt.plot(Alphas - np.pi, Deltas, ".", markersize=1)
plt.savefig(plotfile, dpi=300)
plt.close()
plotfile = os.path.join(outdir, label + "_alpha_hist.png")
print(f"Plotting Alpha distribution of {Ndraws} points to file: {plotfile}")
plt.hist(Alphas, 50)
plt.xlabel("Alpha")
plt.ylabel("draws")
plt.savefig(plotfile, dpi=100)
plt.close()
plotfile = os.path.join(outdir, label + "_delta_hist.png")
print(f"Plotting Delta distribution of {Ndraws} points to file: {plotfile}")
plt.hist(Deltas, 50)
plt.xlabel("Delta")
plt.ylabel("draws")
plt.savefig(plotfile, dpi=100)
plt.close()
plotfile = os.path.join(outdir, label + "_sindelta_hist.png")
print(f"Plotting sin(Delta) distribution of {Ndraws} points to file: {plotfile}")
plt.hist(np.sin(Deltas), 50)
plt.xlabel("sin(Delta)")
plt.ylabel("draws")
plt.savefig(plotfile, dpi=100)
plt.close()
