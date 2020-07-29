#!/usr/bin/env python

import pyfstat
import os
import numpy as np

try:
    from gridcorner import gridcorner
except ImportError:
    raise ImportError(
        "Python module 'gridcorner' not found, please install from "
        "https://gitlab.aei.uni-hannover.de/GregAshton/gridcorner"
    )

outdir = os.path.join("PyFstat_example_data", "PyFstat_example_simple_mcmc_vs_grid")
if not os.path.isdir(outdir) or not np.any(
    [f.endswith(".sft") for f in os.listdir(outdir)]
):
    raise RuntimeError(
        "Please first run PyFstat_example_make_data_for_simple_mcmc_vs_grid.py !"
    )

F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = 0.5
Delta = 1

minStartTime = 1000000000
duration = 2 * 86400
maxStartTime = minStartTime + duration
tref = minStartTime

Tsft = 1800

m = 0.001
dF0 = np.sqrt(12 * m) / (np.pi * duration)
dF1 = np.sqrt(180 * m) / (np.pi * duration ** 2)
DeltaF0 = 500 * dF0
DeltaF1 = 200 * dF1
F0s = [F0 - DeltaF0 / 2.0, F0 + DeltaF0 / 2.0, dF0]
F1s = [F1 - DeltaF1 / 2.0, F1 + DeltaF1 / 2.0, dF1]
F2s = [F2]
Alphas = [Alpha]
Deltas = [Delta]

print("Standard grid search:")
search = pyfstat.GridSearch(
    label="grid_search",
    outdir=outdir,
    sftfilepattern=os.path.join(outdir, "*simulated_signal*sft"),
    F0s=F0s,
    F1s=F1s,
    F2s=F2s,
    Alphas=Alphas,
    Deltas=Deltas,
    tref=tref,
    minStartTime=minStartTime,
    maxStartTime=maxStartTime,
    BSGL=False,
)
search.run()
search.print_max_twoF()
search.save_array_to_disk(search.data)

print("Plotting 2F(F0)...")
search.plot_1D(xkey="F0", xlabel="freq [Hz]", ylabel="$2\mathcal{F}$")

print("Making F0-F1 corner plot of 2F...")
F0_vals = np.unique(search.data[:, 2]) - F0
F1_vals = np.unique(search.data[:, 3]) - F1
twoF = search.data[:, -1].reshape((len(F0_vals), len(F1_vals)))
xyz = [F0_vals, F1_vals]
labels = [
    "$f - f_0$",
    "$\dot{f} - \dot{f}_0$",
    "$\widetilde{2\mathcal{F}}$",
]
fig, axes = gridcorner(
    twoF, xyz, projection="log_mean", labels=labels, whspace=0.1, factor=1.8
)
fig.savefig(os.path.join(outdir, search.label + "_corner.png"))
