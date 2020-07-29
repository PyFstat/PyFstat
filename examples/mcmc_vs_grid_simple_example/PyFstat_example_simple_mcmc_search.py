#!/usr/bin/env python

import pyfstat
import os
import numpy as np

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

theta_prior = {
    "F0": {"type": "unif", "lower": F0 - DeltaF0 / 2.0, "upper": F0 + DeltaF0 / 2.0},
    "F1": {"type": "unif", "lower": F1 - DeltaF1 / 2.0, "upper": F1 + DeltaF1 / 2.0},
    "F2": F2,
    "Alpha": Alpha,
    "Delta": Delta,
}

ntemps = 2
log10beta_min = -1
nwalkers = 100
nsteps = [200, 200]  # [burnin,production]

mcmc = pyfstat.MCMCSearch(
    label="mcmc_search",
    outdir=outdir,
    sftfilepattern=os.path.join(outdir, "*simulated_signal*sft"),
    theta_prior=theta_prior,
    tref=tref,
    minStartTime=minStartTime,
    maxStartTime=maxStartTime,
    nsteps=nsteps,
    nwalkers=nwalkers,
    ntemps=ntemps,
    log10beta_min=log10beta_min,
)
mcmc.run()
mcmc.plot_corner()
mcmc.print_summary()
mcmc.plot_prior_posterior()
